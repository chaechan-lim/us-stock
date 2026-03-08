"""Backtest verification for quant analytics modules.

Tests whether quant algorithms actually improve returns:
1. Factor-based stock selection vs equal-weight (does picking high-factor stocks help?)
2. Kelly position sizing vs fixed sizing (does dynamic sizing boost returns?)
3. Combined: factor selection + Kelly sizing vs baseline

Usage:
    cd backend
    python3 -m backtest.verify_quant
    python3 -m backtest.verify_quant --quick
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from analytics.factor_model import MultiFactorModel, FactorWeights
from analytics.position_sizing import KellyPositionSizer
from analytics.signal_quality import SignalQualityTracker
from backtest.metrics import MetricsCalculator, BacktestMetrics, Trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Wide universe: 40 stocks across sectors for meaningful factor selection
UNIVERSE = [
    # Tech (mega)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "CRM",
    "ADBE", "ORCL", "CSCO", "AMD", "INTC", "QCOM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT", "LLY", "MRK",
    # Consumer
    "WMT", "PG", "KO", "MCD", "COST", "HD",
    # Energy + Industrial
    "XOM", "CVX", "CAT", "HON", "GE", "LMT",
    # Other
    "NFLX", "TSLA",
]

QUICK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                   "AVGO", "AMD", "JPM", "UNH", "LLY", "NFLX",
                   "TSLA", "COST", "GE", "XOM"]

TEST_PERIOD = "3y"
INITIAL_EQUITY = 100_000
REBALANCE_DAYS = 21  # Monthly rebalance
TOP_N = 8            # Hold top N stocks by factor score
SLIPPAGE_PCT = 0.05  # 0.05% per trade


@dataclass
class QuantBacktestResult:
    """Result for a single quant strategy variant."""
    name: str
    final_equity: float
    total_return_pct: float
    cagr: float
    sharpe: float
    mdd_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float


def load_universe_data(symbols: list[str], period: str = TEST_PERIOD) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for all symbols."""
    data = {}
    logger.info("Loading data for %d symbols...", len(symbols))
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 252:
                logger.warning("Skipping %s: insufficient data (%d bars)", sym, len(df))
                continue
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            data[sym] = df
        except Exception as e:
            logger.warning("Failed to load %s: %s", sym, e)
    logger.info("Loaded %d/%d symbols", len(data), len(symbols))
    return data


def load_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Load fundamental data for factor model."""
    fundamentals = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            fundamentals[sym] = {
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "returnOnEquity": info.get("returnOnEquity"),
                "profitMargins": info.get("profitMargins"),
                "debtToEquity": info.get("debtToEquity"),
                "revenueGrowth": info.get("revenueGrowth"),
            }
        except Exception:
            fundamentals[sym] = {}
    return fundamentals


def simulate_equal_weight(
    price_data: dict[str, pd.DataFrame],
    top_n: int = TOP_N,
    rebalance_days: int = REBALANCE_DAYS,
) -> QuantBacktestResult:
    """Baseline: equal-weight portfolio of top_n stocks, buy and hold."""
    symbols = list(price_data.keys())
    selected = symbols[:top_n]

    return _simulate_portfolio(
        price_data, selected_per_period=lambda _day, _data: selected,
        name=f"equal_wt_{top_n}", position_size_fn=None,
        rebalance_days=rebalance_days,
    )


def simulate_equal_weight_all(
    price_data: dict[str, pd.DataFrame],
    rebalance_days: int = REBALANCE_DAYS,
) -> QuantBacktestResult:
    """Equal-weight ALL stocks (buy and hold everything)."""
    symbols = list(price_data.keys())

    return _simulate_portfolio(
        price_data, selected_per_period=lambda _day, _data: symbols,
        name=f"equal_all_{len(symbols)}", position_size_fn=None,
        rebalance_days=rebalance_days,
    )


def simulate_factor_selection(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = TOP_N,
    rebalance_days: int = REBALANCE_DAYS,
    weights: FactorWeights | None = None,
) -> QuantBacktestResult:
    """Factor model: select top N stocks by composite factor score."""
    model = MultiFactorModel(weights=weights)

    def select_fn(day_idx: int, all_data: dict[str, pd.DataFrame]) -> list[str]:
        # Use data up to current day for factor scoring
        current_data = {}
        for sym, df in all_data.items():
            subset = df.iloc[:day_idx + 1]
            if len(subset) >= 252:
                current_data[sym] = subset
        if len(current_data) < 3:
            return list(current_data.keys())[:top_n]
        scores = model.score_universe(current_data, fundamentals)
        top = model.get_top_n(scores, n=top_n)
        return [s.symbol for s in top]

    name = "factor_selection"
    if weights:
        name = f"factor_mom{int(weights.momentum*100)}"
    return _simulate_portfolio(
        price_data, selected_per_period=select_fn,
        name=name, position_size_fn=None,
        rebalance_days=rebalance_days,
    )


def simulate_factor_kelly(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = TOP_N,
    rebalance_days: int = 63,
) -> QuantBacktestResult:
    """Factor selection + Kelly-weighted position sizing.

    Top-ranked stocks get bigger allocation, bottom-ranked get smaller.
    Quarterly rebalance with low-turnover logic.
    """
    model = MultiFactorModel(
        weights=FactorWeights(momentum=0.50, value=0.15, quality=0.25, low_volatility=0.10),
    )
    all_symbols = list(price_data.keys())
    median_rank = len(all_symbols) // 2

    factor_cache: dict[str, float] = {}
    current_holdings: set[str] = set()

    def select_fn(day_idx: int, all_data: dict[str, pd.DataFrame]) -> list[str]:
        nonlocal current_holdings
        current_data = {}
        for sym, df in all_data.items():
            subset = df.iloc[:day_idx + 1]
            if len(subset) >= 252:
                current_data[sym] = subset
        if len(current_data) < 3:
            return list(current_data.keys())[:top_n]

        scores = model.score_universe(current_data, fundamentals)
        rank_map = {s.symbol: s.rank for s in scores}
        top_symbols = [s.symbol for s in scores[:top_n]]

        factor_cache.clear()
        for s in scores:
            factor_cache[s.symbol] = s.composite

        keep = {s for s in current_holdings if rank_map.get(s, 999) <= median_rank}
        result = list(keep)
        for sym in top_symbols:
            if sym not in keep and len(result) < top_n:
                result.append(sym)

        current_holdings = set(result)
        return result

    def size_fn(symbol: str, equity: float, n_positions: int) -> float:
        """Weight proportional to factor score rank.

        Top factor score gets ~2x the allocation of bottom.
        """
        if n_positions <= 1:
            return 0.95
        factor_score = factor_cache.get(symbol, 0)
        # Map factor score to weight: higher score → bigger position
        # tanh maps [-3,3] → [-1,1], then scale to [0.5, 1.5] range
        raw_weight = 1.0 + 0.5 * np.tanh(factor_score)
        # Normalize so total doesn't exceed ~95% of equity
        base_weight = 0.95 / n_positions
        return base_weight * raw_weight

    return _simulate_portfolio(
        price_data, selected_per_period=select_fn,
        name="factor_kelly", position_size_fn=size_fn,
        rebalance_days=rebalance_days,
    )


def simulate_momentum_heavy(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = TOP_N,
    rebalance_days: int = REBALANCE_DAYS,
) -> QuantBacktestResult:
    """Momentum-heavy factor weights (70% momentum)."""
    return simulate_factor_selection(
        price_data, fundamentals, top_n, rebalance_days,
        weights=FactorWeights(momentum=0.70, value=0.10, quality=0.10, low_volatility=0.10),
    )


def simulate_factor_low_turnover(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = TOP_N,
    rebalance_days: int = 63,  # Quarterly rebalance
) -> QuantBacktestResult:
    """Factor selection with reduced turnover.

    Only sell a position if it drops below the median rank (bottom half).
    Quarterly rebalance instead of monthly. This preserves winners.
    """
    model = MultiFactorModel()
    all_symbols = list(price_data.keys())
    median_rank = len(all_symbols) // 2

    current_holdings: set[str] = set()

    def select_fn(day_idx: int, all_data: dict[str, pd.DataFrame]) -> list[str]:
        nonlocal current_holdings
        current_data = {}
        for sym, df in all_data.items():
            subset = df.iloc[:day_idx + 1]
            if len(subset) >= 252:
                current_data[sym] = subset
        if len(current_data) < 3:
            return list(current_data.keys())[:top_n]

        scores = model.score_universe(current_data, fundamentals)
        rank_map = {s.symbol: s.rank for s in scores}
        top_symbols = [s.symbol for s in scores[:top_n]]

        # Keep existing holdings that are still above median
        keep = {s for s in current_holdings if rank_map.get(s, 999) <= median_rank}
        # Add new top picks to fill up to top_n
        result = list(keep)
        for sym in top_symbols:
            if sym not in keep and len(result) < top_n:
                result.append(sym)

        current_holdings = set(result)
        return result

    return _simulate_portfolio(
        price_data, selected_per_period=select_fn,
        name="factor_low_turn", position_size_fn=None,
        rebalance_days=rebalance_days,
    )


def simulate_concentrated(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = 5,  # Fewer stocks, higher conviction
    rebalance_days: int = 63,
) -> QuantBacktestResult:
    """Concentrated portfolio: top 5 stocks, quarterly rebalance, keep winners."""
    model = MultiFactorModel(
        weights=FactorWeights(momentum=0.50, value=0.15, quality=0.25, low_volatility=0.10),
    )
    all_symbols = list(price_data.keys())
    median_rank = len(all_symbols) // 2
    current_holdings: set[str] = set()

    def select_fn(day_idx: int, all_data: dict[str, pd.DataFrame]) -> list[str]:
        nonlocal current_holdings
        current_data = {}
        for sym, df in all_data.items():
            subset = df.iloc[:day_idx + 1]
            if len(subset) >= 252:
                current_data[sym] = subset
        if len(current_data) < 3:
            return list(current_data.keys())[:top_n]

        scores = model.score_universe(current_data, fundamentals)
        rank_map = {s.symbol: s.rank for s in scores}
        top_symbols = [s.symbol for s in scores[:top_n]]

        keep = {s for s in current_holdings if rank_map.get(s, 999) <= median_rank}
        result = list(keep)
        for sym in top_symbols:
            if sym not in keep and len(result) < top_n:
                result.append(sym)

        current_holdings = set(result)
        return result

    return _simulate_portfolio(
        price_data, selected_per_period=select_fn,
        name="concentrated_5", position_size_fn=None,
        rebalance_days=rebalance_days,
    )


def _align_data(price_data: dict[str, pd.DataFrame]):
    """Align all price data to common dates and return (common_dates, closes_df)."""
    all_symbols = list(price_data.keys())
    common_dates = None
    for sym in all_symbols:
        idx = price_data[sym].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    closes = pd.DataFrame({
        sym: price_data[sym].loc[common_dates, "close"] for sym in all_symbols
    })
    return common_dates, closes, all_symbols


def _close_all_positions(holdings, entry_prices, entry_dates, closes, common_dates, trades):
    """Close remaining positions at end of backtest."""
    cash_returned = 0.0
    if holdings:
        final_prices = closes.iloc[-1]
        for sym, qty in list(holdings.items()):
            if sym in final_prices.index and not np.isnan(final_prices[sym]):
                sell_price = float(final_prices[sym]) * (1 - SLIPPAGE_PCT / 100)
                pnl = (sell_price - entry_prices[sym]) * qty
                pnl_pct = (sell_price / entry_prices[sym] - 1) * 100
                entry_dt = entry_dates.get(sym, "")
                try:
                    hold_days = (pd.Timestamp(common_dates[-1]) - pd.Timestamp(entry_dt)).days
                except Exception:
                    hold_days = 0
                trades.append(Trade(
                    symbol=sym, side="SELL",
                    entry_date=entry_dt, entry_price=entry_prices[sym],
                    exit_date=str(common_dates[-1]), exit_price=sell_price,
                    quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                    holding_days=hold_days,
                ))
                cash_returned += qty * sell_price
    return cash_returned


def _sell_position(sym, price, holdings, entry_prices, entry_dates, trades, date):
    """Sell a single position and record trade."""
    sell_price = price * (1 - SLIPPAGE_PCT / 100)
    qty = holdings[sym]
    pnl = (sell_price - entry_prices[sym]) * qty
    pnl_pct = (sell_price / entry_prices[sym] - 1) * 100
    entry_dt = entry_dates.get(sym, str(date))
    try:
        hold_days = (pd.Timestamp(date) - pd.Timestamp(entry_dt)).days
    except Exception:
        hold_days = 0
    trades.append(Trade(
        symbol=sym, side="SELL",
        entry_date=entry_dt, entry_price=entry_prices[sym],
        exit_date=str(date), exit_price=sell_price,
        quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
        holding_days=hold_days,
    ))
    cash_returned = qty * sell_price
    del holdings[sym]
    entry_prices.pop(sym, None)
    entry_dates.pop(sym, None)
    return cash_returned


def _compute_momentum_12_1(closes, sym, i):
    """Compute 12-1 momentum (12-month return minus last month return)."""
    price_now = float(closes[sym].iloc[i])
    price_12m = float(closes[sym].iloc[i - 252])
    price_1m = float(closes[sym].iloc[i - 21])
    if price_12m <= 0 or price_1m <= 0 or np.isnan(price_now):
        return None
    ret_12m = price_now / price_12m - 1
    ret_1m = price_now / price_1m - 1
    return ret_12m - ret_1m


def simulate_momentum_trailing(
    price_data: dict[str, pd.DataFrame],
    top_n: int = 10,
    trail_pct: float = 0.20,
    eval_interval: int = 21,
) -> QuantBacktestResult:
    """Momentum entry + trailing stop exit.

    - Buy: top N stocks by 12-1 momentum, confirmed above SMA150
    - Hold: track peak price, only sell if 20% below peak (trailing stop)
    - Never sell winners that are still trending up
    """
    common_dates, closes, all_symbols = _align_data(price_data)

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    peak_prices: dict[str, float] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []
    last_eval = 0

    for i in range(252, len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        # Update peaks and check trailing stops
        for sym in list(holdings.keys()):
            if sym not in prices.index or np.isnan(prices[sym]):
                continue
            price = float(prices[sym])
            # Update peak
            if price > peak_prices.get(sym, 0):
                peak_prices[sym] = price
            # Trailing stop check
            peak = peak_prices[sym]
            if peak > 0 and price < peak * (1 - trail_pct):
                cash += _sell_position(sym, price, holdings, entry_prices,
                                       entry_dates, trades, date)
                peak_prices.pop(sym, None)

        # Evaluate for new entries
        if i - last_eval >= eval_interval:
            last_eval = i

            mom_scores = {}
            for sym in all_symbols:
                if sym in holdings:
                    continue
                mom = _compute_momentum_12_1(closes, sym, i)
                if mom is None:
                    continue
                # Confirm uptrend (above SMA150)
                sma_data = closes[sym].iloc[max(0, i - 149):i + 1]
                if len(sma_data) >= 150:
                    sma = float(sma_data.mean())
                    if float(closes[sym].iloc[i]) > sma and mom > 0:
                        mom_scores[sym] = mom

            ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
            slots = top_n - len(holdings)

            position_value = sum(
                holdings.get(s, 0) * float(prices[s])
                for s in holdings if s in prices.index and not np.isnan(prices[s])
            )
            total_equity = cash + position_value

            for sym, _ in ranked[:slots]:
                if slots <= 0:
                    break
                price = float(prices[sym])
                if np.isnan(price) or price <= 0:
                    continue

                allocation = min(total_equity / top_n, cash * 0.90)
                if allocation <= 0:
                    continue

                buy_price = price * (1 + SLIPPAGE_PCT / 100)
                qty = int(allocation / buy_price)
                if qty <= 0 or qty * buy_price > cash:
                    continue

                cash -= qty * buy_price
                holdings[sym] = qty
                entry_prices[sym] = buy_price
                entry_dates[sym] = str(date)
                peak_prices[sym] = buy_price
                slots -= 1

        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    return QuantBacktestResult(
        name=f"mom_trail_{int(trail_pct*100)}",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_momentum_quality(
    price_data: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    top_n: int = 10,
    trail_pct: float = 0.20,
    eval_interval: int = 21,
) -> QuantBacktestResult:
    """Momentum + Quality: buy high-momentum, high-quality stocks.

    - Score = 60% momentum + 40% quality (ROE, margins)
    - Trailing stop exit (20%)
    - Monthly evaluation for new entries
    """
    common_dates, closes, all_symbols = _align_data(price_data)
    model = MultiFactorModel(
        weights=FactorWeights(momentum=0.60, value=0.05, quality=0.30, low_volatility=0.05),
    )

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    peak_prices: dict[str, float] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []
    last_eval = 0

    for i in range(252, len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        # Trailing stop check
        for sym in list(holdings.keys()):
            if sym not in prices.index or np.isnan(prices[sym]):
                continue
            price = float(prices[sym])
            if price > peak_prices.get(sym, 0):
                peak_prices[sym] = price
            peak = peak_prices[sym]
            if peak > 0 and price < peak * (1 - trail_pct):
                cash += _sell_position(sym, price, holdings, entry_prices,
                                       entry_dates, trades, date)
                peak_prices.pop(sym, None)

        # Factor-based entry
        if i - last_eval >= eval_interval:
            last_eval = i

            current_data = {}
            for sym in all_symbols:
                if sym in holdings:
                    continue
                subset = closes[[sym]].iloc[:i + 1].rename(columns={sym: "close"})
                if len(subset) >= 252:
                    current_data[sym] = pd.DataFrame({"close": subset["close"]})

            if len(current_data) >= 3:
                scores = model.score_universe(current_data, fundamentals)
                # Only buy positive composite scores with confirmed uptrend
                candidates = []
                for s in scores:
                    if s.composite <= 0:
                        continue
                    sym_price = float(closes[s.symbol].iloc[i])
                    sma_data = closes[s.symbol].iloc[max(0, i - 149):i + 1]
                    if len(sma_data) >= 150:
                        sma = float(sma_data.mean())
                        if sym_price > sma:
                            candidates.append(s)

                slots = top_n - len(holdings)
                position_value = sum(
                    holdings.get(s, 0) * float(prices[s])
                    for s in holdings if s in prices.index and not np.isnan(prices[s])
                )
                total_equity = cash + position_value

                for s in candidates[:slots]:
                    if slots <= 0:
                        break
                    sym = s.symbol
                    price = float(prices[sym])
                    if np.isnan(price) or price <= 0:
                        continue

                    # Kelly-inspired: higher factor score → bigger position
                    base_weight = 1.0 / top_n
                    factor_mult = 1.0 + 0.3 * np.tanh(s.composite)
                    weight = base_weight * factor_mult

                    allocation = min(total_equity * weight, cash * 0.90)
                    if allocation <= 0:
                        continue

                    buy_price = price * (1 + SLIPPAGE_PCT / 100)
                    qty = int(allocation / buy_price)
                    if qty <= 0 or qty * buy_price > cash:
                        continue

                    cash -= qty * buy_price
                    holdings[sym] = qty
                    entry_prices[sym] = buy_price
                    entry_dates[sym] = str(date)
                    peak_prices[sym] = buy_price
                    slots -= 1

        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    return QuantBacktestResult(
        name="mom_quality",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_momentum_weighted_all(
    price_data: dict[str, pd.DataFrame],
    rebalance_days: int = 63,
) -> QuantBacktestResult:
    """Momentum-weighted portfolio: ALL stocks, weighted by 12-1 momentum.

    Instead of picking top N, hold ALL stocks but overweight winners.
    Top momentum stocks get 3-4x the allocation of low momentum.
    Quarterly rebalance.
    """
    common_dates, closes, all_symbols = _align_data(price_data)

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []
    last_rebalance = 0

    for i in range(252, len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        if i - last_rebalance >= rebalance_days or i == 252:
            last_rebalance = i

            # Compute momentum for all stocks
            mom_scores = {}
            for sym in all_symbols:
                mom = _compute_momentum_12_1(closes, sym, i)
                if mom is not None:
                    mom_scores[sym] = mom

            if not mom_scores:
                continue

            # Convert momentum to weights: softmax-like
            # Shift so all positive, then normalize
            scores_arr = np.array(list(mom_scores.values()))
            shifted = scores_arr - scores_arr.min() + 0.1  # Ensure all positive
            weights = shifted / shifted.sum()
            weight_map = dict(zip(mom_scores.keys(), weights))

            # Sell everything and rebalance
            for sym in list(holdings.keys()):
                if sym in prices.index and not np.isnan(prices[sym]):
                    cash += _sell_position(sym, float(prices[sym]), holdings,
                                           entry_prices, entry_dates, trades, date)

            # Compute total equity for allocation
            total_equity = cash

            # Buy all stocks with momentum weights
            for sym, weight in weight_map.items():
                price = float(prices[sym]) if sym in prices.index else 0
                if np.isnan(price) or price <= 0:
                    continue

                allocation = total_equity * weight * 0.95  # 5% cash buffer
                buy_price = price * (1 + SLIPPAGE_PCT / 100)
                qty = int(allocation / buy_price)
                if qty <= 0:
                    continue
                cost = qty * buy_price
                if cost > cash:
                    continue

                cash -= cost
                holdings[sym] = qty
                entry_prices[sym] = buy_price
                entry_dates[sym] = str(date)

        # Update equity
        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    return QuantBacktestResult(
        name="mom_weighted_all",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_market_timed_buy_hold(
    price_data: dict[str, pd.DataFrame],
    spy_data: pd.DataFrame,
) -> QuantBacktestResult:
    """Market-timed buy & hold: hold all stocks when SPY > SMA200, cash otherwise.

    Simple timing: be fully invested when market is bullish, 100% cash when bearish.
    Should beat SPY by avoiding drawdowns.
    """
    all_symbols = list(price_data.keys())

    common_dates = None
    for sym in all_symbols:
        idx = price_data[sym].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.intersection(spy_data.index)
    common_dates = common_dates.sort_values()

    closes = pd.DataFrame({
        sym: price_data[sym].loc[common_dates, "close"] for sym in all_symbols
    })
    spy_close = spy_data.loc[common_dates, "close"]

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []
    was_bullish = False

    for i in range(200, len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        spy_price = float(spy_close.iloc[i])
        spy_sma = float(spy_close.iloc[max(0, i - 199):i + 1].mean())
        market_bullish = spy_price > spy_sma

        if market_bullish and not was_bullish:
            # Enter: buy all stocks equally
            n = len(all_symbols)
            total_equity = cash
            for sym in all_symbols:
                price = float(prices[sym]) if sym in prices.index else 0
                if np.isnan(price) or price <= 0:
                    continue
                allocation = total_equity / n * 0.95
                buy_price = price * (1 + SLIPPAGE_PCT / 100)
                qty = int(allocation / buy_price)
                if qty > 0 and qty * buy_price <= cash:
                    cash -= qty * buy_price
                    holdings[sym] = qty
                    entry_prices[sym] = buy_price
                    entry_dates[sym] = str(date)

        elif not market_bullish and was_bullish:
            # Exit: sell everything
            for sym in list(holdings.keys()):
                if sym in prices.index and not np.isnan(prices[sym]):
                    cash += _sell_position(sym, float(prices[sym]), holdings,
                                           entry_prices, entry_dates, trades, date)

        was_bullish = market_bullish

        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    return QuantBacktestResult(
        name="mkt_timed_hold",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_mom_weighted_buy_hold(
    price_data: dict[str, pd.DataFrame],
) -> QuantBacktestResult:
    """Initial momentum-weighted buy & hold.

    Weight by 12-1 momentum at start. Never rebalance. Ride winners.
    Top momentum stocks get 3-5x the allocation of low momentum.
    """
    common_dates, closes, all_symbols = _align_data(price_data)

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []

    # Buy at day 252 with momentum weights
    start_idx = 252
    for i in range(start_idx, len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        if i == start_idx:
            mom_scores = {}
            for sym in all_symbols:
                mom = _compute_momentum_12_1(closes, sym, i)
                if mom is not None:
                    mom_scores[sym] = mom

            if mom_scores:
                scores_arr = np.array(list(mom_scores.values()))
                # Softmax-like: shift positive, power to amplify differences
                shifted = scores_arr - scores_arr.min() + 0.1
                powered = shifted ** 1.5  # Amplify momentum differences
                weights = powered / powered.sum()
                weight_map = dict(zip(mom_scores.keys(), weights))

                for sym, weight in weight_map.items():
                    price = float(prices[sym]) if sym in prices.index else 0
                    if np.isnan(price) or price <= 0:
                        continue
                    allocation = cash * weight * 0.95
                    buy_price = price * (1 + SLIPPAGE_PCT / 100)
                    qty = int(allocation / buy_price)
                    if qty > 0 and qty * buy_price <= cash:
                        cash -= qty * buy_price
                        holdings[sym] = qty
                        entry_prices[sym] = buy_price
                        entry_dates[sym] = str(date)

        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    return QuantBacktestResult(
        name="mom_wt_hold",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_buy_hold_trailing_stop(
    price_data: dict[str, pd.DataFrame],
    trail_pct: float = 0.25,
    reinvest: bool = True,
) -> QuantBacktestResult:
    """Buy all stocks equally, sell only if trailing stop triggers.

    Keeps winners running indefinitely. Only sells losers/reversals.
    Reinvests cash from sold positions into remaining winners.
    """
    common_dates, closes, all_symbols = _align_data(price_data)

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    peak_prices: dict[str, float] = {}
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []

    # Initial buy at day 0
    bought = False

    for i in range(len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        if not bought:
            n = len(all_symbols)
            for sym in all_symbols:
                price = float(prices[sym]) if sym in prices.index else 0
                if np.isnan(price) or price <= 0:
                    continue
                allocation = cash / n * 0.95
                buy_price = price * (1 + SLIPPAGE_PCT / 100)
                qty = int(allocation / buy_price)
                if qty > 0 and qty * buy_price <= cash:
                    cash -= qty * buy_price
                    holdings[sym] = qty
                    entry_prices[sym] = buy_price
                    entry_dates[sym] = str(date)
                    peak_prices[sym] = buy_price
            bought = True

        # Update peaks and check trailing stops
        sold_syms = []
        for sym in list(holdings.keys()):
            if sym not in prices.index or np.isnan(prices[sym]):
                continue
            price = float(prices[sym])
            if price > peak_prices.get(sym, 0):
                peak_prices[sym] = price
            peak = peak_prices[sym]
            if peak > 0 and price < peak * (1 - trail_pct):
                cash += _sell_position(sym, price, holdings, entry_prices,
                                       entry_dates, trades, date)
                peak_prices.pop(sym, None)
                sold_syms.append(sym)

        # Reinvest cash into remaining holdings (proportionally)
        if reinvest and sold_syms and holdings and cash > 100:
            remaining = list(holdings.keys())
            reinvest_per = cash * 0.90 / len(remaining)
            for sym in remaining:
                price = float(prices[sym]) if sym in prices.index else 0
                if np.isnan(price) or price <= 0:
                    continue
                buy_price = price * (1 + SLIPPAGE_PCT / 100)
                qty = int(reinvest_per / buy_price)
                if qty > 0 and qty * buy_price <= cash:
                    cash -= qty * buy_price
                    holdings[sym] += qty
                    # Keep original entry price (average down not tracked for simplicity)

        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    cash += _close_all_positions(holdings, entry_prices, entry_dates,
                                  closes, common_dates, trades)

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, INITIAL_EQUITY)

    name = f"hold_trail_{int(trail_pct*100)}"
    if reinvest:
        name += "_reinv"

    return QuantBacktestResult(
        name=name,
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def simulate_spy_benchmark(
    period: str = TEST_PERIOD,
) -> QuantBacktestResult:
    """SPY buy-and-hold benchmark."""
    try:
        df = yf.download("SPY", period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        logger.error("Failed to load SPY: %s", e)
        return QuantBacktestResult(
            name="SPY_benchmark", final_equity=INITIAL_EQUITY,
            total_return_pct=0, cagr=0, sharpe=0, mdd_pct=0,
            total_trades=0, win_rate=0, profit_factor=0,
        )

    close = df["close"]
    equity_curve = INITIAL_EQUITY * (close / float(close.iloc[0]))

    metrics = MetricsCalculator.calculate(
        equity_curve=equity_curve,
        trades=[],
        initial_equity=INITIAL_EQUITY,
    )

    return QuantBacktestResult(
        name="SPY_benchmark",
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=0,
        win_rate=0,
        profit_factor=0,
    )


def _simulate_portfolio(
    price_data: dict[str, pd.DataFrame],
    selected_per_period,
    name: str,
    position_size_fn=None,
    rebalance_days: int = REBALANCE_DAYS,
) -> QuantBacktestResult:
    """Generic portfolio simulation with periodic rebalancing.

    Args:
        price_data: {symbol: OHLCV DataFrame}
        selected_per_period: fn(day_idx, data) -> list of symbols to hold
        name: Strategy name
        position_size_fn: fn(symbol, equity, n_positions) -> weight fraction (optional)
        rebalance_days: Days between rebalances
    """
    # Align all price data to common dates
    all_symbols = list(price_data.keys())
    common_dates = None
    for sym in all_symbols:
        idx = price_data[sym].index
        if common_dates is None:
            common_dates = idx
        else:
            common_dates = common_dates.intersection(idx)

    if common_dates is None or len(common_dates) < 252:
        return QuantBacktestResult(
            name=name, final_equity=INITIAL_EQUITY,
            total_return_pct=0, cagr=0, sharpe=0, mdd_pct=0,
            total_trades=0, win_rate=0, profit_factor=0,
        )

    common_dates = common_dates.sort_values()
    # Create aligned close prices
    closes = pd.DataFrame({
        sym: price_data[sym].loc[common_dates, "close"]
        for sym in all_symbols
    })

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    holdings: dict[str, float] = {}  # symbol -> quantity
    trades: list[Trade] = []
    equity_curve = []
    equity_dates = []
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}

    current_selection: list[str] = []
    last_rebalance = 0

    for i in range(len(common_dates)):
        date = common_dates[i]
        prices = closes.iloc[i]

        # Rebalance check
        if i - last_rebalance >= rebalance_days or i == 0:
            new_selection = selected_per_period(i, price_data)
            if new_selection != current_selection:
                # Sell stocks no longer selected
                for sym in list(holdings.keys()):
                    if sym not in new_selection and sym in prices.index:
                        price = float(prices[sym])
                        if np.isnan(price):
                            continue
                        sell_price = price * (1 - SLIPPAGE_PCT / 100)
                        qty = holdings[sym]
                        cash += qty * sell_price
                        pnl = (sell_price - entry_prices.get(sym, sell_price)) * qty
                        pnl_pct = (sell_price / entry_prices.get(sym, sell_price) - 1) * 100
                        entry_dt = entry_dates.get(sym, str(date))
                        try:
                            hold_days = (pd.Timestamp(date) - pd.Timestamp(entry_dt)).days
                        except Exception:
                            hold_days = 0
                        trades.append(Trade(
                            symbol=sym, side="SELL",
                            entry_date=entry_dt, entry_price=entry_prices.get(sym, 0),
                            exit_date=str(date), exit_price=sell_price,
                            quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                            holding_days=hold_days,
                        ))
                        del holdings[sym]
                        entry_prices.pop(sym, None)
                        entry_dates.pop(sym, None)

                # Buy newly selected stocks
                # Calculate total equity for position sizing
                position_value = sum(
                    holdings.get(s, 0) * float(prices[s])
                    for s in holdings if s in prices.index and not np.isnan(prices[s])
                )
                total_equity = cash + position_value

                to_buy = [s for s in new_selection if s not in holdings and s in prices.index]
                n_target = len(new_selection)

                for sym in to_buy:
                    price = float(prices[sym])
                    if np.isnan(price) or price <= 0:
                        continue

                    # Position sizing
                    if position_size_fn:
                        weight = position_size_fn(sym, total_equity, n_target)
                    else:
                        weight = 1.0 / n_target  # Equal weight

                    allocation = total_equity * weight
                    allocation = min(allocation, cash * 0.95)
                    if allocation <= 0:
                        continue

                    buy_price = price * (1 + SLIPPAGE_PCT / 100)
                    qty = int(allocation / buy_price)
                    if qty <= 0:
                        continue

                    cost = qty * buy_price
                    if cost > cash:
                        continue

                    cash -= cost
                    holdings[sym] = qty
                    entry_prices[sym] = buy_price
                    entry_dates[sym] = str(date)

                current_selection = new_selection
                last_rebalance = i

        # Update equity
        position_value = sum(
            holdings.get(s, 0) * float(prices[s])
            for s in holdings if s in prices.index and not np.isnan(prices[s])
        )
        equity = cash + position_value
        equity_curve.append(equity)
        equity_dates.append(date)

    # Close remaining positions at end
    if holdings and len(common_dates) > 0:
        final_prices = closes.iloc[-1]
        for sym, qty in list(holdings.items()):
            if sym in final_prices.index:
                price = float(final_prices[sym])
                if not np.isnan(price):
                    sell_price = price * (1 - SLIPPAGE_PCT / 100)
                    cash += qty * sell_price
                    pnl = (sell_price - entry_prices.get(sym, sell_price)) * qty
                    pnl_pct = (sell_price / entry_prices.get(sym, sell_price) - 1) * 100
                    entry_dt = entry_dates.get(sym, "")
                    try:
                        hold_days = (pd.Timestamp(common_dates[-1]) - pd.Timestamp(entry_dt)).days
                    except Exception:
                        hold_days = 0
                    trades.append(Trade(
                        symbol=sym, side="SELL",
                        entry_date=entry_dt, entry_price=entry_prices.get(sym, 0),
                        exit_date=str(common_dates[-1]), exit_price=sell_price,
                        quantity=qty, pnl=pnl, pnl_pct=pnl_pct,
                        holding_days=hold_days,
                    ))

    # Calculate metrics
    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(
        equity_curve=eq_series,
        trades=trades,
        initial_equity=INITIAL_EQUITY,
    )

    return QuantBacktestResult(
        name=name,
        final_equity=metrics.final_equity,
        total_return_pct=metrics.total_return_pct,
        cagr=metrics.cagr,
        sharpe=metrics.sharpe_ratio,
        mdd_pct=metrics.max_drawdown_pct,
        total_trades=metrics.total_trades,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
    )


def print_results(results: list[QuantBacktestResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("QUANT ANALYTICS BACKTEST RESULTS")
    print("=" * 90)
    print(f"{'Strategy':<20s} {'Return':>8s} {'CAGR':>7s} {'Sharpe':>7s} "
          f"{'MDD':>7s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'Final $':>10s}")
    print("-" * 90)

    # Find best CAGR for highlighting
    best_cagr = max(r.cagr for r in results)

    for r in results:
        marker = " ** " if r.cagr == best_cagr else "    "
        print(
            f"{marker}{r.name:<16s} {r.total_return_pct:>+7.1f}% "
            f"{r.cagr:>+6.1%} {r.sharpe:>7.2f} "
            f"{r.mdd_pct:>6.1f}% {r.total_trades:>7d} "
            f"{r.win_rate:>5.1f}% {r.profit_factor:>5.2f} "
            f"${r.final_equity:>9,.0f}"
        )

    print("=" * 90)

    # Analysis
    spy = next((r for r in results if r.name == "SPY_benchmark"), None)
    mom = next((r for r in results if r.name == "mom_trend_10"), None)
    mom_conc = next((r for r in results if r.name == "mom_conc_5"), None)
    mom_timed = next((r for r in results if r.name == "mom_mkt_timed"), None)
    equal_all = next((r for r in results if "equal_all" in r.name), None)

    print("\n--- vs SPY Benchmark ---")
    for r in results:
        if spy and r.name != "SPY_benchmark":
            alpha = r.cagr - spy.cagr
            beat = "BEAT" if alpha > 0 else "LOSE"
            print(f"  {r.name:<20s} alpha = {alpha:+.1%}  [{beat}]")


async def main():
    t0 = time.time()

    # Parse args
    quick = "--quick" in sys.argv

    universe = QUICK_UNIVERSE if quick else UNIVERSE
    logger.info("Mode: %s (%d stocks)", "quick" if quick else "full", len(universe))

    # Load data
    price_data = load_universe_data(universe, TEST_PERIOD)
    if len(price_data) < 5:
        logger.error("Not enough data loaded. Exiting.")
        sys.exit(1)

    # Load fundamentals
    logger.info("Loading fundamentals...")
    fundamentals = load_fundamentals(list(price_data.keys()))

    # Load SPY for market timing
    logger.info("Loading SPY for benchmark + market timing...")
    spy_raw = yf.download("SPY", period=TEST_PERIOD, progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in spy_raw.columns]
    else:
        spy_raw.columns = [c.lower() for c in spy_raw.columns]

    results: list[QuantBacktestResult] = []

    # 1. SPY benchmark
    logger.info("\n--- SPY Benchmark ---")
    results.append(simulate_spy_benchmark(TEST_PERIOD))

    # 2. Equal weight ALL stocks (buy & hold baseline)
    logger.info("\n--- Equal Weight ALL stocks (buy & hold) ---")
    results.append(simulate_equal_weight_all(price_data))

    # 3. Momentum-weighted buy & hold (initial momentum weighting, no rebalance)
    logger.info("\n--- Momentum-Weighted Buy & Hold ---")
    results.append(simulate_mom_weighted_buy_hold(price_data))

    # 4. Buy & Hold + Trailing Stop 25% (sell losers, reinvest in winners)
    logger.info("\n--- Buy & Hold + Trailing Stop 25%% + Reinvest ---")
    results.append(simulate_buy_hold_trailing_stop(price_data, trail_pct=0.25, reinvest=True))

    # 5. Buy & Hold + Trailing Stop 20% + Reinvest
    logger.info("\n--- Buy & Hold + Trailing Stop 20%% + Reinvest ---")
    results.append(simulate_buy_hold_trailing_stop(price_data, trail_pct=0.20, reinvest=True))

    # 6. Buy & Hold + Trailing Stop 25% (no reinvest)
    logger.info("\n--- Buy & Hold + Trailing Stop 25%% (no reinvest) ---")
    results.append(simulate_buy_hold_trailing_stop(price_data, trail_pct=0.25, reinvest=False))

    # 7. Market-timed buy & hold (SPY > SMA200 filter)
    logger.info("\n--- Market-Timed Buy & Hold ---")
    results.append(simulate_market_timed_buy_hold(price_data, spy_raw))

    # Print results
    print_results(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
