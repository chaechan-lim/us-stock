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

# Broad universe for factor model testing
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "CRM",
    # Financials
    "JPM", "BAC", "GS", "V",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT",
    # Consumer
    "WMT", "PG", "KO", "MCD",
    # Energy + Industrial
    "XOM", "CVX", "CAT", "HON",
]

QUICK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                   "JPM", "JNJ", "XOM", "WMT", "UNH", "PG"]

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
    factor = next((r for r in results if r.name == "factor_selection"), None)
    kelly = next((r for r in results if r.name == "factor_kelly"), None)
    equal_all = next((r for r in results if "equal_all" in r.name), None)

    print("\n--- Analysis ---")
    if spy and factor:
        alpha = factor.cagr - spy.cagr
        print(f"Factor vs SPY:      alpha = {alpha:+.1%}")
    if equal_all and factor:
        alpha = factor.cagr - equal_all.cagr
        print(f"Factor vs EqualAll: alpha = {alpha:+.1%}")
    if factor and kelly:
        delta = kelly.cagr - factor.cagr
        print(f"Kelly vs Factor:    delta = {delta:+.1%}")
    if spy and kelly:
        alpha = kelly.cagr - spy.cagr
        print(f"Kelly vs SPY:       alpha = {alpha:+.1%}")
    if equal_all and kelly:
        alpha = kelly.cagr - equal_all.cagr
        print(f"Kelly vs EqualAll:  alpha = {alpha:+.1%}")


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

    results: list[QuantBacktestResult] = []

    # 1. SPY benchmark
    logger.info("\n--- SPY Benchmark ---")
    results.append(simulate_spy_benchmark(TEST_PERIOD))

    # 2. Equal weight baselines
    logger.info("\n--- Equal Weight Top 8 (baseline) ---")
    results.append(simulate_equal_weight(price_data))
    logger.info("\n--- Equal Weight ALL stocks ---")
    results.append(simulate_equal_weight_all(price_data))

    # 3. Factor selection (default weights, monthly)
    logger.info("\n--- Factor Selection (monthly) ---")
    results.append(simulate_factor_selection(price_data, fundamentals))

    # 4. Factor selection with low turnover (quarterly, keep winners)
    logger.info("\n--- Factor Low Turnover (quarterly) ---")
    results.append(simulate_factor_low_turnover(price_data, fundamentals))

    # 5. Concentrated portfolio (top 5, quarterly)
    logger.info("\n--- Concentrated Top 5 ---")
    results.append(simulate_concentrated(price_data, fundamentals))

    # 6. Momentum-heavy factor selection
    logger.info("\n--- Momentum-Heavy (70%% momentum) ---")
    results.append(simulate_momentum_heavy(price_data, fundamentals))

    # 7. Factor + Kelly-weighted sizing
    logger.info("\n--- Factor + Kelly Weighted ---")
    results.append(simulate_factor_kelly(price_data, fundamentals))

    # Print results
    print_results(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
