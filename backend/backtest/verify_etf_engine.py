"""ETF Engine backtest verification.

Simulates the ETF Engine's two strategies over historical data:
1. Regime switching: SPY vs SMA200 → TQQQ/SQQQ, SOXL/SOXS, UPRO/SPXU
2. Sector rotation: top sector ETFs (XLK, XLF, XLE, etc.)

Compares performance vs buy-and-hold SPY benchmark.

Usage:
    cd backend
    python3 -m backtest.verify_etf_engine
    python3 -m backtest.verify_etf_engine --period 5y
    python3 -m backtest.verify_etf_engine --regime-only
    python3 -m backtest.verify_etf_engine --sector-only
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.metrics import MetricsCalculator, Trade, BacktestMetrics
from data.market_state import MarketStateDetector, MarketRegime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Leveraged pairs: (base, bull, bear)
REGIME_PAIRS = [
    ("QQQ", "TQQQ", "SQQQ"),
    ("SPY", "UPRO", "SPXU"),
    ("SOXX", "SOXL", "SOXS"),
]

# Sector ETFs
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer_Disc": "XLY",
    "Consumer_Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real_Estate": "XLRE",
    "Communications": "XLC",
}

MAX_REGIME_ETFS = 2
MAX_SECTOR_ETFS = 3
MAX_HOLD_DAYS_LEVERAGED = 10
MAX_PORTFOLIO_ETF_PCT = 0.30
MAX_SINGLE_ETF_PCT = 0.15
SLIPPAGE_PCT = 0.05  # 0.05%
TOP_SECTOR_MIN_SCORE = 60

# Variant configs for parameter sensitivity testing
REGIME_VARIANTS = {
    "default": {"max_hold": 10, "bear_mode": "always", "max_pairs": 2, "confirmation": 2, "bear_dist": 0, "bear_size": 1.0},
    "bull_only": {"max_hold": 30, "bear_mode": "never", "max_pairs": 2, "confirmation": 2, "bear_dist": 0, "bear_size": 1.0},
    "smart_bear": {"max_hold": 20, "bear_mode": "qualified", "max_pairs": 2, "confirmation": 2, "bear_dist": -3.0, "bear_size": 0.5},
    "strict_bear": {"max_hold": 20, "bear_mode": "qualified", "max_pairs": 1, "confirmation": 3, "bear_dist": -5.0, "bear_size": 0.4},
}


@dataclass
class ETFPosition:
    symbol: str
    quantity: float
    avg_price: float
    entry_idx: int
    reason: str  # "regime_bull", "regime_bear", "sector"


@dataclass
class ETFSimResult:
    name: str
    metrics: BacktestMetrics
    trades: list[Trade]
    regime_changes: int = 0
    sector_rotations: int = 0


def load_etf_data(symbols: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    """Load historical data for all ETFs."""
    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval="1d")
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for %s (%d bars)", sym, len(df))
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            data[sym] = df
        except Exception as e:
            logger.warning("Failed to load %s: %s", sym, e)
    return data


def _calc_sector_scores(
    sector_data: dict[str, pd.DataFrame], idx: int, lookbacks: dict[str, int],
) -> list[tuple[str, str, float]]:
    """Calculate sector strength scores at a given bar index.

    Returns list of (sector_name, etf_symbol, strength_score) sorted by score desc.
    """
    scores = []
    for sector, etf_sym in SECTOR_ETFS.items():
        df = sector_data.get(etf_sym)
        if df is None or idx >= len(df) or idx < 60:
            continue

        close = df.iloc[idx]["close"]

        # 1-week return (5 bars)
        r1w = 0
        if idx >= 5:
            r1w = (close / df.iloc[idx - 5]["close"] - 1) * 100

        # 1-month return (21 bars)
        r1m = 0
        if idx >= 21:
            r1m = (close / df.iloc[idx - 21]["close"] - 1) * 100

        # 3-month return (63 bars)
        r3m = 0
        if idx >= 63:
            r3m = (close / df.iloc[idx - 63]["close"] - 1) * 100

        # Weighted strength score
        strength = r1w * 0.20 + r1m * 0.40 + r3m * 0.40
        scores.append((sector, etf_sym, strength))

    if not scores:
        return []

    # Normalize to 0-100
    raw = [s[2] for s in scores]
    min_r, max_r = min(raw), max(raw)
    spread = max_r - min_r

    result = []
    for sector, etf_sym, raw_score in scores:
        if spread > 0:
            normalized = ((raw_score - min_r) / spread) * 100
        else:
            normalized = 50.0
        result.append((sector, etf_sym, normalized))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def simulate_regime_switching(
    spy_data: pd.DataFrame,
    etf_data: dict[str, pd.DataFrame],
    initial_equity: float = 100_000,
    max_pairs: int = MAX_REGIME_ETFS,
    max_hold_days: int = MAX_HOLD_DAYS_LEVERAGED,
    bear_mode: str = "always",  # "always", "never", "qualified"
    bear_min_distance: float = -3.0,  # SPY % below SMA200 threshold
    bear_size_ratio: float = 1.0,  # bear position size vs bull
    confirmation_days: int = 2,
) -> ETFSimResult:
    """Simulate regime-based leveraged ETF switching.

    Logic:
    - SPY > SMA200 for N days → Bull regime → Buy bull ETFs (TQQQ, SOXL)
    - SPY < SMA200 for N days → Bear regime → depends on bear_mode:
        - "always": always enter bear ETFs
        - "never": exit to cash, no inverse ETFs
        - "qualified": only enter when SPY distance < threshold & high confidence
    - Leveraged ETFs held > max_hold_days → Force sell and re-enter
    """
    detector = MarketStateDetector(confirmation_days=confirmation_days)
    equity = initial_equity
    cash = initial_equity
    positions: dict[str, ETFPosition] = {}
    trades: list[Trade] = []
    equity_curve: list[float] = []
    equity_dates: list = []
    regime_changes = 0
    last_regime: MarketRegime | None = None

    # Available pairs (need both bull and bear data)
    pairs = []
    for base, bull, bear in REGIME_PAIRS:
        if bull in etf_data and bear in etf_data:
            pairs.append((base, bull, bear))
    pairs = pairs[:max_pairs]

    if not pairs:
        logger.error("No valid ETF pairs found in data")
        return ETFSimResult(
            name="regime_switching",
            metrics=BacktestMetrics(),
            trades=[],
        )

    # Align dates: use SPY's index
    for i in range(200, len(spy_data)):
        date = spy_data.index[i]
        spy_slice = spy_data.iloc[:i + 1]

        # Detect regime
        state = detector.detect(spy_slice)
        regime = state.regime

        # Determine target direction
        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
            target_dir = "bull"
        elif regime == MarketRegime.DOWNTREND:
            if bear_mode == "always":
                target_dir = "bear"
            elif bear_mode == "qualified":
                # Only enter bear if SPY is sufficiently below SMA200
                # and MarketState has high confidence
                if (state.spy_distance_pct <= bear_min_distance
                        and state.confidence >= 0.7):
                    target_dir = "bear"
                else:
                    target_dir = "neutral"
            else:  # "never"
                target_dir = "neutral"
        else:
            target_dir = "neutral"

        # Regime change → switch positions
        if regime != last_regime and last_regime is not None:
            regime_changes += 1

            # Close all current regime positions
            for sym in list(positions.keys()):
                pos = positions[sym]
                if not pos.reason.startswith("regime"):
                    continue
                df = etf_data.get(sym)
                if df is None or date not in df.index:
                    continue
                sell_price = float(df.loc[date, "close"]) * (1 - SLIPPAGE_PCT / 100)
                proceeds = pos.quantity * sell_price
                cash += proceeds
                pnl = (sell_price - pos.avg_price) * pos.quantity
                pnl_pct = (sell_price / pos.avg_price - 1) * 100
                entry_date = spy_data.index[pos.entry_idx]
                trades.append(Trade(
                    symbol=sym, side="SELL",
                    entry_date=str(entry_date), entry_price=pos.avg_price,
                    exit_date=str(date), exit_price=sell_price,
                    quantity=pos.quantity, pnl=pnl, pnl_pct=pnl_pct,
                    holding_days=(date - entry_date).days,
                    strategy_name="regime_switch",
                ))
                del positions[sym]

            # Open new direction positions
            if target_dir != "neutral":
                for base, bull, bear in pairs:
                    target_sym = bull if target_dir == "bull" else bear
                    df = etf_data.get(target_sym)
                    if df is None or date not in df.index:
                        continue
                    if target_sym in positions:
                        continue

                    price = float(df.loc[date, "close"])
                    buy_price = price * (1 + SLIPPAGE_PCT / 100)
                    max_alloc = equity * MAX_SINGLE_ETF_PCT
                    if target_dir == "bear":
                        max_alloc *= bear_size_ratio
                    alloc = min(max_alloc, cash * 0.9)
                    qty = int(alloc / buy_price)
                    if qty <= 0:
                        continue
                    cost = qty * buy_price
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[target_sym] = ETFPosition(
                        symbol=target_sym, quantity=qty,
                        avg_price=buy_price, entry_idx=i,
                        reason=f"regime_{target_dir}",
                    )

        last_regime = regime

        # Max hold days check for leveraged ETFs
        for sym in list(positions.keys()):
            pos = positions[sym]
            hold_days = i - pos.entry_idx
            if hold_days > max_hold_days:
                df = etf_data.get(sym)
                if df is None or date not in df.index:
                    continue
                sell_price = float(df.loc[date, "close"]) * (1 - SLIPPAGE_PCT / 100)
                proceeds = pos.quantity * sell_price
                cash += proceeds
                pnl = (sell_price - pos.avg_price) * pos.quantity
                pnl_pct = (sell_price / pos.avg_price - 1) * 100
                entry_date = spy_data.index[pos.entry_idx]
                trades.append(Trade(
                    symbol=sym, side="SELL",
                    entry_date=str(entry_date), entry_price=pos.avg_price,
                    exit_date=str(date), exit_price=sell_price,
                    quantity=pos.quantity, pnl=pnl, pnl_pct=pnl_pct,
                    holding_days=hold_days,
                    strategy_name="regime_hold_limit",
                ))
                del positions[sym]

                # Re-enter if still in same regime
                if target_dir != "neutral":
                    df_check = etf_data.get(sym)
                    if df_check is not None and date in df_check.index:
                        price = float(df_check.loc[date, "close"])
                        buy_price = price * (1 + SLIPPAGE_PCT / 100)
                        alloc = min(equity * MAX_SINGLE_ETF_PCT, cash * 0.9)
                        qty = int(alloc / buy_price)
                        if qty > 0 and qty * buy_price <= cash:
                            cash -= qty * buy_price
                            positions[sym] = ETFPosition(
                                symbol=sym, quantity=qty,
                                avg_price=buy_price, entry_idx=i,
                                reason=f"regime_{target_dir}",
                            )

        # Update equity
        pos_value = 0
        for sym, pos in positions.items():
            df = etf_data.get(sym)
            if df is not None and date in df.index:
                pos_value += pos.quantity * float(df.loc[date, "close"])
            else:
                pos_value += pos.quantity * pos.avg_price
        equity = cash + pos_value
        equity_curve.append(equity)
        equity_dates.append(date)

    # Close remaining positions at end
    if positions:
        last_date = spy_data.index[-1]
        for sym, pos in list(positions.items()):
            df = etf_data.get(sym)
            if df is not None and last_date in df.index:
                price = float(df.loc[last_date, "close"])
            else:
                price = pos.avg_price
            cash += pos.quantity * price
            pnl = (price - pos.avg_price) * pos.quantity
            entry_date = spy_data.index[pos.entry_idx]
            trades.append(Trade(
                symbol=sym, side="SELL",
                entry_date=str(entry_date), entry_price=pos.avg_price,
                exit_date=str(last_date), exit_price=price,
                quantity=pos.quantity, pnl=pnl,
                pnl_pct=(price / pos.avg_price - 1) * 100,
                holding_days=(last_date - entry_date).days,
                strategy_name="regime_close",
            ))

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, initial_equity)

    return ETFSimResult(
        name="regime_switching",
        metrics=metrics,
        trades=trades,
        regime_changes=regime_changes,
    )


def simulate_sector_rotation(
    etf_data: dict[str, pd.DataFrame],
    initial_equity: float = 100_000,
    max_sectors: int = MAX_SECTOR_ETFS,
    rebalance_interval: int = 21,  # ~monthly
) -> ETFSimResult:
    """Simulate sector ETF rotation based on relative strength.

    Logic:
    - Every rebalance_interval days, rank sectors by momentum
    - Buy top N sector ETFs, sell bottom sectors
    - Hold until next rebalance
    """
    # Find common date range
    all_dfs = {sym: df for sym, df in etf_data.items() if sym in SECTOR_ETFS.values()}
    if len(all_dfs) < 3:
        logger.error("Need at least 3 sector ETFs, got %d", len(all_dfs))
        return ETFSimResult(name="sector_rotation", metrics=BacktestMetrics(), trades=[])

    # Use the ETF with most data as reference index
    ref_sym = max(all_dfs.keys(), key=lambda s: len(all_dfs[s]))
    ref_df = all_dfs[ref_sym]

    equity = initial_equity
    cash = initial_equity
    positions: dict[str, ETFPosition] = {}
    trades: list[Trade] = []
    equity_curve: list[float] = []
    equity_dates: list = []
    rotations = 0
    last_top: list[str] = []

    for i in range(63, len(ref_df)):  # Need 63 bars for 3-month lookback
        date = ref_df.index[i]

        # Rebalance check
        if i % rebalance_interval == 0 or i == 63:
            scores = _calc_sector_scores(all_dfs, i, {"1w": 5, "1m": 21, "3m": 63})
            if not scores:
                continue

            top_sectors = [
                (sector, sym) for sector, sym, score in scores[:max_sectors]
                if score >= TOP_SECTOR_MIN_SCORE
            ]
            top_syms = [sym for _, sym in top_sectors]
            bottom_syms = [sym for _, sym, score in scores[-3:] if score < 40]

            if sorted(top_syms) != sorted(last_top):
                rotations += 1

                # Sell positions not in top sectors
                for sym in list(positions.keys()):
                    if sym not in top_syms:
                        pos = positions[sym]
                        df = all_dfs.get(sym)
                        if df is None or date not in df.index:
                            continue
                        sell_price = float(df.loc[date, "close"]) * (1 - SLIPPAGE_PCT / 100)
                        proceeds = pos.quantity * sell_price
                        cash += proceeds
                        pnl = (sell_price - pos.avg_price) * pos.quantity
                        entry_date = ref_df.index[pos.entry_idx]
                        trades.append(Trade(
                            symbol=sym, side="SELL",
                            entry_date=str(entry_date), entry_price=pos.avg_price,
                            exit_date=str(date), exit_price=sell_price,
                            quantity=pos.quantity, pnl=pnl,
                            pnl_pct=(sell_price / pos.avg_price - 1) * 100,
                            holding_days=(date - entry_date).days,
                            strategy_name="sector_rotation",
                        ))
                        del positions[sym]

                # Buy new top sectors
                for sector, sym in top_sectors:
                    if sym in positions:
                        continue
                    df = all_dfs.get(sym)
                    if df is None or date not in df.index:
                        continue
                    price = float(df.loc[date, "close"])
                    buy_price = price * (1 + SLIPPAGE_PCT / 100)
                    alloc = min(equity * MAX_SINGLE_ETF_PCT, cash * 0.9)
                    qty = int(alloc / buy_price)
                    if qty <= 0 or qty * buy_price > cash:
                        continue
                    cash -= qty * buy_price
                    positions[sym] = ETFPosition(
                        symbol=sym, quantity=qty,
                        avg_price=buy_price, entry_idx=i,
                        reason="sector",
                    )

                last_top = top_syms

        # Update equity
        pos_value = 0
        for sym, pos in positions.items():
            df = all_dfs.get(sym)
            if df is not None and date in df.index:
                pos_value += pos.quantity * float(df.loc[date, "close"])
            else:
                pos_value += pos.quantity * pos.avg_price
        equity = cash + pos_value
        equity_curve.append(equity)
        equity_dates.append(date)

    # Close remaining
    if positions:
        last_date = ref_df.index[-1]
        for sym, pos in list(positions.items()):
            df = all_dfs.get(sym)
            price = float(df.loc[last_date, "close"]) if df is not None and last_date in df.index else pos.avg_price
            cash += pos.quantity * price
            entry_date = ref_df.index[pos.entry_idx]
            trades.append(Trade(
                symbol=sym, side="SELL",
                entry_date=str(entry_date), entry_price=pos.avg_price,
                exit_date=str(last_date), exit_price=price,
                quantity=pos.quantity,
                pnl=(price - pos.avg_price) * pos.quantity,
                pnl_pct=(price / pos.avg_price - 1) * 100,
                holding_days=(last_date - entry_date).days,
                strategy_name="sector_close",
            ))

    eq_series = pd.Series(equity_curve, index=equity_dates)
    metrics = MetricsCalculator.calculate(eq_series, trades, initial_equity)

    return ETFSimResult(
        name="sector_rotation",
        metrics=metrics,
        trades=trades,
        sector_rotations=rotations,
    )


def simulate_spy_benchmark(spy_data: pd.DataFrame, initial_equity: float = 100_000) -> BacktestMetrics:
    """Buy-and-hold SPY benchmark."""
    if spy_data.empty:
        return BacktestMetrics()

    start_price = float(spy_data.iloc[200]["close"])  # Same starting point as regime sim
    qty = int(initial_equity * 0.95 / start_price)
    remaining_cash = initial_equity - qty * start_price

    equity_curve = []
    equity_dates = []
    for i in range(200, len(spy_data)):
        price = float(spy_data.iloc[i]["close"])
        equity_curve.append(remaining_cash + qty * price)
        equity_dates.append(spy_data.index[i])

    eq_series = pd.Series(equity_curve, index=equity_dates)
    return MetricsCalculator.calculate(eq_series, [], initial_equity)


def print_result(result: ETFSimResult, label: str = "") -> None:
    m = result.metrics
    print(f"\n{'='*60}")
    print(f"  {label or result.name}")
    print(f"{'='*60}")
    print(f"  Total Return:   {m.total_return_pct:+.1f}%")
    print(f"  CAGR:           {m.cagr:+.1%}")
    print(f"  Sharpe:         {m.sharpe_ratio:.2f}")
    print(f"  Sortino:        {m.sortino_ratio:.2f}")
    print(f"  Max Drawdown:   {m.max_drawdown_pct:.1f}%")
    print(f"  Total Trades:   {m.total_trades}")
    print(f"  Win Rate:       {m.win_rate:.1f}%")
    print(f"  Profit Factor:  {m.profit_factor:.2f}")
    print(f"  Avg Hold (days):{m.avg_holding_days:.1f}")
    print(f"  Final Equity:   ${m.final_equity:,.0f}")
    if result.regime_changes:
        print(f"  Regime Changes: {result.regime_changes}")
    if result.sector_rotations:
        print(f"  Rotations:      {result.sector_rotations}")

    # Top trades
    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        print(f"\n  Top 5 Winning Trades:")
        for t in sorted_trades[:5]:
            print(f"    {t.symbol:5s} {t.entry_date[:10]} -> {t.exit_date[:10]} "
                  f"PnL=${t.pnl:+,.0f} ({t.pnl_pct:+.1f}%) {t.holding_days}d")
        print(f"\n  Top 5 Losing Trades:")
        for t in sorted_trades[-5:]:
            print(f"    {t.symbol:5s} {t.entry_date[:10]} -> {t.exit_date[:10]} "
                  f"PnL=${t.pnl:+,.0f} ({t.pnl_pct:+.1f}%) {t.holding_days}d")


def main():
    t0 = time.time()

    # Parse args
    period = "3y"
    run_regime = True
    run_sector = True
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--period"):
            period = arg.split("=")[1] if "=" in arg else args[args.index(arg) + 1]
        elif arg == "--regime-only":
            run_sector = False
        elif arg == "--sector-only":
            run_regime = False

    # Collect all symbols to load
    symbols_needed = {"SPY"}
    if run_regime:
        for base, bull, bear in REGIME_PAIRS:
            symbols_needed.update([bull, bear])
    if run_sector:
        symbols_needed.update(SECTOR_ETFS.values())

    logger.info("Loading %d ETFs (period=%s)...", len(symbols_needed), period)
    all_data = load_etf_data(sorted(symbols_needed), period=period)
    logger.info("Loaded %d/%d ETFs", len(all_data), len(symbols_needed))

    if "SPY" not in all_data:
        logger.error("SPY data is required")
        sys.exit(1)

    spy_data = all_data["SPY"]
    logger.info("SPY: %d bars (%s to %s)",
                len(spy_data), spy_data.index[0].date(), spy_data.index[-1].date())

    # Benchmark
    bench = simulate_spy_benchmark(spy_data)
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: SPY Buy & Hold")
    print(f"{'='*60}")
    print(f"  Total Return: {bench.total_return_pct:+.1f}%")
    print(f"  CAGR:         {bench.cagr:+.1%}")
    print(f"  Sharpe:       {bench.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {bench.max_drawdown_pct:.1f}%")

    # Regime switching variants
    regime_results = {}
    if run_regime:
        for variant_name, cfg in REGIME_VARIANTS.items():
            logger.info("Running regime switching [%s]...", variant_name)
            result = simulate_regime_switching(
                spy_data, all_data,
                max_pairs=cfg["max_pairs"],
                max_hold_days=cfg["max_hold"],
                bear_mode=cfg["bear_mode"],
                bear_min_distance=cfg["bear_dist"],
                bear_size_ratio=cfg["bear_size"],
                confirmation_days=cfg["confirmation"],
            )
            regime_results[variant_name] = result
            label = (
                f"REGIME [{variant_name}] hold={cfg['max_hold']}d "
                f"bear={cfg['bear_mode']} "
                f"pairs={cfg['max_pairs']} confirm={cfg['confirmation']}d"
            )
            print_result(result, label)

    # Sector rotation
    sector_result = None
    if run_sector:
        logger.info("Running sector rotation simulation...")
        sector_result = simulate_sector_rotation(all_data)
        print_result(sector_result, "SECTOR ETF ROTATION (XLK, XLF, XLE, ...)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY vs SPY Buy & Hold (CAGR {bench.cagr:+.1%})")
    print(f"{'='*60}")
    if run_regime:
        for name, result in regime_results.items():
            m = result.metrics
            alpha = m.cagr - bench.cagr
            cfg = REGIME_VARIANTS[name]
            print(f"  Regime [{name:13s}]: CAGR {m.cagr:+6.1%} "
                  f"(alpha {alpha:+.1%}) Sharpe {m.sharpe_ratio:5.2f} "
                  f"MDD {m.max_drawdown_pct:6.1f}% "
                  f"(hold={cfg['max_hold']}d bear={cfg['bear_mode']})")
    if sector_result:
        alpha = sector_result.metrics.cagr - bench.cagr
        print(f"  Sector Rotation   : CAGR {sector_result.metrics.cagr:+6.1%} "
              f"(alpha {alpha:+.1%}) Sharpe {sector_result.metrics.sharpe_ratio:5.2f} "
              f"MDD {sector_result.metrics.max_drawdown_pct:6.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
