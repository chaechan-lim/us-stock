"""Infinity Buy backtest — 무한매수법 (averaging-down 분할 매수).

Algorithm:
  1. Capital split into N equal slices.
  2. Each trading day: buy 1 slice at open if not all slices used.
  3. Track running avg cost. When current price >= avg * (1 + profit_target):
     sell ALL → cycle complete. Reset and start over.
  4. If all slices used, hold and wait for profit_target.

Sweep:
  - Symbols: TQQQ (NDX 3x), SOXL (Semi 3x), SPY (1x baseline), QQQ (1x)
  - N (slices): 20, 40, 60
  - profit_target: 0.05, 0.10, 0.15

Per-variant report:
  - Total return % (2y)
  - Max drawdown (peak-to-trough on portfolio value, MTM)
  - Sharpe (daily return-based)
  - Cycles completed
  - Avg cycle length (days)
  - Worst stuck duration (longest open cycle)
  - Final position state (cash + shares value)
"""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass

import yfinance as yf
import pandas as pd

print = functools.partial(print, flush=True)


SYMBOLS = ["TQQQ", "SOXL", "SPY", "QQQ"]
SLICES = [20, 40, 60]
PROFIT_TARGETS = [0.05, 0.10, 0.15]
INITIAL_CAPITAL = 10_000.0
PERIOD = "2y"


@dataclass
class _Result:
    symbol: str
    n: int
    pt: float
    total_return_pct: float
    max_dd_pct: float
    sharpe: float
    cycles: int
    avg_cycle_days: float
    worst_stuck_days: int
    final_value: float


def _load_prices(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period=PERIOD, progress=False,
                     auto_adjust=False, threads=False)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "Close"]].dropna()


def simulate(symbol: str, n: int, pt: float, df: pd.DataFrame) -> _Result:
    """Run one infinity-buy simulation."""
    slice_amount = INITIAL_CAPITAL / n
    cash = INITIAL_CAPITAL
    shares = 0.0
    avg_cost = 0.0
    slices_used = 0  # 0..n
    cycles: list[int] = []  # cycle lengths in days
    cycle_start_idx = 0

    portfolio_values = []  # for MDD + Sharpe

    for idx, (date, row) in enumerate(df.iterrows()):
        open_p = float(row["Open"])
        close_p = float(row["Close"])

        # 1) Decision at OPEN — buy if slices remain
        if slices_used < n and cash >= slice_amount:
            qty = slice_amount / open_p
            new_cost_total = avg_cost * shares + slice_amount
            shares += qty
            avg_cost = new_cost_total / shares if shares > 0 else 0
            cash -= slice_amount
            slices_used += 1

        # 2) Check profit target at OPEN (intraday sell at open)
        #    (Real-world: would also check intraday high, but use open for simplicity)
        if shares > 0 and avg_cost > 0:
            if open_p >= avg_cost * (1 + pt):
                # Sell all at open
                proceeds = shares * open_p
                cash += proceeds
                cycles.append(idx - cycle_start_idx)
                shares = 0.0
                avg_cost = 0.0
                slices_used = 0
                cycle_start_idx = idx

        # 3) End-of-day MTM
        pv = cash + shares * close_p
        portfolio_values.append(pv)

    # Final state
    final_value = portfolio_values[-1] if portfolio_values else INITIAL_CAPITAL
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100

    # MDD on portfolio value curve
    peak = portfolio_values[0]
    max_dd = 0.0
    for v in portfolio_values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

    # Sharpe from daily returns
    returns = []
    for i in range(1, len(portfolio_values)):
        prev = portfolio_values[i - 1]
        if prev > 0:
            returns.append((portfolio_values[i] - prev) / prev)
    if len(returns) >= 30:
        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = math.sqrt(var)
        sharpe = (mu / std) * math.sqrt(252) if std > 0 else 0.0
    else:
        sharpe = 0.0

    avg_cycle = sum(cycles) / len(cycles) if cycles else 0
    # Worst stuck: longest open cycle (ends at current bar if still open)
    open_cycle_len = len(df) - cycle_start_idx if shares > 0 else 0
    worst_stuck = max([*cycles, open_cycle_len]) if (cycles or shares > 0) else 0

    return _Result(
        symbol=symbol, n=n, pt=pt,
        total_return_pct=round(total_return, 2),
        max_dd_pct=round(max_dd, 2),
        sharpe=round(sharpe, 2),
        cycles=len(cycles),
        avg_cycle_days=round(avg_cycle, 1),
        worst_stuck_days=worst_stuck,
        final_value=round(final_value, 2),
    )


def main():
    # Buy-and-hold baseline per symbol (for context)
    print("=" * 110)
    print(f"  Infinity Buy backtest — {PERIOD} window, ${INITIAL_CAPITAL:.0f} capital")
    print("=" * 110)
    print()

    baselines = {}
    for sym in SYMBOLS:
        try:
            df = _load_prices(sym)
            if df.empty:
                continue
            bh_ret = (float(df["Close"].iloc[-1]) / float(df["Open"].iloc[0]) - 1) * 100
            baselines[sym] = (df, round(bh_ret, 2))
            print(f"Buy-and-hold {sym} ({PERIOD}): {bh_ret:+.2f}%")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    print()

    all_results: list[_Result] = []
    for sym, (df, _bh) in baselines.items():
        for n in SLICES:
            for pt in PROFIT_TARGETS:
                r = simulate(sym, n, pt, df)
                all_results.append(r)

    # Group by symbol
    for sym in baselines:
        bh = baselines[sym][1]
        print("=" * 110)
        print(f"  {sym}  (Buy-and-hold {PERIOD}: {bh:+.2f}%)")
        print("=" * 110)
        hdr = (f"{'N':>3} {'PT%':>5}  {'Ret%':>7} {'MDD%':>7} {'Sharpe':>7} "
               f"{'Cycles':>7} {'AvgDays':>8} {'WorstStuck':>11}  {'Δ vs BH':>9}")
        print(hdr)
        print("-" * len(hdr))
        for r in all_results:
            if r.symbol != sym:
                continue
            delta = r.total_return_pct - bh
            print(f"{r.n:>3} {r.pt*100:>5.0f}  "
                  f"{r.total_return_pct:>+7.2f} {r.max_dd_pct:>+7.2f} "
                  f"{r.sharpe:>+7.2f} {r.cycles:>7d} "
                  f"{r.avg_cycle_days:>8.1f} {r.worst_stuck_days:>11d}  "
                  f"{delta:>+9.2f}")
        print()

    # Best per symbol by Sharpe
    print("=" * 110)
    print(f"  BEST per symbol (sorted by Sharpe)")
    print("=" * 110)
    for sym in baselines:
        sym_results = [r for r in all_results if r.symbol == sym]
        sym_results.sort(key=lambda x: -x.sharpe)
        top = sym_results[0]
        bh = baselines[sym][1]
        flag = "✓ BEAT BH" if top.total_return_pct > bh else "✗"
        print(f"  {sym}: N={top.n} PT={top.pt*100:.0f}%  Ret={top.total_return_pct:+.2f}% "
              f"(BH {bh:+.2f}%)  Sharpe={top.sharpe:+.2f}  MDD={top.max_dd_pct:+.2f}%  "
              f"Cycles={top.cycles} (avg {top.avg_cycle_days:.0f}d)  {flag}")


if __name__ == "__main__":
    sys.exit(main() or 0)
