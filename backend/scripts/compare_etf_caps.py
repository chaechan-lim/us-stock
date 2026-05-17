"""ETF cap sweep — finally backtest #52 (US ETF cap 10→20% question).

Uses the new ETFBacktestEngine (backtest/etf_engine_backtest.py). Unlocks
the cap-tuning that was blocked by 'no backtest harness' since 2026-05-07.

Variants (matches the comments in config/etf_universe.yaml):
  V0_10_05:  portfolio 10% / single 5%   (current production, conservative)
  V1_15_08:  portfolio 15% / single 8%
  V2_20_10:  portfolio 20% / single 10%  (prior 2026-05-07 setting)
  V3_30_15:  portfolio 30% / single 15%  (KR-equivalent)

US 2y, $10k baseline.
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx"):
    logging.getLogger(n).setLevel(logging.ERROR)

from backtest.etf_engine_backtest import ETFBacktestConfig, ETFBacktestEngine


VARIANTS = [
    ("V0_10_05", 0.10, 0.05),
    ("V1_15_08", 0.15, 0.08),
    ("V2_20_10", 0.20, 0.10),
    ("V3_30_15", 0.30, 0.15),
]


async def run(name: str, portfolio_pct: float, single_pct: float) -> dict:
    cfg = ETFBacktestConfig(
        initial_capital=10_000,
        period="2y",
        max_portfolio_pct=portfolio_pct,
        max_single_etf_pct=single_pct,
    )
    t0 = time.time()
    res = await ETFBacktestEngine(cfg).run()
    el = time.time() - t0
    return dict(
        name=name,
        portfolio_pct=portfolio_pct,
        single_pct=single_pct,
        ret=res.total_return_pct,
        sharpe=res.sharpe,
        mdd=res.max_drawdown_pct,
        trades=res.trades,
        final=res.final_value,
        regime_flips=res.regime_changes,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 100)
    print("  ETF cap sweep (US 2y, $10k initial)")
    print("=" * 100)

    results = []
    for name, pp, sp in VARIANTS:
        print(f"\n▶ {name}  portfolio={pp:.0%} single={sp:.0%}")
        r = await run(name, pp, sp)
        results.append(r)
        print(f"  Ret={r['ret']:+.2f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.2f}%  Trades={r['trades']}  "
              f"Flips={r['regime_flips']}  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = (f"{'Variant':<12} {'pf':>5} {'sgl':>5} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'Trades':>7} {'Flips':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<12} {r['portfolio_pct']:>5.0%} {r['single_pct']:>5.0%} "
              f"{r['ret']:+7.2f} {r['sharpe']:+7.2f} {r['mdd']:7.2f} "
              f"{r['trades']:7d} {r['regime_flips']:6d}")

    print("\nDelta vs V0_10_05 (current production):")
    v0 = results[0]
    for r in results[1:]:
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        ok = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<12}  ΔRet={d_ret:+5.2f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.2f}pp  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
