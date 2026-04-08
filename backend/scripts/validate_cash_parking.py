"""Validate cash parking effect on US 2y full-pipeline backtest.

The system has -22.4% alpha vs SPY post-donchian-disable. Hypothesis:
parking idle cash in SPY recovers a portion of that gap by giving the
portfolio direct beta exposure during periods when active strategies
hold cash.

Variants:
  V0_no_parking      : current behavior (cash sits idle)
  V1_park_30         : park if cash > 30% of portfolio (10% buffer)
  V2_park_20         : park if cash > 20%
  V3_park_50         : park if cash > 50% (more conservative)
  V4_park_10         : park if cash > 10% (most aggressive)

Run from backend/:
    ../venv/bin/python -u scripts/validate_cash_parking.py
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data", "backtest", "strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig


US_BASE = dict(
    market="US",
    use_wide_universe=True,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=20,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.05,
    volume_adjusted_slippage=True,
    disabled_strategies=[
        "bollinger_squeeze", "volume_profile",
        "cis_momentum", "larry_williams", "bnf_deviation",
        "donchian_breakout",
    ],
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
)


VARIANTS = [
    ("V0_no_parking",  {"enable_cash_parking": False}),
    ("V1_park_30",     {"enable_cash_parking": True, "cash_parking_threshold": 0.30}),
    ("V2_park_20",     {"enable_cash_parking": True, "cash_parking_threshold": 0.20}),
    ("V3_park_50",     {"enable_cash_parking": True, "cash_parking_threshold": 0.50}),
    ("V4_park_10",     {"enable_cash_parking": True, "cash_parking_threshold": 0.10}),
]


async def run_variant(name: str, overrides: dict) -> dict:
    cfg = PipelineConfig(**{**US_BASE, **overrides})
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name,
        ret=m.total_return_pct,
        sharpe=m.sharpe_ratio,
        sortino=m.sortino_ratio,
        mdd=m.max_drawdown_pct,
        pf=m.profit_factor,
        trades=m.total_trades,
        wr=m.win_rate,
        alpha=m.alpha,
        bench=m.benchmark_return_pct,
        elapsed=el,
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 110)
    print("  Cash Parking Validation — US 2Y Wide Universe (donchian disabled baseline)")
    print("=" * 110)

    out = []
    for name, ov in VARIANTS:
        desc = ", ".join(f"{k}={v}" for k, v in ov.items())
        print(f"\n▶ {name}  {desc}")
        try:
            r = await run_variant(name, ov)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:.2f}  MDD={r['mdd']:.1f}%  "
              f"PF={r['pf']:.2f}  Trades={r['trades']}  Alpha={r['alpha']:+.1f}%  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print(f"{'Variant':<18} {'Ret%':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} "
          f"{'PF':>6} {'Trades':>7} {'WR%':>6} {'Alpha%':>8}")
    print("-" * 110)
    for r in out:
        print(f"{r['name']:<18} {r['ret']:>+8.1f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} {r['wr']:>6.1f} "
              f"{r['alpha']:>+8.1f}")
    if out:
        print(f"{'Benchmark (SPY)':<18} {out[0]['bench']:>+8.1f}")

    if len(out) >= 2:
        baseline = out[0]
        print("\nDelta vs V0_no_parking:")
        for r in out[1:]:
            print(f"  {r['name']:<18} ΔRet={r['ret']-baseline['ret']:>+5.1f}pp  "
                  f"ΔSharpe={r['sharpe']-baseline['sharpe']:>+5.2f}  "
                  f"ΔAlpha={r['alpha']-baseline['alpha']:>+5.1f}pp  "
                  f"ΔMDD={r['mdd']-baseline['mdd']:>+5.1f}pp")


if __name__ == "__main__":
    asyncio.run(main())
