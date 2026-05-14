"""P1 (#55) — time-based stale exit sweep (US 2y).

Target: bounce_die cleanup bucket (14 cases, -$268, avg 122h hold) where
positions reached only +1% high then drifted to -5% cleanup. No existing
mechanism (profit_taking +6%, trailing +8%, breakeven 50% of TP) fires
in this range, so positions sit through inevitable drawdown.

This sweep adds a second cleanup trigger based on hold time + pnl:
"if held >= N days AND pnl < threshold AND all-HOLD, force-sell".

Variants:
  V0_off:    stale_time_days=0 (current behavior)
  V1_2d_0:   2d, exit any non-profitable stagnant position
  V2_3d_0:   3d, more patient
  V3_2d_n2:  2d, only exit if down >2% (preserves slight bounces)
  V4_5d_0:   5d, very patient (matches avg cleanup hold)

Backtest baseline (Phase 2): US 2y Ret +30.2%, Sharpe +2.06, MDD -4.8%.
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner",
         "data", "backtest", "strategies", "engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


VARIANTS = [
    ("V0_off",     0, 0.0),
    ("V1_2d_0",    2, 0.0),
    ("V2_3d_0",    3, 0.0),
    ("V3_2d_n2",   2, -0.02),
    ("V4_5d_0",    5, 0.0),
]


def _us_cfg(stale_days: int, stale_thr: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("US")
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=0.10,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        sector_boost_weight=0.2,
        disabled_strategies=disabled,
        # Match live: keep the loss-based cleanup active so both triggers
        # can fire (only one will per position; we measure combined).
        stale_pnl_threshold=-0.05,
        stale_time_days=stale_days,
        stale_time_pnl_threshold=stale_thr,
    )


async def run(name: str, stale_days: int, stale_thr: float) -> dict:
    cfg = PipelineConfig(**_us_cfg(stale_days, stale_thr))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    cleanup_n = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("position_cleanup")
    )
    return dict(
        name=name, stale_days=stale_days, stale_thr=stale_thr,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        cleanup_n=cleanup_n,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 100)
    print("  P1 (#55) time-based stale exit sweep (US 2y)")
    print("=" * 100)
    results = []
    for name, days, thr in VARIANTS:
        thr_str = f"{thr:+.0%}" if thr != 0 else "0%"
        print(f"\n▶ {name}  stale={days}d threshold={thr_str}")
        r = await run(name, days, thr)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  "
              f"Trades={r['trades']}  Cleanups={r['cleanup_n']}  "
              f"({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = (f"{'Variant':<14} {'days':>5} {'thr':>6} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'PF':>6} {'Trades':>7} {'Cleanup':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        thr_str = f"{r['stale_thr']:+.0%}" if r['stale_thr'] != 0 else "0%"
        print(f"{r['name']:<14} {r['stale_days']:>5d} {thr_str:>6} "
              f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:7.1f} "
              f"{r['pf']:6.2f} {r['trades']:7d} {r['cleanup_n']:8d}")

    print("\nDelta vs V0_off (4-dim improvement test — needs all non-negative):")
    v0 = results[0]
    for r in results[1:]:
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        d_pf = r['pf'] - v0['pf']
        d_clean = r['cleanup_n'] - v0['cleanup_n']
        ok = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= 0
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<14}  ΔRet={d_ret:+5.1f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.1f}pp  ΔPF={d_pf:+5.2f}  "
              f"ΔCleanups={d_clean:+3d}  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
