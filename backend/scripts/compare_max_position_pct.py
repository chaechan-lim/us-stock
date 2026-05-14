"""max_position_pct sweep (US 2y) — explores raising the per-position cap.

Live observation (2026-05-14): Kelly sweep showed AvgAlloc plateaus at
~$5.4k regardless of Kelly fraction (binding at 10% cap). To grow per-
position size meaningfully, the cap itself must rise.

KR runs at 0.20 with dual_momentum only (high-conviction single signal);
US at 0.10. This sweep tests US 0.10 / 0.12 / 0.15 / 0.18 / 0.20 to find
the trade-off between concentration risk and capital deployment.

Backtest baseline (Phase 2 + P1, US 2y): Ret +27.0% Sharpe 1.95.
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
    ("V0_10", 0.10),
    ("V1_12", 0.12),
    ("V2_15", 0.15),
    ("V3_18", 0.18),
    ("V4_20", 0.20),
]


def _us_cfg(max_pct: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("US")
    eval_cfg = loader.get_market_evaluation_loop_config("US")
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=max_pct,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight") or 0.2),
        disabled_strategies=disabled,
        # Match live: cleanup + P1 stale_time both enabled
        stale_pnl_threshold=-0.05,
        stale_time_days=int(eval_cfg.get("stale_time_days") or 2),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold") or -0.02),
    )


async def run(name: str, max_pct: float) -> dict:
    cfg = PipelineConfig(**_us_cfg(max_pct))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    avg_alloc = 0.0
    if res.trades:
        sizes = [t.entry_price * t.quantity for t in res.trades if t.quantity > 0]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)
    return dict(
        name=name, max_pct=max_pct,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_alloc=round(avg_alloc, 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 100)
    print("  max_position_pct sweep (US 2y, on Phase 2 + P1 baseline)")
    print("=" * 100)
    results = []
    for name, mp in VARIANTS:
        print(f"\n▶ {name}  max_position_pct={mp:.0%}")
        r = await run(name, mp)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  "
              f"Trades={r['trades']}  AvgAlloc=${r['avg_alloc']:.0f}  "
              f"({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = (f"{'Variant':<10} {'maxPct':>7} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'PF':>6} {'Trades':>7} {'AvgAlloc$':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<10} {r['max_pct']:>7.0%} {r['ret']:+7.1f} "
              f"{r['sharpe']:+7.2f} {r['mdd']:7.1f} {r['pf']:6.2f} "
              f"{r['trades']:7d} {r['avg_alloc']:10.0f}")

    print("\nDelta vs V0_10 (current production):")
    v0 = results[0]
    for r in results[1:]:
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        d_pf = r['pf'] - v0['pf']
        d_alloc = r['avg_alloc'] - v0['avg_alloc']
        ok = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= 0
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<10}  ΔRet={d_ret:+5.1f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.1f}pp  ΔPF={d_pf:+5.2f}  "
              f"ΔAvgAlloc=${d_alloc:+5.0f}  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
