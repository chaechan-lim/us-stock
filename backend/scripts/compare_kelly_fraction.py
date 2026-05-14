"""Kelly fraction sweep (US 2y) — explores larger per-position sizing.

Live observation (2026-05-14): most BUYs land at 1 share even when cap
(max_position_pct=0.10) has room for 2-3. Conclusion: Kelly is binding,
not the cap. Current kelly_fraction=0.40 is quarter-Kelly (conservative).

Variants:
  V0_30:  Kelly 0.30 (more conservative)
  V1_40:  Kelly 0.40 (current — baseline)
  V2_50:  Kelly 0.50 (half-Kelly, common industry default)
  V3_60:  Kelly 0.60
  V4_70:  Kelly 0.70 (aggressive)

Hypothesis: bigger fraction → bigger absolute positions → higher returns
but also bigger drawdowns. Sharpe likely peaks somewhere 0.40-0.60.

Backtest baseline (Phase 2 + P1, US 2y): Ret +27.0% Sharpe 1.95 MDD -4.9%.
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
    ("V0_30",  0.30),
    ("V1_40",  0.40),
    ("V2_50",  0.50),
    ("V3_60",  0.60),
    ("V4_70",  0.70),
]


def _us_cfg(kelly: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("US")
    eval_cfg = loader.get_market_evaluation_loop_config("US")
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
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight") or 0.2),
        disabled_strategies=disabled,
        kelly_fraction=kelly,
        # Match live: cleanup + P1 stale_time both enabled
        stale_pnl_threshold=-0.05,
        stale_time_days=int(eval_cfg.get("stale_time_days") or 2),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold") or -0.02),
    )


async def run(name: str, kelly: float) -> dict:
    cfg = PipelineConfig(**_us_cfg(kelly))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    # Average position size (avg allocation_usd at entry)
    avg_alloc = 0.0
    if res.trades:
        sizes = [t.entry_price * t.quantity for t in res.trades if t.quantity > 0]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)
    return dict(
        name=name, kelly=kelly,
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
    print("  Kelly fraction sweep (US 2y, on Phase 2 + P1 baseline)")
    print("=" * 100)
    results = []
    for name, kelly in VARIANTS:
        print(f"\n▶ {name}  kelly_fraction={kelly:.2f}")
        r = await run(name, kelly)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  "
              f"Trades={r['trades']}  AvgAlloc=${r['avg_alloc']:.0f}  "
              f"({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = (f"{'Variant':<10} {'kelly':>6} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'PF':>6} {'Trades':>7} {'AvgAlloc$':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<10} {r['kelly']:>6.2f} {r['ret']:+7.1f} "
              f"{r['sharpe']:+7.2f} {r['mdd']:7.1f} {r['pf']:6.2f} "
              f"{r['trades']:7d} {r['avg_alloc']:10.0f}")

    print("\nDelta vs V1_40 (current production):")
    v1 = next(r for r in results if r["name"] == "V1_40")
    for r in results:
        if r["name"] == "V1_40":
            continue
        d_ret = r['ret'] - v1['ret']
        d_sharpe = r['sharpe'] - v1['sharpe']
        d_mdd = r['mdd'] - v1['mdd']
        d_pf = r['pf'] - v1['pf']
        d_alloc = r['avg_alloc'] - v1['avg_alloc']
        ok = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= 0
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<10}  ΔRet={d_ret:+5.1f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.1f}pp  ΔPF={d_pf:+5.2f}  "
              f"ΔAvgAlloc=${d_alloc:+5.0f}  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
