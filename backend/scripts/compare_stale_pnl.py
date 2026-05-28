"""US position_cleanup threshold sweep (2y).

Live observation (2026-05-27): cleanup × 8 = −$121, top loser by far.
Cumulative live memory: 45 cleanups / 0 wins / −$651, larger than any
positive contributor. LLM #2 today recommended disabling cleanup
entirely; this script measures the trade-off.

Mechanism: cleanup fires when (all strategies HOLD) AND
(pnl_pct < stale_pnl_threshold). Current threshold -0.05. Variants
test relaxing (-0.07/-0.10), disabling (-1.0 = never), and a control
that loosens slightly less.

baseline: live US config (stale_pnl_threshold = -0.05)
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in (
    "yfinance", "peewee", "urllib3", "httpx", "scanner",
    "data", "backtest", "strategies", "engine",
):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


VARIANTS = [
    ("V0_curr_-5",   -0.05),  # current live
    ("V1_relax_-7",  -0.07),
    ("V2_relax_-10", -0.10),
    ("V3_disable",   -1.00),  # effectively off
]


def _us_cfg(stale_pnl: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("US")
    eval_cfg = loader.get_market_evaluation_loop_config("US")
    risk_cfg = loader.get_market_risk_config("US")
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=float(risk_cfg.get("max_position_pct", 0.15)),
        min_position_pct=float(risk_cfg.get("min_position_pct", 0.03)),
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight") or 0.2),
        disabled_strategies=disabled,
        kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.50)),
        stale_pnl_threshold=stale_pnl,
        # P1 stale_time stays off (disabled 5-19 due to live whipsaw)
        stale_time_days=int(eval_cfg.get("stale_time_days") or 0),
        stale_time_pnl_threshold=float(
            eval_cfg.get("stale_time_pnl_threshold") or -0.02
        ),
    )


async def run(name: str, stale_pnl: float) -> dict:
    cfg = PipelineConfig(**_us_cfg(stale_pnl))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    # Count cleanup sells specifically
    cleanups = 0
    cleanup_pnl = 0.0
    for t in res.trades or []:
        strat = getattr(t, "strategy_name", "") or ""
        if "cleanup" in strat or "stale" in strat or "indifference" in strat:
            cleanups += 1
            cleanup_pnl += getattr(t, "pnl", 0) or 0
    return dict(
        name=name, thr=stale_pnl,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        cleanups=cleanups,
        cleanup_pnl=round(cleanup_pnl, 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 100)
    print("  US stale_pnl_threshold sweep (2y) — does cleanup-firing hurt alpha?")
    print("=" * 100)
    results = []
    for name, thr in VARIANTS:
        print(f"\n▶ {name}  stale_pnl_threshold={thr:.2f}")
        r = await run(name, thr)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trd={r['trades']:>4}  Cleanups={r['cleanups']:>3} (PnL=${r['cleanup_pnl']:+,.0f})  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = (
        f"{'Variant':<14} {'thr':>6} {'Ret%':>7} {'Sharpe':>7} "
        f"{'MDD%':>7} {'PF':>5} {'Trd':>4} {'Cleanups':>10} {'CleanPnL$':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<14} {r['thr']:>6.2f} {r['ret']:+7.1f} {r['sharpe']:+7.2f} "
            f"{r['mdd']:>7.1f} {r['pf']:>5.2f} {r['trades']:>4} "
            f"{r['cleanups']:>10} {r['cleanup_pnl']:>+11,.0f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_curr_-5 (current live):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([
                dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10,
            ])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<14} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
