"""KR sell_cooldown × whipsaw sweep (2y).

User context (2026-06-01): KR funnel shows 230 sell_cooldown rejections /
24h dominating buy attempts. Live cooldown=1d (86400s), whipsaw_max_losses=2.
Question: does loosening cooldown unlock cash deployment without raising
whipsaw losses materially?

Baseline V0 from compare_kr_exposure.py is on Pareto frontier (Sharpe 0.57,
71.5% deployed). Goal: find if cooldown lever moves any axis enough to
justify yaml change.

Variants:
  V0  cd=1d  w=2   baseline (current live)
  V1  cd=0d  w=2   cooldown disabled — fastest re-entry
  V2  cd=2d  w=2   cooldown longer — counterfactual stress
  V3  cd=1d  w=1   stricter whipsaw — paired guard
  V4  cd=1d  w=3   looser whipsaw — paired loosen
  V5  cd=0d  w=3   most permissive
  V6  cd=0d  w=1   cooldown off + whipsaw strict (compromise)
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
    # (name,             sell_cooldown_days, whipsaw_max_losses)
    # V0 = LIVE yaml (cd=3, w=2). Earlier sweep accidentally used cd=1
    # which is materially different (+9pp Ret gap).
    ("V0_cd3_w2",         3, 2),
    ("V1_cd1_w2",         1, 2),
    ("V2_cd2_w2",         2, 2),
    ("V3_cd5_w2",         5, 2),
    ("V4_cd3_w1",         3, 1),
    ("V5_cd3_w3",         3, 3),
    ("V6_cd0_w2",         0, 2),
]


def _kr_cfg(cooldown_days: int, whipsaw: int) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=cooldown_days,
        whipsaw_max_losses=whipsaw,
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )


async def run(name: str, cd: int, w: int) -> dict:
    cfg = PipelineConfig(**_kr_cfg(cd, w))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    snaps = eng._daily_snapshots
    avg_deployed = 0.0
    if snaps:
        avg_deployed = sum(
            (s.equity - s.cash) / s.equity for s in snaps if s.equity > 0
        ) / len(snaps)

    return dict(
        name=name, cd=cd, w=w,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  KR sell_cooldown × whipsaw sweep (2y)")
    print("=" * 110)
    results = []
    for name, cd, w in VARIANTS:
        print(f"\n▶ {name}  cooldown={cd}d  whipsaw={w}")
        r = await run(name, cd, w)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  Deployed={r['avg_deployed_pct']:5.1f}%  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<14} {'cd':>3} {'w':>2} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Deploy%':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<14} {r['cd']:>3} {r['w']:>2} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} {r['avg_deployed_pct']:>7.1f}%"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_cd1_w2 (baseline):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            ddep = r["avg_deployed_pct"] - base["avg_deployed_pct"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<14} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔDeploy={ddep:+5.1f}pp  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
