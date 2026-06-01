"""KR extra lever sweep — P3 / P4 / P5 / P8 batch (2y).

V0 = live post-PR #183 baseline: min_confidence=0.20, stale_time=14d/+2%,
cd=3, w=2, sizing_up on. Tests untested levers from earlier sweeps:

  P5 — V5+V2 pure combo (no max_pos) — confirms stack health
  P3 — max_sector_pct cap 0.4/0.5/0.6 (currently 1.0 = no cap)
  P4 — enable_quality_amplification ON (PF-based weight boost)
  P8 — trailing variation: 6/2 (fast trail) + 10/4 (slow trail)
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
    ("V0_live",                  {}),
    # P5: stale + min_conf already applied to V0 via yaml-loaded eval_cfg
    # (re-confirms the stack baseline after PR #183 merge)
    # P3: sector cap
    ("V1_sector_04",             {"max_sector_pct": 0.40}),
    ("V2_sector_05",             {"max_sector_pct": 0.50}),
    ("V3_sector_06",             {"max_sector_pct": 0.60}),
    # P4: quality amp
    ("V4_quality_amp",           {"enable_quality_amplification": True}),
    # P8: trailing
    ("V5_trail_fast_6_2",        {"trailing_activation_pct": 0.06,
                                  "trailing_trail_pct": 0.02}),
    ("V6_trail_slow_10_4",       {"trailing_activation_pct": 0.10,
                                  "trailing_trail_pct": 0.04}),
]


def _kr_cfg(overrides: dict) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    base = dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 3)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        stale_time_days=int(eval_cfg.get("stale_time_days", 0)),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold", 0.0)),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
        trailing_activation_pct=float(risk.get("default_trailing_activation_pct", 0.08)),
        trailing_trail_pct=float(risk.get("default_trailing_stop_pct", 0.03)),
    )
    base.update(overrides)
    return base


async def run(name: str, overrides: dict) -> dict:
    cfg = PipelineConfig(**_kr_cfg(overrides))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    snaps = eng._daily_snapshots
    avg_deployed = 0.0
    avg_npos = 0.0
    if snaps:
        avg_deployed = sum(
            (s.equity - s.cash) / s.equity for s in snaps if s.equity > 0
        ) / len(snaps)
        avg_npos = sum(s.n_positions for s in snaps) / len(snaps)

    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        avg_npos=round(avg_npos, 1),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 115)
    print("  KR extra lever sweep (2y) — P3/P4/P5/P8 on post-#183 baseline")
    print("=" * 115)
    results = []
    for name, ov in VARIANTS:
        print(f"\n▶ {name}  overrides={ov}")
        r = await run(name, ov)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  Pos={r['avg_npos']:4.1f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY")
    print("=" * 115)
    hdr = (
        f"{'Variant':<26} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Dep%':>6} {'Pos':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<26} {r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} "
            f"{r['pf']:>5.2f} {r['trades']:>4} {r['avg_deployed_pct']:>5.1f}% "
            f"{r['avg_npos']:>5.1f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_live (post-#183 baseline):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            ddep = r["avg_deployed_pct"] - base["avg_deployed_pct"]
            dpos = r["avg_npos"] - base["avg_npos"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<26} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔDep={ddep:+5.1f}pp  ΔPos={dpos:+4.1f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
