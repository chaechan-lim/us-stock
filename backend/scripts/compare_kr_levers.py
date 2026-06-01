"""KR position-size lever sweep (2y).

User context (2026-06-01): live KR positions are tiny (4/5 at 1-3 shares
≈ <2% equity). Backtests of A (sizing-up) and C (cooldown) showed those
levers don't help. This script tests the next round of *untested* knobs
that could grow per-position size or position count:

  E — stale_time_days       Auto-exit non-growing positions after N days
                            → frees capital for fresh, full-sized entries
  J — max_positions         18 → 25, 30  (more slots = same cash / more names
                            but each smaller; we want fewer/bigger though)
  H — min_confidence        0.30 → 0.20  (let weaker signals through →
                            more entries, lower per-entry quality)
  M — vol_scale_min         0.5 → 0.7    (less Kelly downsizing for high-vol
                            symbols; counter to user's "too small" complaint)
  Combo                    Best of E + (J or M) + H

V0 must be the LIVE yaml baseline (cd=3, w=2) — otherwise we mis-attribute
gains/losses to the lever instead of the cooldown drift bug.
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


# Each variant overrides only its target field(s); rest = live yaml baseline.
VARIANTS = [
    # (name,                  overrides)
    ("V0_live",               {}),
    ("V1_stale_10d",          {"stale_time_days": 10, "stale_time_pnl_threshold": 0.02}),
    ("V2_stale_14d",          {"stale_time_days": 14, "stale_time_pnl_threshold": 0.02}),
    ("V3_max_pos_25",         {"max_positions": 25}),
    ("V4_max_pos_30",         {"max_positions": 30}),
    ("V5_min_conf_020",       {"min_confidence": 0.20}),
    ("V6_vol_min_07",         {"vol_scale_min": 0.7}),
    ("V7_combo_stale_pos_conf", {
        "stale_time_days": 14,
        "stale_time_pnl_threshold": 0.02,
        "max_positions": 25,
        "min_confidence": 0.20,
    }),
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
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
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
    print("  KR position-size lever sweep (2y) — D/E/J/H/M + combos")
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
        print("\n  vs V0_live (baseline):")
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
