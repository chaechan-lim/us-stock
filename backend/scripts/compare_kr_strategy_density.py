"""KR strategy-add re-test with CORRECT live baseline (cd=3, w=2).

Earlier `compare_kr_strategy_add.py` hardcoded sell_cooldown_days=1 which
made V0 catastrophically bad (-13.9% Ret); the "improvements" of VA/VB
were just clawing back the cooldown bug.

This re-tests on the actual live yaml baseline (cd=3, w=2, vol_scaling on,
sector_boost=0.3, kelly_fraction yaml-loaded) so the signal-density
question is asked properly: "given the current solid baseline, does
adding strategy X make positions grow / count more?"
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


ALL_STRATEGIES = {
    "supertrend", "trend_following", "donchian_breakout",
    "macd_histogram", "rsi_divergence", "bollinger_squeeze",
    "volume_profile", "regime_switch", "sector_rotation",
    "cis_momentum", "larry_williams", "bnf_deviation",
    "volume_surge", "cross_sectional_momentum",
    "pead_drift", "quality_factor",
}

VARIANTS = [
    # (name,            extra_enabled)
    ("V0_live",         set()),
    ("VA_supertrend",   {"supertrend"}),
    ("VB_tf",           {"trend_following"}),
    ("VC_bnf",          {"bnf_deviation"}),
    ("VD_volsurge",     {"volume_surge"}),
    ("VE_bbsqueeze",    {"bollinger_squeeze"}),
    ("VF_macd",         {"macd_histogram"}),
    ("VG_rsi_div",      {"rsi_divergence"}),
    ("VH_combo_st_bb",  {"supertrend", "bollinger_squeeze"}),
]


def _kr_cfg(extra: set[str]) -> dict:
    loader = StrategyConfigLoader()
    yaml_disabled = set(loader.get_market_disabled_strategies("KR"))
    # Override: remove `extra` from the disabled set
    disabled = sorted(yaml_disabled - extra)
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


async def run(name: str, extra: set[str]) -> dict:
    cfg = PipelineConfig(**_kr_cfg(extra))
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
        name=name, extras=sorted(extra),
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
    print("  KR strategy-add sweep on LIVE baseline (cd=3, w=2)")
    print("=" * 115)
    results = []
    for name, extras in VARIANTS:
        print(f"\n▶ {name}  +{sorted(extras)}")
        r = await run(name, extras)
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
        f"{'Variant':<20} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Dep%':>6} {'Pos':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<20} {r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} "
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
                f"    {r['name']:<20} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔDep={ddep:+5.1f}pp  ΔPos={dpos:+4.1f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
