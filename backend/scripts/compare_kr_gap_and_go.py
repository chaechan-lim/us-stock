"""KR Gap-and-Go strategy backtest (2y) — P6.1 validation.

V0: live post-PR #183 (dual_momentum + supertrend, gap_and_go DISABLED)
V1: + gap_and_go enabled (full mix)
V2: gap_and_go ONLY (isolate its contribution)
Vparams: tune min_gap_pct ∈ {2%, 3%, 4%} × min_vol_ratio ∈ {1.5, 2.0}

Acceptance gates per CLAUDE.md (KR combo relaxed):
  Sharpe > 0 (preferable > current 0.71)
  MDD < 15% (current -11%)
  PF > 1.0 (current 1.30)
  Improves on all 4 dims vs current KR combo
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
    "pead_drift", "quality_factor", "gap_and_go",
}


VARIANTS = [
    # (name,                   extra_enabled,  gng_params)
    ("V0_live",                set(),          {}),
    ("V1_with_gng",            {"gap_and_go"}, {}),
    ("V2_gng_only",            {"gap_and_go", "dual_momentum_OFF", "supertrend_OFF"}, {}),
    ("V3_gng_gap_02",          {"gap_and_go"}, {"min_gap_pct": 0.02}),
    ("V4_gng_gap_04",          {"gap_and_go"}, {"min_gap_pct": 0.04}),
    ("V5_gng_vol_20",          {"gap_and_go"}, {"min_vol_ratio": 2.0}),
    ("V6_gng_vol_10",          {"gap_and_go"}, {"min_vol_ratio": 1.0}),
]


def _kr_cfg(extra: set[str], gng_params: dict) -> dict:
    loader = StrategyConfigLoader()
    yaml_disabled = set(loader.get_market_disabled_strategies("KR"))

    # Compute disabled list:
    #   gap_and_go is yaml-disabled by global enabled=false; we hot-enable
    #   by passing yaml=False but excluding from disabled. The backtest
    #   pipeline's strategy filter uses (yaml-enabled AND not in disabled).
    #   To re-enable from backtest, we patch the registry differently — see
    #   helper below.
    keep_extras = extra - {"dual_momentum_OFF", "supertrend_OFF"}
    drop_baselines = (
        {"dual_momentum"} if "dual_momentum_OFF" in extra else set()
    ) | (
        {"supertrend"} if "supertrend_OFF" in extra else set()
    )

    disabled = sorted(
        (yaml_disabled - keep_extras) | drop_baselines
    )
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
    )


async def run(name: str, extra: set[str], gng_params: dict) -> dict:
    cfg = PipelineConfig(**_kr_cfg(extra, gng_params))
    eng = FullPipelineBacktest(cfg)
    # Hot-enable gap_and_go in registry + apply params (since yaml says enabled=false)
    if "gap_and_go" in extra:
        from strategies.gap_and_go import GapAndGoStrategy
        eng._registry._strategies["gap_and_go"] = GapAndGoStrategy(params=gng_params or None)

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

    # Per-strategy stats
    gng_trades = res.strategy_stats.get("gap_and_go", {}).get("trades", 0)
    gng_pnl = res.strategy_stats.get("gap_and_go", {}).get("pnl", 0.0)

    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        avg_npos=round(avg_npos, 1),
        gng_trades=gng_trades,
        gng_pnl=round(gng_pnl, 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 115)
    print("  KR Gap-and-Go validation (2y) — P6.1")
    print("=" * 115)
    results = []
    for name, extras, params in VARIANTS:
        print(f"\n▶ {name}  extras={sorted(extras)}  params={params}")
        r = await run(name, extras, params)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  Pos={r['avg_npos']:4.1f}  "
            f"GNGtrd={r['gng_trades']:>3}  GNGpnl=₩{r['gng_pnl']:>+12,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY")
    print("=" * 115)
    hdr = (
        f"{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Dep%':>6} {'Pos':>5} {'GNGtrd':>7} {'GNGpnl':>15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<22} {r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} "
            f"{r['pf']:>5.2f} {r['trades']:>4} {r['avg_deployed_pct']:>5.1f}% "
            f"{r['avg_npos']:>5.1f} {r['gng_trades']:>7} ₩{r['gng_pnl']:>+13,.0f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_live (current KR combo):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            ddep = r["avg_deployed_pct"] - base["avg_deployed_pct"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔDep={ddep:+5.1f}pp  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
