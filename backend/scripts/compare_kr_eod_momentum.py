"""KR EOD Momentum strategy backtest (2y) — Tier-2 follow-up after
Gap-and-Go failure. Tests close-on-N-day-high breakout-on-volume pattern.

Same hot-enable pattern as compare_kr_gap_and_go.py — strategy is added
to registry but yaml stays disabled until backtest validates.
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
    # (name,                  extras,                       params)
    ("V0_live",                set(),                       {}),
    ("V1_with_eod",            {"eod_momentum"},            {}),
    ("V2_eod_only",            {"eod_momentum", "dual_momentum_OFF", "supertrend_OFF"}, {}),
    ("V3_eod_breakout_010",    {"eod_momentum"},            {"breakout_buffer": 0.01}),
    ("V4_eod_lookback_10",     {"eod_momentum"},            {"lookback_days": 10}),
    ("V5_eod_lookback_40",     {"eod_momentum"},            {"lookback_days": 40}),
    ("V6_eod_vol_20",          {"eod_momentum"},            {"min_vol_ratio": 2.0}),
    ("V7_eod_relaxed",         {"eod_momentum"},
        {"close_top_pct": 0.50, "prev_proximity": 0.90}),
]


def _kr_cfg(extra: set[str], _params: dict) -> dict:
    loader = StrategyConfigLoader()
    yaml_disabled = set(loader.get_market_disabled_strategies("KR"))
    keep_extras = extra - {"dual_momentum_OFF", "supertrend_OFF"}
    drop_baselines = (
        {"dual_momentum"} if "dual_momentum_OFF" in extra else set()
    ) | (
        {"supertrend"} if "supertrend_OFF" in extra else set()
    )
    disabled = sorted((yaml_disabled - keep_extras) | drop_baselines)
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


async def run(name: str, extra: set[str], params: dict) -> dict:
    cfg = PipelineConfig(**_kr_cfg(extra, params))
    eng = FullPipelineBacktest(cfg)
    if "eod_momentum" in extra:
        from strategies.eod_momentum import EODMomentumStrategy
        eng._registry._strategies["eod_momentum"] = EODMomentumStrategy(
            params=params or None
        )
        # Inject into profiles so combiner picks up the signal
        # (we mutate the in-memory loader, doesn't write yaml)
        for state in ("strong_uptrend", "uptrend", "sideways"):
            profile = eng._registry._config_loader._config["profiles"].get(state, {})
            profile["eod_momentum"] = 0.10
            eng._registry._config_loader._config["profiles"][state] = profile

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

    eod_stats = res.strategy_stats.get("eod_momentum", {})
    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        eod_trades=eod_stats.get("trades", 0),
        eod_pnl=round(eod_stats.get("pnl", 0.0), 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 120)
    print("  KR EOD Momentum validation (2y) — Tier-2 follow-up")
    print("=" * 120)
    results = []
    for name, extras, params in VARIANTS:
        print(f"\n▶ {name}  extras={sorted(extras)}  params={params}")
        r = await run(name, extras, params)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  "
            f"EODtrd={r['eod_trades']:>3}  EODpnl=₩{r['eod_pnl']:>+12,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 120)
    print("  SUMMARY")
    print("=" * 120)
    hdr = (
        f"{'Variant':<26} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Dep%':>6} {'EODtrd':>7} {'EODpnl':>15}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<26} {r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} "
            f"{r['pf']:>5.2f} {r['trades']:>4} {r['avg_deployed_pct']:>5.1f}% "
            f"{r['eod_trades']:>7} ₩{r['eod_pnl']:>+13,.0f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_live:")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<26} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
