"""KR Sector-RS strategy backtest (2y) — Track C.

Tests whether a standalone Sector-RS strategy (BUY when sector is top-N
AND own momentum positive) adds alpha on top of existing dual_momentum +
supertrend + sector_boost combo.

Per Phase B doc this is expected to be low/no alpha (overlap with
sector_boost which already multiplies confidence by sector strength).
Build + test to confirm.
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
    ("V0_live",                set(),               {}),
    ("V1_with_srs",            {"sector_rs"},       {}),
    ("V2_srs_only",            {"sector_rs", "dual_momentum_OFF", "supertrend_OFF"}, {}),
    ("V3_srs_top5",            {"sector_rs"},       {"top_n_sectors": 5}),
    ("V4_srs_loose_mom",       {"sector_rs"},       {"min_own_momentum": 0.01}),
    ("V5_srs_floor70",         {"sector_rs"},       {"strength_pct_floor": 70.0}),
]


def _kr_cfg(extra: set[str], _params: dict) -> dict:
    loader = StrategyConfigLoader()
    yaml_disabled = set(loader.get_market_disabled_strategies("KR"))
    keep_extras = extra - {"dual_momentum_OFF", "supertrend_OFF"}
    drop = (
        {"dual_momentum"} if "dual_momentum_OFF" in extra else set()
    ) | (
        {"supertrend"} if "supertrend_OFF" in extra else set()
    )
    disabled = sorted((yaml_disabled - keep_extras) | drop)
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

    if "sector_rs" in extra:
        from strategies.sector_rs import SectorRSStrategy
        eng._registry._strategies["sector_rs"] = SectorRSStrategy(
            params=params or None,
        )
        # Inject profile weight so combiner doesn't suppress
        for state in ("strong_uptrend", "uptrend", "sideways"):
            profile = eng._registry._config_loader._config["profiles"].get(state, {})
            profile["sector_rs"] = 0.10
            eng._registry._config_loader._config["profiles"][state] = profile

        # Hook engine's per-day evaluation: before each day's analyze,
        # refresh sector snapshot on the strategy class. The full_pipeline
        # engine already has sector_history; we tap into the date loop.
        orig_eval = eng._evaluate_at_date if hasattr(eng, "_evaluate_at_date") else None
        if eng._sector_history is not None:
            # Set once with most-recent score; backtest engine doesn't expose
            # a per-bar sector hook, so we use the latest snapshot. This is
            # a known limitation (we'd need to plumb sector_history into
            # _process_signals to get per-bar accuracy).
            last_date = eng._sector_history.dates[-1] if eng._sector_history.dates else ""
            scores = eng._sector_history.score_at(last_date) if last_date else {}
            symbol_map = eng._sector_history.symbol_sector
            SectorRSStrategy.set_sector_snapshot(scores, symbol_map)

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
    srs_stats = res.strategy_stats.get("sector_rs", {})
    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        srs_trades=srs_stats.get("trades", 0),
        srs_pnl=round(srs_stats.get("pnl", 0.0), 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 115)
    print("  KR Sector-RS validation (2y) — Track C")
    print("=" * 115)
    results = []
    for name, extras, params in VARIANTS:
        print(f"\n▶ {name}  extras={sorted(extras)}  params={params}")
        r = await run(name, extras, params)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  "
            f"SRStrd={r['srs_trades']:>3}  SRSpnl=₩{r['srs_pnl']:>+12,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY vs V0_live")
    print("=" * 115)
    if len(results) > 1:
        base = results[0]
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
