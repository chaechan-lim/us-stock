"""KR bnf_deviation profile weight sweep (2y).

Current profile weights for bnf_deviation:
  strong_uptrend: 0.05  uptrend: 0.05  sideways: 0.05
  weak_downtrend: 0.25  downtrend: 0.40

Live simulation shows bnf generates 9 BUY signals on KR watchlist today
but combiner suppresses most (weight 0.05 + opposing supertrend/dm
SELLs). Question: does raising bnf weight in uptrend/sideways profiles
let more bnf signals through without diluting the proven 14% baseline?

On post-bnf-loose baseline (bnf enabled, params -3/+2, weights all 0.05).
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


# Per variant: weight in (strong_uptrend, uptrend, sideways)
VARIANTS = [
    ("V0_live_05",        0.05),
    ("V1_w_010",          0.10),
    ("V2_w_015",          0.15),
    ("V3_w_020",          0.20),
    ("V4_w_025",          0.25),
]


def _kr_cfg() -> dict:
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


async def run(name: str, weight: float) -> dict:
    cfg = PipelineConfig(**_kr_cfg())
    eng = FullPipelineBacktest(cfg)
    # Mutate profile weights in-memory
    loader = eng._registry._config_loader
    for state in ("strong_uptrend", "uptrend", "sideways"):
        profile = loader._config["profiles"].get(state, {})
        profile["bnf_deviation"] = weight
        loader._config["profiles"][state] = profile

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

    bnf_stats = res.strategy_stats.get("bnf_deviation", {})
    return dict(
        name=name, weight=weight,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        bnf_trades=bnf_stats.get("trades", 0),
        bnf_pnl=round(bnf_stats.get("pnl", 0.0), 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 115)
    print("  KR bnf_deviation profile weight sweep (2y)")
    print("=" * 115)
    results = []
    for name, w in VARIANTS:
        print(f"\n▶ {name}  weight={w}")
        r = await run(name, w)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  "
            f"bnf={r['bnf_trades']}t/₩{r['bnf_pnl']:+,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY vs V0_live_05")
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
                f"    {r['name']:<14} w={r['weight']:.2f}  ΔRet={dret:+5.1f}  "
                f"ΔSharpe={dshp:+5.2f}  ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
