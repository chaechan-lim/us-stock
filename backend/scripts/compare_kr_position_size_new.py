"""KR position size sweep on new baseline (post-bnf-weight-0.20).

Earlier compare_kr_cash_deployment / compare_kr_buy_size / compare_kr_levers
swept kelly_fraction, vol_target, vol_scale_min, min_position_pct on the
OLDER baseline (Ret ~12-14%). The current best baseline is Ret +18.5%
(bnf weight 0.20). Resize levers may behave differently on the new
baseline — re-measure.

User concern: per-position size too small (KR positions $200-400, avg
14 positions). +2.3pp from bnf-loose, +7.3pp from bnf-weight came from
COUNT increase (more trades). Need SIZE increase for actual deploy.
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
    # (name, overrides)
    ("V0_live",                    {}),
    # min_position_pct sweep
    ("V1_min_pos_007",             {"min_position_pct": 0.07}),
    ("V2_min_pos_010",             {"min_position_pct": 0.10}),
    # kelly_fraction sweep
    ("V3_kelly_050",               {"kelly_fraction": 0.50}),
    ("V4_kelly_060",               {"kelly_fraction": 0.60}),
    # vol_scale_min sweep
    ("V5_vol_min_07",              {"vol_scale_min": 0.7}),
    ("V6_vol_min_08",              {"vol_scale_min": 0.8}),
    # vol_target sweep
    ("V7_vol_target_005",          {"vol_scale_target_risk_pct": 0.05}),
    ("V8_vol_target_006",          {"vol_scale_target_risk_pct": 0.06}),
    # max_position_pct: allow bigger single positions
    ("V9_max_pos_025",             {"max_position_pct": 0.25}),
    # combos
    ("V10_combo_aggressive",       {
        "min_position_pct": 0.07,
        "kelly_fraction": 0.50,
        "vol_scale_min": 0.7,
    }),
    ("V11_combo_max_size",         {
        "min_position_pct": 0.10,
        "kelly_fraction": 0.60,
        "vol_scale_min": 0.8,
        "vol_scale_target_risk_pct": 0.06,
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

    # Avg buy size (KRW)
    avg_alloc = 0.0
    if res.trades:
        sizes = [t.entry_price * t.quantity for t in res.trades if t.quantity > 0]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)

    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        avg_npos=round(avg_npos, 1),
        avg_alloc=round(avg_alloc, 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 130)
    print("  KR position size sweep on post-bnf-weight-0.20 baseline (2y)")
    print("=" * 130)
    results = []
    for name, ov in VARIANTS:
        print(f"\n▶ {name}  ov={ov}")
        r = await run(name, ov)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  Pos={r['avg_npos']:4.1f}  "
            f"AvgAlloc=₩{r['avg_alloc']:>11,.0f}  ({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 130)
    print("  SUMMARY vs V0_live")
    print("=" * 130)
    if len(results) > 1:
        base = results[0]
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            dalloc = r["avg_alloc"] - base["avg_alloc"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<26} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔAlloc=₩{dalloc:>+10,.0f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
