"""KR cash deployment sweep (2y) — find calibration that drops idle
cash without losing alpha.

Live 5-28: KR cash 78% (after #59-A/B/C + #168 vol scaling). Funnel
448/449 BUY signals rejected — sell_cooldown (62%) + same_signal_24h
(31%) account for 93%. Idle cash mostly comes from re-firing on the
same name and being throttled.

Variants (KR 2y, on the post-#168 baseline):
  V0_baseline    : current live config
  V1_minpos_5    : min_position_pct 0.04 → 0.05 (16% larger floor)
  V2_minpos_6    : min_position_pct 0.04 → 0.06 (50% larger floor)
  V3_cooldown_12h: sell_cooldown_days 1 → 0.5 (12h)
  V4_limit_15    : daily_buy_limit 10 → 15
  V5_combo_5_12h : V1 + V3
  V6_combo_6_12h : V2 + V3

Operator gate: pick whichever clears 4 dims (Ret, Sharpe, MDD, PF) vs
V0 with the smallest AvgCash% (= most cash actually deployed).
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
    # (name,              min_pos, cooldown_days, daily_buy_limit)
    ("V0_baseline",       0.04, 1.0, 10),
    ("V1_minpos_5",       0.05, 1.0, 10),
    ("V2_minpos_6",       0.06, 1.0, 10),
    ("V3_cooldown_12h",   0.04, 0.5, 10),
    ("V4_limit_15",       0.04, 1.0, 15),
    ("V5_combo_5_12h",    0.05, 0.5, 10),
    ("V6_combo_6_12h",    0.06, 0.5, 10),
]


def _kr_cfg(min_pos: float, cooldown_days: float, limit: int) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk_cfg = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol_cfg = risk_cfg.get("volatility_scaling") or {}
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk_cfg.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk_cfg.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk_cfg.get("max_positions", 18)),
        max_position_pct=float(risk_cfg.get("max_position_pct", 0.20)),
        min_position_pct=min_pos,
        sell_cooldown_days=int(round(cooldown_days)) if cooldown_days >= 1 else 0,
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.50)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol_cfg.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol_cfg.get("min_scale", 0.5)),
        vol_scale_max=float(vol_cfg.get("max_scale", 1.5)),
    )


async def run(name: str, min_pos: float, cooldown: float, limit: int) -> dict:
    cfg = PipelineConfig(**_kr_cfg(min_pos, cooldown, limit))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    avg_alloc = 0.0
    if res.trades:
        sizes = [
            t.entry_price * t.quantity for t in res.trades if t.quantity > 0
        ]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)

    return dict(
        name=name, min_pos=min_pos, cooldown=cooldown, limit=limit,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_alloc=round(avg_alloc, 0),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  KR cash deployment sweep (2y) — find min_pos / cooldown calibration")
    print("=" * 110)
    results = []
    for name, mp, cd, lim in VARIANTS:
        label = f"{name} min_pos={mp:.2f} cooldown={cd:.1f}d limit={lim}"
        print(f"\n▶ {label}")
        r = await run(name, mp, cd, lim)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  AvgAlloc=₩{r['avg_alloc']:,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<18} {'min_p':>6} {'cd':>5} {'limit':>6} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trd':>4} "
        f"{'AvgAlloc':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<18} {r['min_pos']:>6.2f} {r['cooldown']:>5.1f} {r['limit']:>6} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} ₩{r['avg_alloc']:>12,.0f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_baseline:")
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
                f"    {r['name']:<18} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
