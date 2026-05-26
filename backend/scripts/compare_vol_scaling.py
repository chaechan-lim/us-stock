"""KR vol scaling sweep (2y) — find the right target_risk_pct / min_scale
for risk-parity sizing when ATR% of the universe sits 4-7% (KR mid-caps).

Live observation (2026-05-26): KR equity ₩29M, cash 89% idle. Funnel
gates fire as designed; the actual bottleneck is `apply_volatility_
scaling` (evaluation_loop.py:2106). target_risk_pct=0.02 with floor 0.3
was US-calibrated and clamps almost every KR buy to 1 share. The
backtest path didn't apply this scaling at all until #X, so prior KR
backtest numbers under-represent the live drag.

Variants (KR 2y, on the post-#X backtest path with enable_vol_scaling=True):
  V0_no_scale  : baseline — vol scaling off (matches pre-#X behavior)
  V1_us_2_30   : 0.02 / 0.30 (live US default — what KR currently runs)
  V2_us_2_50   : 0.02 / 0.50 (raise floor only)
  V3_kr_3_50   : 0.03 / 0.50
  V4_kr_4_50   : 0.04 / 0.50 (proposed live default)
  V5_kr_4_60   : 0.04 / 0.60

Operator gate: pick whichever clears 4 dims (Ret, Sharpe, MDD, PF) vs
V0 with the largest exposure boost (= smallest residual cash).
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
    # (name,            enable, target, min, max)
    ("V0_no_scale",     False, 0.02, 0.30, 1.5),
    ("V1_us_2_30",      True,  0.02, 0.30, 1.5),
    ("V2_us_2_50",      True,  0.02, 0.50, 1.5),
    ("V3_kr_3_50",      True,  0.03, 0.50, 1.5),
    ("V4_kr_4_50",      True,  0.04, 0.50, 1.5),
    ("V5_kr_4_60",      True,  0.04, 0.60, 1.5),
]


def _kr_cfg(enable: bool, target: float, lo: float, hi: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    risk_cfg = loader.get_market_risk_config("KR")
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk_cfg.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk_cfg.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk_cfg.get("max_positions", 18)),
        max_position_pct=float(risk_cfg.get("max_position_pct", 0.20)),
        min_position_pct=float(risk_cfg.get("min_position_pct", 0.04)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 1)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.50)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=enable,
        vol_scale_target_risk_pct=target,
        vol_scale_min=lo,
        vol_scale_max=hi,
    )


async def run(name: str, enable: bool, target: float, lo: float, hi: float) -> dict:
    cfg = PipelineConfig(**_kr_cfg(enable, target, lo, hi))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    # Average position size (KRW) at entry — proxy for cash deployment.
    avg_alloc = 0.0
    if res.trades:
        sizes = [
            t.entry_price * t.quantity
            for t in res.trades if t.quantity > 0
        ]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)

    # Skip cash% — backtest result shape varies; AvgAlloc is the
    # actionable proxy for deployment.
    avg_cash_pct = None

    return dict(
        name=name, enable=enable, target=target, lo=lo, hi=hi,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_alloc=round(avg_alloc, 0),
        avg_cash_pct=avg_cash_pct,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  KR vol scaling sweep (2y) — find the right risk-parity calibration")
    print("=" * 110)
    results = []
    for name, en, target, lo, hi in VARIANTS:
        label = f"{name} enable={en} target={target} min={lo} max={hi}"
        print(f"\n▶ {label}")
        r = await run(name, en, target, lo, hi)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  AvgAlloc=₩{r['avg_alloc']:,.0f}  "
            f"AvgCash={r['avg_cash_pct']}%  ({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<14} {'en':>3} {'targ':>5} {'min':>5} {'max':>4} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trd':>4} "
        f"{'AvgAlloc':>11} {'Cash%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        en = "✓" if r["enable"] else "✗"
        cash = f"{r['avg_cash_pct']:.0f}" if r["avg_cash_pct"] is not None else "—"
        print(
            f"{r['name']:<14} {en:>3} {r['target']:>5.2f} {r['lo']:>5.2f} {r['hi']:>4.1f} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} ₩{r['avg_alloc']:>10,.0f} {cash:>5}%"
        )

    # 4-dim check vs V0 baseline
    if len(results) > 1:
        base = results[0]
        print("\n  vs V0 (baseline, vol-scale off):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([
                dret > -1.0,  # within 1pp ret regression
                dshp > -0.10,
                dmdd > -3.0,  # MDD doesn't blow out 3pp
                dpf > -0.10,
            ])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<14} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  4-dim={improves}/4  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
