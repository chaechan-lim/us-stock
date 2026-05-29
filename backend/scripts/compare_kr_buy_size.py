"""KR buy size sweep (2y) — kelly_fraction × vol_scale.target_risk_pct.

User suggestion (2026-05-29): 14d KR average buy size only ₩300K-1M
(1-3% of equity) while min_pos_pct=5% → ₩1.5M intended floor. Means
Kelly is undersized + vol_scale is cutting it more. Bigger buys = same
cash deployed in fewer trades = less same-symbol churn.

Variants (KR 2y, on post-#173 baseline: sell_cd=3d, vol_scale 0.04/0.5,
min_pos=0.05, kelly=0.40, rsi_div enabled):

  V0_baseline       kelly=0.40 vol_target=0.04
  V1_kelly_0.50     kelly=0.50 vol_target=0.04
  V2_kelly_0.60     kelly=0.60 vol_target=0.04
  V3_vol_0.05       kelly=0.40 vol_target=0.05
  V4_vol_0.06       kelly=0.40 vol_target=0.06
  V5_combo_k0.5_v0.05
  V6_combo_k0.6_v0.06
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
    # (name,                    kelly, vol_target)
    ("V0_baseline",             0.40, 0.04),
    ("V1_kelly_0.50",           0.50, 0.04),
    ("V2_kelly_0.60",           0.60, 0.04),
    ("V3_vol_0.05",             0.40, 0.05),
    ("V4_vol_0.06",             0.40, 0.06),
    ("V5_combo_k0.5_v0.05",     0.50, 0.05),
    ("V6_combo_k0.6_v0.06",     0.60, 0.06),
]


def _kr_cfg(kelly: float, vol_target: float) -> dict:
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
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=kelly,
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=vol_target,
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )


async def run(name: str, kelly: float, vt: float) -> dict:
    cfg = PipelineConfig(**_kr_cfg(kelly, vt))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    avg_alloc = 0.0
    if res.trades:
        sizes = [t.entry_price * t.quantity for t in res.trades if t.quantity > 0]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)

    return dict(
        name=name, kelly=kelly, vt=vt,
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
    print("  KR buy size sweep (2y) — kelly_fraction × vol_target")
    print("=" * 110)
    results = []
    for name, k, vt in VARIANTS:
        print(f"\n▶ {name}  kelly={k} vol_target={vt}")
        r = await run(name, k, vt)
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
        f"{'Variant':<24} {'kelly':>6} {'volT':>5} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'AvgAlloc':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<24} {r['kelly']:>6.2f} {r['vt']:>5.2f} "
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
            d_alloc = r["avg_alloc"] - base["avg_alloc"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<24} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  Δalloc=₩{d_alloc:>+10,.0f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
