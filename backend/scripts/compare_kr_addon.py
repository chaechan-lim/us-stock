"""KR holding add-on (sizing-up) sweep (2y).

User context (2026-06-01): live KR has 4/5 positions at 1-3 shares
(placeholder level, <2% of equity), but the original strategies don't
re-fire on the held names so they stay tiny. The live `sizing_up` path
converts a fresh BUY on a held + undersized symbol into an add-on. This
backtest enables the equivalent code path in `full_pipeline.py` (gated
by `enable_holding_addon=False` by default, so all existing scripts
are unaffected).

Variants (threshold = the under-sized cutoff as a fraction of min_pos):
  V0_off                          add-on disabled (baseline)
  V1_thr0.3                       only top-up when value < 0.3 × min_pos × eq
  V2_thr0.5                       only top-up when value < 0.5 × min_pos × eq
  V3_thr0.7                       only top-up when value < 0.7 × min_pos × eq
  V4_thr1.0                       always top up below the min_pos floor
  V5_thr0.5_conf0.30              looser conf gate (lower bar for add-ons)
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
    # (name,           enabled, threshold, min_conf)
    ("V0_off",         False,   0.5,       0.50),
    ("V1_thr0.3",      True,    0.3,       0.50),
    ("V2_thr0.5",      True,    0.5,       0.50),
    ("V3_thr0.7",      True,    0.7,       0.50),
    ("V4_thr1.0",      True,    1.0,       0.50),
    ("V5_thr0.5_lc",   True,    0.5,       0.30),
]


def _kr_cfg(enabled: bool, threshold: float, min_conf: float) -> dict:
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
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 1)),
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
        enable_holding_addon=enabled,
        holding_addon_threshold=threshold,
        holding_addon_min_confidence=min_conf,
    )


async def run(name: str, enabled: bool, threshold: float, min_conf: float) -> dict:
    cfg = PipelineConfig(**_kr_cfg(enabled, threshold, min_conf))
    eng = FullPipelineBacktest(cfg)
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

    return dict(
        name=name, enabled=enabled, threshold=threshold, min_conf=min_conf,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  KR holding add-on sweep (2y)")
    print("=" * 110)
    results = []
    for name, en, thr, mc in VARIANTS:
        print(f"\n▶ {name}  enabled={en}  thr={thr}  min_conf={mc}")
        r = await run(name, en, thr, mc)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  Deployed={r['avg_deployed_pct']:5.1f}%  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<14} {'en':>3} {'thr':>5} {'mc':>5} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Deploy%':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<14} {str(r['enabled'])[0]:>3} {r['threshold']:>5.2f} {r['min_conf']:>5.2f} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} {r['avg_deployed_pct']:>7.1f}%"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_off (baseline):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            ddep = r["avg_deployed_pct"] - base["avg_deployed_pct"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<14} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔDeploy={ddep:+5.1f}pp  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
