"""KR exposure sweep (2y) — direct attack on chronic under-deployment.

Live observation 2026-05-29 EOD:
  KR funnel: 239 signals → 0 placed (fill 0%)
    sell_cooldown:        173 (72.4%)
    same_signal_24h:      64  (26.8%)
    sizing_too_high:      2   (0.8%)
  KR deployed_pct: 22.6%, slot_fill 32%, idle_days 1
  8 of 14 KR positions are 1-share placeholders

Variants test which yaml-level levers actually deploy cash:

  V0_baseline            sell_cd=3 min_pos=0.05 max_pos=18
  V1_cd1                 sell_cd=1 (revert #173 — undoes churn-protection)
  V2_min3                min_pos=0.03 (smaller slot, more names fit)
  V3_pos25               max_pos=25 (more concurrent slots)
  V4_cd1_min3            V1 + V2 (aggressive)
  V5_cd1_pos25           V1 + V3
  V6_all                 V1 + V2 + V3 (max aggressive)
  V7_min2_pos30          min_pos=0.02 max_pos=30 (extreme)

Primary metric: **avg deployed_pct over 2y** (cash deployment proxy).
Secondary: Ret/Sharpe/MDD/PF/Trades to verify no catastrophic
regression in risk-adjusted return.
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
    # (name,                    sell_cd, min_pos, max_pos)
    ("V0_baseline",                3,    0.05,    18),
    ("V1_cd1",                     1,    0.05,    18),
    ("V2_min3",                    3,    0.03,    18),
    ("V3_pos25",                   3,    0.05,    25),
    ("V4_cd1_min3",                1,    0.03,    18),
    ("V5_cd1_pos25",               1,    0.05,    25),
    ("V6_all",                     1,    0.03,    25),
    ("V7_min2_pos30",              1,    0.02,    30),
]


def _kr_cfg(sell_cd: int, min_pos: float, max_pos: int) -> dict:
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
        max_positions=max_pos,
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=min_pos,
        sell_cooldown_days=sell_cd,
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(eval_cfg.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )


async def run(name: str, sell_cd: int, min_pos: float, max_pos: int) -> dict:
    cfg = PipelineConfig(**_kr_cfg(sell_cd, min_pos, max_pos))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    # Compute avg deployed_pct over the backtest window from daily snapshots.
    snaps = eng._daily_snapshots
    if snaps:
        deployments = [
            (s.equity - s.cash) / s.equity if s.equity > 0 else 0.0
            for s in snaps
        ]
        avg_deployed = sum(deployments) / len(deployments)
        min_deployed = min(deployments)
        max_deployed = max(deployments)
        # Avg position count over window
        avg_n_pos = sum(s.n_positions for s in snaps) / len(snaps)
    else:
        avg_deployed = min_deployed = max_deployed = avg_n_pos = 0.0

    avg_alloc = 0.0
    if res.trades:
        sizes = [t.entry_price * t.quantity for t in res.trades if t.quantity > 0]
        if sizes:
            avg_alloc = sum(sizes) / len(sizes)

    return dict(
        name=name, sell_cd=sell_cd, min_pos=min_pos, max_pos=max_pos,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_alloc=round(avg_alloc, 0),
        deployed=round(avg_deployed * 100, 1),
        deployed_min=round(min_deployed * 100, 1),
        deployed_max=round(max_deployed * 100, 1),
        avg_n_pos=round(avg_n_pos, 1),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 115)
    print("  KR exposure sweep (2y) — direct attack on chronic under-deployment")
    print("=" * 115)
    results = []
    for name, sd, mp, xp in VARIANTS:
        print(f"\n▶ {name}  sell_cd={sd}d min_pos={mp} max_pos={xp}")
        r = await run(name, sd, mp, xp)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  AvgAlloc=₩{r['avg_alloc']:,.0f}  "
            f"Deployed={r['deployed']:.1f}% (min {r['deployed_min']:.1f} max {r['deployed_max']:.1f})  "
            f"AvgPos={r['avg_n_pos']:.1f}  ({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY (sorted by Deployed%)")
    print("=" * 115)
    hdr = (
        f"{'Variant':<22} {'sell_cd':>7} {'min_pos':>7} {'max_pos':>7} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'Deployed%':>10} {'AvgPos':>7} {'AvgAlloc':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: -x["deployed"]):
        print(
            f"{r['name']:<22} {r['sell_cd']:>7d} {r['min_pos']:>7.2f} {r['max_pos']:>7d} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} {r['deployed']:>9.1f}% {r['avg_n_pos']:>7.1f} "
            f"₩{r['avg_alloc']:>12,.0f}"
        )

    base = next(r for r in results if r["name"] == "V0_baseline")
    print("\n  vs V0_baseline:")
    for r in results:
        if r["name"] == "V0_baseline":
            continue
        d_deploy = r["deployed"] - base["deployed"]
        d_ret = r["ret"] - base["ret"]
        d_shp = r["sharpe"] - base["sharpe"]
        d_mdd = r["mdd"] - base["mdd"]
        d_pf = r["pf"] - base["pf"]
        d_trades = r["trades"] - base["trades"]
        # Tag: ★ = strictly better deployment AND no catastrophic loss
        improves_deploy = d_deploy > 5.0
        no_crash = d_shp >= -0.20 and d_mdd <= 3.0 and d_pf >= -0.10
        tag = "★" if improves_deploy and no_crash else ("△" if d_deploy > 0 else "✗")
        print(
            f"    {r['name']:<22} ΔDeploy={d_deploy:+5.1f}pp  "
            f"ΔRet={d_ret:+5.1f}  ΔSharpe={d_shp:+5.2f}  "
            f"ΔMDD={d_mdd:+5.1f}  ΔPF={d_pf:+5.2f}  Δtrd={d_trades:+4}  {tag}"
        )


if __name__ == "__main__":
    asyncio.run(main())
