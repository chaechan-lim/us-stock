"""KR anti-churn sweep — min_hold_days variation on bnf-w-0.20 baseline.

Live observed bnf↔supertrend conflict: BUY 09:31 → SELL 10:26 (55min),
4 stocks round-tripped same day. User concern this is churn-only with
no alpha. Backtest tests whether tightening min_hold reduces churn
*without* losing alpha.

Backtest grain is daily so min_hold_days controls inter-day re-sell:
  1d  (current live):  positions can be sold the day after buy
  2d  next-day re-sell blocked
  3d  must hold 3 trading days minimum
  5d  must hold 1 week minimum
  + held_sell_bias variants — reduce SELL prob for held positions
  + held_min_confidence increase — make SELLs harder when held
"""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies","engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


VARIANTS = [
    ("V0_live",                  {}),
    ("V1_min_hold_2d",           {"min_hold_days": 2}),
    ("V2_min_hold_3d",           {"min_hold_days": 3}),
    ("V3_min_hold_5d",           {"min_hold_days": 5}),
    # held_sell_bias: reduce SELL probability for held positions
    ("V4_held_bias_05",          {"held_sell_bias": -0.05}),
    ("V5_held_bias_10",          {"held_sell_bias": -0.10}),
    # held_min_confidence: raise SELL threshold when held
    ("V6_held_conf_055",         {"held_min_confidence": 0.55}),
    ("V7_held_conf_070",         {"held_min_confidence": 0.70}),
    # Combos
    ("V8_2d_bias_05",            {"min_hold_days": 2, "held_sell_bias": -0.05}),
    ("V9_3d_conf_055",           {"min_hold_days": 3, "held_min_confidence": 0.55}),
]


def _kr_cfg(ov):
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    base = dict(
        market="KR", initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 3)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08, volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        stale_time_days=int(eval_cfg.get("stale_time_days", 0)),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold", 0.0)),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True, enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )
    base.update(ov)
    return base


async def run(name, ov):
    cfg = PipelineConfig(**_kr_cfg(ov))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    dep = sum((s.equity-s.cash)/s.equity for s in snaps if s.equity>0)/len(snaps) if snaps else 0
    sizes = [t.entry_price*t.quantity for t in res.trades if t.quantity>0]
    avg_alloc = sum(sizes)/len(sizes) if sizes else 0
    return dict(name=name, ret=round(m.total_return_pct,2), sharpe=round(m.sharpe_ratio,2),
                mdd=round(m.max_drawdown_pct,2), pf=round(m.profit_factor,2),
                trades=m.total_trades, dep=round(dep*100,1), avg_alloc=round(avg_alloc,0),
                elapsed=round(el,1))


async def main():
    print("=" * 120); print("  KR anti-churn sweep on bnf-w-0.20 baseline"); print("=" * 120)
    results = []
    for name, ov in VARIANTS:
        print(f"\n▶ {name}  ov={ov}")
        r = await run(name, ov); results.append(r)
        print(f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  MDD={r['mdd']:6.1f}%  "
              f"PF={r['pf']:.2f}  Trd={r['trades']:>4}  Dep={r['dep']:5.1f}%  "
              f"AvgAlloc=₩{r['avg_alloc']:>11,.0f}  ({r['elapsed']:.0f}s)")
    base = results[0]
    print(f"\n  vs V0:")
    for r in results[1:]:
        dret = r["ret"]-base["ret"]; dshp = r["sharpe"]-base["sharpe"]
        dmdd = r["mdd"]-base["mdd"]; dpf = r["pf"]-base["pf"]
        dtrd = r["trades"]-base["trades"]
        imp = sum([dret>-1.0, dshp>-0.10, dmdd>-3.0, dpf>-0.10])
        tag = "✓" if imp==4 else "△" if imp>=3 else "✗"
        print(f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
              f"ΔMDD={dmdd:+5.1f}  ΔTrd={dtrd:+4d}  ΔPF={dpf:+5.2f}  {tag}")


if __name__ == "__main__":
    asyncio.run(main())
