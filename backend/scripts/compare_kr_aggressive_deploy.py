"""KR aggressive deploy sweep — V1 vs V2 vs cash_parking re-enable.

User priority: 더 많은 deploy. V1 (min_pos 0.07) gave 77.7% backtest
deploy. User wants more. Tests:
  V1 (current live)         min_pos 0.07
  V2 (min_pos 0.10)         더 큰 사이즈
  V_park (V1 + cash_park)   유휴 cash → KODEX 200 ETF
  V_park_v2                 V2 + parking
  V_park_only               cash_parking 만 추가 (V0 baseline)
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
    ("V0_base",                    {"min_position_pct": 0.05}),
    ("V1_min07_live",              {"min_position_pct": 0.07}),
    ("V2_min10",                   {"min_position_pct": 0.10}),
    ("V_parking_only",             {
        "min_position_pct": 0.05,
        "enable_cash_parking": True,
        "cash_parking_symbol": "069500.KS",   # KODEX 200
        "cash_parking_threshold": 0.50,        # park when cash >= 50%
        "cash_parking_min_hold_days": 5,
        "cash_parking_enable_unpark": True,
        "cash_parking_max_pct": 0.50,
    }),
    ("V_V1_parking",               {
        "min_position_pct": 0.07,
        "enable_cash_parking": True,
        "cash_parking_symbol": "069500.KS",
        "cash_parking_threshold": 0.50,
        "cash_parking_min_hold_days": 5,
        "cash_parking_enable_unpark": True,
        "cash_parking_max_pct": 0.50,
    }),
    ("V_V2_parking",               {
        "min_position_pct": 0.10,
        "enable_cash_parking": True,
        "cash_parking_symbol": "069500.KS",
        "cash_parking_threshold": 0.50,
        "cash_parking_min_hold_days": 5,
        "cash_parking_enable_unpark": True,
        "cash_parking_max_pct": 0.50,
    }),
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
    print("=" * 120); print("  KR aggressive deploy sweep"); print("=" * 120)
    results = []
    for name, ov in VARIANTS:
        print(f"\n▶ {name}  ov={ov}")
        r = await run(name, ov); results.append(r)
        print(f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  MDD={r['mdd']:6.1f}%  "
              f"PF={r['pf']:.2f}  Trd={r['trades']:>4}  Dep={r['dep']:5.1f}%  "
              f"AvgAlloc=₩{r['avg_alloc']:>11,.0f}  ({r['elapsed']:.0f}s)")
    base = results[0]
    print(f"\n  vs V0_base:")
    for r in results[1:]:
        dret = r["ret"]-base["ret"]; dshp = r["sharpe"]-base["sharpe"]
        dmdd = r["mdd"]-base["mdd"]; dpf = r["pf"]-base["pf"]
        ddep = r["dep"]-base["dep"]
        imp = sum([dret>-1.0, dshp>-0.10, dmdd>-3.0, dpf>-0.10])
        tag = "✓" if imp==4 else "△" if imp>=3 else "✗"
        print(f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
              f"ΔMDD={dmdd:+5.1f}  ΔDep={ddep:+5.1f}pp  ΔPF={dpf:+5.2f}  {tag}")

if __name__ == "__main__":
    asyncio.run(main())
