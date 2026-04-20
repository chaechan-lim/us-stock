"""KR cash utilization: parking vs bigger positions vs fewer positions."""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

ALL_DISABLED = [
    "supertrend","macd_histogram","rsi_divergence","bollinger_squeeze",
    "volume_profile","regime_switch","sector_rotation","cis_momentum",
    "larry_williams","bnf_deviation","volume_surge","cross_sectional_momentum",
    "pead_drift","quality_factor","trend_following","donchian_breakout",
]

KR_BASE = dict(
    market="KR",
    initial_equity=100_000_000,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=12,
    max_position_pct=0.20,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.08,
    volume_adjusted_slippage=True,
    min_confidence=0.30,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
    disabled_strategies=[s for s in ALL_DISABLED if s != "dual_momentum"],
)

VARIANTS = [
    ("V0_current",        {**KR_BASE}),
    ("V1_park_kodex200",  {**KR_BASE, "enable_cash_parking": True, "cash_parking_symbol": "069500.KS",
                           "cash_parking_threshold": 0.50}),
    ("V2_25pct_pos",      {**KR_BASE, "max_position_pct": 0.25}),
    ("V3_30pct_pos",      {**KR_BASE, "max_position_pct": 0.30}),
    ("V4_8pos_25pct",     {**KR_BASE, "max_positions": 8, "max_position_pct": 0.25}),
    ("V5_6pos_30pct",     {**KR_BASE, "max_positions": 6, "max_position_pct": 0.30}),
]


async def run(name, kw):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    park = res.strategy_stats.get("cash_parking", {"trades": 0, "pnl": 0})
    return dict(name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
                mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
                wr=m.win_rate, alpha=m.alpha, park_pnl=park["pnl"],
                elapsed=el, strat_stats=res.strategy_stats)


async def main():
    print("=" * 100)
    print("  KR Cash Utilization (dual_momentum only, 2Y, 100M KRW)")
    print("=" * 100)
    out = []
    for name, kw in VARIANTS:
        print(f"\n▶ {name}")
        try:
            r = await run(name, kw)
        except Exception as e:
            print(f"  ✗ {e}"); continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.1f}% "
              f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% "
              f"Park={r['park_pnl']:+,.0f} ({r['elapsed']:.0f}s)")

    print(f"\n{'Variant':<20} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'WR%':>5} {'Park PnL':>10}")
    print("-" * 85)
    for r in out:
        print(f"{r['name']:<20} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['wr']:>5.0f}% {r['park_pnl']:>+10,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
