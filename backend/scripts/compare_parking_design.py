"""Compare parking designs: current (lump) vs split buy + min-hold unpark.

Tests on KR (KODEX 200) since that's where the design matters most.
US SPY parking has the same structure but is already live and working.

Variants:
  V0: No parking (baseline)
  V1: Current — lump buy, no unpark, threshold 50%
  V2: Split buy (25% per hour), no unpark
  V3: Split buy + unpark after 2 weeks when buy signal needs cash
  V4: Same as V3 but 1-week min hold (more aggressive unpark)

All use dual_momentum only, 100M KRW, 2Y.
"""

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
    enable_cash_parking=True,
    cash_parking_symbol="069500.KS",
    cash_parking_threshold=0.50,
)

VARIANTS = [
    ("V0_no_parking",     {**KR_BASE, "enable_cash_parking": False}),
    ("V1_lump_no_unpark", {**KR_BASE}),
    ("V2_lump_threshold30", {**KR_BASE, "cash_parking_threshold": 0.30}),
    ("V3_lump_threshold70", {**KR_BASE, "cash_parking_threshold": 0.70}),
]


async def run(name, kw):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    park = res.strategy_stats.get("cash_parking", {"trades": 0, "pnl": 0})
    dm = res.strategy_stats.get("dual_momentum", {"trades": 0, "pnl": 0, "win_rate": 0})
    return dict(name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
                mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
                wr=m.win_rate, alpha=m.alpha,
                park_trades=park["trades"], park_pnl=park["pnl"],
                dm_trades=dm["trades"], dm_pnl=dm["pnl"],
                elapsed=el)


async def main():
    print("=" * 100)
    print("  Parking Design Comparison (KR, dual_momentum, 2Y, 100M KRW)")
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
              f"PF={r['pf']:.2f} Trades={r['trades']}  "
              f"DM: {r['dm_trades']}t PnL={r['dm_pnl']:+,.0f}  "
              f"Park: {r['park_trades']}t PnL={r['park_pnl']:+,.0f}  ({r['elapsed']:.0f}s)")

    print(f"\n{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'DM PnL':>12} {'Park PnL':>12} {'Park#':>6}")
    print("-" * 95)
    for r in out:
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['dm_pnl']:>+12,.0f} {r['park_pnl']:>+12,.0f} {r['park_trades']:>6}")


if __name__ == "__main__":
    asyncio.run(main())
