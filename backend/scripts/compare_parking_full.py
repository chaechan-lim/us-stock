"""Full parking design comparison: lump vs split, with/without unpark.

Tests on both US and KR.
"""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

# ── KR ──
KR_DISABLED = [
    "supertrend","macd_histogram","rsi_divergence","bollinger_squeeze",
    "volume_profile","regime_switch","sector_rotation","cis_momentum",
    "larry_williams","bnf_deviation","volume_surge","cross_sectional_momentum",
    "pead_drift","quality_factor","trend_following","donchian_breakout",
]
KR_BASE = dict(
    market="KR", initial_equity=100_000_000,
    default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
    max_positions=12, max_position_pct=0.20,
    sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
    slippage_pct=0.08, volume_adjusted_slippage=True,
    min_confidence=0.30, held_sell_bias=0.05, held_min_confidence=0.40,
    disabled_strategies=[s for s in KR_DISABLED if s != "dual_momentum"],
    enable_cash_parking=True, cash_parking_symbol="069500.KS",
    cash_parking_threshold=0.70,
)

# ── US ──
US_DISABLED = [
    "bollinger_squeeze","volume_profile","cis_momentum","larry_williams",
    "bnf_deviation","donchian_breakout","dual_momentum","quality_factor",
]
US_BASE = dict(
    market="US", initial_equity=100_000,
    default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
    max_positions=20, max_position_pct=0.08,
    sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
    slippage_pct=0.05, volume_adjusted_slippage=True,
    min_confidence=0.30, held_sell_bias=0.05, held_min_confidence=0.40,
    disabled_strategies=US_DISABLED,
    enable_cash_parking=True, cash_parking_threshold=0.50,
)

VARIANTS = [
    # KR variants
    ("KR_V0_no_park",           {**KR_BASE, "enable_cash_parking": False}),
    ("KR_V1_lump_no_unpark",    {**KR_BASE}),
    ("KR_V2_lump_unpark_2w",    {**KR_BASE, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 10}),
    ("KR_V3_split25_unpark_2w", {**KR_BASE, "cash_parking_split_ratio": 0.25, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 10}),
    ("KR_V4_split25_unpark_1w", {**KR_BASE, "cash_parking_split_ratio": 0.25, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 5}),
    # US variants
    ("US_V0_no_park",           {**US_BASE, "enable_cash_parking": False}),
    ("US_V1_lump_no_unpark",    {**US_BASE}),
    ("US_V2_lump_unpark_2w",    {**US_BASE, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 10}),
    ("US_V3_split25_unpark_2w", {**US_BASE, "cash_parking_split_ratio": 0.25, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 10}),
    ("US_V4_split25_unpark_1w", {**US_BASE, "cash_parking_split_ratio": 0.25, "cash_parking_enable_unpark": True, "cash_parking_min_hold_days": 5}),
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
                wr=m.win_rate, park_trades=park["trades"], park_pnl=park["pnl"],
                elapsed=el)


async def main():
    print("=" * 110)
    print("  Full Parking Design: lump vs split, with/without unpark (2Y)")
    print("=" * 110)
    out = []
    for name, kw in VARIANTS:
        print(f"\n▶ {name}")
        try:
            r = await run(name, kw)
        except Exception as e:
            print(f"  ✗ {e}"); continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.1f}% "
              f"PF={r['pf']:.2f} Trades={r['trades']} Park={r['park_trades']}t "
              f"PnL={r['park_pnl']:+,.0f} ({r['elapsed']:.0f}s)")

    for market in ("KR", "US"):
        group = [r for r in out if r["name"].startswith(market)]
        if not group:
            continue
        print(f"\n{'─'*10} {market} {'─'*80}")
        print(f"{'Variant':<25} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Park#':>6} {'Park PnL':>12}")
        print("-" * 95)
        for r in group:
            print(f"{r['name']:<25} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
                  f"{r['pf']:>6.2f} {r['trades']:>7d} {r['park_trades']:>6} {r['park_pnl']:>+12,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
