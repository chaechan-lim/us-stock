"""Compare KR with/without ETF engine + allocation variants."""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

KR_BASE = dict(
    market="KR",
    initial_equity=100_000_000,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=15,
    max_position_pct=0.10,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.08,
    volume_adjusted_slippage=True,
    min_confidence=0.40,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
    disabled_strategies=[
        "supertrend","macd_histogram","rsi_divergence","bollinger_squeeze",
        "volume_profile","regime_switch","sector_rotation","cis_momentum",
        "larry_williams","bnf_deviation","volume_surge","cross_sectional_momentum",
        "pead_drift","quality_factor","trend_following","donchian_breakout",
    ],
)

VARIANTS = [
    ("V0_dm_only",         {**KR_BASE}),
    ("V1_dm_lower_conf",   {**KR_BASE, "min_confidence": 0.30}),
    ("V2_dm_bigger_pos",   {**KR_BASE, "max_position_pct": 0.15}),
    ("V3_dm_bigger+low",   {**KR_BASE, "max_position_pct": 0.15, "min_confidence": 0.30}),
    ("V4_dm_20pct_pos",    {**KR_BASE, "max_position_pct": 0.20, "min_confidence": 0.30}),
]


async def run(name, kw):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
                mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
                wr=m.win_rate, alpha=m.alpha, elapsed=el, strat_stats=res.strategy_stats)


async def main():
    print("=" * 100)
    print("  KR Strategy Variants (dual_momentum only, 2Y, 100M KRW)")
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
              f"PF={r['pf']:.2f} Trades={r['trades']} Alpha={r['alpha']:+.1f}% ({r['elapsed']:.0f}s)")

    print(f"\n{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Alpha%':>8}")
    print("-" * 80)
    for r in out:
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['alpha']:>+8.1f}")

    if out:
        best = max(out, key=lambda x: x['sharpe'])
        print(f"\nBest ({best['name']}) strategy breakdown:")
        for sn, st in sorted(best['strat_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            if st['trades'] > 0:
                print(f"  {sn:<25} trades={st['trades']:>3d} WR={st['win_rate']:>5.1f}% PnL={st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
