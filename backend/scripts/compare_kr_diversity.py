"""KR diversity: supertrend re-enable + dual_momentum lookback variants."""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

# All KR disabled except what we explicitly enable
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
)

def disabled_except(keep: list[str]) -> list[str]:
    return [s for s in ALL_DISABLED if s not in keep]

VARIANTS = [
    ("V0_dm_only",            {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])}),
    ("V1_dm+supertrend",      {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "supertrend"])}),
    ("V2_dm+super+volsurge",  {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "supertrend", "volume_surge"])}),
    ("V3_supertrend_only",    {**KR_BASE, "disabled_strategies": disabled_except(["supertrend"])}),
    ("V4_dm_lookback6m",      {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])}),
    ("V5_dm6m+supertrend",    {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "supertrend"])}),
]


async def run(name, kw):
    cfg = PipelineConfig(**kw)
    # V4/V5: override dual_momentum lookback to 6 months
    if "6m" in name:
        from strategies.registry import StrategyRegistry
        from strategies.config_loader import StrategyConfigLoader
        loader = StrategyConfigLoader()
        reg = StrategyRegistry(config_loader=loader)
        dm = reg.get("dual_momentum")
        if dm:
            dm.set_params({"lookback_months": 6})

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
    print("  KR Diversity: supertrend + lookback variants (2Y, 100M KRW, 20% pos)")
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
              f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% ({r['elapsed']:.0f}s)")

    print(f"\n{'Variant':<25} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'WR%':>5}")
    print("-" * 80)
    for r in out:
        print(f"{r['name']:<25} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['wr']:>5.0f}%")

    for r in out:
        if r['strat_stats']:
            print(f"\n  {r['name']}:")
            for sn, st in sorted(r['strat_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
                if st['trades'] > 0:
                    print(f"    {sn:<25} trades={st['trades']:>3d} WR={st['win_rate']:>5.1f}% PnL={st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
