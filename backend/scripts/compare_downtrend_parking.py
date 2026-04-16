"""Test cash_parking behavior during downtrend regimes."""

import asyncio, functools, logging, sys, time
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data", "backtest", "strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

US_BASE = dict(
    market="US", initial_equity=100_000,
    default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
    max_positions=20, max_position_pct=0.08,
    sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
    slippage_pct=0.05, volume_adjusted_slippage=True,
    min_confidence=0.30, held_sell_bias=0.05, held_min_confidence=0.40,
    enable_cash_parking=True, cash_parking_threshold=0.50,
    disabled_strategies=["bollinger_squeeze","volume_profile","cis_momentum",
                         "larry_williams","bnf_deviation","donchian_breakout",
                         "dual_momentum","quality_factor"],
)

VARIANTS = [
    ("V0_always_park",   {**US_BASE}),
    ("V1_skip_downtrend",{**US_BASE, "cash_parking_skip_downtrend": True}),
    ("V2_sell_downtrend",{**US_BASE, "cash_parking_skip_downtrend": True, "cash_parking_sell_on_downtrend": True}),
]


async def run(name: str, kw: dict) -> dict:
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    park = res.strategy_stats.get("cash_parking", {"trades": 0, "pnl": 0})
    return dict(name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
                mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
                alpha=m.alpha, park_trades=park["trades"], park_pnl=park["pnl"], elapsed=el)


async def main():
    print("=" * 110)
    print("  Downtrend Parking Strategies (min_conf=0.30, threshold=50%)")
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
              f"Alpha={r['alpha']:+.1f}% SPY trades={r['park_trades']} PnL=${r['park_pnl']:+,.0f} ({r['elapsed']:.0f}s)")

    print(f"\n{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'Alpha%':>8} {'SPY':>5} {'SPY PnL':>10}")
    print("-" * 80)
    for r in out:
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['alpha']:>+8.1f} {r['park_trades']:>5} ${r['park_pnl']:>+9,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
