"""Compare fixed SPY allocation ratios vs current threshold-based parking.

Tests: what % of portfolio should be held in SPY permanently?
Combined with V1 min_confidence=0.30 (best from combiner test).

Run from backend/:
    ../venv/bin/python -u scripts/compare_spy_allocation.py
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data", "backtest", "strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

US_BASE = dict(
    market="US",
    initial_equity=100_000,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=20,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.05,
    volume_adjusted_slippage=True,
    min_confidence=0.30,  # V1 best
    held_sell_bias=0.05,
    held_min_confidence=0.40,
    disabled_strategies=[
        "bollinger_squeeze", "volume_profile", "cis_momentum",
        "larry_williams", "bnf_deviation", "donchian_breakout",
        "dual_momentum", "quality_factor",
    ],
)

VARIANTS = [
    ("V0_no_parking",     {**US_BASE, "enable_cash_parking": False}),
    ("V1_park_30pct",     {**US_BASE, "enable_cash_parking": True, "cash_parking_threshold": 0.30}),
    ("V2_park_20pct",     {**US_BASE, "enable_cash_parking": True, "cash_parking_threshold": 0.20}),
    ("V3_park_10pct",     {**US_BASE, "enable_cash_parking": True, "cash_parking_threshold": 0.10}),
    ("V4_park_50pct",     {**US_BASE, "enable_cash_parking": True, "cash_parking_threshold": 0.50}),
]


async def run_variant(name: str, cfg_kw: dict) -> dict:
    cfg = PipelineConfig(**cfg_kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    # Count cash_parking trades
    park_stats = res.strategy_stats.get("cash_parking", {"trades": 0, "pnl": 0, "win_rate": 0})

    return dict(
        name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
        wr=m.win_rate, alpha=m.alpha, elapsed=el,
        park_trades=park_stats.get("trades", 0),
        park_pnl=park_stats.get("pnl", 0),
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 110)
    print("  SPY Allocation Comparison (all with min_confidence=0.30)")
    print("=" * 110)

    out = []
    for name, cfg_kw in VARIANTS:
        print(f"\n▶ {name}")
        try:
            r = await run_variant(name, cfg_kw)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:.2f}  MDD={r['mdd']:.1f}%  "
              f"Trades={r['trades']}  Alpha={r['alpha']:+.1f}%  "
              f"SPY trades={r['park_trades']} PnL=${r['park_pnl']:+,.0f}  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print(f"{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Alpha%':>8} {'SPY PnL':>10}")
    print("-" * 110)
    for r in out:
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['alpha']:>+8.1f} ${r['park_pnl']:>+9,.0f}")

    if out:
        print(f"\nBenchmark SPY: +{out[0].get('alpha', 0) + out[0].get('ret', 0):.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
