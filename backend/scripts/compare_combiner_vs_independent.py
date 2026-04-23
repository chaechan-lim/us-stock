"""Compare combiner (weighted vote) vs independent strategy execution.

Tests whether removing the SignalCombiner and letting each strategy
trade independently improves returns. Uses the current US config
(donchian/dual_momentum/quality_factor disabled).

Variants:
  V0_combiner:    Current — all signals go through SignalCombiner
  V1_independent: Each strategy fires independently if conf >= threshold
  V2_best_wins:   Highest-confidence signal per symbol wins (no averaging)

Run from backend/:
    ../venv/bin/python -u scripts/compare_combiner_vs_independent.py
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
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
    enable_cash_parking=True,
    cash_parking_threshold=0.30,
    disabled_strategies=[
        "bollinger_squeeze", "volume_profile", "cis_momentum",
        "larry_williams", "bnf_deviation", "donchian_breakout",
        "dual_momentum", "quality_factor",
    ],
)

VARIANTS = [
    ("V0_combiner",    {**US_BASE, "min_confidence": 0.50}),
    ("V1_low_conf",    {**US_BASE, "min_confidence": 0.30}),
    ("V2_very_low",    {**US_BASE, "min_confidence": 0.20}),
    ("V3_no_active_ratio", {**US_BASE, "min_confidence": 0.30, "min_active_ratio": 0.0}),
]


async def run_variant(name: str, cfg_kw: dict) -> dict:
    cfg = PipelineConfig(**cfg_kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name, ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor, trades=m.total_trades,
        wr=m.win_rate, alpha=m.alpha, elapsed=el,
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 110)
    print("  Combiner threshold comparison — does lowering min_confidence help?")
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
              f"PF={r['pf']:.2f}  Trades={r['trades']}  Alpha={r['alpha']:+.1f}%  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print(f"{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Alpha%':>8}")
    print("-" * 110)
    for r in out:
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['alpha']:>+8.1f}")

    # Show per-strategy for best variant
    if out:
        best = max(out, key=lambda x: x['sharpe'])
        print(f"\nBest ({best['name']}) strategy breakdown:")
        for sn, st in sorted(best['strat_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            if st['trades'] > 0:
                print(f"  {sn:<28} trades={st['trades']:>3d} WR={st['win_rate']:>5.1f}% PnL=${st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
