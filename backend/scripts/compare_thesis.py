"""Compare 3 strategy theses + current baseline on US 2y wide universe.

Each thesis disables everything else and uses the same exit thresholds /
risk params as live. All variants use cash parking enabled (V1_park_30
winning config) since that's the live target.

Theses:
  A. Sector ETF rotation       — sector_rotation + regime_switch only
  B. Quality momentum          — dual_momentum + trend_following + supertrend
  C. Mean reversion            — rsi_divergence + bollinger_squeeze + bnf_deviation
  D. Current (post-donchian)   — 11 strategies after donchian US disable

Goal: pick the thesis with the highest Sharpe AND positive Alpha (or
the smallest negative alpha if none reach +).

Run from backend/:
    ../venv/bin/python -u scripts/compare_thesis.py
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

ALL_STRATEGIES = {
    "trend_following", "donchian_breakout", "supertrend", "macd_histogram",
    "dual_momentum", "rsi_divergence", "bollinger_squeeze", "volume_profile",
    "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
    "bnf_deviation", "volume_surge", "cross_sectional_momentum",
    "quality_factor", "pead_drift",
}


def disable_except(keep: set[str]) -> list[str]:
    return sorted(ALL_STRATEGIES - keep)


US_BASE = dict(
    market="US",
    use_wide_universe=True,
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
)


VARIANTS = [
    ("A_sector_rotation",  disable_except({"sector_rotation", "regime_switch"})),
    ("B_quality_momentum", disable_except({"dual_momentum", "trend_following", "supertrend"})),
    ("C_mean_reversion",   disable_except({"rsi_divergence", "bollinger_squeeze", "bnf_deviation"})),
    ("D_current_11",       ["bollinger_squeeze", "volume_profile", "cis_momentum",
                            "larry_williams", "bnf_deviation", "donchian_breakout"]),
]


async def run_variant(name: str, disabled: list[str]) -> dict:
    cfg = PipelineConfig(**{**US_BASE, "disabled_strategies": disabled})
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name,
        n_enabled=len(ALL_STRATEGIES) - len(disabled),
        ret=m.total_return_pct,
        sharpe=m.sharpe_ratio,
        sortino=m.sortino_ratio,
        mdd=m.max_drawdown_pct,
        pf=m.profit_factor,
        trades=m.total_trades,
        wr=m.win_rate,
        alpha=m.alpha,
        bench=m.benchmark_return_pct,
        elapsed=el,
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 110)
    print("  Thesis Comparison — US 2Y Wide Universe + Cash Parking 30%")
    print("=" * 110)

    out = []
    for name, disabled in VARIANTS:
        n_enabled = len(ALL_STRATEGIES) - len(disabled)
        print(f"\n▶ {name} (enabled={n_enabled})")
        try:
            r = await run_variant(name, disabled)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:.2f}  MDD={r['mdd']:.1f}%  "
              f"PF={r['pf']:.2f}  Trades={r['trades']}  Alpha={r['alpha']:+.1f}%  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print(f"{'Thesis':<22} {'#':>3} {'Ret%':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} "
          f"{'PF':>6} {'Trades':>7} {'WR%':>6} {'Alpha%':>8}")
    print("-" * 110)
    for r in out:
        print(f"{r['name']:<22} {r['n_enabled']:>3} {r['ret']:>+8.1f} {r['sharpe']:>7.2f} "
              f"{r['sortino']:>8.2f} {r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} "
              f"{r['wr']:>6.1f} {r['alpha']:>+8.1f}")
    if out:
        print(f"{'Benchmark (SPY)':<22} {'-':>3} {out[0]['bench']:>+8.1f}")

    if out:
        print("\nPer-strategy contribution by thesis (positive PnL only):")
        for r in out:
            print(f"\n  {r['name']}:")
            for sname, st in sorted(r["strat_stats"].items(),
                                     key=lambda x: x[1]["pnl"], reverse=True):
                if st["trades"] > 0:
                    sign = "+" if st["pnl"] >= 0 else ""
                    print(f"    {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                          f"PnL={sign}${st['pnl']:,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
