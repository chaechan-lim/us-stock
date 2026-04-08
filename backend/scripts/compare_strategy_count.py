"""Compare strategy count: pre-f5693c7 (2-strategy) vs current (11) on identical period.

Variants:
  - V1_legacy_2:    dual_momentum + volume_surge (yesterday's "2 strategy" PROVISIONAL state)
  - V2_alt_2:       dual_momentum + trend_following (alternative 2-strategy combo)
  - V3_minimal_3:   dual_momentum + trend_following + supertrend (best 3 by WR)
  - V4_current_11:  current strategies.yaml after my donchian disable
  - V5_no_4_07_new: current minus the 4-07 additions (cs_momentum, quality_factor, pead_drift)
"""

import asyncio
import sys
import time
import functools

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data", "backtest", "strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

ALL_OPTIONAL = [
    "trend_following", "dual_momentum", "donchian_breakout", "supertrend",
    "macd_histogram", "rsi_divergence", "bollinger_squeeze", "volume_profile",
    "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
    "bnf_deviation", "volume_surge", "cross_sectional_momentum",
    "quality_factor", "pead_drift",
]


def disabled_except(keep: list[str]) -> list[str]:
    return [s for s in ALL_OPTIONAL if s not in keep]


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
)

VARIANTS = [
    ("V1_legacy_2",     disabled_except(["dual_momentum", "volume_surge"])),
    ("V2_alt_2",        disabled_except(["dual_momentum", "trend_following"])),
    ("V3_minimal_3",    disabled_except(["dual_momentum", "trend_following", "supertrend"])),
    ("V4_current_11",   ["bollinger_squeeze", "volume_profile", "cis_momentum",
                         "larry_williams", "bnf_deviation", "donchian_breakout"]),
    ("V5_no_4_07_new",  ["bollinger_squeeze", "volume_profile", "cis_momentum",
                         "larry_williams", "bnf_deviation", "donchian_breakout",
                         "cross_sectional_momentum", "quality_factor", "pead_drift"]),
]


async def main():
    print("=" * 110)
    print("  Strategy Count Comparison — US 2Y Pipeline Backtest")
    print("=" * 110)

    out = []
    for name, disabled in VARIANTS:
        cfg = PipelineConfig(**{**US_BASE, "disabled_strategies": disabled})
        n_enabled = len(ALL_OPTIONAL) - len(disabled)
        eng = FullPipelineBacktest(cfg)
        t0 = time.time()
        try:
            res = await eng.run(period="2y")
        except Exception as e:
            print(f"\n{name}: FAILED ({type(e).__name__}: {e})")
            continue
        el = time.time() - t0
        m = res.metrics
        print(f"\n{name}  (enabled={n_enabled})")
        print(f"  Ret={m.total_return_pct:+.1f}%  CAGR={(m.cagr or 0)*100:+.1f}%  "
              f"Sharpe={m.sharpe_ratio:.2f}  MDD={m.max_drawdown_pct:.1f}%  "
              f"PF={m.profit_factor:.2f}  Trades={m.total_trades}  "
              f"WR={m.win_rate:.1f}%  Alpha={m.alpha:+.1f}%  ({el:.0f}s)")
        out.append((name, n_enabled, res, m))

    print("\n" + "=" * 110)
    print(f"{'Variant':<22} {'#':>3} {'Ret%':>8} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'WR%':>6} {'Alpha%':>8}")
    print("-" * 110)
    for name, n, _, m in out:
        print(f"{name:<22} {n:>3} {m.total_return_pct:>+8.1f} {m.sharpe_ratio:>7.2f} "
              f"{m.max_drawdown_pct:>7.1f} {m.profit_factor:>6.2f} {m.total_trades:>7d} "
              f"{m.win_rate:>6.1f} {m.alpha:>+8.1f}")

    print("\nPer-strategy contribution (best variant by Sharpe):")
    if out:
        best = max(out, key=lambda x: x[3].sharpe_ratio)
        print(f"  → {best[0]}")
        for sname, st in sorted(best[2].strategy_stats.items(),
                                 key=lambda x: x[1]["pnl"], reverse=True):
            if st["trades"] > 0:
                print(f"  {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                      f"PnL=${st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
