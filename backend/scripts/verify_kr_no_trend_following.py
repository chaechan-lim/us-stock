"""Verify hypothesis: KR backtest 0 trades is caused by trend_following diluting
dual_momentum signals. Test by adding trend_following to KR disabled list.
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


KR_BASE = dict(
    market="KR",
    initial_equity=100_000_000,  # 100M KRW (~$80k USD); KR uses raw KRW price units
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=15,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.08,
    volume_adjusted_slippage=True,
    min_confidence=0.40,  # KR market override
    held_sell_bias=0.05,
    held_min_confidence=0.40,
)

# Mirror KR yaml disabled list
KR_DISABLED_BASE = [
    "supertrend", "macd_histogram", "rsi_divergence", "bollinger_squeeze",
    "volume_profile", "regime_switch", "sector_rotation", "cis_momentum",
    "larry_williams", "bnf_deviation", "volume_surge",
]


VARIANTS = [
    ("KR_baseline", KR_DISABLED_BASE),
    ("KR_no_trend_following", KR_DISABLED_BASE + ["trend_following"]),
    ("KR_no_donchian", KR_DISABLED_BASE + ["donchian_breakout"]),
    ("KR_no_tf_no_donchian", KR_DISABLED_BASE + ["trend_following", "donchian_breakout"]),
    ("KR_only_dual_mom", KR_DISABLED_BASE + ["trend_following", "donchian_breakout",
                                              "cross_sectional_momentum",
                                              "quality_factor", "pead_drift"]),
    ("KR_lower_minconf", KR_DISABLED_BASE),  # below
]


async def main():
    print("=" * 100)
    print("  KR Backtest Diagnostic — Test trend_following hypothesis (2Y)")
    print("=" * 100)

    out = []
    for name, disabled in VARIANTS:
        cfg_kw = {**KR_BASE, "disabled_strategies": disabled}
        if name == "KR_lower_minconf":
            cfg_kw["min_confidence"] = 0.30
        cfg = PipelineConfig(**cfg_kw)
        eng = FullPipelineBacktest(cfg)
        t0 = time.time()
        try:
            res = await eng.run(period="2y")
        except Exception as e:
            print(f"\n{name}: FAILED ({type(e).__name__}: {e})")
            continue
        el = time.time() - t0
        m = res.metrics
        print(f"\n{name}: disabled={len(disabled)} strats")
        print(f"  Ret={m.total_return_pct:+.1f}%  Sharpe={m.sharpe_ratio:.2f}  "
              f"MDD={m.max_drawdown_pct:.1f}%  PF={m.profit_factor:.2f}  "
              f"Trades={m.total_trades}  WR={m.win_rate:.1f}%  Alpha={m.alpha:+.1f}%  ({el:.0f}s)")
        out.append((name, res, m))

    print("\n" + "=" * 100)
    for name, _, m in out:
        print(f"  {name:<25} Trades={m.total_trades:>4d}  Ret={m.total_return_pct:>+6.1f}%  "
              f"Sharpe={m.sharpe_ratio:>5.2f}  Alpha={m.alpha:>+6.1f}%")

    # Print strategy contribution for the best variant
    if out:
        best = max(out, key=lambda x: x[2].total_trades)
        print(f"\nStrategy contribution ({best[0]}):")
        for sname, st in sorted(best[1].strategy_stats.items(),
                                 key=lambda x: x[1]["pnl"], reverse=True):
            if st["trades"] > 0:
                print(f"  {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                      f"PnL=${st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
