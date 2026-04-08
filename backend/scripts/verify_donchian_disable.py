"""Verify donchian_breakout disable improves US 2y pipeline backtest.

Re-runs the same US baseline as validate_new_strategy.py, but the
strategies.yaml has already been edited to add donchian_breakout to the
US disabled_strategies list.

Usage:
    cd backend && ../venv/bin/python -u scripts/verify_donchian_disable.py
"""

import asyncio
import sys
import time
import logging
import functools

print = functools.partial(print, flush=True)

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
for n in ("httpx", "urllib3", "yfinance", "peewee", "backtest", "strategies", "data", "scanner"):
    logging.getLogger(n).setLevel(logging.WARNING)


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
    disabled_strategies=[
        "bollinger_squeeze", "volume_profile",
        "cis_momentum", "larry_williams", "bnf_deviation",
        "donchian_breakout",  # <-- NEW
    ],
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
)


async def main():
    print("=" * 100)
    print("  Verify donchian_breakout disable — US 2Y full pipeline")
    print("=" * 100)
    cfg = PipelineConfig(**US_BASE)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    print(f"\nRet={m.total_return_pct:+.1f}%  CAGR={(m.cagr or 0)*100:+.1f}%  "
          f"Sharpe={m.sharpe_ratio:.2f}  MDD={m.max_drawdown_pct:.1f}%  "
          f"PF={m.profit_factor:.2f}  Trades={m.total_trades}  "
          f"WR={m.win_rate:.1f}%  Alpha={m.alpha:+.1f}%  ({el:.0f}s)")

    print("\nBaseline (donchian ON, prior run): Ret=+8.0%  Sharpe=0.47  Alpha=-24.0%  Trades=341")
    print(f"Delta:                              Ret={m.total_return_pct-8.0:+.1f}pp "
          f"Sharpe={m.sharpe_ratio-0.47:+.2f}  Alpha={m.alpha-(-24.0):+.1f}pp  "
          f"Trades={m.total_trades-341:+d}")

    print("\nStrategy contribution (donchian OFF):")
    for sname, st in sorted(res.strategy_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        if st["trades"] > 0:
            print(f"  {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                  f"PnL=${st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
