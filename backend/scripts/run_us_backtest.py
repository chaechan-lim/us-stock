"""Run US market full pipeline backtest for comparison.

Usage:
    cd backend && ../venv/bin/python scripts/run_us_backtest.py
"""

import asyncio
import sys
import logging

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for name in ("httpx", "urllib3", "yfinance", "peewee", "backtest", "strategies", "data"):
    logging.getLogger(name).setLevel(logging.WARNING)


async def main():
    config = PipelineConfig(
        market="US",
        initial_equity=100_000,
        # Current live-equivalent settings
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=0.08,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
    )

    bt = FullPipelineBacktest(config)
    result = await bt.run(period="2y")

    print("\n" + "=" * 60)
    print("  US Backtest (current live config)")
    print("=" * 60)
    print(result.summary())

    if result.strategy_stats:
        print("Strategy breakdown:")
        for name, stats in sorted(
            result.strategy_stats.items(),
            key=lambda x: x[1]["pnl"], reverse=True,
        ):
            if stats["trades"] > 0:
                print(
                    f"  {name:25s} trades={stats['trades']:3d}  "
                    f"WR={stats['win_rate']:5.1f}%  "
                    f"PnL=${stats['pnl']:+,.0f}"
                )


if __name__ == "__main__":
    asyncio.run(main())
