"""Compare current live KR config vs optimized config.

Usage:
    cd backend && ../venv/bin/python scripts/run_kr_comparison.py
"""

import asyncio
import sys
import logging

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for name in ("httpx", "urllib3", "yfinance", "peewee"):
    logging.getLogger(name).setLevel(logging.WARNING)


async def run_config(label: str, config: PipelineConfig):
    bt = FullPipelineBacktest(config)
    result = await bt.run(period="2y")
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(result.summary())

    if result.strategy_stats:
        print("Strategy breakdown:")
        for name, stats in sorted(
            result.strategy_stats.items(),
            key=lambda x: x[1]["trades"], reverse=True,
        ):
            if stats["trades"] > 0:
                print(
                    f"  {name:25s} trades={stats['trades']:3d}  "
                    f"WR={stats['win_rate']:5.1f}%  "
                    f"PnL=₩{stats['pnl']:+,.0f}"
                )
    return result


async def main():
    # A. Current live config (SL12/TP20, dynamic SL/TP)
    live_config = PipelineConfig(
        market="KR",
        initial_equity=5_000_000,
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.20,
        dynamic_sl_tp=True,
        max_positions=15,
        max_position_pct=0.10,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.10,
        volume_adjusted_slippage=True,
    )

    # B. Optimized config (SL10/TP15, static SL/TP, lower min_confidence)
    opt_config = PipelineConfig(
        market="KR",
        initial_equity=5_000_000,
        default_stop_loss_pct=0.10,
        default_take_profit_pct=0.15,
        dynamic_sl_tp=False,  # Static SL/TP
        min_confidence=0.40,  # Lower threshold
        min_position_pct=0.05,  # Bigger minimum positions
        max_positions=15,
        max_position_pct=0.10,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.10,
        volume_adjusted_slippage=True,
    )

    # Universe is resolved in __init__, so create a temp instance to check size
    print(f"\nKR Backtest Comparison")
    print(f"Initial equity: ₩{live_config.initial_equity:,.0f}\n")

    result_a = await run_config("A. Current Live (SL12/TP20, dynamic)", live_config)
    result_b = await run_config("B. Optimized (SL10/TP15, static, minConf=0.40)", opt_config)

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    ma = result_a.metrics
    mb = result_b.metrics
    print(f"  {'':25s} {'Current':>12s} {'Optimized':>12s}")
    print(f"  {'Return':25s} {ma.total_return_pct:>+11.1f}% {mb.total_return_pct:>+11.1f}%")
    print(f"  {'CAGR':25s} {ma.cagr:>11.1%} {mb.cagr:>11.1%}")
    print(f"  {'Sharpe':25s} {ma.sharpe_ratio:>12.2f} {mb.sharpe_ratio:>12.2f}")
    print(f"  {'Sortino':25s} {ma.sortino_ratio:>12.2f} {mb.sortino_ratio:>12.2f}")
    print(f"  {'MDD':25s} {ma.max_drawdown_pct:>11.1f}% {mb.max_drawdown_pct:>11.1f}%")
    print(f"  {'Trades':25s} {ma.total_trades:>12d} {mb.total_trades:>12d}")
    print(f"  {'Win Rate':25s} {ma.win_rate:>11.1f}% {mb.win_rate:>11.1f}%")
    print(f"  {'Profit Factor':25s} {ma.profit_factor:>12.2f} {mb.profit_factor:>12.2f}")
    print(f"  {'Final Equity':25s} ₩{ma.final_equity:>11,.0f} ₩{mb.final_equity:>11,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
