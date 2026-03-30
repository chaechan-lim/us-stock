"""KR aggressive config sweep — find meaningful return levels.

Usage:
    cd backend && ../venv/bin/python scripts/run_kr_aggressive.py
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
# Only show results, suppress all info logs
for name in ("httpx", "urllib3", "yfinance", "peewee", "backtest", "strategies", "data"):
    logging.getLogger(name).setLevel(logging.WARNING)


async def run_config(label: str, config: PipelineConfig):
    bt = FullPipelineBacktest(config)
    result = await bt.run(period="2y")
    m = result.metrics

    strat_summary = ""
    if result.strategy_stats:
        top = sorted(
            result.strategy_stats.items(),
            key=lambda x: x[1]["pnl"], reverse=True,
        )
        parts = []
        for name, s in top:
            if s["trades"] > 0:
                parts.append(f"{name}={s['trades']}t/₩{s['pnl']:+,.0f}")
        strat_summary = " | ".join(parts[:5])

    print(
        f"{label:45s} "
        f"Ret={m.total_return_pct:>+6.1f}%  "
        f"Sharpe={m.sharpe_ratio:>+5.2f}  "
        f"MDD={m.max_drawdown_pct:>5.1f}%  "
        f"Trades={m.total_trades:>3d}  "
        f"WR={m.win_rate:>5.1f}%  "
        f"PF={m.profit_factor:>5.2f}  "
        f"Equity=₩{m.final_equity:>,.0f}"
    )
    print(f"  {strat_summary}")
    return result


async def main():
    print("=" * 120)
    print("KR Aggressive Config Sweep — ₩5,000,000 initial, 2Y period")
    print("=" * 120)

    # Baseline: current optimized
    await run_config("A. Baseline (SL10/TP15/static/minConf0.40)", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40, min_position_pct=0.05,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
    ))

    # B. Remove dual_momentum (유일한 손실 전략)
    await run_config("B. A + exclude dual_momentum", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40, min_position_pct=0.05,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # C. Aggressive sizing: half Kelly, 8% min position
    await run_config("C. B + kelly=0.5, minPos=8%", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # D. Concentrated: fewer bigger positions
    await run_config("D. C + max 8 positions, 12% max pos", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=8, max_position_pct=0.12,
        slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # E. More confidence = more trades
    await run_config("E. C + minConf=0.30", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.30,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # F. Wider TP (let winners run)
    await run_config("F. C + TP=25%", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.25,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # G. No TP (only strategy signal or SL exits)
    await run_config("G. C + no TP (999)", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=9.99,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # H. Full aggressive: concentrated + no TP + tight SL
    await run_config("H. Concentrated + SL7% + no TP", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.07, default_take_profit_pct=9.99,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.10,
        max_positions=8, max_position_pct=0.15,
        slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # I. No screening (more stocks pass through)
    await run_config("I. C + no screening (grade D)", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        min_screen_grade="D",
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        disabled_strategies=["dual_momentum"],
    ))

    # J. No cooldown/whipsaw (more aggressive re-entry)
    await run_config("J. C + no cooldown/whipsaw", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.10, default_take_profit_pct=0.15,
        dynamic_sl_tp=False, min_confidence=0.40,
        kelly_fraction=0.50, min_position_pct=0.08,
        max_positions=15, slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=0, whipsaw_max_losses=0, min_hold_days=0,
        disabled_strategies=["dual_momentum"],
    ))

    # K. Best combo attempt
    await run_config("K. Best combo (conc+noTP+noGates+minConf30)", PipelineConfig(
        market="KR", initial_equity=5_000_000,
        default_stop_loss_pct=0.08, default_take_profit_pct=9.99,
        dynamic_sl_tp=False, min_confidence=0.30,
        kelly_fraction=0.50, min_position_pct=0.10,
        max_positions=10, max_position_pct=0.15,
        slippage_pct=0.10, volume_adjusted_slippage=True,
        sell_cooldown_days=0, whipsaw_max_losses=0, min_hold_days=0,
        disabled_strategies=["dual_momentum"],
    ))

    print("\n" + "=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
