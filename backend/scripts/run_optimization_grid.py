"""Grid search for 30%+ returns — sizing, SL/TP, weights, concentration.

Usage:
    cd backend && ../venv/bin/python scripts/run_optimization_grid.py
"""

import asyncio
import sys
import logging
import itertools

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(level=logging.WARNING)
for name in ("httpx", "urllib3", "yfinance", "peewee", "backtest", "strategies", "data"):
    logging.getLogger(name).setLevel(logging.WARNING)

ALL_STRATEGIES = [
    "trend_following", "donchian_breakout", "supertrend", "macd_histogram",
    "dual_momentum", "rsi_divergence", "bollinger_squeeze", "volume_profile",
    "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
    "bnf_deviation", "volume_surge",
]


def disable_except(keep):
    return [s for s in ALL_STRATEGIES if s not in keep]


async def run(config):
    bt = FullPipelineBacktest(config)
    result = await bt.run(period="2y")
    return result.metrics


async def main():
    # ═══════════════════════════════════════════════
    # KR: supertrend + dual_momentum
    # ═══════════════════════════════════════════════
    print("=" * 120)
    print("KR OPTIMIZATION GRID — supertrend + dual_momentum")
    print("=" * 120)
    print(f"{'Config':55s} {'Ret':>7s} {'Sharpe':>7s} {'MDD':>7s} {'Trades':>6s} {'WR':>6s} {'PF':>6s}")
    print("-" * 120)

    kr_keep = ["supertrend", "dual_momentum"]
    kr_disabled = disable_except(kr_keep)
    kr_results = []

    # Grid: kelly × max_pos_pct × max_positions × TP × SL
    for kelly, max_pct, max_pos, tp, sl in itertools.product(
        [0.50, 0.75, 1.00],        # kelly_fraction
        [0.12, 0.15, 0.20],        # max_position_pct
        [5, 8],                     # max_positions
        [0.12, 0.15, 0.20, 0.30],  # take_profit
        [0.07, 0.10],              # stop_loss
    ):
        label = f"K={kelly:.2f} pos={max_pct:.0%}×{max_pos} SL={sl:.0%}/TP={tp:.0%}"
        try:
            m = await run(PipelineConfig(
                market="KR", initial_equity=5_000_000,
                default_stop_loss_pct=sl, default_take_profit_pct=tp,
                dynamic_sl_tp=False,
                kelly_fraction=kelly, min_position_pct=max_pct * 0.6,
                max_positions=max_pos, max_position_pct=max_pct,
                min_confidence=0.30, min_active_ratio=0.0,
                slippage_pct=0.10, volume_adjusted_slippage=True,
                sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
                disabled_strategies=kr_disabled,
            ))
            print(
                f"  {label:53s} {m.total_return_pct:>+6.1f}% "
                f"{m.sharpe_ratio:>+6.2f} {m.max_drawdown_pct:>6.1f}% "
                f"{m.total_trades:>5d} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f}"
            )
            kr_results.append((label, m.total_return_pct, m.sharpe_ratio, m.max_drawdown_pct, m.total_trades))
        except Exception as e:
            print(f"  {label:53s} ERROR: {e}")

    # Top 5 KR
    kr_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nKR TOP 5:")
    for label, ret, sharpe, mdd, trades in kr_results[:5]:
        print(f"  {label:53s} Ret={ret:+.1f}%  Sharpe={sharpe:+.2f}  MDD={mdd:.1f}%  Trades={trades}")

    # ═══════════════════════════════════════════════
    # US: sector_rotation + volume_profile + volume_surge
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("US OPTIMIZATION GRID — sector_rotation + volume_profile + volume_surge")
    print("=" * 120)
    print(f"{'Config':55s} {'Ret':>7s} {'Sharpe':>7s} {'MDD':>7s} {'Trades':>6s} {'WR':>6s} {'PF':>6s}")
    print("-" * 120)

    us_keep = ["sector_rotation", "volume_profile", "volume_surge"]
    us_disabled = disable_except(us_keep)
    us_results = []

    for kelly, max_pct, max_pos, tp, sl in itertools.product(
        [0.50, 0.75, 1.00],
        [0.12, 0.15, 0.20],
        [5, 8],
        [0.12, 0.15, 0.20, 0.30],
        [0.07, 0.10],
    ):
        label = f"K={kelly:.2f} pos={max_pct:.0%}×{max_pos} SL={sl:.0%}/TP={tp:.0%}"
        try:
            m = await run(PipelineConfig(
                market="US", initial_equity=100_000,
                default_stop_loss_pct=sl, default_take_profit_pct=tp,
                dynamic_sl_tp=False,
                kelly_fraction=kelly, min_position_pct=max_pct * 0.6,
                max_positions=max_pos, max_position_pct=max_pct,
                min_confidence=0.30, min_active_ratio=0.0,
                slippage_pct=0.05, volume_adjusted_slippage=True,
                sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
                disabled_strategies=us_disabled,
            ))
            print(
                f"  {label:53s} {m.total_return_pct:>+6.1f}% "
                f"{m.sharpe_ratio:>+6.2f} {m.max_drawdown_pct:>6.1f}% "
                f"{m.total_trades:>5d} {m.win_rate:>5.1f}% {m.profit_factor:>5.2f}"
            )
            us_results.append((label, m.total_return_pct, m.sharpe_ratio, m.max_drawdown_pct, m.total_trades))
        except Exception as e:
            print(f"  {label:53s} ERROR: {e}")

    # Top 5 US
    us_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nUS TOP 5:")
    for label, ret, sharpe, mdd, trades in us_results[:5]:
        print(f"  {label:53s} Ret={ret:+.1f}%  Sharpe={sharpe:+.2f}  MDD={mdd:.1f}%  Trades={trades}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
