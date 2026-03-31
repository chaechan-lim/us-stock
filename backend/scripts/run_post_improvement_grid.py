"""Post-improvement grid search — find 30%+ return configs.

Strategy code was improved (STOCK-67~76), need to re-optimize
sizing, SL/TP, and strategy combos for higher returns.

Usage:
    cd backend && ../venv/bin/python scripts/run_post_improvement_grid.py
"""

import asyncio
import itertools
import logging
import sys

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(level=logging.WARNING)
for name in (
    "httpx", "urllib3", "yfinance", "peewee",
    "backtest", "strategies", "data",
):
    logging.getLogger(name).setLevel(logging.WARNING)

ALL_STRATEGIES = [
    "trend_following", "donchian_breakout", "supertrend",
    "macd_histogram", "dual_momentum", "rsi_divergence",
    "bollinger_squeeze", "volume_profile", "regime_switch",
    "sector_rotation", "cis_momentum", "larry_williams",
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
    # KR: supertrend + dual_momentum (proven best)
    # ═══════════════════════════════════════════════
    print("=" * 130)
    print("KR GRID — supertrend + dual_momentum")
    print("=" * 130)
    hdr = (
        f"{'Config':60s} {'Ret':>7s} {'Sharpe':>7s} "
        f"{'MDD':>7s} {'Trades':>6s} {'WR':>6s} {'PF':>6s}"
    )
    print(hdr)
    print("-" * 130)

    kr_keep = ["supertrend", "dual_momentum"]
    kr_disabled = disable_except(kr_keep)
    kr_results = []

    for kelly, max_pct, max_pos, tp, sl, min_conf in itertools.product(
        [0.75, 1.00, 1.50],         # kelly — more aggressive
        [0.15, 0.20, 0.25, 0.30],   # max_position_pct — bigger
        [5, 8],                      # max_positions
        [0.15, 0.20, 0.30, 0.50],   # take_profit — wider
        [0.07, 0.10, 0.15],         # stop_loss
        [0.20, 0.30],               # min_confidence — lower
    ):
        label = (
            f"K={kelly:.2f} pos={max_pct:.0%}x{max_pos} "
            f"SL={sl:.0%}/TP={tp:.0%} conf={min_conf}"
        )
        try:
            m = await run(PipelineConfig(
                market="KR",
                initial_equity=5_000_000,
                default_stop_loss_pct=sl,
                default_take_profit_pct=tp,
                dynamic_sl_tp=False,
                kelly_fraction=kelly,
                min_position_pct=max_pct * 0.5,
                max_positions=max_pos,
                max_position_pct=max_pct,
                min_confidence=min_conf,
                min_active_ratio=0.0,
                slippage_pct=0.10,
                volume_adjusted_slippage=True,
                sell_cooldown_days=1,
                whipsaw_max_losses=2,
                min_hold_days=1,
                disabled_strategies=kr_disabled,
            ))
            print(
                f"  {label:58s} {m.total_return_pct:>+6.1f}% "
                f"{m.sharpe_ratio:>+6.2f} {m.max_drawdown_pct:>6.1f}% "
                f"{m.total_trades:>5d} {m.win_rate:>5.1f}% "
                f"{m.profit_factor:>5.2f}"
            )
            kr_results.append((
                label, m.total_return_pct, m.sharpe_ratio,
                m.max_drawdown_pct, m.total_trades, m.win_rate,
                m.profit_factor,
            ))
        except Exception as e:
            print(f"  {label:58s} ERROR: {e}")

    kr_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nKR TOP 10:")
    for r in kr_results[:10]:
        print(
            f"  {r[0]:58s} Ret={r[1]:+.1f}%  Sharpe={r[2]:+.2f}  "
            f"MDD={r[3]:.1f}%  Trades={r[4]}  WR={r[5]:.1f}%  "
            f"PF={r[6]:.2f}"
        )

    # ═══════════════════════════════════════════════
    # US: multiple combos
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("US GRID — sector_rotation + volume_profile + volume_surge")
    print("=" * 130)
    print(hdr)
    print("-" * 130)

    us_combos = [
        ("Top3", ["sector_rotation", "volume_profile", "volume_surge"]),
        (
            "Top5+dm",
            [
                "sector_rotation", "volume_profile", "volume_surge",
                "regime_switch", "cis_momentum", "dual_momentum",
            ],
        ),
        (
            "Top5+rsi",
            [
                "sector_rotation", "volume_profile", "volume_surge",
                "regime_switch", "cis_momentum", "rsi_divergence",
            ],
        ),
    ]
    us_results = []

    for combo_name, keep in us_combos:
        us_disabled = disable_except(keep)
        for kelly, max_pct, max_pos, tp, sl, min_conf in itertools.product(
            [0.75, 1.00, 1.50],
            [0.15, 0.20, 0.25, 0.30],
            [5, 8],
            [0.15, 0.20, 0.30, 0.50],
            [0.07, 0.10, 0.15],
            [0.20, 0.30],
        ):
            label = (
                f"{combo_name} K={kelly:.2f} pos={max_pct:.0%}x{max_pos} "
                f"SL={sl:.0%}/TP={tp:.0%} conf={min_conf}"
            )
            try:
                m = await run(PipelineConfig(
                    market="US",
                    initial_equity=100_000,
                    default_stop_loss_pct=sl,
                    default_take_profit_pct=tp,
                    dynamic_sl_tp=False,
                    kelly_fraction=kelly,
                    min_position_pct=max_pct * 0.5,
                    max_positions=max_pos,
                    max_position_pct=max_pct,
                    min_confidence=min_conf,
                    min_active_ratio=0.0,
                    slippage_pct=0.05,
                    volume_adjusted_slippage=True,
                    sell_cooldown_days=1,
                    whipsaw_max_losses=2,
                    min_hold_days=1,
                    disabled_strategies=us_disabled,
                ))
                print(
                    f"  {label:58s} {m.total_return_pct:>+6.1f}% "
                    f"{m.sharpe_ratio:>+6.2f} "
                    f"{m.max_drawdown_pct:>6.1f}% "
                    f"{m.total_trades:>5d} {m.win_rate:>5.1f}% "
                    f"{m.profit_factor:>5.2f}"
                )
                us_results.append((
                    label, m.total_return_pct, m.sharpe_ratio,
                    m.max_drawdown_pct, m.total_trades, m.win_rate,
                    m.profit_factor,
                ))
            except Exception as e:
                print(f"  {label:58s} ERROR: {e}")

    us_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nUS TOP 10:")
    for r in us_results[:10]:
        print(
            f"  {r[0]:58s} Ret={r[1]:+.1f}%  Sharpe={r[2]:+.2f}  "
            f"MDD={r[3]:.1f}%  Trades={r[4]}  WR={r[5]:.1f}%  "
            f"PF={r[6]:.2f}"
        )

    print("\n" + "=" * 130)


if __name__ == "__main__":
    asyncio.run(main())
