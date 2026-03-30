"""Test market-specific optimal strategy combos vs all-14.

Usage:
    cd backend && ../venv/bin/python scripts/run_market_optimal.py
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

ALL_STRATEGIES = [
    "trend_following", "donchian_breakout", "supertrend", "macd_histogram",
    "dual_momentum", "rsi_divergence", "bollinger_squeeze", "volume_profile",
    "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
    "bnf_deviation", "volume_surge",
]


def disable_except(keep: list[str]) -> list[str]:
    return [s for s in ALL_STRATEGIES if s not in keep]


async def run(label: str, config: PipelineConfig):
    bt = FullPipelineBacktest(config)
    result = await bt.run(period="2y")
    m = result.metrics

    strats = ""
    if result.strategy_stats:
        parts = []
        for name, s in sorted(result.strategy_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            if s["trades"] > 0:
                parts.append(f"{name}={s['trades']}t/WR{s['win_rate']:.0f}%/PnL{s['pnl']:+,.0f}")
        strats = " | ".join(parts)

    print(
        f"  {label:50s} Ret={m.total_return_pct:>+6.1f}%  "
        f"Sharpe={m.sharpe_ratio:>+5.2f}  MDD={m.max_drawdown_pct:>5.1f}%  "
        f"Trades={m.total_trades:>3d}  WR={m.win_rate:>5.1f}%  "
        f"PF={m.profit_factor:>5.2f}"
    )
    print(f"    {strats}")
    return result


async def main():
    # Common concentrated config
    common = dict(
        default_stop_loss_pct=0.10,
        default_take_profit_pct=0.15,
        dynamic_sl_tp=False,
        max_positions=8,
        max_position_pct=0.12,
        kelly_fraction=0.50,
        min_position_pct=0.08,
        min_confidence=0.30,
        min_active_ratio=0.0,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        volume_adjusted_slippage=True,
    )

    # ── KR ──
    print("=" * 130)
    print("  KR MARKET — ₩5,000,000")
    print("=" * 130)

    kr = dict(**common, market="KR", initial_equity=5_000_000, slippage_pct=0.10)

    # Baseline: all 14
    await run("KR-A. All 14 strategies", PipelineConfig(**kr))

    # Top 2 only
    await run("KR-B. Top2 (supertrend + dual_momentum)", PipelineConfig(
        **kr, disabled_strategies=disable_except(["supertrend", "dual_momentum"]),
    ))

    # Top 3
    await run("KR-C. Top3 (+macd_histogram)", PipelineConfig(
        **kr, disabled_strategies=disable_except(["supertrend", "dual_momentum", "macd_histogram"]),
    ))

    # Top 4
    await run("KR-D. Top4 (+donchian_breakout)", PipelineConfig(
        **kr, disabled_strategies=disable_except(["supertrend", "dual_momentum", "macd_histogram", "donchian_breakout"]),
    ))

    # Top 5
    await run("KR-E. Top5 (+bnf_deviation)", PipelineConfig(
        **kr, disabled_strategies=disable_except(["supertrend", "dual_momentum", "macd_histogram", "donchian_breakout", "bnf_deviation"]),
    ))

    # Without weak strategies only
    await run("KR-F. All minus weak (tf/vs/rsi/boll/regime)", PipelineConfig(
        **kr, disabled_strategies=["trend_following", "volume_surge", "rsi_divergence", "bollinger_squeeze", "regime_switch"],
    ))

    # ── US ──
    print(f"\n{'=' * 130}")
    print("  US MARKET — $100,000")
    print("=" * 130)

    us = dict(**common, market="US", initial_equity=100_000, slippage_pct=0.05)

    # Baseline: all 14
    await run("US-A. All 14 strategies", PipelineConfig(**us))

    # Top 3
    await run("US-B. Top3 (sector_rot + vol_profile + vol_surge)", PipelineConfig(
        **us, disabled_strategies=disable_except(["sector_rotation", "volume_profile", "volume_surge"]),
    ))

    # Top 5
    await run("US-C. Top5 (+regime_switch + cis_momentum)", PipelineConfig(
        **us, disabled_strategies=disable_except(["sector_rotation", "volume_profile", "volume_surge", "regime_switch", "cis_momentum"]),
    ))

    # Top 5 + rsi_divergence (decent individual)
    await run("US-D. Top5 + rsi_divergence", PipelineConfig(
        **us, disabled_strategies=disable_except(["sector_rotation", "volume_profile", "volume_surge", "regime_switch", "cis_momentum", "rsi_divergence"]),
    ))

    # Without weak strategies only
    await run("US-E. All minus weak (tf/super/macd/boll/bnf)", PipelineConfig(
        **us, disabled_strategies=["trend_following", "supertrend", "macd_histogram", "bollinger_squeeze", "bnf_deviation"],
    ))

    # Top 5 + dual_momentum (decent in US standalone)
    await run("US-F. Top5 + dual_momentum", PipelineConfig(
        **us, disabled_strategies=disable_except(["sector_rotation", "volume_profile", "volume_surge", "regime_switch", "cis_momentum", "dual_momentum"]),
    ))

    print("\n" + "=" * 130)


if __name__ == "__main__":
    asyncio.run(main())
