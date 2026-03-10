"""Compare full pipeline backtest: default vs tuned parameters.

Usage:
    cd backend && python -m backtest.compare_regime_sell [period]
    cd backend && python -m backtest.compare_regime_sell 5y
"""

import asyncio
import logging
import sys

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Broad universe for meaningful backtest
TEST_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "CRM", "ADBE", "ORCL",
    # Finance
    "JPM", "BAC", "GS", "WFC",
    # Healthcare
    "UNH", "JNJ", "LLY", "MRK",
    # Consumer
    "WMT", "HD", "COST", "NKE",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial
    "CAT", "BA", "HON",
]


def fmt(label, val_off, val_on, suffix="", higher_better=True):
    diff = val_on - val_off
    direction = "+" if diff > 0 else ""
    better = (diff > 0) == higher_better
    marker = " <<" if better and abs(diff) > 0.1 else ""
    return (
        f"  {label:20s}  {val_off:>10{suffix}}  {val_on:>10{suffix}}"
        f"  ({direction}{diff:{suffix}}){marker}"
    )


async def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "3y"

    base = dict(
        universe=list(TEST_UNIVERSE),
        initial_equity=100_000,
        enable_regime_sells=True,
    )

    # ── A: Conservative defaults (current live system) ────────────────
    cfg_a = PipelineConfig(
        **base,
        min_active_ratio=0.15,
        min_confidence=0.50,
        min_screen_grade="B",
        screen_interval=20,
        max_positions=20,
        max_watchlist=30,
    )

    # ── B: Tuned (proven parameters, no complex features) ────────────
    cfg_b = PipelineConfig(
        **base,
        min_active_ratio=0.05,        # allow single-strategy conviction
        min_confidence=0.35,          # lower bar for combined signal
        min_screen_grade="C",         # wider candidate pool
        screen_interval=10,           # bi-weekly screening
        max_positions=15,
        max_watchlist=25,
        max_position_pct=0.10,        # up to 10% per position
        max_exposure_pct=0.95,        # use more capital
        kelly_fraction=0.35,          # more aggressive Kelly (35% vs 25%)
        confidence_exponent=1.5,      # less penalty for moderate confidence
        min_position_pct=0.03,        # 3% minimum position
    )

    # ── C: Concentrated — fewer positions, bigger size, wider TP ─────
    cfg_c = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.35,
        min_screen_grade="C",
        screen_interval=10,
        max_positions=10,             # fewer, bigger positions
        max_watchlist=25,
        max_position_pct=0.15,        # up to 15% per position
        max_exposure_pct=0.95,
        kelly_fraction=0.40,          # 40% Kelly
        confidence_exponent=1.2,      # minimal confidence penalty
        min_position_pct=0.04,        # 4% minimum position
        default_stop_loss_pct=0.10,   # slightly wider SL
        default_take_profit_pct=0.35, # much wider TP: let winners run
    )

    # ── D: Wide TP + concentrated + recovery ────────────────────────
    cfg_d = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.35,
        min_screen_grade="C",
        screen_interval=10,
        max_positions=10,             # concentrated
        max_watchlist=25,
        max_position_pct=0.15,        # 15% per position
        max_exposure_pct=0.95,
        kelly_fraction=0.40,
        confidence_exponent=1.2,
        min_position_pct=0.05,        # 5% minimum position
        default_stop_loss_pct=0.12,   # wide SL
        default_take_profit_pct=0.50, # very wide TP: let winners run far
        recovery_watch_days=30,       # longer recovery window
    )

    configs = {"CONSERVATIVE": cfg_a, "TUNED": cfg_b, "CONCENTRATED": cfg_c, "WIDE_TP": cfg_d}
    results = {}

    for name, cfg in configs.items():
        logger.info("\n" + "=" * 60)
        logger.info(f"Running {name} config...")
        logger.info("=" * 60)
        engine = FullPipelineBacktest(cfg)
        results[name] = await engine.run(period=period)

    # ── Compare ───────────────────────────────────────────────────────
    config_names = list(results.keys())
    metrics_list = [results[n].metrics for n in config_names]

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    header = f"  {'':20s}" + "".join(f"  {n:>10s}" for n in config_names)
    logger.info(header)
    logger.info(f"  {'-' * (20 + 12 * len(config_names))}")

    def row(label, values, suffix=""):
        parts = f"  {label:20s}"
        for v in values:
            parts += f"  {v:>10{suffix}}"
        return parts

    logger.info(row("CAGR %", [m.cagr * 100 for m in metrics_list], ".1f"))
    logger.info(row("Total Return %", [m.total_return_pct for m in metrics_list], ".1f"))
    logger.info(row("Sharpe", [m.sharpe_ratio for m in metrics_list], ".2f"))
    logger.info(row("Sortino", [m.sortino_ratio for m in metrics_list], ".2f"))
    logger.info(row("MDD %", [m.max_drawdown_pct for m in metrics_list], ".1f"))
    logger.info(row("MDD Days", [m.max_drawdown_days for m in metrics_list], ".0f"))
    logger.info(row("Win Rate %", [m.win_rate for m in metrics_list], ".1f"))
    logger.info(row("Profit Factor", [m.profit_factor for m in metrics_list], ".2f"))
    logger.info(row("Total Trades", [m.total_trades for m in metrics_list], ".0f"))
    logger.info(row("Final Equity", [m.final_equity for m in metrics_list], ",.0f"))
    logger.info(row("Alpha %", [m.alpha for m in metrics_list], ".1f"))
    logger.info(row("SPY %", [m.benchmark_return_pct for m in metrics_list], ".1f"))

    # Strategy breakdown for each config
    for cname in config_names:
        result = results[cname]
        regime_sells = [t for t in result.trades if t.strategy_name == "regime_protect"]
        logger.info(f"\n  [{cname}] {result.metrics.total_trades} trades" +
                     (f", regime_sells={len(regime_sells)}" if regime_sells else ""))
        for sname, stats in sorted(
            result.strategy_stats.items(),
            key=lambda x: x[1]["pnl"], reverse=True,
        ):
            if stats["trades"] > 0:
                logger.info(
                    f"    {sname:20s}: {stats['trades']:3d} trades, "
                    f"WR={stats['win_rate']:4.0f}%, PnL=${stats['pnl']:+,.0f}"
                )


if __name__ == "__main__":
    asyncio.run(main())
