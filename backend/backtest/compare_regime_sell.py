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


async def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "3y"

    base = dict(
        universe=list(TEST_UNIVERSE),
        initial_equity=100_000,
        enable_regime_sells=True,
    )

    # ── A: Current best (WIDE_TP) ────────────────────────────────
    cfg_a = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.35,
        min_screen_grade="C",
        screen_interval=10,
        max_positions=10,
        max_watchlist=25,
        max_position_pct=0.15,
        max_exposure_pct=0.95,
        kelly_fraction=0.40,
        confidence_exponent=1.2,
        min_position_pct=0.05,
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.50,
        recovery_watch_days=30,
    )

    # ── B: Park in QQQ instead of SPY ────────────────────────────
    cfg_b = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.35,
        min_screen_grade="C",
        screen_interval=10,
        max_positions=10,
        max_watchlist=25,
        max_position_pct=0.15,
        max_exposure_pct=0.95,
        kelly_fraction=0.40,
        confidence_exponent=1.2,
        min_position_pct=0.05,
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.50,
        recovery_watch_days=30,
        enable_cash_parking=True,
        cash_parking_symbol="QQQ",
        cash_parking_threshold=0.15,
    )

    # ── C: Ultra aggressive: 25% positions, 5 max, 60% Kelly ────
    cfg_c = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.30,          # lower bar
        min_screen_grade="C",
        screen_interval=10,
        max_positions=5,
        max_watchlist=25,
        max_position_pct=0.25,        # 25% per position!
        max_exposure_pct=0.95,
        kelly_fraction=0.60,          # 60% Kelly
        confidence_exponent=1.0,      # no penalty
        min_position_pct=0.10,        # 10% minimum
        default_stop_loss_pct=0.15,
        default_take_profit_pct=9.99, # no TP
        recovery_watch_days=30,
        enable_cash_parking=True,
        cash_parking_symbol="QQQ",
        cash_parking_threshold=0.15,
    )

    # ── D: Like C but 8 positions + 20% max ──────────────────────
    cfg_d = PipelineConfig(
        **base,
        min_active_ratio=0.05,
        min_confidence=0.30,
        min_screen_grade="C",
        screen_interval=10,
        max_positions=8,
        max_watchlist=25,
        max_position_pct=0.20,
        max_exposure_pct=0.95,
        kelly_fraction=0.50,
        confidence_exponent=1.0,
        min_position_pct=0.08,
        default_stop_loss_pct=0.15,
        default_take_profit_pct=9.99,
        recovery_watch_days=30,
        enable_cash_parking=True,
        cash_parking_symbol="QQQ",
        cash_parking_threshold=0.15,
    )

    configs = {"WIDE_TP": cfg_a, "QQQ_PARK": cfg_b, "ULTRA_AGG": cfg_c, "AGG_8POS": cfg_d}
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

    # Strategy breakdown for top 2 configs
    sorted_configs = sorted(
        config_names, key=lambda n: results[n].metrics.total_return_pct, reverse=True,
    )
    for cname in sorted_configs[:2]:
        result = results[cname]
        logger.info(f"\n  [{cname}] {result.metrics.total_trades} trades")
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
