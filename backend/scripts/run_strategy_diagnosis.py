"""Diagnose individual strategy edge — US + KR.

Runs full pipeline with only 1 strategy enabled at a time.
This shows each strategy's standalone performance without combiner dilution.

Usage:
    cd backend && ../venv/bin/python scripts/run_strategy_diagnosis.py
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
    "trend_following",
    "donchian_breakout",
    "supertrend",
    "macd_histogram",
    "dual_momentum",
    "rsi_divergence",
    "bollinger_squeeze",
    "volume_profile",
    "regime_switch",
    "sector_rotation",
    "cis_momentum",
    "larry_williams",
    "bnf_deviation",
    "volume_surge",
]


async def run_single_strategy(strategy_name: str, market: str, base_config: dict):
    """Run backtest with only one strategy enabled."""
    others = [s for s in ALL_STRATEGIES if s != strategy_name]
    config = PipelineConfig(
        **base_config,
        disabled_strategies=others,
        # Single strategy: no combiner consensus needed
        min_confidence=0.30,  # Low bar since only 1 strategy
        min_active_ratio=0.0,  # Always active (only 1 strategy)
    )
    bt = FullPipelineBacktest(config)
    try:
        result = await bt.run(period="2y")
        m = result.metrics
        return {
            "strategy": strategy_name,
            "market": market,
            "return_pct": m.total_return_pct,
            "cagr": m.cagr,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "mdd": m.max_drawdown_pct,
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "avg_hold": m.avg_holding_days,
            "final_equity": m.final_equity,
        }
    except Exception as e:
        return {
            "strategy": strategy_name,
            "market": market,
            "return_pct": 0,
            "error": str(e),
        }


async def main():
    kr_base = dict(
        market="KR",
        initial_equity=5_000_000,
        default_stop_loss_pct=0.10,
        default_take_profit_pct=0.15,
        dynamic_sl_tp=False,
        max_positions=8,
        max_position_pct=0.12,
        kelly_fraction=0.50,
        min_position_pct=0.08,
        slippage_pct=0.10,
        volume_adjusted_slippage=True,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
    )

    us_base = dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.10,
        default_take_profit_pct=0.15,
        dynamic_sl_tp=False,
        max_positions=8,
        max_position_pct=0.12,
        kelly_fraction=0.50,
        min_position_pct=0.08,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
    )

    print("=" * 130)
    print("INDIVIDUAL STRATEGY DIAGNOSIS — 2Y backtest, concentrated config (8 pos, 12% max)")
    print("=" * 130)

    for market, base, currency in [("KR", kr_base, "₩"), ("US", us_base, "$")]:
        print(f"\n{'─' * 130}")
        print(f"  {market} MARKET  (initial: {currency}{base['initial_equity']:,.0f})")
        print(f"{'─' * 130}")
        print(
            f"  {'Strategy':25s} {'Return':>8s} {'CAGR':>7s} {'Sharpe':>7s} "
            f"{'Sortino':>8s} {'MDD':>7s} {'Trades':>7s} {'WR':>6s} "
            f"{'PF':>6s} {'Hold':>5s} {'Final Equity':>15s}"
        )
        print(f"  {'─' * 118}")

        results = []
        for strat in ALL_STRATEGIES:
            r = await run_single_strategy(strat, market, base)
            results.append(r)
            if "error" in r:
                print(f"  {strat:25s} ERROR: {r['error'][:80]}")
            else:
                print(
                    f"  {strat:25s} {r['return_pct']:>+7.1f}% {r['cagr']:>6.1%} "
                    f"{r['sharpe']:>+7.2f} {r['sortino']:>+8.2f} "
                    f"{r['mdd']:>6.1f}% {r['trades']:>7d} {r['win_rate']:>5.1f}% "
                    f"{r['profit_factor']:>6.2f} {r['avg_hold']:>4.0f}d "
                    f"{currency}{r['final_equity']:>14,.0f}"
                )

        # Summary
        valid = [r for r in results if "error" not in r and r["trades"] > 0]
        if valid:
            positive = [r for r in valid if r["return_pct"] > 0]
            negative = [r for r in valid if r["return_pct"] <= 0]
            best = max(valid, key=lambda r: r["return_pct"])
            worst = min(valid, key=lambda r: r["return_pct"])
            print(f"\n  {market} Summary:")
            print(f"    Profitable: {len(positive)}/{len(valid)} strategies")
            print(f"    Best:  {best['strategy']} ({best['return_pct']:+.1f}%, Sharpe {best['sharpe']:.2f})")
            print(f"    Worst: {worst['strategy']} ({worst['return_pct']:+.1f}%, Sharpe {worst['sharpe']:.2f})")

    print("\n" + "=" * 130)


if __name__ == "__main__":
    asyncio.run(main())
