"""F1 position_cleanup threshold sweep.

`_stale_pnl_threshold` is the floor below which a held position with a HOLD
signal is force-sold by the synthetic 'position_cleanup' strategy. Live
runs use -0.05 (5% loss). Backtest's PipelineConfig default is 0
(disabled), creating a backtest/live mismatch on top of the actual
threshold-tuning question.

This sweep enables cleanup at varying thresholds:
  off      : disabled (current backtest default — no force-sell)
  -0.05    : current live behaviour
  -0.08    : looser, gives burst names more rope
  -0.10    : even looser

Per-market baseline + 4 variants → 8 backtests, ~5 min total.
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner",
         "data", "backtest", "strategies", "engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig  # noqa
from strategies.config_loader import StrategyConfigLoader  # noqa


def _kr_cfg(disabled: list[str], stale: float) -> dict:
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.20,
        max_positions=12,
        max_position_pct=0.20,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        disabled_strategies=disabled,
        sector_boost_weight=0.3,  # match live
        stale_pnl_threshold=stale,
    )


def _us_cfg(disabled: list[str], stale: float) -> dict:
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=0.10,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        disabled_strategies=disabled,
        sector_boost_weight=0.2,  # match live
        stale_pnl_threshold=stale,
    )


VARIANTS = [
    ("off",   0.0),
    ("-0.05", -0.05),
    ("-0.08", -0.08),
    ("-0.10", -0.10),
]


async def _run(market: str, name: str, stale: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies(market)
    kw = _kr_cfg(disabled, stale) if market == "KR" else _us_cfg(disabled, stale)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    cleanup_trades = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("position_cleanup")
    )
    return dict(
        market=market, name=name, stale=stale,
        ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor,
        trades=m.total_trades, cleanup=cleanup_trades,
        elapsed=el,
    )


async def main():
    print("=" * 90)
    print("  F1 position_cleanup threshold sweep (2y)")
    print("=" * 90)
    results: list[dict] = []
    for market in ("KR", "US"):
        for name, stale in VARIANTS:
            print(f"\n▶ {market} cleanup={name}")
            try:
                r = await _run(market, name, stale)
            except Exception as e:
                print(f"  ✗ {e}")
                import traceback; traceback.print_exc(); continue
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} "
                  f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
                  f"Trades={r['trades']} cleanup={r['cleanup']} ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    for market in ("KR", "US"):
        print(f"\n{market}:")
        print(f"  {'cleanup':>8} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} "
              f"{'Trades':>7} {'CleanupTrd':>11}")
        print("  " + "-" * 70)
        for r in results:
            if r["market"] != market:
                continue
            print(f"  {r['name']:>8} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} "
                  f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} "
                  f"{r['cleanup']:>11d}")


if __name__ == "__main__":
    asyncio.run(main())
