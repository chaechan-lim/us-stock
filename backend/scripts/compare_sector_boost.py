"""D1 sector-strength BUY boost — sweep sector_boost_weight.

Runs the KR + US 2y backtest at several boost weights. Positive finding
looks like: Return/Sharpe/PF all improve as weight moves up to some
optimum, then roll over. MDD should not worsen dramatically.

Usage:
    cd backend && python scripts/compare_sector_boost.py
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data",
         "backtest", "strategies", "engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig  # noqa
from strategies.config_loader import StrategyConfigLoader  # noqa


WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _base_kw(market: str, disabled: list[str]) -> dict:
    if market == "KR":
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
        )
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
    )


async def _run_one(market: str, weight: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies(market)
    kw = _base_kw(market, disabled)
    kw["sector_boost_weight"] = weight
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        market=market, weight=weight,
        ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor,
        trades=m.total_trades, elapsed=el,
    )


async def main():
    print("=" * 80)
    print("  D1 sector_boost_weight sweep (2y, both markets)")
    print("=" * 80)

    results: list[dict] = []
    for market in ("KR", "US"):
        for w in WEIGHTS:
            print(f"\n▶ {market} weight={w:.2f}")
            try:
                r = await _run_one(market, w)
            except Exception as e:
                print(f"  ✗ {e}")
                import traceback
                traceback.print_exc()
                continue
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} "
                  f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
                  f"Trades={r['trades']} ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for market in ("KR", "US"):
        print(f"\n{market}:")
        print(f"  {'weight':>7} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7}")
        print("  " + "-" * 60)
        base = None
        for r in results:
            if r["market"] != market:
                continue
            if r["weight"] == 0.0:
                base = r
            improvements = ""
            if base and r["weight"] > 0.0:
                d_ret = r["ret"] - base["ret"]
                d_sharpe = r["sharpe"] - base["sharpe"]
                d_mdd = abs(r["mdd"]) - abs(base["mdd"])
                d_pf = r["pf"] - base["pf"]
                flags = ""
                flags += "✓" if d_ret > 0 else "✗"
                flags += "✓" if d_sharpe > 0 else "✗"
                flags += "✓" if d_mdd < 0 else "✗"  # lower MDD = better
                flags += "✓" if d_pf > 0 else "✗"
                improvements = f"  {flags}"
            print(f"  {r['weight']:>7.2f} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} "
                  f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d}{improvements}")


if __name__ == "__main__":
    asyncio.run(main())
