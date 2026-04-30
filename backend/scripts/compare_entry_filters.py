"""F2 entry-filter sweep — root-cause fix for "buy then immediately drop".

Tests two new strategy params introduced 2026-04-30:
  - supertrend.pullback_max_pct  : reject BUY when close is more than
    X% above the supertrend line. Forces entry near support.
  - dual_momentum.max_5d_gain    : reject BUY when 5-day return exceeds
    X. Avoids buying exhausted vertical moves.

Variants per market:
  Baseline   : current
  Pullback5  : supertrend pullback_max_pct=0.05
  Pullback7  : supertrend pullback_max_pct=0.07
  AntiOB15   : dual_momentum max_5d_gain=0.15
  AntiOB20   : dual_momentum max_5d_gain=0.20
  Combined   : pullback=0.05 + max_5d_gain=0.15

KR runs only dual_momentum so pullback variants there are no-ops.
US runs trend_following + supertrend so max_5d_gain has no effect.
We sweep all in both markets for completeness; no-op variants will
match Baseline within stochastic noise.
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


def _kr_cfg(disabled: list[str]) -> dict:
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
        sector_boost_weight=0.3,
    )


def _us_cfg(disabled: list[str]) -> dict:
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
        sector_boost_weight=0.2,
    )


VARIANTS = [
    ("Baseline",   {}),
    ("Pullback5",  {"supertrend": {"pullback_max_pct": 0.05}}),
    ("Pullback7",  {"supertrend": {"pullback_max_pct": 0.07}}),
    ("AntiOB15",   {"dual_momentum": {"max_5d_gain": 0.15}}),
    ("AntiOB20",   {"dual_momentum": {"max_5d_gain": 0.20}}),
    ("Combined",   {"supertrend": {"pullback_max_pct": 0.05},
                    "dual_momentum": {"max_5d_gain": 0.15}}),
]


async def _run(market: str, name: str, overrides: dict) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies(market)
    kw = _kr_cfg(disabled) if market == "KR" else _us_cfg(disabled)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    for strat_name, params in overrides.items():
        s = eng._registry.get(strat_name)
        if s:
            s.set_params(params)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        market=market, name=name,
        ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor,
        trades=m.total_trades, elapsed=el,
    )


async def main():
    print("=" * 90)
    print("  F2 entry-filter sweep — pullback + anti-overbought (2y)")
    print("=" * 90)
    results: list[dict] = []
    for market in ("KR", "US"):
        for name, overrides in VARIANTS:
            print(f"\n▶ {market} {name}")
            try:
                r = await _run(market, name, overrides)
            except Exception as e:
                print(f"  ✗ {e}")
                import traceback; traceback.print_exc(); continue
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} "
                  f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
                  f"Trades={r['trades']} ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 90)
    print("  SUMMARY (vs Baseline per market)")
    print("=" * 90)
    for market in ("KR", "US"):
        base = next((r for r in results if r["market"] == market and r["name"] == "Baseline"), None)
        print(f"\n{market}:")
        print(f"  {'variant':<11} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} "
              f"{'Trades':>7} {'vs base':>9}")
        print("  " + "-" * 70)
        for r in results:
            if r["market"] != market:
                continue
            flags = ""
            if base and r["name"] != "Baseline":
                flags += "✓" if r["ret"] > base["ret"] else "✗"
                flags += "✓" if r["sharpe"] > base["sharpe"] else "✗"
                flags += "✓" if abs(r["mdd"]) < abs(base["mdd"]) else "✗"
                flags += "✓" if r["pf"] > base["pf"] else "✗"
            print(f"  {r['name']:<11} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} "
                  f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} {flags:>9}")


if __name__ == "__main__":
    asyncio.run(main())
