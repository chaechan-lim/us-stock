"""E1/E2/E3 (supertrend entry hardening) + D1 (sector boost) sweep.

Variants:
  Baseline    : current yaml (supertrend conf_bars=1, no vol filter, no sector boost)
  E1          : supertrend confirmation_bars 1 → 2 (require 2-bar confirmation)
  E2          : supertrend volatility_filter=True, max_volatility_pct=3
  E3          : E1 + E2
  D1          : sector_boost_weight (KR 0.3, US 0.2)
  E3+D1       : all three

KR doesn't run supertrend (disabled per Option A), so E1/E2/E3 are
meaningful only for US. KR only gets baseline + D1.

Per CLAUDE.md "measure relative, not absolute": look at direction and
ordering, not absolute numbers.
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


def _kr_cfg(disabled: list[str], sector_weight: float = 0.0) -> dict:
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
        sector_boost_weight=sector_weight,
    )


def _us_cfg(disabled: list[str], sector_weight: float = 0.0) -> dict:
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
        sector_boost_weight=sector_weight,
    )


async def _run(market: str, name: str, kw: dict,
               supertrend_override: dict | None = None) -> dict:
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    if supertrend_override:
        strat = eng._registry.get("supertrend")
        if strat:
            strat.set_params(supertrend_override)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        market=market, variant=name,
        ret=m.total_return_pct, sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct, pf=m.profit_factor,
        trades=m.total_trades, elapsed=el,
    )


async def main():
    loader = StrategyConfigLoader()
    kr_disabled = loader.get_market_disabled_strategies("KR")
    us_disabled = loader.get_market_disabled_strategies("US")

    runs: list[tuple[str, str, dict, dict | None]] = [
        # KR
        ("KR", "Baseline",  _kr_cfg(kr_disabled), None),
        ("KR", "D1(0.3)",   _kr_cfg(kr_disabled, sector_weight=0.3), None),
        # US — all 5 entry/sector variants
        ("US", "Baseline",  _us_cfg(us_disabled), None),
        ("US", "E1_conf2",  _us_cfg(us_disabled),
                            {"confirmation_bars": 2}),
        ("US", "E2_volfilt",_us_cfg(us_disabled),
                            {"volatility_filter": True,
                             "max_volatility_pct": 3.0}),
        ("US", "E3_both",   _us_cfg(us_disabled),
                            {"confirmation_bars": 2,
                             "volatility_filter": True,
                             "max_volatility_pct": 3.0}),
        ("US", "D1(0.2)",   _us_cfg(us_disabled, sector_weight=0.2), None),
        ("US", "E3+D1",     _us_cfg(us_disabled, sector_weight=0.2),
                            {"confirmation_bars": 2,
                             "volatility_filter": True,
                             "max_volatility_pct": 3.0}),
    ]

    print("=" * 90)
    print("  E1/E2/E3 + D1 sweep (2y, US supertrend + sector boost)")
    print("=" * 90)

    results: list[dict] = []
    for market, name, kw, st_override in runs:
        print(f"\n▶ {market} {name}")
        try:
            r = await _run(market, name, kw, st_override)
        except Exception as e:
            print(f"  ✗ {e}")
            import traceback
            traceback.print_exc()
            continue
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} "
              f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
              f"Trades={r['trades']} ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 90)
    print("  SUMMARY (vs Baseline per market)")
    print("=" * 90)
    for market in ("KR", "US"):
        base = next((r for r in results if r["market"] == market
                     and r["variant"] == "Baseline"), None)
        print(f"\n{market}:")
        print(f"  {'variant':<14} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} "
              f"{'PF':>6} {'Trades':>7} {'vs base':>8}")
        print("  " + "-" * 70)
        for r in results:
            if r["market"] != market:
                continue
            if r["variant"] == "Baseline" or base is None:
                flags = ""
            else:
                d_ret = r["ret"] - base["ret"]
                d_sh = r["sharpe"] - base["sharpe"]
                d_mdd = abs(r["mdd"]) - abs(base["mdd"])
                d_pf = r["pf"] - base["pf"]
                flags = ""
                flags += "✓" if d_ret > 0 else "✗"
                flags += "✓" if d_sh > 0 else "✗"
                flags += "✓" if d_mdd < 0 else "✗"
                flags += "✓" if d_pf > 0 else "✗"
            print(f"  {r['variant']:<14} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} "
                  f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} {flags:>8}")


if __name__ == "__main__":
    asyncio.run(main())
