"""CI Backtest Gate — catch strategy/config regressions in PRs.

Runs a short (6-month) KR + US pipeline backtest with the current
config/strategies.yaml, compares key metrics against the committed
baseline in tests/backtest_baselines.json, and exits non-zero if any
metric has regressed beyond the defined tolerance.

Intentional performance changes should be accompanied by an update to
backtest_baselines.json in the same PR (makes the change explicit).

Thresholds (failure if exceeded):
- Sharpe drop by > 0.30 (abs)
- Profit factor drop by > 0.20 (abs)
- MDD (drawdown) worsens by > 5 percentage points
- Trades count drop by > 50% (detects silent disable bugs)

Return drop is NOT gated — live vs backtest return has high variance
and regression would be too noisy.

Usage:
    cd backend && python scripts/ci_backtest_gate.py
    cd backend && python scripts/ci_backtest_gate.py --update-baseline
"""

import argparse
import asyncio
import functools
import json
import logging
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx", "scanner", "data",
         "backtest", "strategies", "engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig  # noqa
from backtest.gate import THRESHOLDS, compare  # noqa
from strategies.config_loader import StrategyConfigLoader  # noqa

BASELINE_PATH = Path(__file__).parent.parent / "tests" / "backtest_baselines.json"


def _kr_config(disabled: list[str]) -> dict:
    """Matches live KR evaluation loop config (markets.KR in strategies.yaml)."""
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


def _us_config(disabled: list[str]) -> dict:
    """Matches live US evaluation loop config (markets.US in strategies.yaml)."""
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

async def _run(market: str) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies(market)
    kw = _kr_config(disabled) if market == "KR" else _us_config(disabled)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    # 2y — engine requires 250+ bars of SPY/regime data for SMA200 warmup
    # (common_start = spy_dates[250] in full_pipeline.py). 1y (244 bars)
    # and 6mo throw IndexError. Runtime is ~20-30s per market.
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        elapsed_sec=round(el, 1),
    )


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def _save_baseline(data: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(data, indent=2) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true",
                        help="Run backtest and save results as new baseline.")
    parser.add_argument("--markets", nargs="+", default=["KR", "US"],
                        help="Markets to run (default: KR US).")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for market in args.markets:
        print(f"▶ Running {market} 2y backtest...")
        results[market] = await _run(market)
        r = results[market]
        print(f"  {market}: Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} "
              f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} Trades={r['trades']} "
              f"({r['elapsed_sec']:.0f}s)")

    if args.update_baseline:
        _save_baseline(results)
        print(f"\n✓ Baseline updated: {BASELINE_PATH}")
        return 0

    baseline = _load_baseline()
    all_failures: list[str] = []
    for market, current in results.items():
        baseline_market = baseline.get(market, {})
        failures = compare(baseline_market, current, market)
        all_failures.extend(failures)

    print("\n" + "=" * 72)
    if all_failures:
        print("  BACKTEST GATE — REGRESSION DETECTED")
        print("=" * 72)
        for f in all_failures:
            print(f"  ✗ {f}")
        print(
            "\nIf this change is intentional (e.g. deliberately tighter risk "
            "settings), rerun with --update-baseline and commit the updated "
            "backtest_baselines.json in the same PR."
        )
        return 1

    print("  BACKTEST GATE — PASSED (no metric regression)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
