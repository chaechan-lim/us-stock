"""P2 A/B backtest — cold tracker vs DB-seeded tracker.

Validates that boot-seeding the SignalQualityTracker (P2 PR #142) does not
regress metrics vs current live behavior (which boots cold). Same yaml
config and universe; only difference is whether Kelly sizing has access
to per-strategy edge stats from day one.

V0_cold:    signal_quality_seed_path=None → tracker boots empty
V1_seeded:  signal_quality_seed_path=data/signal_quality_snapshot.json

Snapshot was generated from the live trades DB by
scripts/snapshot_signal_quality.py.
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

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


SNAPSHOT_PATH = "../data/signal_quality_snapshot.json"


def _us_cfg(seed: str | None) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("US")
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
        sector_boost_weight=0.2,
        disabled_strategies=disabled,
        signal_quality_seed_path=seed,
    )


async def run(name: str, seed: str | None) -> dict:
    cfg = PipelineConfig(**_us_cfg(seed))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        wr=round(m.win_rate * 100, 1),
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 90)
    print("  P2 A/B — cold tracker vs DB-seeded (US 2y)")
    print("=" * 90)

    results = []
    for name, seed in (("V0_cold", None), ("V1_seeded", SNAPSHOT_PATH)):
        print(f"\n▶ {name}  seed={seed or '(none)'}")
        r = await run(name, seed)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} "
              f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
              f"Trades={r['trades']} WR={r['wr']:.0f}% ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    hdr = f"{'Variant':<14} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'WR%':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<14} {r['ret']:+7.1f} {r['sharpe']:+7.2f} "
              f"{r['mdd']:7.1f} {r['pf']:6.2f} {r['trades']:7d} {r['wr']:5.0f}")

    if len(results) == 2:
        v0, v1 = results
        print("\nDelta V1 vs V0:")
        print(f"  ΔRet={v1['ret']-v0['ret']:+5.1f}pp  "
              f"ΔSharpe={v1['sharpe']-v0['sharpe']:+5.2f}  "
              f"ΔMDD={v1['mdd']-v0['mdd']:+5.1f}pp  "
              f"ΔPF={v1['pf']-v0['pf']:+5.2f}  "
              f"ΔTrades={v1['trades']-v0['trades']:+d}")
        regressed = (
            v1['ret'] < v0['ret'] - 1.0
            or v1['sharpe'] < v0['sharpe'] - 0.10
            or v1['mdd'] < v0['mdd'] - 2.0
            or v1['pf'] < v0['pf'] - 0.10
        )
        if regressed:
            print("  ⚠️  REGRESSION on at least one dimension — investigate before merging P2")
        else:
            print("  ✓  No material regression — safe to merge P2")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
