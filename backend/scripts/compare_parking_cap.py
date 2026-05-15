"""Cash parking max_pct cap sweep (US 2y).

2026-04-28: cash_parking was disabled because parking grew to 57% of US
positions, eating the exposure cap and rejecting strategy BUYs.

2026-05-15 (P3): introduce max_pct cap so parking can re-enable safely.
This sweep tests the new cap vs prior "uncapped" behavior.

Baseline: parking disabled (current production).
Variants:
  V0_off:           no parking (current live)
  V1_cap_15:        parking 30% threshold, cap 15% of equity
  V2_cap_25:        cap 25% of equity (recommended starting point)
  V3_cap_40:        cap 40% of equity
  V4_uncapped:      no cap (= prior production behavior that failed)
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


VARIANTS = [
    ("V0_off",       False, 0.0,  1.0),
    ("V1_cap_15",    True,  0.30, 0.15),
    ("V2_cap_25",    True,  0.30, 0.25),
    ("V3_cap_40",    True,  0.30, 0.40),
    ("V4_uncapped",  True,  0.30, 1.00),
]


def _cfg(parking: bool, threshold: float, max_pct: float) -> dict:
    loader = StrategyConfigLoader()
    eval_cfg = loader.get_market_evaluation_loop_config("US")
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=20,
        max_position_pct=0.15,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight") or 0.2),
        disabled_strategies=loader.get_market_disabled_strategies("US"),
        kelly_fraction=0.50,
        stale_pnl_threshold=-0.05,
        stale_time_days=int(eval_cfg.get("stale_time_days") or 2),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold") or -0.02),
        # Parking knobs
        enable_cash_parking=parking,
        cash_parking_symbol="SPY",
        cash_parking_threshold=threshold,
        cash_parking_max_pct=max_pct,
        cash_parking_split_ratio=1.0,
        cash_parking_enable_unpark=True,
    )


async def run(name: str, parking: bool, thr: float, cap: float) -> dict:
    cfg = PipelineConfig(**_cfg(parking, thr, cap))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    # Count parking trades
    park_trades = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("cash_parking")
    )
    return dict(
        name=name, parking=parking, thr=thr, cap=cap,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        park_trades=park_trades,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  Cash parking max_pct cap sweep (US 2y, Phase 2+P1 baseline)")
    print("=" * 110)
    results = []
    for name, parking, thr, cap in VARIANTS:
        cap_str = f"{cap:.0%}" if parking else "—"
        print(f"\n▶ {name}  parking={parking} thr={thr:.0%} cap={cap_str}")
        r = await run(name, parking, thr, cap)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  "
              f"Trades={r['trades']}  Park={r['park_trades']}  "
              f"({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (f"{'Variant':<14} {'cap':>5} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'PF':>6} {'Trades':>7} {'Park':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        cap_str = f"{r['cap']:.0%}" if r['parking'] else "—"
        print(f"{r['name']:<14} {cap_str:>5} {r['ret']:+7.1f} "
              f"{r['sharpe']:+7.2f} {r['mdd']:7.1f} {r['pf']:6.2f} "
              f"{r['trades']:7d} {r['park_trades']:6d}")

    print("\nDelta vs V0_off (no parking):")
    v0 = results[0]
    for r in results[1:]:
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        d_pf = r['pf'] - v0['pf']
        ok = d_ret > 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= -0.05
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<14}  ΔRet={d_ret:+5.1f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.1f}pp  ΔPF={d_pf:+5.2f}  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
