"""Bear-market parking value test (multiple historical windows).

User question (2026-05-15): "2y backtest is mostly bull market → parking
value not measured. Need a bear-market test."

3 bear-period windows + 1 bull control:
  A) 2022 full year         SPY -19.4% (rate-hike bear)
  B) 2020-Q1 + Q2 recovery  SPY -33% → +37% (COVID V)
  C) 2018-Q4                SPY -19% (Powell pivot)
  D) 2024 (bull control)    SPY +25%

Each window, compare V0 (parking off) vs V1 (parking 40% cap, P3+P3-A).
ETF Engine OFF (already shown -7.3pp; no reason to retest).
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


# Windows: (name, start, end, spy_context)
# All windows include ~250-bar SMA200 warmup before the target period.
# Trading happens after warmup → the target market regime is captured.
WINDOWS = [
    ("2022_bear",       "2020-12-01", "2023-06-30", "warmup + 2022 -19% + 2023 recovery"),
    ("2020_covid",      "2019-06-01", "2020-12-31", "warmup + COVID -33% V-recovery"),
    ("2018_pivot_bear", "2017-09-01", "2019-06-30", "warmup + 2018Q4 -19% + recovery"),
    ("2024_bull_ctrl",  "2023-01-01", "2024-12-31", "warmup + 2024 +25% bull control"),
]


def _cfg(parking_on: bool) -> dict:
    loader = StrategyConfigLoader()
    eval_cfg = loader.get_market_evaluation_loop_config("US")
    kw = dict(
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
    )
    if parking_on:
        kw.update(
            enable_cash_parking=True,
            cash_parking_symbol="SPY",
            cash_parking_threshold=0.30,
            cash_parking_max_pct=0.40,
            cash_parking_per_cycle_pct=0.10,
            cash_parking_split_ratio=1.0,
            cash_parking_enable_unpark=True,
        )
    return kw


async def run(label: str, start: str, end: str, parking: bool) -> dict:
    cfg = PipelineConfig(**_cfg(parking))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(start=start, end=end)
    el = time.time() - t0
    m = res.metrics
    park_trades = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("cash_parking")
    )
    return dict(
        label=label,
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
    print("  Bear-market parking value test")
    print("=" * 110)

    all_rows = []
    for name, start, end, ctx in WINDOWS:
        print(f"\n────── {name}  {start} → {end}  ({ctx}) ──────")
        v0 = await run(f"{name}_V0_off", start, end, parking=False)
        v1 = await run(f"{name}_V1_park40", start, end, parking=True)
        all_rows.append((name, ctx, v0, v1))
        print(f"  V0 (off):         Ret={v0['ret']:+.2f}%  Sharpe={v0['sharpe']:+.2f}  "
              f"MDD={v0['mdd']:+.2f}%  PF={v0['pf']:.2f}  Trades={v0['trades']}")
        print(f"  V1 (parking 40%): Ret={v1['ret']:+.2f}%  Sharpe={v1['sharpe']:+.2f}  "
              f"MDD={v1['mdd']:+.2f}%  PF={v1['pf']:.2f}  Trades={v1['trades']}  "
              f"Park={v1['park_trades']}")
        d_ret = v1['ret'] - v0['ret']
        d_sharpe = v1['sharpe'] - v0['sharpe']
        d_mdd = v1['mdd'] - v0['mdd']
        d_pf = v1['pf'] - v0['pf']
        print(f"  ΔV1-V0:           ΔRet={d_ret:+.2f}pp  ΔSharpe={d_sharpe:+.2f}  "
              f"ΔMDD={d_mdd:+.2f}pp  ΔPF={d_pf:+.2f}")

    print("\n" + "=" * 110)
    print("  CROSS-WINDOW SUMMARY (Δ = V1 parking 40% − V0 off)")
    print("=" * 110)
    hdr = f"{'Window':<14} {'ctx':<25} {'ΔRet':>8} {'ΔSharpe':>8} {'ΔMDD':>8} {'ΔPF':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, ctx, v0, v1 in all_rows:
        d_ret = v1['ret'] - v0['ret']
        d_sharpe = v1['sharpe'] - v0['sharpe']
        d_mdd = v1['mdd'] - v0['mdd']
        d_pf = v1['pf'] - v0['pf']
        verdict = "✓" if (d_ret > 0 and d_sharpe >= -0.05) else "✗" if d_ret < -1 else "~"
        print(f"{name:<14} {ctx:<25} {d_ret:>+7.2f}pp {d_sharpe:>+8.2f} "
              f"{d_mdd:>+7.2f}pp {d_pf:>+7.2f}  {verdict}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
