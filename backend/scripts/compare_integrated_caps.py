"""Integrated cash_parking + ETF Engine cap sweep (US 2y).

Tests the realistic capital-competition scenario the user flagged:
"parking cap 40% + ETF cap 20% = 60% reserved, where does strategy fit?"

Variants (all on Phase 2 + P1 baseline):
  V0_park0_etf0:   no parking, no ETF (Phase 2+P1 baseline)
  V1_park40_etf0:  parking 40% only (P3+P3-A — last sweep winner)
  V2_park0_etf20:  ETF 20% only
  V3_park40_etf20: BOTH at full cap (the question — do they conflict?)
  V4_park25_etf20: parking 25% + ETF 20% (compromise A)
  V5_park40_etf10: parking 40% + ETF 10% conservative (compromise B)
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
    # (name, parking_max_pct (None = off), etf_max_portfolio_pct (None = off))
    ("V0_park0_etf0",   None, None),
    ("V1_park40_etf0",  0.40, None),
    ("V2_park0_etf20",  None, 0.20),
    ("V3_park40_etf20", 0.40, 0.20),
    ("V4_park25_etf20", 0.25, 0.20),
    ("V5_park40_etf10", 0.40, 0.10),
]


def _cfg(parking_max: float | None, etf_max_portfolio: float | None) -> dict:
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
    if parking_max is not None:
        kw.update(
            enable_cash_parking=True,
            cash_parking_symbol="SPY",
            cash_parking_threshold=0.30,
            cash_parking_max_pct=parking_max,
            cash_parking_per_cycle_pct=0.10,
            cash_parking_split_ratio=1.0,
            cash_parking_enable_unpark=True,
        )
    if etf_max_portfolio is not None:
        kw.update(
            enable_etf_engine=True,
            etf_max_portfolio_pct=etf_max_portfolio,
            etf_max_single_pct=etf_max_portfolio / 2,
        )
    return kw


async def run(name: str, park: float | None, etf: float | None) -> dict:
    cfg = PipelineConfig(**_cfg(park, etf))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    park_trades = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("cash_parking")
    )
    etf_trades = sum(
        1 for t in res.trades
        if (t.strategy_name or "").startswith("etf_")
    )
    return dict(
        name=name, park=park, etf=etf,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        park_trades=park_trades,
        etf_trades=etf_trades,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  Integrated cash_parking + ETF Engine sweep (US 2y, Phase 2+P1 baseline)")
    print("=" * 110)
    results = []
    for name, park, etf in VARIANTS:
        park_s = f"{park:.0%}" if park else "—"
        etf_s = f"{etf:.0%}" if etf else "—"
        print(f"\n▶ {name}  parking={park_s} etf={etf_s}")
        r = await run(name, park, etf)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}%  Sharpe={r['sharpe']:+.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  "
              f"Trades={r['trades']}  Park={r['park_trades']} ETF={r['etf_trades']}  "
              f"({r['elapsed']:.0f}s)")

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (f"{'Variant':<18} {'park':>6} {'etf':>5} {'Ret%':>7} {'Sharpe':>7} "
           f"{'MDD%':>7} {'PF':>6} {'Trades':>7} {'Park':>5} {'ETF':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ps = f"{r['park']:.0%}" if r['park'] else "—"
        es = f"{r['etf']:.0%}" if r['etf'] else "—"
        print(f"{r['name']:<18} {ps:>6} {es:>5} {r['ret']:+7.1f} "
              f"{r['sharpe']:+7.2f} {r['mdd']:7.1f} {r['pf']:6.2f} "
              f"{r['trades']:7d} {r['park_trades']:5d} {r['etf_trades']:4d}")

    print("\nDelta vs V0 (no parking, no ETF):")
    v0 = results[0]
    for r in results[1:]:
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        d_pf = r['pf'] - v0['pf']
        ok = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= -0.05
        tag = "✓ ADOPT" if ok else "✗"
        print(f"  {r['name']:<18}  ΔRet={d_ret:+5.1f}pp  ΔSharpe={d_sharpe:+5.2f}  "
              f"ΔMDD={d_mdd:+5.1f}pp  ΔPF={d_pf:+5.2f}  {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
