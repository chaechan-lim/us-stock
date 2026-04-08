"""Validate the post-4-06 strategy overhaul on US + KR via full pipeline backtest.

Compares the current strategies.yaml baseline against two anti-whipsaw variants
to confirm whether the recent exit overhaul (hard_sl -7→-15, profit_protect 15→25,
trailing 4/3 → 8/4) actually improves alpha.

Usage:
    cd backend && ../venv/bin/python scripts/validate_new_strategy.py
"""

import asyncio
import sys
import time
import logging
import functools

# Force line-buffered stdout (so progress shows up under nohup)
print = functools.partial(print, flush=True)

sys.path.insert(0, ".")

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
for n in ("httpx", "urllib3", "yfinance", "peewee", "backtest", "strategies", "data", "scanner"):
    logging.getLogger(n).setLevel(logging.WARNING)


# US base — mirrors current live settings
US_BASE = dict(
    market="US",
    initial_equity=100_000,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=20,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.05,
    volume_adjusted_slippage=True,
    disabled_strategies=[
        "bollinger_squeeze", "volume_profile",
        "cis_momentum", "larry_williams", "bnf_deviation",
    ],
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
)

KR_BASE = dict(
    market="KR",
    initial_equity=100_000,
    default_stop_loss_pct=0.12,
    default_take_profit_pct=0.20,
    max_positions=15,
    max_position_pct=0.08,
    sell_cooldown_days=1,
    whipsaw_max_losses=2,
    min_hold_days=1,
    slippage_pct=0.08,  # KR is wider
    volume_adjusted_slippage=True,
    min_confidence=0.50,
    held_sell_bias=0.05,
    held_min_confidence=0.40,
)


VARIANTS = [
    ("US_new",        {**US_BASE}),
    ("US_anti_whip",  {**US_BASE, "sell_cooldown_days": 5, "stale_pnl_threshold": -0.07}),
    ("KR_new",        {**KR_BASE}),
    ("KR_anti_whip",  {**KR_BASE, "sell_cooldown_days": 5, "stale_pnl_threshold": -0.07}),
]


async def run_variant(name: str, overrides: dict, period: str = "2y") -> dict:
    cfg = PipelineConfig(**overrides)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period=period)
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name,
        ret=m.total_return_pct,
        cagr=(m.cagr or 0) * 100,
        sharpe=m.sharpe_ratio,
        sortino=m.sortino_ratio,
        mdd=m.max_drawdown_pct,
        trades=m.total_trades,
        wr=m.win_rate,
        pf=m.profit_factor,
        alpha=m.alpha,
        bench=m.benchmark_return_pct,
        avg_hold=m.avg_holding_days,
        elapsed=el,
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 100)
    print("  New Strategy Validation — US + KR Full Pipeline (2Y)")
    print("=" * 100)

    out = []
    for name, ov in VARIANTS:
        desc = ", ".join(f"{k}={v}" for k, v in ov.items() if k not in US_BASE and k not in KR_BASE)
        print(f"\n▶ {name}  {desc or '(baseline)'}")
        try:
            r = await run_variant(name, ov)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            continue
        out.append(r)
        print(f"  Ret={r['ret']:+.1f}%  CAGR={r['cagr']:+.1f}%  Sharpe={r['sharpe']:.2f}  "
              f"MDD={r['mdd']:.1f}%  PF={r['pf']:.2f}  Trades={r['trades']}  "
              f"Alpha={r['alpha']:+.1f}%  ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print(f"{'Variant':<18} {'Ret%':>7} {'CAGR%':>7} {'Sharpe':>7} {'MDD%':>7} "
          f"{'PF':>6} {'Trades':>7} {'WR%':>6} {'Alpha%':>8} {'Hold':>5}")
    print("-" * 100)
    for r in out:
        print(f"{r['name']:<18} {r['ret']:>+7.1f} {r['cagr']:>+7.1f} {r['sharpe']:>7.2f} "
              f"{r['mdd']:>7.1f} {r['pf']:>6.2f} {r['trades']:>7d} {r['wr']:>6.1f} "
              f"{r['alpha']:>+8.1f} {r['avg_hold']:>5.0f}")

    print("\nStrategy contribution (US_new):")
    for r in out:
        if r["name"] != "US_new":
            continue
        for sname, st in sorted(r["strat_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            if st["trades"] > 0:
                print(f"  {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                      f"PnL=${st['pnl']:+,.0f}")

    print("\nStrategy contribution (KR_new):")
    for r in out:
        if r["name"] != "KR_new":
            continue
        for sname, st in sorted(r["strat_stats"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            if st["trades"] > 0:
                print(f"  {sname:<28} trades={st['trades']:>3d}  WR={st['win_rate']:>5.1f}%  "
                      f"PnL={st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
