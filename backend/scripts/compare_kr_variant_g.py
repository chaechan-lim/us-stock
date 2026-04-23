"""KR variant G: B+C merge — dual_momentum lookback=3 + volatility_filter=False.

Reference points from compare_kr_strategy_mix_2026_04.py run at 12:05:
  A_baseline_dm18       Ret=+1.8%  Sharpe=-0.17  MDD=-7.0%  PF=1.06  FAIL
  B_dm_lookback3        Ret=+5.2%  Sharpe=+0.20  MDD=-5.2%  PF=1.14  PASS
  C_dm_no_volfilter     Ret=+6.8%  Sharpe=+0.32  MDD=-6.6%  PF=1.14  PASS

Hypothesis: merging B+C into one set of params may compound the gains.
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

ALL_STRATS = [
    "supertrend", "macd_histogram", "rsi_divergence", "bollinger_squeeze",
    "volume_profile", "regime_switch", "sector_rotation", "cis_momentum",
    "larry_williams", "bnf_deviation", "volume_surge", "cross_sectional_momentum",
    "pead_drift", "quality_factor", "trend_following", "donchian_breakout",
    "dual_momentum",
]


def disabled_except(keep: list[str]) -> list[str]:
    return [s for s in ALL_STRATS if s not in keep]


KR_BASE = dict(
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
)


async def run(name: str, kw: dict, overrides: dict | None):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    if overrides:
        for strat_name, params in overrides.items():
            strat = eng._registry.get(strat_name)
            if strat:
                strat.set_params(params)

    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    return dict(
        name=name,
        ret=m.total_return_pct,
        sharpe=m.sharpe_ratio,
        mdd=m.max_drawdown_pct,
        pf=m.profit_factor,
        trades=m.total_trades,
        wr=m.win_rate,
        elapsed=el,
        strat_stats=res.strategy_stats,
    )


async def main():
    print("=" * 100)
    print("  KR Variant G — dm lookback=3 + vol_filter=False (B+C merge)")
    print("=" * 100)

    name = "G_dm_lookback3_no_volfilter"
    kw = {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])}
    overrides = {"dual_momentum": {"lookback_months": 3, "volatility_filter": False}}

    print(f"\n▶ {name}")
    r = await run(name, kw, overrides)
    print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.1f}% "
          f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% ({r['elapsed']:.0f}s)")

    # Compare to earlier runs
    print("\n" + "=" * 100)
    print("  COMPARISON TABLE (previous results from 12:05 run)")
    print("=" * 100)
    ref = [
        ("A_baseline_dm18",            +1.8, -0.17, -7.0,  1.06, "FAIL"),
        ("B_dm_lookback3",             +5.2, +0.20, -5.2,  1.14, "PASS"),
        ("C_dm_no_volfilter",          +6.8, +0.32, -6.6,  1.14, "PASS"),
        ("E_dm+supertrend",            +4.1, +0.10, -11.6, 1.09, "PASS"),
        (r["name"], r["ret"], r["sharpe"], r["mdd"], r["pf"],
            ("PASS" if (r["sharpe"] > 0 and r["mdd"] < 15 and r["pf"] > 1.0) else "FAIL")),
    ]
    print(f"\n{'Variant':<32} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Floor':>6}")
    print("-" * 80)
    for rec in ref:
        marker = " ←" if rec[0] == r["name"] else ""
        print(f"{rec[0]:<32} {rec[1]:>+7.1f} {rec[2]:>7.2f} {rec[3]:>7.1f} "
              f"{rec[4]:>6.2f} {rec[5]:>6}{marker}")

    # baseline compare
    base = (1.8, -0.17, -7.0, 1.06)
    imp = (r["ret"] > base[0], r["sharpe"] > base[1], r["mdd"] < base[2], r["pf"] > base[3])
    wins = sum(imp)
    flags = "".join(("✓" if x else "✗") for x in imp)
    print(f"\n  Improvement vs baseline A: Ret/Sharpe/MDD/PF = {flags}  ({wins}/4)")
    if wins == 4 and r["sharpe"] > 0 and r["mdd"] < 15 and r["pf"] > 1.0:
        print("  ← ADOPT CANDIDATE (all 4 dims improve + floor pass)")

    # strategy breakdown
    if r["strat_stats"]:
        print(f"\n  {r['name']}:")
        for sn, st in sorted(r["strat_stats"].items(),
                             key=lambda x: x[1]["pnl"], reverse=True):
            if st["trades"] > 0:
                print(f"    {sn:<25} trades={st['trades']:>3d} "
                      f"WR={st['win_rate']:>5.1f}% PnL={st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
