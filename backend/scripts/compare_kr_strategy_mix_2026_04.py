"""KR strategy diversification — baseline vs 5 variants.

Baseline A: dual_momentum only (yaml defaults: lookback=18, vol_filter=3%)
Variant B: dual_momentum lookback 18 → 3 (단기 모멘텀화)
Variant C: dual_momentum vol_filter disabled
Variant D: baseline + cis_momentum
Variant E: baseline + supertrend
Variant F: baseline + sector_rotation

Runs 2y KR pipeline backtest. Reports Ret/Sharpe/MDD/PF.

Judging criteria (per CLAUDE.md):
- Combo activation floor: Sharpe > 0, MDD < 15%, PF > 1.0
- Ret / Sharpe / MDD / PF all four dimensions improve vs A
- Long-run targets: Sharpe > 1.0, CAGR > 12%
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

# Every non-target strategy disabled. Baseline keeps dual_momentum only.
ALL_STRATS = [
    "supertrend", "macd_histogram", "rsi_divergence", "bollinger_squeeze",
    "volume_profile", "regime_switch", "sector_rotation", "cis_momentum",
    "larry_williams", "bnf_deviation", "volume_surge", "cross_sectional_momentum",
    "pead_drift", "quality_factor", "trend_following", "donchian_breakout",
    "dual_momentum",
]


def disabled_except(keep: list[str]) -> list[str]:
    return [s for s in ALL_STRATS if s not in keep]


# Mirror current live KR settings (config/strategies.yaml markets.KR overrides)
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


VARIANTS = [
    ("A_baseline_dm18",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])},
        None),
    ("B_dm_lookback3",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])},
        {"dual_momentum": {"lookback_months": 3}}),
    ("C_dm_no_volfilter",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum"])},
        {"dual_momentum": {"volatility_filter": False}}),
    ("D_dm+cis_momentum",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "cis_momentum"])},
        None),
    ("E_dm+supertrend",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "supertrend"])},
        None),
    ("F_dm+sector_rotation",
        {**KR_BASE, "disabled_strategies": disabled_except(["dual_momentum", "sector_rotation"])},
        None),
]


async def run(name: str, kw: dict, overrides: dict | None):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    # Param overrides applied to backtest's internal registry
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
        alpha=m.alpha,
        elapsed=el,
        strat_stats=res.strategy_stats,
    )


def activation_floor(r) -> str:
    """CLAUDE.md combo activation floor: Sharpe>0, MDD<15%, PF>1.0."""
    ok = r["sharpe"] > 0 and r["mdd"] < 15.0 and r["pf"] > 1.0
    return "PASS" if ok else "FAIL"


def compare_to_baseline(r, base) -> dict:
    """All 4 dims must improve vs baseline to adopt combo."""
    return {
        "ret": r["ret"] > base["ret"],
        "sharpe": r["sharpe"] > base["sharpe"],
        "mdd": r["mdd"] < base["mdd"],  # lower is better
        "pf": r["pf"] > base["pf"],
    }


async def main():
    print("=" * 100)
    print("  KR Strategy Mix 2026-04 — baseline + 5 variants (2Y, 100M KRW, 20% pos)")
    print("=" * 100)
    results = []
    for name, kw, overrides in VARIANTS:
        print(f"\n▶ {name}")
        try:
            r = await run(name, kw, overrides)
        except Exception as e:
            print(f"  ✗ {e}")
            import traceback
            traceback.print_exc()
            continue
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.1f}% "
              f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% "
              f"alpha={r['alpha']:+.1f}% ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"\n{'Variant':<22} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} "
          f"{'Trades':>7} {'WR%':>5} {'Floor':>6}")
    print("-" * 80)
    baseline = next((r for r in results if r["name"].startswith("A_")), None)
    for r in results:
        floor = activation_floor(r)
        print(f"{r['name']:<22} {r['ret']:>+7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} "
              f"{r['pf']:>6.2f} {r['trades']:>7d} {r['wr']:>5.0f}% {floor:>6}")

    if baseline:
        print(f"\nImprovement vs baseline A ({baseline['name']}):")
        for r in results:
            if r is baseline:
                continue
            imp = compare_to_baseline(r, baseline)
            wins = sum(imp.values())
            flags = "".join(("✓" if imp[k] else "✗") for k in ("ret", "sharpe", "mdd", "pf"))
            mark = " ← ADOPT CANDIDATE" if wins == 4 and activation_floor(r) == "PASS" else ""
            print(f"  {r['name']:<22}  Ret/Sharpe/MDD/PF = {flags}  ({wins}/4){mark}")

    # Per-strategy PnL breakdown
    print("\n" + "=" * 100)
    print("  PER-STRATEGY PnL BREAKDOWN")
    print("=" * 100)
    for r in results:
        if r["strat_stats"]:
            print(f"\n  {r['name']}:")
            for sn, st in sorted(r["strat_stats"].items(),
                                 key=lambda x: x[1]["pnl"], reverse=True):
                if st["trades"] > 0:
                    print(f"    {sn:<25} trades={st['trades']:>3d} "
                          f"WR={st['win_rate']:>5.1f}% PnL={st['pnl']:+,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
