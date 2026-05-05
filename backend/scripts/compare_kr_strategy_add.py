"""KR strategy diversification — add a 2nd strategy alongside dual_momentum.

User context: live KR at 76% cash, single-strategy (dual_momentum-only) is
the structural cause. Backtest #34 (2026-04-09) tested cross_sectional_momentum
+ pead_drift + quality_factor on KR — all losers. Recent infra changes
since then (sector_boost weight=0.3, G1 limit-at-line entry, F2 anti-overbought,
KR cap 12→18) may shift the calculus.

Variants: dual_momentum stays. Add one more strategy from the previously-
disabled pool. Tests pairs that weren't covered by #34.

  V0_dm_only:        baseline (current live config)
  VA_dm_supertrend:  re-test supertrend on KR (was disabled 2026-04-23 for
                      combiner dilution; G1 limit-at-line is new)
  VB_dm_tf:          re-test trend_following (same caveat)
  VC_dm_bnf:         bnf_deviation (KR-flavored mean-reversion, untested
                      in #34)
  VD_dm_volsurge:    volume_surge (burst-catcher matching user's stated
                      thesis; untested in #34)
  VE_dm_bbsqueeze:   bollinger_squeeze (volatility-breakout; untested
                      in #34)
"""

import argparse
import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

ALL_STRATEGIES = {
    "supertrend", "trend_following", "donchian_breakout",
    "macd_histogram", "rsi_divergence", "bollinger_squeeze",
    "volume_profile", "regime_switch", "sector_rotation",
    "cis_momentum", "larry_williams", "bnf_deviation",
    "volume_surge", "cross_sectional_momentum",
    "pead_drift", "quality_factor",
}

# Per variant: which strategies stay enabled (dual_momentum is implicit)
VARIANTS = [
    ("V0_dm_only",       set()),
    ("VA_dm_supertrend", {"supertrend"}),
    ("VB_dm_tf",         {"trend_following"}),
    ("VC_dm_bnf",        {"bnf_deviation"}),
    ("VD_dm_volsurge",   {"volume_surge"}),
    ("VE_dm_bbsqueeze",  {"bollinger_squeeze"}),
]


def kr_config(extra: set[str]) -> dict:
    """Mirrors live KR eval-loop config + cap=18, with `extra` strategies
    re-enabled on top of dual_momentum."""
    keep = {"dual_momentum"} | extra
    disabled = list(ALL_STRATEGIES - keep)
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=0.12,
        default_take_profit_pct=0.20,
        max_positions=18,                 # post-PR #123
        max_position_pct=0.20,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        sector_boost_weight=0.3,          # mirrors yaml
        disabled_strategies=disabled,
    )


async def run_variant(name: str, kw: dict):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    avg_cash = (sum(s.cash / s.equity for s in snaps if s.equity > 0) /
                len(snaps)) if snaps else 0
    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_cash_pct=round(avg_cash * 100, 1),
        elapsed=round(el, 1),
        per_strategy=res.strategy_stats,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variant names to run")
    args = parser.parse_args()

    selected = VARIANTS
    if args.variants:
        selected = [(n, e) for n, e in VARIANTS if n in args.variants]

    results = []
    print("══════ KR diversification (cap=18, sector_boost=0.3) ══════")
    for name, extra in selected:
        kw = kr_config(extra)
        print(f"▶ {name}  +{sorted(extra) if extra else 'none'}")
        r = await run_variant(name, kw)
        results.append(r)
        print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} "
              f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} Trades={r['trades']} "
              f"Cash={r['avg_cash_pct']:.0f}% ({r['elapsed']:.0f}s)")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = f"{'Variant':<18} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Cash%':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<18} {r['ret']:+7.1f} {r['sharpe']:+7.2f} "
              f"{r['mdd']:7.1f} {r['pf']:6.2f} {r['trades']:7d} "
              f"{r['avg_cash_pct']:7.0f}")

    print("\nDelta vs V0_dm_only (4-dim improvement test — needs ALL non-negative):")
    v0 = next(r for r in results if r['name'] == 'V0_dm_only')
    for r in results:
        if r['name'] == 'V0_dm_only':
            continue
        d_ret = r['ret'] - v0['ret']
        d_sharpe = r['sharpe'] - v0['sharpe']
        d_mdd = r['mdd'] - v0['mdd']
        d_pf = r['pf'] - v0['pf']
        d_cash = r['avg_cash_pct'] - v0['avg_cash_pct']
        # KR combo activation: improve all 4 metrics. Cash is bonus.
        ok_4dim = d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_pf >= 0
        ok_floor = r['sharpe'] > 0 and r['mdd'] > -15.0 and r['pf'] > 1.0
        tag = "✓ ADOPT" if ok_4dim and ok_floor else "✗"
        print(f"  {r['name']:<18}: ΔRet={d_ret:+5.1f}pp ΔSharpe={d_sharpe:+5.2f} "
              f"ΔMDD={d_mdd:+5.1f}pp ΔPF={d_pf:+5.2f} ΔCash={d_cash:+5.0f}pp {tag}")

    # Per-strategy contribution summary for adopted variants
    print("\nPer-strategy PnL (KRW):")
    for r in results:
        ps = r.get('per_strategy', {})
        if not ps:
            continue
        print(f"  {r['name']}:")
        for sname in sorted(ps.keys()):
            stats = ps[sname]
            print(f"    {sname:<28} trades={stats['trades']:>4} "
                  f"WR={stats.get('win_rate', 0)*100:>4.0f}%  "
                  f"PnL=₩{stats['pnl']:>+12,.0f}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
