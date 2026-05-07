"""Trailing-stop activation_pct sweep — does triggering the trail later
let winners run further?

Live obs 2026-05-06: 5 KR `dual_momentum:trailing_stop` exits in one day,
each at +0.3% to +11%. supertrend day-1 rounds (5/6→5/7) showed 247540
+7.8% trail-out and 003670 partial profit-taking with similar early trail
behavior. Hypothesis: 4% activation triggers too early on noisy intraday
moves, capping winners that would have gone +15-25% if given more rope.

Variants (KR + US 2y, current live config — cap=18 KR / 20 US,
sector_boost, supertrend re-enabled, P1+P2 sizing):
  V0_baseline: activation=0.04 trail=0.03 (live default)
  V1_6_3:      activation=0.06 trail=0.03
  V2_8_3:      activation=0.08 trail=0.03
  V3_8_5:     activation=0.08 trail=0.05  (let winners breathe more)
  V4_10_5:    activation=0.10 trail=0.05
"""

import argparse, asyncio, functools, logging, sys, time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies","engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

import engine.risk_manager as rm
from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

ALL_DISABLED_KR = [
    "trend_following","donchian_breakout","macd_histogram","rsi_divergence",
    "bollinger_squeeze","volume_profile","regime_switch","sector_rotation",
    "cis_momentum","larry_williams","bnf_deviation","volume_surge",
    "cross_sectional_momentum","pead_drift","quality_factor",
]
ALL_DISABLED_US = [
    "dual_momentum","donchian_breakout","macd_histogram","rsi_divergence",
    "bollinger_squeeze","volume_profile","regime_switch","sector_rotation",
    "cis_momentum","larry_williams","bnf_deviation","volume_surge",
    "cross_sectional_momentum","pead_drift","quality_factor",
]
US_VA_OVERRIDE = {
    "strong_uptrend": 0.100, "uptrend": 0.085, "sideways": 0.065,
    "weak_downtrend": 0.040, "downtrend": 0.020,
}

VARIANTS = [
    ("V0_baseline", 0.04, 0.03),
    ("V1_6_3",      0.06, 0.03),
    ("V2_8_3",      0.08, 0.03),
    ("V3_8_5",      0.08, 0.05),
    ("V4_10_5",     0.10, 0.05),
]

def kr_cfg(activation, trail):
    return dict(
        market="KR", initial_equity=100_000_000,
        default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
        max_positions=18, max_position_pct=0.20, min_position_pct=0.04,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.08, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.3,
        disabled_strategies=ALL_DISABLED_KR,
        trailing_activation_pct=activation, trailing_trail_pct=trail,
    )

def us_cfg(activation, trail):
    return dict(
        market="US", initial_equity=100_000,
        default_stop_loss_pct=0.08, default_take_profit_pct=0.20,
        max_positions=20, max_position_pct=0.10, min_position_pct=0.05,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.05, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.2,
        disabled_strategies=ALL_DISABLED_US,
        trailing_activation_pct=activation, trailing_trail_pct=trail,
    )


async def run_variant(name, market, activation, trail):
    if market == "US":
        rm.REGIME_POSITION_PCT = dict(US_VA_OVERRIDE)
    else:
        # restore default for KR
        rm.REGIME_POSITION_PCT = {
            "strong_uptrend": 0.08, "uptrend": 0.07, "sideways": 0.06,
            "weak_downtrend": 0.04, "downtrend": 0.03,
        }
    kw = kr_cfg(activation, trail) if market == "KR" else us_cfg(activation, trail)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    avg_cash = (sum(s.cash/s.equity for s in snaps if s.equity > 0) / len(snaps)) if snaps else 0
    return dict(
        name=name, market=market,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        wr=round(m.win_rate*100, 0),
        cash=round(avg_cash*100, 1),
        elapsed=round(el, 1),
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["KR","US"])
    args = parser.parse_args()

    results = []
    for market in args.markets:
        print(f"\n══ {market} ══")
        for (name, activation, trail) in VARIANTS:
            print(f"▶ {name}  act={activation:.2f} trail={trail:.2f}")
            r = await run_variant(name, market, activation, trail)
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} MDD={r['mdd']:.1f}% "
                  f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% Cash={r['cash']:.0f}% ({r['elapsed']:.0f}s)")

    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)
    print(f"{'Variant':<13} {'Mkt':<3} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trades':>7} {'WR%':>4} {'Cash%':>6}")
    print("-"*72)
    for r in results:
        print(f"{r['name']:<13} {r['market']:<3} {r['ret']:+7.1f} {r['sharpe']:+7.2f} "
              f"{r['mdd']:7.1f} {r['pf']:5.2f} {r['trades']:7d} {r['wr']:4.0f} {r['cash']:6.0f}")
    print("\nDelta vs V0 baseline:")
    for market in args.markets:
        v0 = next(r for r in results if r['market']==market and r['name']=='V0_baseline')
        for name in ("V1_6_3","V2_8_3","V3_8_5","V4_10_5"):
            v = next(r for r in results if r['market']==market and r['name']==name)
            d = lambda k: v[k]-v0[k]
            ok = d('ret')>=0 and d('sharpe')>=-0.05 and d('mdd')>=-2.0 and d('pf')>=0
            tag = "✓" if ok else "✗"
            print(f"  {market} {name:<13}: ΔRet={d('ret'):+5.1f}pp ΔSharpe={d('sharpe'):+5.2f} "
                  f"ΔMDD={d('mdd'):+5.1f}pp ΔPF={d('pf'):+5.2f} {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
