"""KR symbol churn sweep (2y).

Live 5-29 (post-#171): same KOSPI blue chips churning 4 round-trips
in 14 days — 005930/005935/005380/006400/018260/403870 etc. Several
generate small wins (삼성SDS +80K) but the losers swamp them
(006400 −111K, 403870 −55K, 028260 −54K).

Hypotheses we test:
  H1: min_hold_days 1 → 3 stops short-term churn
  H2: sell_cooldown_days 1 → 3/5 prevents same-day re-entry
  H3: whipsaw_max_losses 2 → 1 cuts off symbols quicker
  H4: recovery_watch_days 7 → 3 frees up universe slots faster

Variants:
  V0_baseline    : current live config
  V1_hold_3      : min_hold_days 1 → 3
  V2_cd_3        : sell_cooldown_days 1 → 3
  V3_cd_5        : sell_cooldown_days 1 → 5
  V4_whips_1     : whipsaw_max_losses 2 → 1
  V5_combo_h3cd3 : V1 + V2
"""

import asyncio
import functools
import logging
import sys
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in (
    "yfinance", "peewee", "urllib3", "httpx", "scanner",
    "data", "backtest", "strategies", "engine",
):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


VARIANTS = [
    # (name,            min_hold, sell_cooldown, whipsaw_max)
    ("V0_baseline",         1,  1, 2),
    ("V1_hold_3",           3,  1, 2),
    ("V2_cd_3",             1,  3, 2),
    ("V3_cd_5",             1,  5, 2),
    ("V4_whips_1",          1,  1, 1),
    ("V5_combo_h3cd3",      3,  3, 2),
]


def _kr_cfg(hold: int, cd: int, whips: int) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=cd,
        whipsaw_max_losses=whips,
        min_hold_days=hold,
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.50)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )


async def run(name: str, hold: int, cd: int, whips: int) -> dict:
    cfg = PipelineConfig(**_kr_cfg(hold, cd, whips))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    # Symbol churn proxy: max trades-per-symbol and unique symbols traded
    from collections import Counter
    if res.trades:
        sym_count = Counter()
        for t in res.trades:
            sym_count[t.symbol] += 1
        max_churn = max(sym_count.values()) if sym_count else 0
        n_unique = len(sym_count)
        avg_per_sym = sum(sym_count.values()) / max(n_unique, 1)
    else:
        max_churn = n_unique = 0
        avg_per_sym = 0

    return dict(
        name=name, hold=hold, cd=cd, whips=whips,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        unique_syms=n_unique,
        avg_per_sym=round(avg_per_sym, 1),
        max_churn=max_churn,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 110)
    print("  KR churn sweep (2y) — does min_hold / sell_cooldown / whipsaw_max")
    print("  reduce same-symbol round-trips?")
    print("=" * 110)
    results = []
    for name, hold, cd, whips in VARIANTS:
        print(f"\n▶ {name}  hold={hold}d cooldown={cd}d whips={whips}")
        r = await run(name, hold, cd, whips)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  uniq={r['unique_syms']:>3}  "
            f"avg/sym={r['avg_per_sym']:.1f}  max_churn={r['max_churn']:>3}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<18} {'hold':>5} {'cd':>4} {'whp':>4} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} "
        f"{'Trd':>4} {'uniq':>5} {'avg/sym':>8} {'maxC':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<18} {r['hold']:>5} {r['cd']:>4} {r['whips']:>4} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} {r['unique_syms']:>5} {r['avg_per_sym']:>8.1f} {r['max_churn']:>5}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_baseline:")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            d_unique = r["unique_syms"] - base["unique_syms"]
            d_avg = r["avg_per_sym"] - base["avg_per_sym"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<18} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  "
                f"Δuniq={d_unique:+3}  Δavg/sym={d_avg:+4.1f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
