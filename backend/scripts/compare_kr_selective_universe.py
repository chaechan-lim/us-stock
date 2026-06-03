"""KR selective universe expansion (2y) — 10 top-signal missing names.

Yesterday's compare_kr_universe.py forced ALL 38 missing seed names
into the universe → -18pp Ret (alpha-destructive). This re-tests with
only the top 10 names that show strong multi-strategy BUY signals
today (live simulation).

Hypothesis: blanket expansion adds noise but cherry-picked high-signal
names may be additive. Backtest measures whether selective expansion
preserves or improves the baseline.

Top 10 candidates (from 2026-06-03 live simulation, 2+ BUY signals
with confidence >= 0.65 from any strategy):
  042660 한화오션, 000990 DB하이텍, 006260 LS, 079550 LIG넥스원,
  329180 현대로템, 018880 한온시스템, 131970 테스나, 010140 삼성중공업,
  012450 한화에어로스페이스, 009540 HD한국조선해양

Defense/shipbuilding/semicon-parts cluster — sectors not represented
in current 41-symbol watchlist.
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

from backtest.full_pipeline import (
    DEFAULT_KR_UNIVERSE, FullPipelineBacktest, PipelineConfig,
)
from strategies.config_loader import StrategyConfigLoader


# Top 10 high-signal additions (yfinance suffixes)
SELECTIVE_ADDITIONS = [
    "042660.KS",  # 한화오션
    "000990.KS",  # DB하이텍
    "006260.KS",  # LS
    "079550.KS",  # LIG넥스원
    "329180.KS",  # 현대로템
    "018880.KS",  # 한온시스템
    "131970.KQ",  # 테스나 KOSDAQ
    "010140.KS",  # 삼성중공업
    "012450.KS",  # 한화에어로스페이스
    "009540.KS",  # HD한국조선해양
]


def _kr_cfg(universe: list[str]) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    return dict(
        market="KR",
        universe=universe,
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 3)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        stale_time_days=int(eval_cfg.get("stale_time_days", 0)),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold", 0.0)),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
    )


async def run(name: str, universe: list[str]) -> dict:
    cfg = PipelineConfig(**_kr_cfg(universe))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    snaps = eng._daily_snapshots
    avg_deployed = 0.0
    avg_npos = 0.0
    if snaps:
        avg_deployed = sum(
            (s.equity - s.cash) / s.equity for s in snaps if s.equity > 0
        ) / len(snaps)
        avg_npos = sum(s.n_positions for s in snaps) / len(snaps)
    return dict(
        name=name, n_symbols=len(universe),
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        avg_npos=round(avg_npos, 1),
        elapsed=round(el, 1),
    )


async def main():
    default = list(DEFAULT_KR_UNIVERSE)
    expanded = default + [s for s in SELECTIVE_ADDITIONS if s not in default]

    print("=" * 115)
    print(f"  KR selective universe expansion — DEFAULT={len(default)} vs +TOP10={len(expanded)}")
    print("=" * 115)

    variants = [
        ("V0_default41",        default),
        ("V1_top10_added",      expanded),
    ]
    results = []
    for name, uni in variants:
        print(f"\n▶ {name}  n={len(uni)}")
        r = await run(name, uni)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  Pos={r['avg_npos']:4.1f}  "
            f"({r['elapsed']:.0f}s)"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_default41:")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(f"    {r['name']}  ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                  f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔPos={r['avg_npos']-base['avg_npos']:+4.1f}  {tag}")


if __name__ == "__main__":
    asyncio.run(main())
