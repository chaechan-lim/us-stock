"""KR universe expansion backtest (2y).

Live KR has 41 active watchlist symbols vs 79 in the seed universe
(_KR_UNIVERSE in scanner/kr_screener.py). Indicator screener filters
38 names with min_grade=B; this script tests whether forcing the wider
79-name universe into the backtest changes the picture.

V0_live41:  current backtest default (DEFAULT_KR_UNIVERSE, ~41 yf-symbols)
V1_wide79:  all 79 _KR_UNIVERSE entries converted to yfinance format
V2_wide_no_grade_filter: V1 with min_grade='C' (less strict screener)

If V1 improves on any of {Ret, Sharpe, MDD, PF} without degrading the
others, universe expansion is a free deploy lever (yaml-config or
watchlist seed change). If it regresses, the screener filter is doing
its job — the 38 excluded names are noise.
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


def _full_kr_universe_yf() -> list[str]:
    """Convert _KR_UNIVERSE (KRX+name tuples) to yfinance suffix format."""
    from scanner.kr_screener import _KR_UNIVERSE
    out = []
    for sym, exch, _name in _KR_UNIVERSE:
        suffix = ".KS" if exch == "KRX" else ".KQ"
        out.append(f"{sym}{suffix}")
    return out


def _kr_cfg(universe: list[str], min_grade: str = "B") -> dict:
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
        min_screen_grade=min_grade,
    )


async def run(name: str, universe: list[str], min_grade: str) -> dict:
    cfg = PipelineConfig(**_kr_cfg(universe, min_grade))
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
        name=name, n_symbols=len(universe), min_grade=min_grade,
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
    full = _full_kr_universe_yf()
    print("=" * 115)
    print(f"  KR universe expansion (2y) — DEFAULT={len(DEFAULT_KR_UNIVERSE)} vs FULL={len(full)}")
    print("=" * 115)

    variants = [
        ("V0_default41",      list(DEFAULT_KR_UNIVERSE), "B"),
        ("V1_full79",         full,                     "B"),
        ("V2_full79_gradeC",  full,                     "C"),
    ]
    results = []
    for name, uni, grade in variants:
        print(f"\n▶ {name}  n={len(uni)}  min_grade={grade}")
        r = await run(name, uni, grade)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  Pos={r['avg_npos']:4.1f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 115)
    print("  SUMMARY")
    print("=" * 115)
    hdr = (
        f"{'Variant':<22} {'N':>4} {'Grade':>6} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} "
        f"{'PF':>5} {'Trd':>4} {'Dep%':>6} {'Pos':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<22} {r['n_symbols']:>4} {r['min_grade']:>6} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} "
            f"{r['pf']:>5.2f} {r['trades']:>4} {r['avg_deployed_pct']:>5.1f}% "
            f"{r['avg_npos']:>5.1f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_default41 (current backtest baseline):")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  ΔPos={r['avg_npos']-base['avg_npos']:+4.1f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
