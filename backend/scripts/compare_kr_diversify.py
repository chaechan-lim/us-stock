"""KR strategy diversification + max_positions sweep (2y).

Live 5-28: KR cash 69% after #168 + #169. Funnel: sell_cooldown 71%
+ same_signal_24h 28% = 99% of rejections. cooldown reduction backtest
hurts (V3 -3pp Ret); the residual cash drag is signal-diversity: KR
only runs dual_momentum + supertrend right now. Same names get dual
signals → dedupe → fewer fills.

This sweep tries:
  - Raising max_positions 18 → 25 (B option from operator chat)
  - Adding each idle-but-defined strategy back into the KR mix
  - Combinations of additions

Each variant runs on the current live baseline (post-#168 vol scaling,
post-#170 stale_pnl -0.10) so the comparison reflects actual deploy.
Variants are exercised by writing a temp strategies.yaml that
selectively flips `enabled: true` for candidate strategies and pruning
`markets.US.disabled_strategies` so US is unaffected (we set the
candidates in US.disabled_strategies for the duration of the run).
"""

import asyncio
import copy
import functools
import logging
import os
import sys
import tempfile
import time

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in (
    "yfinance", "peewee", "urllib3", "httpx", "scanner",
    "data", "backtest", "strategies", "engine",
):
    logging.getLogger(n).setLevel(logging.WARNING)

import yaml  # noqa: E402

from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


REAL_YAML = os.path.join(os.path.dirname(__file__), "..", "..", "config", "strategies.yaml")


VARIANTS = [
    # (name,                     max_positions, extra_enabled_for_kr)
    ("V0_baseline",              18, []),
    ("V1_max25",                 25, []),
    ("V2_+donchian",             18, ["donchian_breakout"]),
    ("V3_+bbs",                  18, ["bollinger_squeeze"]),
    ("V4_+csm",                  18, ["cross_sectional_momentum"]),
    ("V5_+vol_surge",            18, ["volume_surge"]),
    ("V6_+macd",                 18, ["macd_histogram"]),
    ("V7_+rsi_div",              18, ["rsi_divergence"]),
    ("V8_max25_+best",           25, []),  # filled in main after V2-V7 ranked
]


def _make_temp_yaml(extra_enabled: list[str]) -> str:
    """Write a temp strategies.yaml that flips selected strategies to
    enabled:true, and adds the same names to markets.US.disabled_
    strategies so the US backtest doesn't see them (we're testing
    KR-only diversification)."""
    with open(REAL_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data = copy.deepcopy(data)
    strategies = data.get("strategies", {})
    for name in extra_enabled:
        if name not in strategies:
            raise SystemExit(f"strategy '{name}' not in yaml/strategies")
        strategies[name]["enabled"] = True
    # Keep US safe: add candidates to its disabled list.
    us_dis = list(data.get("markets", {}).get("US", {}).get("disabled_strategies", []))
    for name in extra_enabled:
        if name not in us_dis:
            us_dis.append(name)
    data["markets"]["US"]["disabled_strategies"] = us_dis
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="w", encoding="utf-8",
    )
    yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    return tmp.name


def _kr_cfg(max_pos: int, yaml_path: str | None) -> dict:
    loader = StrategyConfigLoader(yaml_path) if yaml_path else StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk_cfg = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol_cfg = risk_cfg.get("volatility_scaling") or {}
    return dict(
        market="KR",
        initial_equity=100_000_000,
        default_stop_loss_pct=float(risk_cfg.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk_cfg.get("default_take_profit_pct", 0.20)),
        max_positions=max_pos,
        max_position_pct=float(risk_cfg.get("max_position_pct", 0.20)),
        min_position_pct=float(risk_cfg.get("min_position_pct", 0.05)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 1)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08,
        volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.50)),
        enforce_min_position_pct_floor=True,
        enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol_cfg.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol_cfg.get("min_scale", 0.5)),
        vol_scale_max=float(vol_cfg.get("max_scale", 1.5)),
        strategy_config_path=yaml_path,
    )


async def run(name: str, max_pos: int, extra: list[str]) -> dict:
    yaml_path = _make_temp_yaml(extra) if extra else None
    try:
        cfg = PipelineConfig(**_kr_cfg(max_pos, yaml_path))
        eng = FullPipelineBacktest(cfg)
        t0 = time.time()
        res = await eng.run(period="2y")
        el = time.time() - t0
        m = res.metrics
        avg_alloc = 0.0
        if res.trades:
            sizes = [
                t.entry_price * t.quantity for t in res.trades if t.quantity > 0
            ]
            if sizes:
                avg_alloc = sum(sizes) / len(sizes)
        return dict(
            name=name, max_pos=max_pos, extra=extra,
            ret=round(m.total_return_pct, 2),
            sharpe=round(m.sharpe_ratio, 2),
            mdd=round(m.max_drawdown_pct, 2),
            pf=round(m.profit_factor, 2),
            trades=m.total_trades,
            avg_alloc=round(avg_alloc, 0),
            elapsed=round(el, 1),
        )
    finally:
        if yaml_path:
            try: os.unlink(yaml_path)
            except OSError: pass


async def main():
    print("=" * 110)
    print("  KR diversification sweep (2y) — max_positions + strategy addition")
    print("=" * 110)
    results = []
    for name, mp, ex in VARIANTS:
        if name == "V8_max25_+best":
            # Pick the best ΔRet vs V0 from V2..V7
            base_ret = results[0]["ret"]
            ranked = sorted(
                [r for r in results if r["name"].startswith(("V2","V3","V4","V5","V6","V7"))],
                key=lambda r: r["ret"] - base_ret, reverse=True,
            )
            if ranked:
                ex = ranked[0]["extra"]
                name = f"V8_max25_{'+'.join(ex)}"
            else:
                continue
        print(f"\n▶ {name}  max_pos={mp}  extra={ex}")
        r = await run(name, mp, ex)
        results.append(r)
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  "
            f"Trades={r['trades']:>4}  AvgAlloc=₩{r['avg_alloc']:,.0f}  "
            f"({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = (
        f"{'Variant':<26} {'maxP':>5} "
        f"{'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trd':>4} "
        f"{'AvgAlloc':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<26} {r['max_pos']:>5} "
            f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:>7.1f} {r['pf']:>5.2f} "
            f"{r['trades']:>4} ₩{r['avg_alloc']:>12,.0f}"
        )

    if len(results) > 1:
        base = results[0]
        print("\n  vs V0_baseline:")
        for r in results[1:]:
            dret = r["ret"] - base["ret"]
            dshp = r["sharpe"] - base["sharpe"]
            dmdd = r["mdd"] - base["mdd"]
            dpf = r["pf"] - base["pf"]
            improves = sum([dret > -1.0, dshp > -0.10, dmdd > -3.0, dpf > -0.10])
            tag = "✓" if improves == 4 else "△" if improves >= 3 else "✗"
            print(
                f"    {r['name']:<26} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
                f"ΔMDD={dmdd:+5.1f}  ΔPF={dpf:+5.2f}  {tag}"
            )


if __name__ == "__main__":
    asyncio.run(main())
