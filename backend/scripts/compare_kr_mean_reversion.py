"""KR mean-reversion strategy re-examination (2y).

Existing rsi_divergence is already live (+2.8pp per #170 sweep).
This tests the two other mean-reversion strategies currently
yaml-disabled: bollinger_squeeze and bnf_deviation. Both produced
0 trades in compare_kr_strategy_density.py because they're disabled
at the strategy-block level (enabled:false), not just market-disabled.

Hot-enables them in the registry (gap_and_go pattern) + injects profile
weights. Tests baseline + each strategy alone + combined + param tuning.

On post-#183 live baseline (min_conf=0.20, stale=14, trailing 10/4, rsi_div on).
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
    # (name,                       hot_enable: set,             params: dict)
    ("V0_live",                    set(),                       {}),
    ("V1_bb_squeeze",              {"bollinger_squeeze"},       {}),
    ("V2_bnf_deviation",           {"bnf_deviation"},           {}),
    ("V3_both",                    {"bollinger_squeeze", "bnf_deviation"}, {}),
    # bnf param tuning
    ("V4_bnf_loose",               {"bnf_deviation"},
        {"bnf_deviation": {"buy_deviation": -3.0, "sell_deviation": 2.0}}),
    ("V5_bnf_tight",               {"bnf_deviation"},
        {"bnf_deviation": {"buy_deviation": -7.0, "sell_deviation": 4.0}}),
    # bb_squeeze param tuning
    ("V6_bb_short_period",         {"bollinger_squeeze"},
        {"bollinger_squeeze": {"bb_period": 10, "keltner_period": 10}}),
    ("V7_bb_wider_std",            {"bollinger_squeeze"},
        {"bollinger_squeeze": {"bb_std": 2.5}}),
]


def _kr_cfg(extra: set[str]) -> dict:
    loader = StrategyConfigLoader()
    yaml_disabled = set(loader.get_market_disabled_strategies("KR"))
    disabled = sorted(yaml_disabled - extra)
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


async def run(name: str, extra: set[str], params_overrides: dict) -> dict:
    cfg = PipelineConfig(**_kr_cfg(extra))
    eng = FullPipelineBacktest(cfg)

    if extra:
        from strategies.bollinger_squeeze import BollingerSqueezeStrategy
        from strategies.bnf_deviation import BNFDeviationStrategy

        loader = eng._registry._config_loader
        strategy_map = {
            "bollinger_squeeze": BollingerSqueezeStrategy,
            "bnf_deviation": BNFDeviationStrategy,
        }
        for sname in extra:
            cls = strategy_map.get(sname)
            if cls is None:
                continue
            yaml_params = loader.get_strategy_params(sname) or {}
            yaml_params.update(params_overrides.get(sname, {}))
            eng._registry._strategies[sname] = cls(params=yaml_params)
            # Inject profile weights so combiner doesn't suppress
            for state in ("strong_uptrend", "uptrend", "sideways"):
                profile = loader._config["profiles"].get(state, {})
                profile[sname] = 0.10
                loader._config["profiles"][state] = profile

    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics

    snaps = eng._daily_snapshots
    avg_deployed = 0.0
    if snaps:
        avg_deployed = sum(
            (s.equity - s.cash) / s.equity for s in snaps if s.equity > 0
        ) / len(snaps)

    extra_stats = {s: res.strategy_stats.get(s, {}) for s in extra}
    return dict(
        name=name,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_deployed_pct=round(avg_deployed * 100, 1),
        extra_stats=extra_stats,
        elapsed=round(el, 1),
    )


async def main():
    print("=" * 120)
    print("  KR Mean-Reversion re-examination (2y) — bollinger_squeeze + bnf_deviation")
    print("=" * 120)
    results = []
    for name, extras, params in VARIANTS:
        print(f"\n▶ {name}  hot={sorted(extras)}  override={params}")
        r = await run(name, extras, params)
        results.append(r)
        # Format extras stats
        ext_str = " ".join(
            f"{s}={st.get('trades', 0)}t/₩{st.get('pnl', 0):+,.0f}"
            for s, st in r['extra_stats'].items()
        )
        print(
            f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  "
            f"MDD={r['mdd']:6.1f}%  PF={r['pf']:.2f}  Trd={r['trades']:>4}  "
            f"Dep={r['avg_deployed_pct']:5.1f}%  {ext_str}  ({r['elapsed']:.0f}s)"
        )

    print("\n" + "=" * 120)
    print("  SUMMARY vs V0_live")
    print("=" * 120)
    if len(results) > 1:
        base = results[0]
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
