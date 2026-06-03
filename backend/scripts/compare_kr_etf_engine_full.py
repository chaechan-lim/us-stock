"""KR ETF Engine backtest on bnf-w-0.20 baseline.

Tests adding the KR ETF engine (regime-based KODEX/KOSDAQ leveraged
pairs + sector ETFs) on top of the current KR strategy combo. Shares
cash pool with individual stocks via `enable_etf_engine=True` in
PipelineConfig.

Setup:
  1. Read config/kr_etf_universe.yaml
  2. Rewrite 6-digit codes → 6-digit.KS / .KQ for yfinance
  3. Pass temp yaml to PipelineConfig.etf_universe_config_path
  4. Run on KR baseline + variants of etf_max_portfolio_pct
"""

import asyncio, functools, logging, sys, tempfile, time
from pathlib import Path
print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies","engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

import yaml
from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader


def _make_kr_etf_yfinance_yaml() -> str:
    """Read kr_etf_universe.yaml and rewrite codes to .KS / .KQ format.

    KOSPI200 base/bull/bear: KS. KOSDAQ150 base/bull/bear: KQ.
    Sector ETFs: KS.
    Safe haven (148070 KOSEF 국고채): KS.
    """
    src = Path("/home/chans/us-stock/config/kr_etf_universe.yaml")
    with src.open() as f:
        data = yaml.safe_load(f)

    def with_kr(code: str, exch: str = "KS") -> str:
        if not code or code.endswith((".KS", ".KQ")):
            return code
        return f"{code}.{exch}"

    # leveraged_pairs
    for key, pair in (data.get("leveraged_pairs") or {}).items():
        suffix = "KQ" if "KOSDAQ" in key else "KS"
        for k in ("bull", "bear", "base"):
            if pair.get(k):
                pair[k] = with_kr(pair[k], suffix)

    # sectors
    for _name, sect in (data.get("sectors") or {}).items():
        if sect.get("etf"):
            sect["etf"] = with_kr(sect["etf"], "KS")

    # safe_haven + volatility
    data["safe_haven"] = [with_kr(s, "KS") for s in data.get("safe_haven", [])]
    data["volatility"] = [with_kr(s, "KS") for s in data.get("volatility", [])]

    out = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="kr_etf_yf_",
    )
    yaml.safe_dump(data, out, allow_unicode=True)
    out.close()
    return out.name


KR_ETF_YAML = _make_kr_etf_yfinance_yaml()
print(f"Temp KR ETF universe (yfinance suffixes): {KR_ETF_YAML}")


VARIANTS = [
    # (name, etf_enable, etf_max_pct, etf_single_pct)
    ("V0_no_etf",            False, 0.00, 0.00),
    ("V1_etf_10_05",         True,  0.10, 0.05),
    ("V2_etf_20_10",         True,  0.20, 0.10),
    ("V3_etf_30_15",         True,  0.30, 0.15),
    ("V4_etf_40_20",         True,  0.40, 0.20),
    ("V5_etf_50_25",         True,  0.50, 0.25),
]


def _kr_cfg(etf_on: bool, etf_max_pct: float, etf_single_pct: float) -> dict:
    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies("KR")
    risk = loader.get_market_risk_config("KR")
    eval_cfg = loader.get_market_evaluation_loop_config("KR")
    vol = risk.get("volatility_scaling") or {}
    return dict(
        market="KR", initial_equity=100_000_000,
        default_stop_loss_pct=float(risk.get("default_stop_loss_pct", 0.12)),
        default_take_profit_pct=float(risk.get("default_take_profit_pct", 0.20)),
        max_positions=int(risk.get("max_positions", 18)),
        max_position_pct=float(risk.get("max_position_pct", 0.20)),
        min_position_pct=float(risk.get("min_position_pct", 0.05)),
        sell_cooldown_days=int(eval_cfg.get("sell_cooldown_days", 3)),
        whipsaw_max_losses=int(eval_cfg.get("whipsaw_max_losses", 2)),
        min_hold_days=int(eval_cfg.get("min_hold_days", 1)),
        slippage_pct=0.08, volume_adjusted_slippage=True,
        min_confidence=float(eval_cfg.get("min_confidence") or 0.30),
        stale_time_days=int(eval_cfg.get("stale_time_days", 0)),
        stale_time_pnl_threshold=float(eval_cfg.get("stale_time_pnl_threshold", 0.0)),
        sector_boost_weight=float(eval_cfg.get("sector_boost_weight", 0.3)),
        disabled_strategies=disabled,
        kelly_fraction=float(risk.get("kelly_fraction", 0.40)),
        enforce_min_position_pct_floor=True, enable_vol_scaling=True,
        vol_scale_target_risk_pct=float(vol.get("target_risk_pct", 0.04)),
        vol_scale_min=float(vol.get("min_scale", 0.5)),
        vol_scale_max=float(vol.get("max_scale", 1.5)),
        # ETF engine knobs
        enable_etf_engine=etf_on,
        etf_max_portfolio_pct=etf_max_pct,
        etf_max_single_pct=etf_single_pct,
        etf_universe_config_path=KR_ETF_YAML if etf_on else None,
    )


async def run(name, etf_on, max_pct, single_pct):
    cfg = PipelineConfig(**_kr_cfg(etf_on, max_pct, single_pct))
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    dep = sum((s.equity-s.cash)/s.equity for s in snaps if s.equity>0)/len(snaps) if snaps else 0
    # ETF stats
    etf_stats = {
        k: v for k, v in res.strategy_stats.items()
        if "etf" in k.lower()
    }
    return dict(name=name, ret=round(m.total_return_pct,2), sharpe=round(m.sharpe_ratio,2),
                mdd=round(m.max_drawdown_pct,2), pf=round(m.profit_factor,2),
                trades=m.total_trades, dep=round(dep*100,1),
                etf_stats=etf_stats, elapsed=round(el,1))


async def main():
    print("=" * 120); print("  KR ETF Engine sweep on bnf-w-0.20 baseline"); print("=" * 120)
    results = []
    for name, etf_on, mp, sp in VARIANTS:
        print(f"\n▶ {name}  etf={etf_on}  max_pct={mp} single={sp}")
        r = await run(name, etf_on, mp, sp); results.append(r)
        etf_summary = ""
        if r['etf_stats']:
            for k, st in r['etf_stats'].items():
                etf_summary += f" {k}:{st.get('trades',0)}t/₩{st.get('pnl',0):+,.0f}"
        print(f"  Ret={r['ret']:+6.1f}%  Sharpe={r['sharpe']:+5.2f}  MDD={r['mdd']:6.1f}%  "
              f"PF={r['pf']:.2f}  Trd={r['trades']:>4}  Dep={r['dep']:5.1f}%"
              f"{etf_summary}  ({r['elapsed']:.0f}s)")
    base = results[0]
    print(f"\n  vs V0_no_etf:")
    for r in results[1:]:
        dret = r["ret"]-base["ret"]; dshp = r["sharpe"]-base["sharpe"]
        dmdd = r["mdd"]-base["mdd"]; dpf = r["pf"]-base["pf"]
        ddep = r["dep"]-base["dep"]
        imp = sum([dret>-1.0, dshp>-0.10, dmdd>-3.0, dpf>-0.10])
        tag = "✓" if imp==4 else "△" if imp>=3 else "✗"
        print(f"    {r['name']:<22} ΔRet={dret:+5.1f}  ΔSharpe={dshp:+5.2f}  "
              f"ΔMDD={dmdd:+5.1f}  ΔDep={ddep:+5.1f}pp  ΔPF={dpf:+5.2f}  {tag}")


if __name__ == "__main__":
    asyncio.run(main())
