"""One-shot per-strategy backtest report (KR + US, 2y, current config).

Mirrors live config:
  KR: dual_momentum + supertrend (trend_following disabled)
  US: supertrend + trend_following (dual_momentum disabled)
"""

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
    "donchian_breakout","macd_histogram","rsi_divergence","bollinger_squeeze",
    "volume_profile","regime_switch","sector_rotation","cis_momentum",
    "larry_williams","bnf_deviation","volume_surge","cross_sectional_momentum",
    "pead_drift","quality_factor",
}


def kr_cfg():
    keep = {"dual_momentum", "supertrend"}
    return dict(
        market="KR", initial_equity=100_000_000,
        default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
        max_positions=18, max_position_pct=0.20,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.08, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.3,
        disabled_strategies=list(ALL_STRATEGIES | {"trend_following"} - keep),
    )


def us_cfg():
    keep = {"supertrend", "trend_following"}
    return dict(
        market="US", initial_equity=100_000,
        default_stop_loss_pct=0.08, default_take_profit_pct=0.20,
        max_positions=20, max_position_pct=0.10,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.05, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.2,
        disabled_strategies=list(ALL_STRATEGIES | {"dual_momentum"} - keep),
    )


async def run(name, kw):
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    return name, res, el


async def main():
    out = []
    for name, kw in [("KR", kr_cfg()), ("US", us_cfg())]:
        print(f"▶ {name} (2y)…")
        out.append(await run(name, kw))

    print("\n" + "=" * 80)
    print("  AGGREGATE (2y, current live config)")
    print("=" * 80)
    print(f"{'Mkt':<3} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trades':>7} {'WR%':>5} {'Alpha':>7}")
    print("-" * 60)
    for name, res, el in out:
        m = res.metrics
        print(f"{name:<3} {m.total_return_pct:+7.2f} {m.sharpe_ratio:+7.2f} "
              f"{m.max_drawdown_pct:7.1f} {m.profit_factor:5.2f} "
              f"{m.total_trades:7d} {m.win_rate*100:5.0f} {m.alpha:+7.1f}")

    print("\n" + "=" * 80)
    print("  PER-STRATEGY CONTRIBUTION")
    print("=" * 80)
    for name, res, _ in out:
        currency = "₩" if name == "KR" else "$"
        print(f"\n  {name}:")
        ps = res.strategy_stats
        for sname in sorted(ps.keys()):
            stats = ps[sname]
            wr = stats.get("win_rate", 0)
            wr_str = f"{wr*100:>4.0f}%" if isinstance(wr, (int, float)) else "  -- "
            print(f"    {sname:<24} trades={stats['trades']:>4} "
                  f"WR={wr_str}  PnL={currency}{stats['pnl']:>+12,.0f}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
