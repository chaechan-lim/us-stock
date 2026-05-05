"""Daily buy limit + confidence escalation variants.

Live observation 2026-05-04 US session: with limit=5 + escalation
0.65/0.75/0.90, only 3-5 fresh BUYs fired before the conf bar shut out
0.64-confidence signals (NVDA-tier 0.81 OK, but 0.64 candidates piled up
unfilled). User wants the bar relaxed; this script tests whether the
relaxation breaks performance in backtest.

Variants:
  V0_current: limit=5, esc 0.65/0.75/0.90 (matches live evaluation_loop)
  V1_10_relaxed: limit=10, esc 0.50/0.60/0.90
  V2_disabled: limit=0 (no cap)
  V3_15_relaxed: limit=15, esc 0.50/0.60/0.90

Reports Ret/Sharpe/MDD/PF/Trades/Cash% per variant per market.
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
for n in ("yfinance","peewee","urllib3","httpx","scanner","data","backtest","strategies","engine"):
    logging.getLogger(n).setLevel(logging.WARNING)

from backtest import full_pipeline as fp
from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader

_ORIG_CHECK = fp.FullPipelineBacktest._check_daily_buy_allowed


def patch_thresholds(low_threshold: float, high_threshold: float, override: float = 0.90):
    """Replace the hardcoded 0.65/0.75/0.90 with given values."""
    def _check(self, confidence: float) -> bool:
        cfg = self._config
        limit = cfg.daily_buy_limit
        if limit <= 0:
            return True
        if self._daily_buy_count >= limit:
            if cfg.enable_confidence_escalation and confidence >= override:
                return True
            return False
        if not cfg.enable_confidence_escalation:
            return True
        usage_ratio = self._daily_buy_count / limit
        if usage_ratio >= 0.8:
            return confidence >= high_threshold
        elif usage_ratio >= 0.6:
            return confidence >= low_threshold
        return True
    fp.FullPipelineBacktest._check_daily_buy_allowed = _check


def restore_check():
    fp.FullPipelineBacktest._check_daily_buy_allowed = _ORIG_CHECK


def _market_config(market: str, disabled: list[str], daily_buy_limit: int,
                   enable_escalation: bool) -> dict:
    if market == "KR":
        kw = dict(
            market="KR", initial_equity=100_000_000,
            default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
            max_positions=18, max_position_pct=0.20,
            sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
            slippage_pct=0.08, volume_adjusted_slippage=True,
            min_confidence=0.30, disabled_strategies=disabled,
        )
    else:
        kw = dict(
            market="US", initial_equity=100_000,
            default_stop_loss_pct=0.08, default_take_profit_pct=0.20,
            max_positions=20, max_position_pct=0.10,
            sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
            slippage_pct=0.05, volume_adjusted_slippage=True,
            min_confidence=0.30, disabled_strategies=disabled,
        )
    kw["daily_buy_limit"] = daily_buy_limit
    kw["enable_confidence_escalation"] = enable_escalation
    return kw


VARIANTS = [
    # (name, daily_buy_limit, escalation_enabled, low_threshold, high_threshold)
    ("V0_current",      5,  True,  0.65, 0.75),
    ("V1_10_relaxed",   10, True,  0.50, 0.60),
    ("V2_disabled",     0,  False, 0.0,  0.0),
    ("V3_15_relaxed",   15, True,  0.50, 0.60),
]


async def run_variant(name: str, market: str, kw: dict,
                       low: float, high: float):
    patch_thresholds(low, high)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    if snaps:
        cash_pcts = [s.cash / s.equity for s in snaps if s.equity > 0]
        avg_cash = sum(cash_pcts) / len(cash_pcts) if cash_pcts else 0.0
    else:
        avg_cash = 0.0
    return dict(
        name=name, market=market,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        avg_cash_pct=round(avg_cash * 100, 1),
        elapsed=round(el, 1),
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["KR", "US"])
    args = parser.parse_args()

    loader = StrategyConfigLoader()
    results = []

    for market in args.markets:
        disabled = loader.get_market_disabled_strategies(market)
        print(f"\n══════ {market} ══════")
        for (name, limit, esc, low, high) in VARIANTS:
            kw = _market_config(market, disabled, limit, esc)
            print(f"▶ {name}  limit={limit} esc={esc} bars={low}/{high}")
            r = await run_variant(name, market, kw, low, high)
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} "
                  f"MDD={r['mdd']:.1f}% PF={r['pf']:.2f} "
                  f"Trades={r['trades']} Cash={r['avg_cash_pct']:.0f}% "
                  f"({r['elapsed']:.0f}s)")

    restore_check()

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    hdr = f"{'Variant':<16} {'Mkt':<3} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Cash%':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<16} {r['market']:<3} "
              f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:7.1f} "
              f"{r['pf']:6.2f} {r['trades']:7d} {r['avg_cash_pct']:7.0f}")

    print("\nDelta vs V0 baseline (per market):")
    for market in args.markets:
        v0 = next(r for r in results if r['market']==market and r['name']=='V0_current')
        for name in ("V1_10_relaxed", "V2_disabled", "V3_15_relaxed"):
            v = next(r for r in results if r['market']==market and r['name']==name)
            d_ret = v['ret'] - v0['ret']
            d_sharpe = v['sharpe'] - v0['sharpe']
            d_mdd = v['mdd'] - v0['mdd']
            d_cash = v['avg_cash_pct'] - v0['avg_cash_pct']
            ok = (d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_cash <= 0)
            tag = "✓" if ok else "✗"
            print(f"  {market} {name:<16}: ΔRet={d_ret:+5.1f}pp ΔSharpe={d_sharpe:+5.2f} "
                  f"ΔMDD={d_mdd:+5.1f}pp ΔCash={d_cash:+5.0f}pp {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
