"""Position sizing variants — compare current behavior vs proposed fixes.

User reported 72% idle cash on live (₩13M cash / ₩18M equity). Diagnosed in
risk_manager.py: REGIME_POSITION_PCT hardcoded at 0.07 for uptrend, dwarfing
yaml `max_position_pct: 0.20`. Fixed-sizing fallback also lacks
`min_position_pct` floor. Result: 4-6% positions, never the intended 15-20%.

Variants:
  V0_baseline: current behavior (regime table dominant, no floor in fallback).
  VA_yaml_base: regime becomes a multiplier of yaml max_position_pct.
                base = max_position_pct × regime_mult (uptrend=0.85, ...).
  VB_floor:    keep regime table but enforce min_position_pct as a floor in
                the fixed-sizing fallback (the path most live trades hit).
  VC_combined: VA + VB + relaxed conf_mult (0.7 + 0.3·conf instead of
                0.4 + 0.6·conf) so low-conf signals still get meaningful size.

Reports per variant: Ret, Sharpe, MDD, PF, Trades, WR, Alpha, Avg Cash %
(daily mean of cash/equity over the backtest — the user's primary metric).
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

import engine.risk_manager as rm
from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig
from strategies.config_loader import StrategyConfigLoader

# Snapshot the original module values so each variant can restore.
_ORIG_REGIME_PCT = dict(rm.REGIME_POSITION_PCT)
_ORIG_KELLY_CALC = rm.RiskManager.calculate_kelly_position_size


# ── Variant patches ─────────────────────────────────────────────────────

def patch_baseline():
    """Restore everything — V0 sanity."""
    rm.REGIME_POSITION_PCT = dict(_ORIG_REGIME_PCT)
    rm.RiskManager.calculate_kelly_position_size = _ORIG_KELLY_CALC


def patch_va_yaml_base(max_pct: float):
    """Regime becomes a multiplier; base = yaml max_position_pct."""
    multipliers = {
        "strong_uptrend": 1.0,
        "uptrend":        0.85,
        "sideways":       0.65,
        "weak_downtrend": 0.40,
        "downtrend":      0.20,
    }
    rm.REGIME_POSITION_PCT = {k: round(max_pct * v, 4) for k, v in multipliers.items()}
    rm.RiskManager.calculate_kelly_position_size = _ORIG_KELLY_CALC


def _wrap_with_floor(orig_calc):
    """Wrap calculate_kelly_position_size to enforce min_position_pct
    when the fallback path returned a too-small allocation."""
    def wrapped(self, *args, **kwargs):
        result = orig_calc(self, *args, **kwargs)
        if not result.allowed or result.quantity <= 0:
            return result
        # Floor the allocation to min_position_pct of portfolio_value
        portfolio_value = kwargs.get("portfolio_value")
        if portfolio_value is None and len(args) >= 3:
            portfolio_value = args[2]
        price = kwargs.get("price")
        if price is None and len(args) >= 2:
            price = args[1]
        cash_available = kwargs.get("cash_available")
        if cash_available is None and len(args) >= 4:
            cash_available = args[3]
        if not portfolio_value or not price or not cash_available:
            return result
        min_alloc = portfolio_value * self._params.min_position_pct
        current_alloc = result.quantity * price
        if current_alloc >= min_alloc:
            return result
        # Bump to floor (capped by cash, max_position_pct enforced upstream)
        target_alloc = min(min_alloc, cash_available * 0.95,
                          portfolio_value * self._params.max_position_pct)
        new_qty = int(target_alloc / price)
        if new_qty <= result.quantity:
            return result
        result.quantity = new_qty
        result.allocation_usd = new_qty * price
        result.reason = f"{result.reason} +floor"
        return result
    return wrapped


def patch_vb_floor():
    """Enforce min_position_pct floor in the sizing fallback path."""
    rm.REGIME_POSITION_PCT = dict(_ORIG_REGIME_PCT)
    rm.RiskManager.calculate_kelly_position_size = _wrap_with_floor(_ORIG_KELLY_CALC)


def patch_vc_combined(max_pct: float):
    """VA base + VB floor."""
    patch_va_yaml_base(max_pct)
    rm.RiskManager.calculate_kelly_position_size = _wrap_with_floor(_ORIG_KELLY_CALC)


# ── Backtest harness ─────────────────────────────────────────────────────

def _market_config(market: str, max_positions: int, max_pct: float, disabled: list[str]) -> dict:
    if market == "KR":
        return dict(
            market="KR",
            initial_equity=100_000_000,
            default_stop_loss_pct=0.12,
            default_take_profit_pct=0.20,
            max_positions=max_positions,
            max_position_pct=max_pct,
            sell_cooldown_days=1,
            whipsaw_max_losses=2,
            min_hold_days=1,
            slippage_pct=0.08,
            volume_adjusted_slippage=True,
            min_confidence=0.30,
            disabled_strategies=disabled,
        )
    return dict(
        market="US",
        initial_equity=100_000,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
        max_positions=max_positions,
        max_position_pct=max_pct,
        sell_cooldown_days=1,
        whipsaw_max_losses=2,
        min_hold_days=1,
        slippage_pct=0.05,
        volume_adjusted_slippage=True,
        min_confidence=0.30,
        disabled_strategies=disabled,
    )


async def run_variant(name: str, market: str, kw: dict, patcher, max_pct: float):
    if patcher is patch_baseline:
        patcher()
    elif patcher is patch_va_yaml_base or patcher is patch_vc_combined:
        patcher(max_pct)
    else:
        patcher()

    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    # Average cash utilization: cash/equity per snapshot, mean
    snaps = eng._daily_snapshots
    if snaps:
        cash_pcts = [s.cash / s.equity for s in snaps if s.equity > 0]
        avg_cash_pct = sum(cash_pcts) / len(cash_pcts) if cash_pcts else 0.0
    else:
        avg_cash_pct = 0.0
    avg_positions = sum(s.n_positions for s in snaps) / len(snaps) if snaps else 0
    return dict(
        name=name, market=market,
        ret=round(m.total_return_pct, 2),
        sharpe=round(m.sharpe_ratio, 2),
        mdd=round(m.max_drawdown_pct, 2),
        pf=round(m.profit_factor, 2),
        trades=m.total_trades,
        wr=round(m.win_rate * 100, 1),
        alpha=round(m.alpha, 1),
        avg_cash_pct=round(avg_cash_pct * 100, 1),
        avg_positions=round(avg_positions, 1),
        elapsed=round(el, 1),
    )


VARIANTS = [
    ("V0_baseline",    patch_baseline),
    ("VA_yaml_base",   patch_va_yaml_base),
    ("VG_18pos",       patch_baseline),       # raise cap (test "more slots"
    ("VH_24pos",       patch_baseline),       #  hypothesis from live morning
    ("VI_18pos_VA",    patch_va_yaml_base),   #  rejections)
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["KR", "US"])
    args = parser.parse_args()

    loader = StrategyConfigLoader()
    results = []

    for market in args.markets:
        disabled = loader.get_market_disabled_strategies(market)
        # Match live config: KR max_positions=12 max_pct=0.20; US 20/0.10
        if market == "KR":
            max_positions, max_pct = 12, 0.20
        else:
            max_positions, max_pct = 20, 0.10
        kw = _market_config(market, max_positions, max_pct, disabled)

        print(f"\n══════ {market} ══════")
        for name, patcher in VARIANTS:
            # Override max_positions for cap-variants
            kw_v = dict(kw)
            if name in ("VG_18pos", "VI_18pos_VA"):
                kw_v["max_positions"] = 18
            elif name == "VH_24pos":
                kw_v["max_positions"] = 24
            print(f"▶ {name}...")
            r = await run_variant(name, market, kw_v, patcher, max_pct)
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} MDD={r['mdd']:.1f}% "
                  f"PF={r['pf']:.2f} Trades={r['trades']} WR={r['wr']:.0f}% "
                  f"Alpha={r['alpha']:+.1f}% Cash={r['avg_cash_pct']:.0f}% "
                  f"AvgPos={r['avg_positions']:.1f} ({r['elapsed']:.0f}s)")

    # Restore module so subsequent imports aren't poisoned.
    patch_baseline()

    # Summary table
    print("\n" + "=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    hdr = f"{'Variant':<16} {'Mkt':<3} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Trades':>7} {'Cash%':>7} {'AvgPos':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<16} {r['market']:<3} "
              f"{r['ret']:+7.1f} {r['sharpe']:+7.2f} {r['mdd']:7.1f} {r['pf']:6.2f} "
              f"{r['trades']:7d} {r['avg_cash_pct']:7.0f} {r['avg_positions']:7.1f}")

    print("\nDelta vs V0 baseline (per market):")
    for market in args.markets:
        v0 = next(r for r in results if r['market']==market and r['name']=='V0_baseline')
        for name in ("VA_yaml_base", "VG_18pos", "VH_24pos", "VI_18pos_VA"):
            v = next(r for r in results if r['market']==market and r['name']==name)
            d_ret = v['ret'] - v0['ret']
            d_sharpe = v['sharpe'] - v0['sharpe']
            d_mdd = v['mdd'] - v0['mdd']  # more negative = worse
            d_cash = v['avg_cash_pct'] - v0['avg_cash_pct']
            ok = (d_ret >= 0 and d_sharpe >= -0.05 and d_mdd >= -2.0 and d_cash <= 0)
            tag = "✓" if ok else "✗"
            print(f"  {market} {name:<14}: ΔRet={d_ret:+5.1f}pp ΔSharpe={d_sharpe:+5.2f} "
                  f"ΔMDD={d_mdd:+5.1f}pp ΔCash={d_cash:+5.0f}pp {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
