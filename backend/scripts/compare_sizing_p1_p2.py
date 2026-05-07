"""P1 (alloc round-up for high-price) + P2 (min_position_pct floor) variants.

Live observation 2026-05-07: KR cash 78% despite supertrend active. Two
sizing-level cash drags:

  P1: "Price too high for allocation" — 18 rejects/16h on KR blue chips
       (LGES, 현대차, SK하이닉스 etc.). alloc < share price → qty=0.
  P2: Fixed-sizing fallback doesn't enforce min_position_pct floor — low-
       conf signals get sized below 4% on KR / 5% on US.

This script tests four variants on current live config (KR dm+supertrend
cap=18, US sizing VA override). Both P1 and P2 are in-process patches;
no live code changes yet.
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

US_VA_OVERRIDE = {  # PR #123 — US live regime table
    "strong_uptrend": 0.100, "uptrend": 0.085, "sideways": 0.065,
    "weak_downtrend": 0.040, "downtrend": 0.020,
}

_ORIG_KELLY = rm.RiskManager.calculate_kelly_position_size
_ORIG_REGIME = dict(rm.REGIME_POSITION_PCT)


def patch_p1_round_up():
    """If alloc < price but alloc ≥ 50% of price AND 1 share ≤ max_position_pct
    of portfolio_value, buy 1 share anyway. Wraps the original sizing fn."""
    def wrapped(self, *args, **kwargs):
        result = _ORIG_KELLY(self, *args, **kwargs)
        if result.allowed and result.quantity > 0:
            return result
        if result.reason and "Price too high" not in result.reason:
            return result
        # Need to recompute: was alloc close to price?
        portfolio_value = kwargs.get("portfolio_value") or (args[2] if len(args) >= 3 else 0)
        cash_available = kwargs.get("cash_available") or (args[3] if len(args) >= 4 else 0)
        price = kwargs.get("price") or (args[1] if len(args) >= 2 else 0)
        if portfolio_value <= 0 or price <= 0:
            return result
        # Check: 1 share fits within max_position_pct
        one_share_pct = price / portfolio_value
        if one_share_pct > self._params.max_position_pct:
            return result  # 1 share too large, reject correctly
        if cash_available < price:
            return result  # actually no cash
        # Allow 1 share
        return rm.PositionSizeResult(
            quantity=1,
            allocation_usd=price,
            risk_per_share=price * self._params.default_stop_loss_pct,
            reason="Round-up to 1 share (P1)",
            allowed=True,
        )
    rm.RiskManager.calculate_kelly_position_size = wrapped


def patch_p2_floor():
    """Enforce min_position_pct floor in fallback path (when Kelly doesn't fire)."""
    def wrapped(self, *args, **kwargs):
        result = _ORIG_KELLY(self, *args, **kwargs)
        if not result.allowed or result.quantity <= 0:
            return result
        portfolio_value = kwargs.get("portfolio_value") or (args[2] if len(args) >= 3 else 0)
        cash_available = kwargs.get("cash_available") or (args[3] if len(args) >= 4 else 0)
        price = kwargs.get("price") or (args[1] if len(args) >= 2 else 0)
        if portfolio_value <= 0 or price <= 0:
            return result
        min_alloc = portfolio_value * self._params.min_position_pct
        if result.quantity * price >= min_alloc:
            return result
        max_alloc = portfolio_value * self._params.max_position_pct
        target = min(min_alloc, cash_available * 0.95, max_alloc)
        new_qty = int(target / price)
        if new_qty <= result.quantity:
            return result
        result.quantity = new_qty
        result.allocation_usd = new_qty * price
        result.reason = f"{result.reason} +floor"
        return result
    rm.RiskManager.calculate_kelly_position_size = wrapped


def patch_p1_p2_combined():
    """Apply P1 first (allow high-price), then P2 floor where it bumps qty."""
    def wrapped(self, *args, **kwargs):
        result = _ORIG_KELLY(self, *args, **kwargs)
        portfolio_value = kwargs.get("portfolio_value") or (args[2] if len(args) >= 3 else 0)
        cash_available = kwargs.get("cash_available") or (args[3] if len(args) >= 4 else 0)
        price = kwargs.get("price") or (args[1] if len(args) >= 2 else 0)
        if portfolio_value <= 0 or price <= 0:
            return result
        # P1: round up rejected high-price to 1 share if it fits max_pct
        if not result.allowed and "Price too high" in (result.reason or ""):
            one_share_pct = price / portfolio_value
            if one_share_pct <= self._params.max_position_pct and cash_available >= price:
                return rm.PositionSizeResult(
                    quantity=1, allocation_usd=price,
                    risk_per_share=price * self._params.default_stop_loss_pct,
                    reason="P1 round-up",
                    allowed=True,
                )
            return result
        if not result.allowed or result.quantity <= 0:
            return result
        # P2: floor
        min_alloc = portfolio_value * self._params.min_position_pct
        if result.quantity * price >= min_alloc:
            return result
        max_alloc = portfolio_value * self._params.max_position_pct
        target = min(min_alloc, cash_available * 0.95, max_alloc)
        new_qty = int(target / price)
        if new_qty > result.quantity:
            result.quantity = new_qty
            result.allocation_usd = new_qty * price
            result.reason = f"{result.reason} +P1P2"
        return result
    rm.RiskManager.calculate_kelly_position_size = wrapped


def reset():
    rm.RiskManager.calculate_kelly_position_size = _ORIG_KELLY
    rm.REGIME_POSITION_PCT = dict(_ORIG_REGIME)


def kr_cfg():
    return dict(
        market="KR", initial_equity=100_000_000,
        default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
        max_positions=18, max_position_pct=0.20,
        min_position_pct=0.04,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.08, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.3,
        disabled_strategies=ALL_DISABLED_KR,
    )


def us_cfg():
    return dict(
        market="US", initial_equity=100_000,
        default_stop_loss_pct=0.08, default_take_profit_pct=0.20,
        max_positions=20, max_position_pct=0.10,
        min_position_pct=0.05,
        sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
        slippage_pct=0.05, volume_adjusted_slippage=True,
        min_confidence=0.30, sector_boost_weight=0.2,
        disabled_strategies=ALL_DISABLED_US,
    )


VARIANTS = [
    ("V0_baseline", reset),
    ("VP1_round_up", patch_p1_round_up),
    ("VP2_floor",    patch_p2_floor),
    ("VP1P2_both",   patch_p1_p2_combined),
]


async def run_variant(name, market, kw, patcher):
    patcher()
    # US gets the regime override per PR #123
    if market == "US":
        rm.REGIME_POSITION_PCT = dict(US_VA_OVERRIDE)
    cfg = PipelineConfig(**kw)
    eng = FullPipelineBacktest(cfg)
    t0 = time.time()
    res = await eng.run(period="2y")
    el = time.time() - t0
    m = res.metrics
    snaps = eng._daily_snapshots
    avg_cash = (sum(s.cash/s.equity for s in snaps if s.equity > 0) / len(snaps)) if snaps else 0
    return dict(name=name, market=market,
        ret=round(m.total_return_pct,2), sharpe=round(m.sharpe_ratio,2),
        mdd=round(m.max_drawdown_pct,2), pf=round(m.profit_factor,2),
        trades=m.total_trades, cash=round(avg_cash*100,1),
        elapsed=round(el,1))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["KR","US"])
    args = parser.parse_args()

    results = []
    for market in args.markets:
        kw = kr_cfg() if market == "KR" else us_cfg()
        print(f"\n══ {market} ══")
        for name, patcher in VARIANTS:
            print(f"▶ {name}…")
            r = await run_variant(name, market, kw, patcher)
            results.append(r)
            print(f"  Ret={r['ret']:+.1f}% Sharpe={r['sharpe']:+.2f} MDD={r['mdd']:.1f}% "
                  f"PF={r['pf']:.2f} Trades={r['trades']} Cash={r['cash']:.0f}% ({r['elapsed']:.0f}s)")
    reset()

    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)
    print(f"{'Variant':<14} {'Mkt':<3} {'Ret%':>7} {'Sharpe':>7} {'MDD%':>7} {'PF':>5} {'Trades':>7} {'Cash%':>6}")
    print("-"*68)
    for r in results:
        print(f"{r['name']:<14} {r['market']:<3} {r['ret']:+7.1f} {r['sharpe']:+7.2f} "
              f"{r['mdd']:7.1f} {r['pf']:5.2f} {r['trades']:7d} {r['cash']:6.0f}")
    print("\nDelta vs V0 baseline:")
    for market in args.markets:
        v0 = next(r for r in results if r['market']==market and r['name']=='V0_baseline')
        for name in ("VP1_round_up","VP2_floor","VP1P2_both"):
            v = next(r for r in results if r['market']==market and r['name']==name)
            d = lambda k: v[k]-v0[k]
            ok = d('ret')>=-0.5 and d('sharpe')>=-0.05 and d('mdd')>=-2 and d('cash')<=0
            tag = "✓" if ok else "✗"
            print(f"  {market} {name:<14}: ΔRet={d('ret'):+5.1f}pp ΔSharpe={d('sharpe'):+5.2f} "
                  f"ΔMDD={d('mdd'):+5.1f}pp ΔCash={d('cash'):+5.0f}pp {tag}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
