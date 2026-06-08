"""KR ETF: EW 7-sector default + partial hedge in bear regime.

Thesis (post 2026-06-08 cascade discussion):
  - Bull-mode default = equal-weight 7 sectors buy & hold
    (162% Ret over 2y; the bot must beat this baseline now)
  - Bear-mode = partial hedge: shift weight from sectors → cash/inverse
    based on REGIME CLASSIFIER confidence
  - Rotation logic is removed: the bot's value-add is no longer
    sector-picking, it's recognising regime transitions early.

Regime classifier (multi-signal, weighted):
  S1: 20d realized vol of KODEX 200 > trailing-1y 80th percentile
      (elevated fear, like VKOSPI spike proxy)
  S2: KS200 5d ROC < -3% (recent weakness)
  S3: < 30% sectors above own SMA50 (breadth deterioration)

  Hedge ratio = 0.20 × n_signals_active where n ∈ {0,1,2,3}
  i.e. 0/20/40/60% of equity shifts from sector basket to
  (half cash + half inverse 114800) when triggered.

This is a "do less" thesis: rotation friction killed 121pp; the
new system reduces rotation to zero and only acts on regime change.

Variants tested:
  A  baseline (current live engine, kept for comparison)
  B  EW pure (no hedge, just sector basket + hold)
  C  EW + cash-only hedge (no inverse)
  D  EW + inverse-only hedge (no cash)
  E  EW + partial hedge (cash + inverse split, user's choice)
  E2 same as E but 0.33 per signal (60% max hedge)
  E3 same as E but vol-only signal (single S1 trigger)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"

SECTOR_ETFS = ["091160", "305720", "091180", "244580", "091170", "315930", "117680"]
REGIME_PROXY = "069500"  # KODEX 200 (use returns/ratios only — absolute price dirty)
INVERSE_ETF = "114800"   # KODEX 인버스

INITIAL_KRW = 30_000_000
COMMISSION_KRW = 1_000
SLIPPAGE_BPS = 5
REBALANCE_FREQ_DAYS = 5   # check regime + rebalance weekly

VOL_LOOKBACK = 20
VOL_PERCENTILE_LB = 252   # 1y rolling percentile
VOL_PCT_THRESHOLD = 80
ROC_5D_THRESHOLD = -0.03
SMA50_THRESHOLD = 0.30
SECTOR_SMA = 50

HEDGE_PER_SIGNAL = 0.20   # 20% per active signal (max 60% with all 3)
INVERSE_VS_CASH_RATIO = 0.5  # 50% of hedge into inverse, 50% cash


def _load_ohlc(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = DATA_DIR / f"{sym}.KS__2y__1d.csv"
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.set_index("Date").sort_index()
        out[sym] = df[["Open", "High", "Low", "Close"]].dropna()
    common = None
    for df in out.values():
        common = df.index if common is None else common.intersection(df.index)
    return {k: v.reindex(common).ffill() for k, v in out.items()}


def _compute_regime_signals(
    prices: dict[str, pd.DataFrame], i: int,
) -> tuple[bool, bool, bool]:
    """Return (vol_spike, weak_roc, low_breadth) booleans at bar `i`."""
    ks = prices.get(REGIME_PROXY)
    if ks is None or i < max(VOL_LOOKBACK + VOL_PERCENTILE_LB, SECTOR_SMA + 5):
        return False, False, False
    close = ks["Close"]

    # S1: realized vol percentile
    rets = close.pct_change().dropna()
    cur_vol = rets.iloc[i - VOL_LOOKBACK : i].std()
    historical_vols = rets.rolling(VOL_LOOKBACK).std().iloc[
        i - VOL_PERCENTILE_LB : i
    ].dropna()
    vol_pct = (historical_vols < cur_vol).mean() * 100 if len(historical_vols) else 0
    s1 = vol_pct >= VOL_PCT_THRESHOLD

    # S2: 5d ROC
    if i >= 5:
        roc = (close.iloc[i] / close.iloc[i - 5]) - 1
        s2 = roc <= ROC_5D_THRESHOLD
    else:
        s2 = False

    # S3: breadth — fraction of sector ETFs above own SMA50
    above = 0
    counted = 0
    for sym in SECTOR_ETFS:
        df = prices.get(sym)
        if df is None or i < SECTOR_SMA:
            continue
        sec_close = df["Close"]
        sma = sec_close.iloc[i - SECTOR_SMA : i].mean()
        counted += 1
        if sec_close.iloc[i] >= sma:
            above += 1
    s3 = (counted > 0) and (above / counted) < SMA50_THRESHOLD

    return bool(s1), bool(s2), bool(s3)


@dataclass
class _VariantCfg:
    label: str
    hedge_enabled: bool = True
    hedge_into_cash: bool = True       # use cash component
    hedge_into_inverse: bool = True    # use inverse component
    hedge_per_signal: float = HEDGE_PER_SIGNAL
    use_signals: tuple[bool, bool, bool] = (True, True, True)  # S1,S2,S3
    min_signals_required: int = 1      # gate: need ≥N active before any hedge
    aggressive_hedge_value: float = 0.0  # if >0, override per-signal and use this when gate hits


@dataclass
class _Result:
    label: str
    ret_pct: float
    sharpe: float
    mdd_pct: float
    rebalances: int
    bear_days: int          # days with any signal active
    full_bear_days: int     # days with all 3 signals active
    avg_hedge_ratio: float  # average hedge ratio across all bars


def _run(cfg: _VariantCfg, prices: dict[str, pd.DataFrame]) -> _Result:
    dates = list(next(iter(prices.values())).index)
    cash = float(INITIAL_KRW)
    holdings: dict[str, int] = {}  # symbol → qty
    daily_eq: list[float] = []

    rebalances = 0
    bear_days = 0
    full_bear_days = 0
    hedge_ratios: list[float] = []

    sector_alloc = (1.0 - 0.0) / len(SECTOR_ETFS)  # equal weight within sector basket

    def _o(s: str, idx: int) -> float:
        return float(prices[s]["Open"].iloc[idx])

    def _c(s: str, idx: int) -> float:
        return float(prices[s]["Close"].iloc[idx])

    warmup = max(VOL_LOOKBACK + VOL_PERCENTILE_LB, SECTOR_SMA + 5)
    for i in range(warmup, len(dates)):
        d = dates[i].date()

        # Compute regime signals + target hedge ratio
        s1, s2, s3 = _compute_regime_signals(prices, i)
        active = [
            s and use for s, use in zip((s1, s2, s3), cfg.use_signals)
        ]
        n_active = sum(active)
        # Strict gate: only activate hedge if at least N signals are on.
        # When triggered, use aggressive_hedge_value if set (override
        # per-signal ramp). Otherwise scale linearly.
        if cfg.hedge_enabled and n_active >= cfg.min_signals_required:
            if cfg.aggressive_hedge_value > 0:
                hedge_ratio = cfg.aggressive_hedge_value
            else:
                hedge_ratio = cfg.hedge_per_signal * n_active
        else:
            hedge_ratio = 0.0
        hedge_ratio = min(hedge_ratio, 0.80)  # cap at 80%
        hedge_ratios.append(hedge_ratio)
        if n_active > 0:
            bear_days += 1
        if all(active):
            full_bear_days += 1

        # Compute split: hedge_ratio split between cash and inverse
        if cfg.hedge_enabled and (cfg.hedge_into_cash or cfg.hedge_into_inverse):
            if cfg.hedge_into_cash and cfg.hedge_into_inverse:
                inv_ratio = hedge_ratio * INVERSE_VS_CASH_RATIO
                cash_ratio = hedge_ratio * (1 - INVERSE_VS_CASH_RATIO)
            elif cfg.hedge_into_inverse:
                inv_ratio = hedge_ratio
                cash_ratio = 0.0
            else:
                inv_ratio = 0.0
                cash_ratio = hedge_ratio
        else:
            inv_ratio, cash_ratio = 0.0, 0.0
        sector_basket_ratio = 1.0 - inv_ratio - cash_ratio

        # Rebalance once per REBALANCE_FREQ_DAYS or when regime changes
        # by ≥1 signal (we re-eval every bar but only rebalance on
        # schedule to keep transaction costs sane).
        should_rebalance = (
            (i - warmup) % REBALANCE_FREQ_DAYS == 0
            or n_active == 0 and any(h > 0 for s, h in holdings.items()
                                     if s == INVERSE_ETF)
            or n_active == 3 and holdings.get(INVERSE_ETF, 0) == 0
        )
        if not should_rebalance:
            eq = cash + sum(
                qty * _c(sym, i) for sym, qty in holdings.items()
            )
            daily_eq.append(eq)
            continue

        # Compute target positions at OPEN
        equity = cash + sum(qty * _o(sym, i) for sym, qty in holdings.items())
        target: dict[str, float] = {}
        # Sector basket: EW across 7 sectors
        per_sector = (equity * sector_basket_ratio) / len(SECTOR_ETFS)
        for sym in SECTOR_ETFS:
            target[sym] = per_sector
        # Inverse hedge
        target[INVERSE_ETF] = equity * inv_ratio
        # Cash residual = equity * cash_ratio (implicit)

        # Convert targets to qty and adjust positions
        rebalanced_any = False
        # First, sell anything not in target or over-target
        for sym in list(holdings.keys()):
            cur_qty = holdings[sym]
            cur_value = cur_qty * _o(sym, i)
            tgt_value = target.get(sym, 0.0)
            if tgt_value <= 0 and cur_qty > 0:
                # Full exit
                exec_p = _o(sym, i) * (1 - SLIPPAGE_BPS / 10000)
                cash += cur_qty * exec_p - COMMISSION_KRW
                del holdings[sym]
                rebalanced_any = True
            elif cur_value > tgt_value * 1.05:
                # Partial trim
                excess_value = cur_value - tgt_value
                exec_p = _o(sym, i) * (1 - SLIPPAGE_BPS / 10000)
                trim_qty = int(excess_value / exec_p)
                if trim_qty > 0:
                    cash += trim_qty * exec_p - COMMISSION_KRW
                    holdings[sym] = cur_qty - trim_qty
                    rebalanced_any = True
                    if holdings[sym] <= 0:
                        del holdings[sym]
        # Then, buy under-target symbols
        for sym, tgt_value in target.items():
            if tgt_value <= 0:
                continue
            cur_qty = holdings.get(sym, 0)
            cur_value = cur_qty * _o(sym, i)
            if cur_value < tgt_value * 0.95:
                op = _o(sym, i)
                exec_p = op * (1 + SLIPPAGE_BPS / 10000)
                need_value = tgt_value - cur_value
                add_qty = int(need_value / exec_p)
                if add_qty > 0:
                    cost = add_qty * exec_p + COMMISSION_KRW
                    if cost <= cash:
                        cash -= cost
                        holdings[sym] = cur_qty + add_qty
                        rebalanced_any = True
        if rebalanced_any:
            rebalances += 1

        eq = cash + sum(qty * _c(sym, i) for sym, qty in holdings.items())
        daily_eq.append(eq)

    # Metrics
    final = daily_eq[-1] if daily_eq else INITIAL_KRW
    ret_pct = (final / INITIAL_KRW - 1) * 100
    rets = []
    for j in range(1, len(daily_eq)):
        prev = daily_eq[j - 1]
        if prev > 0:
            rets.append((daily_eq[j] - prev) / prev)
    if len(rets) >= 30:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
        sd = math.sqrt(var)
        sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    peak = daily_eq[0] if daily_eq else INITIAL_KRW
    mdd = 0.0
    for val in daily_eq:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak * 100
            if dd < mdd:
                mdd = dd
    avg_hedge = sum(hedge_ratios) / len(hedge_ratios) if hedge_ratios else 0.0

    return _Result(
        label=cfg.label,
        ret_pct=round(ret_pct, 2),
        sharpe=round(sharpe, 2),
        mdd_pct=round(mdd, 2),
        rebalances=rebalances,
        bear_days=bear_days,
        full_bear_days=full_bear_days,
        avg_hedge_ratio=round(avg_hedge, 3),
    )


def main() -> None:
    symbols = SECTOR_ETFS + [REGIME_PROXY, INVERSE_ETF]
    prices = _load_ohlc(symbols)
    print(f"Loaded {len(prices)} symbols, "
          f"{len(next(iter(prices.values())))} bars")
    print()

    variants = [
        _VariantCfg(label="B EW pure (no hedge)", hedge_enabled=False),
        _VariantCfg(
            label="C EW + cash-only hedge",
            hedge_into_cash=True, hedge_into_inverse=False,
        ),
        _VariantCfg(
            label="D EW + inverse-only hedge",
            hedge_into_cash=False, hedge_into_inverse=True,
        ),
        _VariantCfg(
            label="E EW + partial 50/50",
            hedge_into_cash=True, hedge_into_inverse=True,
        ),
        _VariantCfg(
            label="E2 partial, 33%/signal (90% max)",
            hedge_per_signal=0.30,
        ),
        _VariantCfg(
            label="E3 partial, vol-signal only",
            use_signals=(True, False, False),
        ),
        _VariantCfg(
            label="E4 partial, ROC+breadth only",
            use_signals=(False, True, True),
        ),
        # Strict-gate variants: only hedge on high-confidence bear
        _VariantCfg(
            label="F1 2-of-3 gate, 50% hedge",
            min_signals_required=2,
            aggressive_hedge_value=0.50,
        ),
        _VariantCfg(
            label="F2 3-of-3 gate, 70% hedge",
            min_signals_required=3,
            aggressive_hedge_value=0.70,
        ),
        _VariantCfg(
            label="F3 ROC+breadth 2-gate, 50%",
            use_signals=(False, True, True),
            min_signals_required=2,
            aggressive_hedge_value=0.50,
        ),
    ]
    results = [_run(v, prices) for v in variants]

    print(
        f"{'config':36} {'Ret%':>8} {'Sharpe':>7} {'MDD%':>7} "
        f"{'Rebal':>6} {'BearD':>6} {'FullB':>6} {'AvgHdg':>7}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.label:36} {r.ret_pct:>8.2f} {r.sharpe:>7.2f} "
            f"{r.mdd_pct:>7.2f} {r.rebalances:>6d} {r.bear_days:>6d} "
            f"{r.full_bear_days:>6d} {r.avg_hedge_ratio*100:>6.1f}%"
        )

    # Reference: pure EW b&h (no logic at all)
    print()
    print("Reference: pure EW 7-sector buy-and-hold = +162.57% Ret / 2.71 "
          "Sharpe / -18.98% MDD (from prior backtest)")
    print()
    print("=" * 60)
    print("Δ vs EW pure (does hedge add or subtract value?)")
    print("=" * 60)
    pure = results[0]
    for r in results[1:]:
        d_ret = r.ret_pct - pure.ret_pct
        d_sh = r.sharpe - pure.sharpe
        d_mdd = r.mdd_pct - pure.mdd_pct
        verdict = "✓" if (d_sh > 0 and d_mdd >= -1) else "?"
        print(
            f"  {verdict} {r.label:36} ΔRet={d_ret:+6.2f}pp  "
            f"ΔSharpe={d_sh:+.2f}  ΔMDD={d_mdd:+.2f}pp"
        )


if __name__ == "__main__":
    main()
