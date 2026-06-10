"""Backtest 4 thesis variants for KR ETF engine.

After 2026-06-08 cascade discussion: bot is too defensive in
weak markets, sells at local lows, and underperforms buy-and-hold.
Test 4 different responses:

  V1: no SL on sector ETFs — rank-based exit only
  V2: SL 8→15% widened + drop_day guard (skip SL on −2% KS200 days)
  V3: bear-mode sit out — no new ETF BUY when KS200 < SMA200
  V4: aggressive inverse ETF — relax bear gating, deploy 114800
      (KODEX 인버스) in downtrend regime to "short" the index

Universe: 7 sector ETFs + KODEX 200 (069500, regime proxy) + KODEX
인버스 (114800, V4 only).

Compared against:
  - A baseline (current live engine behaviour, with today's cap_trim
    -3% skip + 12h cooldown applied)
  - KS200 buy-and-hold benchmark (the "doing nothing" comparator)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"

SECTOR_ETFS = {
    "091160": "반도체",
    "305720": "2차전지",
    "091180": "자동차",
    "244580": "바이오",
    "091170": "금융",
    "315930": "IT",
    "117680": "철강소재",
}
REGIME_PROXY = "069500"   # KODEX 200 → KS200 SMA200/momentum proxy
INVERSE_ETF = "114800"    # KODEX 인버스 (1x short KS200)

INITIAL_KRW = 30_000_000
COMMISSION_KRW = 1_000
SLIPPAGE_BPS = 5
DEFAULT_SL_PCT = -0.08
MAX_SINGLE_PCT = 0.15
MAX_PORTFOLIO_PCT = 0.30
MAX_HOLD_DAYS = 10
TOP_N_SECTORS = 3
MIN_SCORE = 60.0
SCORE_W = (0.20, 0.40, 0.40)
LOOKBACKS = (5, 21, 63)
SMA_REGIME = 200
DROP_DAY_PCT = -0.02      # V2: skip SL on KS200 days <= -2%
INVERSE_ALLOC_PCT = 0.15  # V4: 15% sleeve into inverse in downtrend


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


def _score_sectors(prices: dict[str, pd.DataFrame], i: int) -> list[str]:
    raw: list[tuple[str, float]] = []
    lb_1w, lb_1m, lb_3m = LOOKBACKS
    w_1w, w_1m, w_3m = SCORE_W
    for sym in SECTOR_ETFS:
        df = prices.get(sym)
        if df is None or i < lb_3m or i >= len(df):
            continue
        s = df["Close"]
        try:
            cur, p1w, p1m, p3m = (
                float(s.iloc[i]),
                float(s.iloc[i - lb_1w]),
                float(s.iloc[i - lb_1m]),
                float(s.iloc[i - lb_3m]),
            )
            if min(p1w, p1m, p3m) <= 0:
                continue
            r1w = (cur / p1w - 1) * 100
            r1m = (cur / p1m - 1) * 100
            r3m = (cur / p3m - 1) * 100
            raw.append((sym, r1w * w_1w + r1m * w_1m + r3m * w_3m))
        except Exception:
            continue
    if not raw:
        return []
    vals = [v for _, v in raw]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    scored = (
        [(s, 100.0 if v > 0 else 0.0) for s, v in raw] if spread == 0
        else [(s, (v - lo) / spread * 100) for s, v in raw]
    )
    scored.sort(key=lambda x: -x[1])
    return [s for s, sc in scored[:TOP_N_SECTORS] if sc >= MIN_SCORE]


def _regime_at(prices: dict[str, pd.DataFrame], i: int) -> str:
    """Simple regime: above/below KODEX 200 SMA200."""
    df = prices.get(REGIME_PROXY)
    if df is None or i < SMA_REGIME:
        return "uptrend"
    s = df["Close"]
    sma = s.iloc[i - SMA_REGIME : i].mean()
    cur = float(s.iloc[i])
    return "uptrend" if cur >= sma else "downtrend"


def _ks200_return_pct(prices: dict[str, pd.DataFrame], i: int) -> float:
    """Same-bar return of KODEX 200 (proxy for KS200 daily change)."""
    df = prices.get(REGIME_PROXY)
    if df is None or i >= len(df):
        return 0.0
    o = float(df["Open"].iloc[i])
    c = float(df["Close"].iloc[i])
    return (c / o) - 1 if o > 0 else 0.0


@dataclass
class _Pos:
    symbol: str
    qty: int
    entry_price: float
    entry_date: date


@dataclass
class _Variant:
    label: str
    sl_pct: float = DEFAULT_SL_PCT        # 0 disables; -0.15 widens
    drop_day_skip_sl: bool = False        # V2
    sector_sl_disabled: bool = False      # V1
    bear_no_new_buys: bool = False        # V3
    aggressive_inverse: bool = False      # V4
    captrim_skip_pnl: float = -0.03
    sell_cooldown_days: float = 0.5       # 12h ≈ 0.5d


@dataclass
class _Result:
    label: str
    ret_pct: float
    sharpe: float
    mdd_pct: float
    trades: int
    sl_hits: int
    sl_skipped_dropday: int
    bear_days_sitout: int
    inverse_days_held: int


def _run(v: _Variant, prices: dict[str, pd.DataFrame]) -> _Result:
    dates = list(next(iter(prices.values())).index)
    cash = float(INITIAL_KRW)
    positions: dict[str, _Pos] = {}
    last_sell_date: dict[str, date] = {}
    daily_eq: list[float] = []

    trades = sl_hits = sl_skipped_dd = bear_sitout = inv_days = 0

    def _o(s: str, idx: int) -> float:
        return float(prices[s]["Open"].iloc[idx])

    def _l(s: str, idx: int) -> float:
        return float(prices[s]["Low"].iloc[idx])

    def _c(s: str, idx: int) -> float:
        return float(prices[s]["Close"].iloc[idx])

    warmup = max(SMA_REGIME, 63)
    for i in range(warmup, len(dates)):
        d = dates[i].date()
        regime = _regime_at(prices, i)
        ks200_ret = _ks200_return_pct(prices, i)
        is_drop_day = ks200_ret <= DROP_DAY_PCT

        # ---- 1) Open-bar SL + max_hold exits
        to_close: list[tuple[str, str]] = []
        for sym, p in positions.items():
            op = _o(sym, i)
            pnl_at_open = (op / p.entry_price) - 1
            days_held = (d - p.entry_date).days

            # Determine SL threshold for this position
            is_inverse = sym == INVERSE_ETF
            is_sector = sym in SECTOR_ETFS
            if is_inverse:
                # Inverse ETF: always SL at default (no widening)
                sl_threshold = DEFAULT_SL_PCT
            elif is_sector and v.sector_sl_disabled:
                # V1: no SL on sector ETFs
                sl_threshold = -10.0  # never triggers
            else:
                sl_threshold = v.sl_pct

            sl_triggered = pnl_at_open <= sl_threshold
            # V2: skip SL on drop_day
            if sl_triggered and v.drop_day_skip_sl and is_drop_day:
                sl_skipped_dd += 1
                sl_triggered = False

            if sl_triggered:
                to_close.append((sym, "sl"))
                sl_hits += 1
            elif days_held >= MAX_HOLD_DAYS:
                to_close.append((sym, "hold"))
        for sym, _ in to_close:
            pos = positions.pop(sym)
            op = _o(sym, i)
            exec_p = op * (1 - SLIPPAGE_BPS / 10000)
            cash += pos.qty * exec_p - COMMISSION_KRW
            last_sell_date[sym] = d
            trades += 1

        # ---- 2a) V4: inverse-ETF rotation when downtrend
        held_inverse = INVERSE_ETF in positions
        if v.aggressive_inverse and regime == "downtrend":
            inv_days += 1
            # Sell all sector ETFs first (mutual exclusivity with inverse)
            for sym in list(positions):
                if sym in SECTOR_ETFS:
                    pos = positions.pop(sym)
                    op = _o(sym, i)
                    exec_p = op * (1 - SLIPPAGE_BPS / 10000)
                    cash += pos.qty * exec_p - COMMISSION_KRW
                    last_sell_date[sym] = d
                    trades += 1
            # Buy inverse if not held
            if not held_inverse and INVERSE_ETF in prices:
                ls = last_sell_date.get(INVERSE_ETF)
                if ls is None or (d - ls).days >= v.sell_cooldown_days:
                    equity = cash + sum(
                        p.qty * _o(p.symbol, i) for p in positions.values()
                    )
                    target = equity * INVERSE_ALLOC_PCT
                    op = _o(INVERSE_ETF, i)
                    exec_p = op * (1 + SLIPPAGE_BPS / 10000)
                    qty = int(target / exec_p)
                    if qty > 0 and qty * exec_p + COMMISSION_KRW <= cash:
                        cash -= qty * exec_p + COMMISSION_KRW
                        positions[INVERSE_ETF] = _Pos(INVERSE_ETF, qty, exec_p, d)
                        trades += 1
        else:
            # Not in inverse mode → exit any held inverse
            if held_inverse:
                pos = positions.pop(INVERSE_ETF)
                op = _o(INVERSE_ETF, i)
                exec_p = op * (1 - SLIPPAGE_BPS / 10000)
                cash += pos.qty * exec_p - COMMISSION_KRW
                last_sell_date[INVERSE_ETF] = d
                trades += 1

        # ---- 2b) Sector rotation
        # V3: bear sit-out — sell all sector ETFs and don't buy new
        # V4: aggressive inverse already handled sector exits above
        in_sector_mode = (
            not (v.bear_no_new_buys and regime == "downtrend")
            and not (v.aggressive_inverse and regime == "downtrend")
        )
        if v.bear_no_new_buys and regime == "downtrend":
            bear_sitout += 1
            # Sell all sector ETFs, no new buys
            for sym in list(positions):
                if sym in SECTOR_ETFS:
                    pos = positions.pop(sym)
                    op = _o(sym, i)
                    exec_p = op * (1 - SLIPPAGE_BPS / 10000)
                    cash += pos.qty * exec_p - COMMISSION_KRW
                    last_sell_date[sym] = d
                    trades += 1

        if in_sector_mode:
            top_n = _score_sectors(prices, i - 1)
            for sym in list(positions):
                if sym in SECTOR_ETFS and sym not in top_n:
                    pos = positions.pop(sym)
                    op = _o(sym, i)
                    exec_p = op * (1 - SLIPPAGE_BPS / 10000)
                    cash += pos.qty * exec_p - COMMISSION_KRW
                    last_sell_date[sym] = d
                    trades += 1

            equity = cash + sum(p.qty * _o(p.symbol, i) for p in positions.values())
            for sym in top_n:
                if sym in positions:
                    continue
                ls = last_sell_date.get(sym)
                if ls is not None and (d - ls).days < v.sell_cooldown_days:
                    continue
                op = _o(sym, i)
                single_cap = equity * MAX_SINGLE_PCT
                cur_etf_value = sum(
                    p.qty * _o(p.symbol, i) for p in positions.values()
                )
                port_room = equity * MAX_PORTFOLIO_PCT - cur_etf_value
                if port_room <= 0:
                    continue
                target = min(single_cap, port_room)
                exec_p = op * (1 + SLIPPAGE_BPS / 10000)
                qty = int(target / exec_p)
                if qty <= 0 or qty * exec_p + COMMISSION_KRW > cash:
                    continue
                cash -= qty * exec_p + COMMISSION_KRW
                positions[sym] = _Pos(sym, qty, exec_p, d)
                trades += 1

        # ---- 3) Cap-trim with −3% skip (matches live)
        equity_at_open = cash + sum(
            p.qty * _o(p.symbol, i) for p in positions.values()
        )
        etf_val_open = sum(p.qty * _o(p.symbol, i) for p in positions.values())
        if equity_at_open > 0 and etf_val_open / equity_at_open > MAX_PORTFOLIO_PCT:
            over = etf_val_open - equity_at_open * MAX_PORTFOLIO_PCT
            sorted_p = sorted(
                positions.values(),
                key=lambda p: p.qty * _o(p.symbol, i),
                reverse=True,
            )
            for p in sorted_p:
                if over <= 0:
                    break
                low_p = _l(p.symbol, i)
                pnl_at_low = (low_p / p.entry_price) - 1
                if pnl_at_low < v.captrim_skip_pnl:
                    continue
                exec_p = low_p * (1 - SLIPPAGE_BPS / 10000)
                trim_qty = int(min(p.qty, over / exec_p + 1))
                trim_qty = max(1, trim_qty)
                cash += trim_qty * exec_p - COMMISSION_KRW
                trades += 1
                if trim_qty >= p.qty:
                    last_sell_date[p.symbol] = d
                    del positions[p.symbol]
                else:
                    p.qty -= trim_qty
                over -= trim_qty * exec_p

        # ---- 4) Intra-bar SL for older positions (skip if drop_day in V2)
        new_today = {sym for sym in positions if positions[sym].entry_date == d}
        intra_sl: list[str] = []
        for sym, p in positions.items():
            if sym in new_today:
                continue
            low_p = _l(sym, i)
            pnl_at_low = (low_p / p.entry_price) - 1
            is_inverse = sym == INVERSE_ETF
            is_sector = sym in SECTOR_ETFS
            if is_inverse:
                threshold = DEFAULT_SL_PCT
            elif is_sector and v.sector_sl_disabled:
                threshold = -10.0
            else:
                threshold = v.sl_pct
            if pnl_at_low <= threshold:
                if v.drop_day_skip_sl and is_drop_day:
                    sl_skipped_dd += 1
                    continue
                intra_sl.append(sym)
        for sym in intra_sl:
            pos = positions.pop(sym)
            sl_px = pos.entry_price * (1 + max(threshold, -0.50))
            # use this position's threshold (recompute for accuracy)
            sym_thresh = (
                DEFAULT_SL_PCT if sym == INVERSE_ETF
                else (-10.0 if (sym in SECTOR_ETFS and v.sector_sl_disabled) else v.sl_pct)
            )
            sl_px = pos.entry_price * (1 + sym_thresh)
            exec_p = sl_px * (1 - SLIPPAGE_BPS / 10000)
            cash += pos.qty * exec_p - COMMISSION_KRW
            last_sell_date[sym] = d
            trades += 1
            sl_hits += 1

        # ---- 5) MTM
        eq = cash + sum(p.qty * _c(p.symbol, i) for p in positions.values())
        daily_eq.append(eq)

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
    return _Result(
        label=v.label,
        ret_pct=round(ret_pct, 2),
        sharpe=round(sharpe, 2),
        mdd_pct=round(mdd, 2),
        trades=trades,
        sl_hits=sl_hits,
        sl_skipped_dropday=sl_skipped_dd,
        bear_days_sitout=bear_sitout,
        inverse_days_held=inv_days,
    )


def _benchmark_equal_weight(
    prices: dict[str, pd.DataFrame], label: str = "EW 7-sector b&h",
) -> _Result:
    """Equal-weight buy-and-hold across the 7 sector ETFs. Honest
    'do nothing' comparator — splits initial capital evenly across
    every sector ETF the bot trades, then holds. Tells us whether
    the bot's rotation logic adds any value over naive diversification.

    NB: 069500 (KODEX 200) data is dirty (3.5x price move in 2y
    inconsistent with KOSPI 200 reality, likely a back-adjustment
    bug in yfinance). Sector ETFs match today's live fills, so
    those are trustworthy."""
    sector_dfs = {sym: prices[sym] for sym in SECTOR_ETFS if sym in prices}
    if not sector_dfs:
        return _Result(label, 0, 0, 0, 0, 0, 0, 0, 0)
    dates = list(next(iter(sector_dfs.values())).index)
    warmup = max(SMA_REGIME, 63)
    per_sym_capital = (INITIAL_KRW * 0.95) / len(sector_dfs)
    holdings: dict[str, int] = {}
    cash_left = INITIAL_KRW
    for sym, df in sector_dfs.items():
        start_p = float(df["Open"].iloc[warmup])
        qty = int(per_sym_capital / (start_p * (1 + SLIPPAGE_BPS / 10000)))
        holdings[sym] = qty
        cash_left -= qty * start_p * (1 + SLIPPAGE_BPS / 10000) + COMMISSION_KRW

    daily_eq = []
    for i in range(warmup, len(dates)):
        eq = cash_left + sum(
            qty * float(sector_dfs[sym]["Close"].iloc[i])
            for sym, qty in holdings.items()
        )
        daily_eq.append(eq)

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
    return _Result(label, round(ret_pct, 2), round(sharpe, 2),
                   round(mdd, 2), len(sector_dfs), 0, 0, 0, 0)


def main() -> None:
    syms = list(SECTOR_ETFS) + [REGIME_PROXY, INVERSE_ETF]
    prices = _load_ohlc(syms)
    print(f"Loaded {len(prices)} symbols, "
          f"{len(next(iter(prices.values())))} bars")
    print()

    variants = [
        _Variant(label="A baseline (current live)"),
        _Variant(label="V1 sector_no_SL", sector_sl_disabled=True),
        _Variant(label="V2 SL15%+drop_day", sl_pct=-0.15, drop_day_skip_sl=True),
        _Variant(label="V3 bear_sitout", bear_no_new_buys=True),
        _Variant(label="V4 aggressive_inverse", aggressive_inverse=True),
        # Bonus: combos
        _Variant(label="V1+V2 combined", sector_sl_disabled=True,
                 sl_pct=-0.15, drop_day_skip_sl=True),
        _Variant(label="V3+V4 combined", bear_no_new_buys=True,
                 aggressive_inverse=True),
    ]
    results = [_run(v, prices) for v in variants]
    bench = _benchmark_equal_weight(prices)

    print(
        f"{'config':28} {'Ret%':>8} {'Sharpe':>7} {'MDD%':>7} "
        f"{'Trades':>7} {'SL':>5} {'SLskip':>7} {'BearOut':>8} {'InvDays':>8}"
    )
    print("-" * 110)
    print(
        f"{bench.label:28} {bench.ret_pct:>8.2f} {bench.sharpe:>7.2f} "
        f"{bench.mdd_pct:>7.2f} {bench.trades:>7d} "
        f"{bench.sl_hits:>5d} {bench.sl_skipped_dropday:>7d} "
        f"{bench.bear_days_sitout:>8d} {bench.inverse_days_held:>8d}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r.label:28} {r.ret_pct:>8.2f} {r.sharpe:>7.2f} "
            f"{r.mdd_pct:>7.2f} {r.trades:>7d} "
            f"{r.sl_hits:>5d} {r.sl_skipped_dropday:>7d} "
            f"{r.bear_days_sitout:>8d} {r.inverse_days_held:>8d}"
        )
    print()
    a = results[0]
    print("=" * 60)
    print("Δ vs baseline (alpha vs current live engine)")
    print("=" * 60)
    for r in results[1:]:
        d_ret = r.ret_pct - a.ret_pct
        d_sh = r.sharpe - a.sharpe
        d_mdd = r.mdd_pct - a.mdd_pct
        verdict = "✓" if (d_ret > 0.5 and d_mdd >= -1.0) else "?"
        print(
            f"  {verdict} {r.label:26} ΔRet={d_ret:+6.2f}pp  "
            f"ΔSharpe={d_sh:+.2f}  ΔMDD={d_mdd:+.2f}pp"
        )
    print()
    print("=" * 60)
    print("Δ vs equal-weight 7-sector buy&hold")
    print("(naive diversification — does rotation add value?)")
    print("=" * 60)
    for r in [a] + results[1:]:
        d_ret = r.ret_pct - bench.ret_pct
        d_sh = r.sharpe - bench.sharpe
        d_mdd = r.mdd_pct - bench.mdd_pct
        verdict = "✓" if (d_ret > 0 and d_mdd >= -1.0) else "✗"
        print(
            f"  {verdict} {r.label:26} ΔRet={d_ret:+6.2f}pp  "
            f"ΔSharpe={d_sh:+.2f}  ΔMDD={d_mdd:+.2f}pp"
        )


if __name__ == "__main__":
    main()
