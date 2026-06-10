"""KR capital allocation: individual-stock book vs EW ETF sleeve.

Resolves the exposure-vs-whipsaw tension (2026-06-09 discussion):
blocking entries to cut churn creates idle cash (the deployment
problem the user already fought). The real lever is routing capital
into the NO-CHURN EW basket instead of the churny stock book.

This builds ONE portfolio at several sleeve splits and reports
portfolio-level return + Sharpe + MDD + DEPLOYMENT% (idle cash) +
CHURN (trade count). The deployment metric is the whole point: does a
bigger EW sleeve keep capital deployed while cutting churn?

Two sub-portfolios, combined at sleeve_pct:
  1. Stock book  (1 - sleeve_pct of capital):
       supertrend(7,2.0) entries, live SL stack (KR dynamic ATR SL
       clamp[5,20]%, trailing 10/4, hard 12%, TP dynamic), equal-weight
       up to max_positions=18. Churny.
  2. EW sleeve  (sleeve_pct of capital):
       equal-weight 7 KR sector ETFs, buy & hold, weekly rebalance.
       No stop-loss. ~zero churn.

Splits tested: 0% / 30% (current live) / 50% / 70% / 100% EW sleeve.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandas_ta as ta

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"

INITIAL = 30_000_000.0
COMMISSION = 0.0005
SLIPPAGE = 0.0005

# Supertrend + SL stack (matches live KR config)
ST_LENGTH, ST_MULT = 7, 2.0
SL_ATR_MULT, SL_CLAMP = 2.5, (0.05, 0.20)
TP_ATR_MULT, TP_CLAMP = 4.0, (0.08, 0.25)
HARD_SL = 0.12
TRAIL_ACT, TRAIL_PCT = 0.10, 0.04
ATR_LEN = 14
MAX_POSITIONS = 18

SECTOR_ETFS = ["091160", "305720", "091180", "244580", "091170", "315930", "117680"]
EW_REBALANCE_DAYS = 5

ETF_SKIP = {
    "069500", "091160", "091170", "091180", "114800", "122630",
    "132030", "148070", "244580", "261240", "305720", "315930",
    "117680", "233740", "229200", "251340", "950160",
}


def _load(sym: str) -> pd.DataFrame | None:
    for suffix in ("KS", "KQ"):
        path = DATA_DIR / f"{sym}.{suffix}__2y__1d.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None).dt.normalize()
            df = df.set_index("Date").sort_index()
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close"]].dropna()
    return None


def _stock_symbols() -> list[str]:
    syms = set()
    for f in DATA_DIR.glob("*__2y__1d.csv"):
        sym = f.name.split("__")[0].split(".")[0]
        if sym not in ETF_SKIP:
            syms.add(sym)
    return sorted(syms)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    st = ta.supertrend(df["high"], df["low"], df["close"], length=ST_LENGTH, multiplier=ST_MULT)
    if st is None or st.empty:
        return pd.DataFrame()
    dcol = next((c for c in st.columns if c.startswith("SUPERTd")), None)
    lcol = next((c for c in st.columns if c.startswith("SUPERT_")), None)
    if not dcol or not lcol:
        return pd.DataFrame()
    df["st_dir"] = st[dcol]
    df["st_line"] = st[lcol]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)
    return df.dropna()


def _dyn_sl(price, atr):
    if price <= 0 or atr <= 0:
        return HARD_SL
    return max(SL_CLAMP[0], min(SL_CLAMP[1], (atr / price) * SL_ATR_MULT))


def _dyn_tp(price, atr):
    if price <= 0 or atr <= 0:
        return TP_CLAMP[1]
    return max(TP_CLAMP[0], min(TP_CLAMP[1], (atr / price) * TP_ATR_MULT))


@dataclass
class _Held:
    entry_px: float
    peak: float
    sl_pct: float
    tp_pct: float
    shares: int


def _master_calendar(frames: dict[str, pd.DataFrame]) -> list:
    # Use the most common (longest) index — 005930 if present else longest
    if "005930" in frames:
        return list(frames["005930"].index)
    longest = max(frames.values(), key=len)
    return list(longest.index)


def _simulate(sleeve_pct: float, stock_frames, etf_frames, calendar,
              close_map) -> dict:
    """close_map[sym] = pd.Series of close reindexed to calendar + ffilled,
    used for MTM so calendar-misaligned days don't drop holdings to 0
    (phantom drawdown). Signal/exit logic still uses the native frames."""
    stock_cap0 = INITIAL * (1.0 - sleeve_pct)
    sleeve_cap0 = INITIAL * sleeve_pct

    # ---- Stock book state
    cash = stock_cap0
    held: dict[str, _Held] = {}
    per_pos_target = stock_cap0 / MAX_POSITIONS if MAX_POSITIONS else 0
    trades = 0

    # ---- EW sleeve state
    ew_cash = sleeve_cap0
    ew_holdings: dict[str, int] = {}
    last_rebal = -999

    daily_equity = []
    deploy_samples = []

    avail_etfs = [s for s in SECTOR_ETFS if s in etf_frames]

    for di, d in enumerate(calendar):
        # ===== STOCK BOOK =====
        # 1. exits
        for sym in list(held.keys()):
            df = stock_frames.get(sym)
            if df is None or d not in df.index:
                continue
            row = df.loc[d]
            h = held[sym]
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            st_dir = row["st_dir"]
            h.peak = max(h.peak, hi)
            exit_px = None
            stop_px = max(h.entry_px * (1 - HARD_SL), h.entry_px * (1 - h.sl_pct))
            if lo <= stop_px:
                exit_px = stop_px
            if exit_px is None and (h.peak - h.entry_px) / h.entry_px >= TRAIL_ACT:
                tpx = h.peak * (1 - TRAIL_PCT)
                if lo <= tpx:
                    exit_px = tpx
            if exit_px is None and hi >= h.entry_px * (1 + h.tp_pct):
                exit_px = h.entry_px * (1 + h.tp_pct)
            if exit_px is None and st_dir == -1:
                exit_px = cl
            if exit_px is not None:
                cash += h.shares * exit_px * (1 - SLIPPAGE) * (1 - COMMISSION)
                del held[sym]
                trades += 1

        # 2. entries (supertrend bull flip, price > line)
        if len(held) < MAX_POSITIONS:
            for sym, df in stock_frames.items():
                if sym in held or len(held) >= MAX_POSITIONS:
                    continue
                if d not in df.index:
                    continue
                pos = df.index.get_loc(d)
                if pos < 1:
                    continue
                row = df.iloc[pos]
                prev = df.iloc[pos - 1]
                if row["st_dir"] == 1 and prev["st_dir"] == -1 and row["close"] > row["st_line"]:
                    price = float(row["close"]) * (1 + SLIPPAGE)
                    budget = min(per_pos_target, cash)
                    shares = int(budget / price) if price > 0 else 0
                    if shares <= 0:
                        continue
                    cost = shares * price * (1 + COMMISSION)
                    if cost > cash:
                        continue
                    cash -= cost
                    held[sym] = _Held(
                        entry_px=price, peak=float(row["close"]),
                        sl_pct=_dyn_sl(row["close"], row["atr"]),
                        tp_pct=_dyn_tp(row["close"], row["atr"]),
                        shares=shares,
                    )
                    trades += 1

        # stock book MTM (use ffilled close_map — never drops a holding)
        stock_mtm = cash
        for sym, h in held.items():
            cs = close_map.get(sym)
            px = float(cs.iloc[di]) if cs is not None and not pd.isna(cs.iloc[di]) else h.entry_px
            stock_mtm += h.shares * px

        # ===== EW SLEEVE ===== (all prices via ffilled close_map)
        if sleeve_cap0 > 0 and avail_etfs:
            def _etf_px(s):
                cs = close_map.get(s)
                return float(cs.iloc[di]) if cs is not None and not pd.isna(cs.iloc[di]) else 0.0

            ew_equity = ew_cash + sum(q * _etf_px(s) for s, q in ew_holdings.items())
            if di - last_rebal >= EW_REBALANCE_DAYS or last_rebal < 0:
                last_rebal = di
                per_etf = (ew_equity * 0.98) / len(avail_etfs)
                for s in avail_etfs:
                    px = _etf_px(s)
                    if px <= 0:
                        continue
                    cur_q = ew_holdings.get(s, 0)
                    cur_v = cur_q * px
                    if cur_v > per_etf * 1.05:
                        sell_q = int((cur_v - per_etf) / px)
                        if sell_q > 0:
                            ew_cash += sell_q * px * (1 - SLIPPAGE)
                            ew_holdings[s] = cur_q - sell_q
                    elif cur_v < per_etf * 0.95:
                        buy_q = int((per_etf - cur_v) / (px * (1 + SLIPPAGE)))
                        cost = buy_q * px * (1 + SLIPPAGE)
                        if buy_q > 0 and cost <= ew_cash:
                            ew_cash -= cost
                            ew_holdings[s] = cur_q + buy_q
            ew_mtm = ew_cash + sum(q * _etf_px(s) for s, q in ew_holdings.items())
        else:
            ew_mtm = sleeve_cap0  # no sleeve

        total_eq = stock_mtm + ew_mtm
        daily_equity.append(total_eq)
        # deployment = invested / equity (stock invested + ew invested)
        stock_invested = stock_mtm - cash
        ew_invested = ew_mtm - ew_cash
        deploy = (stock_invested + ew_invested) / total_eq if total_eq > 0 else 0
        deploy_samples.append(deploy)

    # ---- metrics
    final = daily_equity[-1] if daily_equity else INITIAL
    ret = (final / INITIAL - 1) * 100
    rets = [
        (daily_equity[i] - daily_equity[i - 1]) / daily_equity[i - 1]
        for i in range(1, len(daily_equity)) if daily_equity[i - 1] > 0
    ]
    if len(rets) >= 30:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    peak = daily_equity[0] if daily_equity else INITIAL
    mdd = 0.0
    for v in daily_equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v - peak) / peak * 100)
    avg_deploy = sum(deploy_samples) / len(deploy_samples) * 100 if deploy_samples else 0

    return {
        "ret": round(ret, 2), "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 2), "trades": trades,
        "deploy": round(avg_deploy, 1), "final": round(final, 0),
    }


def main():
    print("Loading KR stock + ETF data + computing supertrend...")
    stock_frames = {}
    for sym in _stock_symbols():
        raw = _load(sym)
        if raw is None or len(raw) < 120:
            continue
        ind = _indicators(raw)
        if not ind.empty and len(ind) >= 100:
            stock_frames[sym] = ind
    etf_frames = {s: _load(s) for s in SECTOR_ETFS}
    etf_frames = {s: f for s, f in etf_frames.items() if f is not None}
    calendar = _master_calendar(stock_frames)
    # restrict to post-warmup region (supertrend needs ~14 bars)
    calendar = calendar[20:]
    cal_idx = pd.DatetimeIndex(calendar)

    # ffilled close lookup for MTM (stock + ETF), aligned to calendar
    close_map = {}
    for sym, f in stock_frames.items():
        close_map[sym] = f["close"].reindex(cal_idx).ffill()
    for s, f in etf_frames.items():
        close_map[s] = f["close"].reindex(cal_idx).ffill()

    print(f"Stock universe: {len(stock_frames)} | ETFs: {len(etf_frames)} | "
          f"bars: {len(calendar)}\n")

    splits = [0.0, 0.30, 0.50, 0.70, 1.0]
    print(f"{'EW sleeve':>10} {'Ret%':>8} {'Sharpe':>7} {'MDD%':>7} "
          f"{'Trades':>7} {'Deploy%':>8}")
    print("-" * 56)
    rows = []
    for sp in splits:
        r = _simulate(sp, stock_frames, etf_frames, calendar, close_map)
        rows.append((sp, r))
        label = f"{int(sp*100)}%"
        print(f"{label:>10} {r['ret']:>8.2f} {r['sharpe']:>7.2f} "
              f"{r['mdd']:>7.2f} {r['trades']:>7} {r['deploy']:>7.1f}%")

    print()
    print("EW sleeve = % of capital in no-churn EW basket; rest in "
          "supertrend stock book")
    print("Deploy% = avg capital deployed (100 - idle cash). Higher = "
          "less idle cash.")
    print("Trades = total churn (lower = calmer)")
    print()
    base = rows[1][1]  # 30% = current live
    print("=" * 60)
    print("Δ vs 30% sleeve (current live)")
    print("=" * 60)
    for sp, r in rows:
        if sp == 0.30:
            continue
        print(
            f"  EW {int(sp*100):>3}%: ΔRet={r['ret']-base['ret']:+6.2f}pp  "
            f"ΔSharpe={r['sharpe']-base['sharpe']:+.2f}  "
            f"ΔMDD={r['mdd']-base['mdd']:+.2f}pp  "
            f"ΔTrades={r['trades']-base['trades']:+d}  "
            f"ΔDeploy={r['deploy']-base['deploy']:+.1f}pp"
        )


if __name__ == "__main__":
    main()
