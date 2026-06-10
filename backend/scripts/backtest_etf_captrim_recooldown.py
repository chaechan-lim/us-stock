"""KR ETF backtest: cap_trim < -5% skip + sector ETF 24h re-entry block.

Question: do today's two proposed live changes improve realized PnL on
2y of KR sector ETF data?

Variants compared:
  A) baseline — current live (cap_trim trims any breach, 4h cooldown)
  B) cap_trim_skip_loser — skip cap_trim when position PnL < −5%
  C) reentry_24h — sector ETF cooldown bumped to 24h
  D) B+C combined

Simulator covers what matters for this question:
  - 7 KR sector ETFs (the actual live universe)
  - Daily top-N sector rotation (1w 0.2 + 1m 0.4 + 3m 0.4 score,
    min_score 60, top_3)
  - 8% stop-loss per position (matches default_stop_loss_pct)
  - max_single_etf_pct 15%, max_portfolio_pct 30%
  - cap_trim sells the largest over-cap ETF; variants B/D add the
    < −5% skip
  - sell_cooldown_hours (variants C/D: 24, others: 4)
  - max_hold_days 10
  - slippage 5 bps + ₩1,000 commission per order

Skips full regime detection and leveraged pair logic — those weren't
involved in today's incident. The cap_trim path that fired today was
on sector ETFs only.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"

# KR sector ETF universe (matches config/kr_etf_universe.yaml)
SECTOR_ETFS = {
    "091160": "반도체",
    "305720": "2차전지",
    "091180": "자동차",
    "244580": "바이오",
    "091170": "금융",
    "315930": "IT",
    "117680": "철강소재",
}

INITIAL_KRW = 30_000_000  # 3천만원 (KR sleeve typical size)
COMMISSION_KRW = 1_000
SLIPPAGE_BPS = 5  # 0.05%
STOP_LOSS_PCT = -0.08  # -8%
MAX_SINGLE_PCT = 0.15
MAX_PORTFOLIO_PCT = 0.30
MAX_HOLD_DAYS = 10
TOP_N_SECTORS = 3
MIN_SCORE = 60.0
SCORE_W = (0.20, 0.40, 0.40)  # 1w / 1m / 3m
LOOKBACKS = (5, 21, 63)


def _load_ohlc() -> dict[str, pd.DataFrame]:
    """Load full OHLC for each sector ETF. Used to model intra-day drops:
    morning BUY at Open, then intra-bar Low triggers stop_loss / cap_trim
    before the bar closes. Critical for today's incident — pure-Close
    backtest hides the BUY→drop→trim sequence we saw."""
    out: dict[str, pd.DataFrame] = {}
    for sym in SECTOR_ETFS:
        path = DATA_DIR / f"{sym}.KS__2y__1d.csv"
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.set_index("Date").sort_index()
        out[sym] = df[["Open", "High", "Low", "Close"]].dropna()
    if not out:
        raise RuntimeError("No KR ETF data loaded")
    common = None
    for df in out.values():
        common = df.index if common is None else common.intersection(df.index)
    return {k: v.reindex(common).ffill() for k, v in out.items()}


def _score_sectors(prices: dict[str, pd.DataFrame], i: int) -> list[str]:
    raw: list[tuple[str, float]] = []
    lb_1w, lb_1m, lb_3m = LOOKBACKS
    w_1w, w_1m, w_3m = SCORE_W
    for sym, df in prices.items():
        s = df["Close"]
        if i < lb_3m or i >= len(s):
            continue
        try:
            cur = float(s.iloc[i])
            p1w = float(s.iloc[i - lb_1w])
            p1m = float(s.iloc[i - lb_1m])
            p3m = float(s.iloc[i - lb_3m])
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
    scored = []
    if spread == 0:
        scored = [(s, 100.0 if v > 0 else 0.0) for s, v in raw]
    else:
        scored = [(s, (v - lo) / spread * 100) for s, v in raw]
    scored.sort(key=lambda x: -x[1])
    return [s for s, sc in scored[:TOP_N_SECTORS] if sc >= MIN_SCORE]


@dataclass
class _Pos:
    symbol: str
    qty: int
    entry_price: float
    entry_date: date


@dataclass
class _Config:
    label: str
    # B/D: skip cap_trim when position unrealized PnL < skip_threshold
    captrim_skip_pnl_pct: float | None = None
    # C/D: sell_cooldown in days (live default = 4h = 0.167d, our variants 0.167 or 1.0)
    sell_cooldown_days: float = 4 / 24


@dataclass
class _Result:
    label: str
    final_krw: float
    ret_pct: float
    sharpe: float
    mdd_pct: float
    trades: int
    stop_loss_hits: int
    captrim_fires: int
    captrim_skipped_loser: int
    reentry_blocked_count: int
    avg_trade_pnl_krw: float


def _run(cfg: _Config, prices: dict[str, pd.DataFrame]) -> _Result:
    """Intra-bar OHLC simulation. Sequence per day:
       1) at OPEN: stop-loss exit if prior-bar close already triggered SL
       2) sector rotation BUY at OPEN
       3) cap-trim if BUYs pushed exposure over cap at OPEN, but assess
          per-position PnL at intra-bar LOW (matches today's pattern
          where morning BUYs at OPEN, prices drift to LOW, then cap_trim
          + SL fire at the low)
       4) intra-bar SL: positions held from prior days exit at SL price
          if LOW touched it
       5) close MTM"""
    dates = list(next(iter(prices.values())).index)
    cash = float(INITIAL_KRW)
    positions: dict[str, _Pos] = {}
    last_sell_date: dict[str, date] = {}
    daily_eq: list[float] = []

    trades = 0
    sl_hits = 0
    captrim_fires = 0
    captrim_skipped = 0
    reentry_blocked = 0
    realized_pnls: list[float] = []

    def _o(sym: str, idx: int) -> float:
        return float(prices[sym]["Open"].iloc[idx])

    def _l(sym: str, idx: int) -> float:
        return float(prices[sym]["Low"].iloc[idx])

    def _c(sym: str, idx: int) -> float:
        return float(prices[sym]["Close"].iloc[idx])

    warmup = 63
    for i in range(warmup, len(dates)):
        d = dates[i].date()

        # ---- 1) Open-bar exits: stop-loss (if OPEN already below SL — gap-down)
        #         and max_hold_days
        to_close: list[str] = []
        for sym, p in positions.items():
            op = _o(sym, i)
            pnl_at_open = (op / p.entry_price) - 1
            days_held = (d - p.entry_date).days
            if pnl_at_open <= STOP_LOSS_PCT:
                to_close.append(sym)
                sl_hits += 1
            elif days_held >= MAX_HOLD_DAYS:
                to_close.append(sym)
        for sym in to_close:
            pos = positions.pop(sym)
            op = _o(sym, i)
            exec_p = op * (1 - SLIPPAGE_BPS / 10000)
            proceeds = pos.qty * exec_p - COMMISSION_KRW
            realized_pnls.append(proceeds - pos.qty * pos.entry_price)
            cash += proceeds
            last_sell_date[sym] = d
            trades += 1

        # ---- 2) Sector rotation: rank from PRIOR bar close so today's
        #         decision uses information available at the open.
        top_n = _score_sectors(prices, i - 1)
        out_of_top = [sym for sym in list(positions) if sym not in top_n]
        for sym in out_of_top:
            pos = positions.pop(sym)
            op = _o(sym, i)
            exec_p = op * (1 - SLIPPAGE_BPS / 10000)
            proceeds = pos.qty * exec_p - COMMISSION_KRW
            realized_pnls.append(proceeds - pos.qty * pos.entry_price)
            cash += proceeds
            last_sell_date[sym] = d
            trades += 1

        equity = cash + sum(p.qty * _o(p.symbol, i) for p in positions.values())
        for sym in top_n:
            if sym in positions:
                continue
            ls = last_sell_date.get(sym)
            if ls is not None and (d - ls).days < cfg.sell_cooldown_days:
                reentry_blocked += 1
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
            if qty <= 0:
                continue
            cost = qty * exec_p + COMMISSION_KRW
            if cost > cash:
                continue
            cash -= cost
            positions[sym] = _Pos(sym, qty, exec_p, d)
            trades += 1

        # ---- 3) Cap-trim: post-BUY check. Equity at OPEN.
        equity_at_open = cash + sum(
            p.qty * _o(p.symbol, i) for p in positions.values()
        )
        etf_val_open = sum(p.qty * _o(p.symbol, i) for p in positions.values())
        if equity_at_open > 0 and etf_val_open / equity_at_open > MAX_PORTFOLIO_PCT:
            captrim_fires += 1
            over = etf_val_open - equity_at_open * MAX_PORTFOLIO_PCT
            sorted_p = sorted(
                positions.values(),
                key=lambda p: p.qty * _o(p.symbol, i),
                reverse=True,
            )
            for p in sorted_p:
                if over <= 0:
                    break
                # Assess skip at intra-bar LOW (worst-case during the bar).
                # This matches today's case: BUY at 09:02 → LOW around 10am
                # → cap_trim hits the position when it's already underwater.
                low_p = _l(p.symbol, i)
                pnl_at_low = (low_p / p.entry_price) - 1
                if (
                    cfg.captrim_skip_pnl_pct is not None
                    and pnl_at_low < cfg.captrim_skip_pnl_pct
                ):
                    captrim_skipped += 1
                    continue
                # Trim at LOW (pessimistic intra-bar fill)
                exec_p = low_p * (1 - SLIPPAGE_BPS / 10000)
                trim_qty = int(min(p.qty, over / exec_p + 1))
                trim_qty = max(1, trim_qty)
                proceeds = trim_qty * exec_p - COMMISSION_KRW
                realized_pnls.append(proceeds - trim_qty * p.entry_price)
                cash += proceeds
                trades += 1
                if trim_qty >= p.qty:
                    last_sell_date[p.symbol] = d
                    del positions[p.symbol]
                else:
                    p.qty -= trim_qty
                over -= trim_qty * exec_p

        # ---- 4) Intra-bar stop-loss for prior-day positions (exit at SL level)
        new_today = {sym for sym in positions if positions[sym].entry_date == d}
        intra_sl: list[str] = []
        for sym, p in positions.items():
            if sym in new_today:
                continue
            low_p = _l(sym, i)
            pnl_at_low = (low_p / p.entry_price) - 1
            if pnl_at_low <= STOP_LOSS_PCT:
                intra_sl.append(sym)
        for sym in intra_sl:
            pos = positions.pop(sym)
            sl_px = pos.entry_price * (1 + STOP_LOSS_PCT)
            exec_p = sl_px * (1 - SLIPPAGE_BPS / 10000)
            proceeds = pos.qty * exec_p - COMMISSION_KRW
            realized_pnls.append(proceeds - pos.qty * pos.entry_price)
            cash += proceeds
            last_sell_date[sym] = d
            trades += 1
            sl_hits += 1

        # ---- 5) MTM at Close
        eq = cash + sum(p.qty * _c(p.symbol, i) for p in positions.values())
        daily_eq.append(eq)

    # ---- Metrics
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
    for v in daily_eq:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak * 100
            if dd < mdd:
                mdd = dd

    avg_pnl = sum(realized_pnls) / len(realized_pnls) if realized_pnls else 0.0

    return _Result(
        label=cfg.label,
        final_krw=round(final, 0),
        ret_pct=round(ret_pct, 2),
        sharpe=round(sharpe, 2),
        mdd_pct=round(mdd, 2),
        trades=trades,
        stop_loss_hits=sl_hits,
        captrim_fires=captrim_fires,
        captrim_skipped_loser=captrim_skipped,
        reentry_blocked_count=reentry_blocked,
        avg_trade_pnl_krw=round(avg_pnl, 0),
    )


def main() -> None:
    prices = _load_ohlc()
    print(f"Universe: {len(prices)} KR sector ETFs (intra-bar OHLC sim)")
    print(f"Bars: {len(next(iter(prices.values())))} days "
          f"({next(iter(prices.values())).index[0].date()} → "
          f"{next(iter(prices.values())).index[-1].date()})")
    print()

    # NB: sell_cooldown_days=1.0 in a daily-bar sim is functionally
    # equivalent to "next trading day" (gap.days >= 1 → allowed).
    # The live incident was 58min apart on the SAME day — daily bars
    # can't model that. To probe the next-day re-buy case, use 2.0d
    # which actually blocks the next session. Also test 5.0d for
    # comparison.
    configs = [
        _Config(label="A baseline"),
        # Cap-trim skip threshold sweep (looking for live -10.76% incident analog)
        _Config(label="B captrim_skip<-2%", captrim_skip_pnl_pct=-0.02),
        _Config(label="C captrim_skip<-3%", captrim_skip_pnl_pct=-0.03),
        _Config(label="D captrim_skip<-5%", captrim_skip_pnl_pct=-0.05),
        # Re-entry cooldown sweep
        _Config(label="E reentry_2d", sell_cooldown_days=2.0),
        _Config(label="F reentry_3d", sell_cooldown_days=3.0),
        # Combos worth trying
        _Config(
            label="G skip<-3% + reentry_2d",
            captrim_skip_pnl_pct=-0.03,
            sell_cooldown_days=2.0,
        ),
        _Config(
            label="H skip<-2% + reentry_2d",
            captrim_skip_pnl_pct=-0.02,
            sell_cooldown_days=2.0,
        ),
    ]
    results = [_run(c, prices) for c in configs]

    # Print table
    print(
        f"{'config':22} {'Ret%':>8} {'Sharpe':>7} {'MDD%':>7} "
        f"{'Trades':>7} {'SL hits':>8} {'CapTrim':>8} "
        f"{'CT skip':>8} {'ReEntBlk':>9} {'AvgPnL₩':>10}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"{r.label:22} {r.ret_pct:>8.2f} {r.sharpe:>7.2f} {r.mdd_pct:>7.2f} "
            f"{r.trades:>7d} {r.stop_loss_hits:>8d} {r.captrim_fires:>8d} "
            f"{r.captrim_skipped_loser:>8d} {r.reentry_blocked_count:>9d} "
            f"{r.avg_trade_pnl_krw:>10,.0f}"
        )
    print()
    a = results[0]
    for r in results[1:]:
        d_ret = r.ret_pct - a.ret_pct
        d_sh = r.sharpe - a.sharpe
        d_mdd = r.mdd_pct - a.mdd_pct
        sign = "✓" if (d_ret > 0 and d_mdd >= -0.5) else "✗"
        print(
            f"  {sign} {r.label} vs baseline: "
            f"ΔRet={d_ret:+.2f}pp ΔSharpe={d_sh:+.2f} ΔMDD={d_mdd:+.2f}pp"
        )


if __name__ == "__main__":
    main()
