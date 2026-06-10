"""KR Opening Range Breakout (ORB) backtest — Track A proof-of-concept.

Tests whether 09:30-12:00 KST ORB entry has positive alpha on KR daily
1-min OHLCV. Uses yfinance for 5-day window (the maximum yfinance allows
for free intraday). Real validation requires KIS 90-day backfill +
proper IntradayBacktestEngine — this is just a structural smoke test.

ORB logic per symbol per day:
  Range: high/low of bars 09:00-09:29 KST (first 30 min)
  Entry: first bar in 09:30-12:00 where close > range_high * 1.005
         AND bar_volume > range_avg_volume * 1.5
  Exit:  SL = range_low (loose: -range/range_high≈3% typical)
         TP = range_high + (range_high - range_low) * 1.5
         Or end-of-session (15:30 KST close)

Reports per-day breakouts + Win rate + avg %PnL + total trades.

NOTE: 5-day window = 25 trading sessions max if 79 symbols × ~30%
trigger rate = ~60 trades. Statistical power is borderline. Treat as
"does the framework work" not "is ORB profitable".
"""

import functools
import logging
import sys
import time
import warnings
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
for n in ("yfinance", "peewee", "urllib3", "httpx"):
    logging.getLogger(n).setLevel(logging.ERROR)


# ORB parameters
RANGE_MIN_START = 0   # KST 09:00 → UTC 00:00 (minute of UTC day)
RANGE_MIN_END = 30    # KST 09:30 → UTC 00:30
ENTRY_WINDOW_END = 180  # KST 12:00 → UTC 03:00
SESSION_END = 390     # KST 15:30 → UTC 06:30
BREAKOUT_BUFFER = 0.005  # +0.5% above range_high
VOLUME_MULTIPLIER = 1.5   # bar vol > range_avg_vol * 1.5
TP_R_MULTIPLE = 1.5       # TP = entry + range * 1.5
SLIPPAGE_PCT = 0.001      # 10 bps


@dataclass
class Trade:
    symbol: str
    date: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    pnl_pct: float


def _load_intraday(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Bulk-download 5-day 1-min OHLCV from yfinance."""
    yf_symbols = [
        f"{s}.KS" if not s.endswith(".KS") and not s.endswith(".KQ") else s
        for s in symbols
    ]
    out: dict[str, pd.DataFrame] = {}
    print(f"Downloading 1-min OHLCV for {len(yf_symbols)} symbols (5d)...")
    for sym in yf_symbols:
        try:
            df = yf.download(sym, period="5d", interval="1m",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 60:
                continue
            # Flatten multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            out[sym] = df
        except Exception as e:
            print(f"  skip {sym}: {e}")
    print(f"  loaded {len(out)} symbols")
    return out


def _day_session(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """Filter to one trading day in UTC. day is 'YYYY-MM-DD'."""
    return df[df.index.strftime("%Y-%m-%d") == day]


def _minute_of_session(ts) -> int:
    """Minute of UTC day. KR 09:00 KST = 00:00 UTC = minute 0."""
    return ts.hour * 60 + ts.minute


def _backtest_one_day(
    symbol: str, day_df: pd.DataFrame, day: str,
) -> Trade | None:
    """Run ORB on one symbol-day. Returns Trade if entry triggered."""
    if len(day_df) < 35:
        return None

    # Establish range from bars in [0, 30)
    range_bars = day_df[
        day_df.index.map(_minute_of_session).map(
            lambda m: RANGE_MIN_START <= m < RANGE_MIN_END
        )
    ]
    if len(range_bars) < 15:
        return None
    range_high = float(range_bars["high"].max())
    range_low = float(range_bars["low"].min())
    range_avg_vol = float(range_bars["volume"].mean())
    if range_high <= range_low or range_avg_vol <= 0:
        return None

    entry_threshold = range_high * (1 + BREAKOUT_BUFFER)
    vol_threshold = range_avg_vol * VOLUME_MULTIPLIER

    # Scan entry window [30, 180)
    entry_bars = day_df[
        day_df.index.map(_minute_of_session).map(
            lambda m: RANGE_MIN_END <= m < ENTRY_WINDOW_END
        )
    ]
    entry_price = None
    entry_time = None
    for ts, row in entry_bars.iterrows():
        if (
            float(row["close"]) > entry_threshold
            and float(row["volume"]) > vol_threshold
        ):
            entry_price = float(row["close"]) * (1 + SLIPPAGE_PCT)
            entry_time = ts.strftime("%H:%M")
            break
    if entry_price is None:
        return None

    # Walk forward to exit
    after_entry = day_df[day_df.index > ts]
    range_size = range_high - range_low
    tp_price = range_high + range_size * TP_R_MULTIPLE
    sl_price = range_low

    for ts2, row2 in after_entry.iterrows():
        high = float(row2["high"])
        low = float(row2["low"])
        if low <= sl_price:
            exit_price = sl_price * (1 - SLIPPAGE_PCT)
            return Trade(
                symbol=symbol, date=day,
                entry_time=entry_time, entry_price=entry_price,
                exit_time=ts2.strftime("%H:%M"), exit_price=exit_price,
                exit_reason="SL",
                pnl_pct=(exit_price - entry_price) / entry_price * 100,
            )
        if high >= tp_price:
            exit_price = tp_price * (1 - SLIPPAGE_PCT)
            return Trade(
                symbol=symbol, date=day,
                entry_time=entry_time, entry_price=entry_price,
                exit_time=ts2.strftime("%H:%M"), exit_price=exit_price,
                exit_reason="TP",
                pnl_pct=(exit_price - entry_price) / entry_price * 100,
            )

    # End of session close
    if len(after_entry) > 0:
        last_row = after_entry.iloc[-1]
        exit_price = float(last_row["close"]) * (1 - SLIPPAGE_PCT)
        return Trade(
            symbol=symbol, date=day,
            entry_time=entry_time, entry_price=entry_price,
            exit_time=after_entry.index[-1].strftime("%H:%M"),
            exit_price=exit_price, exit_reason="EOS",
            pnl_pct=(exit_price - entry_price) / entry_price * 100,
        )
    return None


def main():
    from scanner.kr_screener import _KR_UNIVERSE
    symbols = [s[0] for s in _KR_UNIVERSE]

    t0 = time.time()
    data = _load_intraday(symbols)
    print(f"  ({time.time()-t0:.0f}s)")

    # Find unique trading days
    all_dates: set[str] = set()
    for df in data.values():
        for d in df.index.strftime("%Y-%m-%d").unique():
            all_dates.add(d)
    days = sorted(all_dates)
    print(f"Trading days: {days}")

    trades: list[Trade] = []
    for sym, df in data.items():
        for day in days:
            day_df = _day_session(df, day)
            t = _backtest_one_day(sym, day_df, day)
            if t is not None:
                trades.append(t)

    print(f"\n{'='*100}")
    print(f"ORB backtest: {len(trades)} trades from {len(data)} symbols × {len(days)} days")
    print(f"{'='*100}")

    if not trades:
        print("No entries triggered.")
        return

    # Per-trade detail
    trades.sort(key=lambda t: (t.date, t.entry_time))
    for t in trades[:15]:
        print(f"  {t.date} {t.symbol:>10} {t.entry_time}@{t.entry_price:>9.0f} "
              f"→ {t.exit_time}@{t.exit_price:>9.0f} {t.exit_reason:>3} "
              f"{t.pnl_pct:+6.2f}%")
    if len(trades) > 15:
        print(f"  ... ({len(trades) - 15} more)")

    # Aggregate
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    total_pnl = sum(t.pnl_pct for t in trades)
    win_rate = len(wins) / len(trades) * 100

    # Exit reason breakdown
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    print(f"\nAggregate:")
    print(f"  Trades:     {len(trades)}")
    print(f"  Win rate:   {win_rate:.1f}%")
    print(f"  Avg win:    {avg_win:+.2f}%")
    print(f"  Avg loss:   {avg_loss:+.2f}%")
    print(f"  Sum PnL%:   {total_pnl:+.2f}%")
    print(f"  Avg/trade:  {total_pnl/len(trades):+.3f}%")
    print(f"  Exits: {by_reason}")
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    print(f"  Expectancy: {expectancy:+.3f}% per trade")


if __name__ == "__main__":
    main()
