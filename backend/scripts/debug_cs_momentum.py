"""Debug cross_sectional_momentum: count BUY/SELL signals over 2y on US universe.

Bypasses pipeline screening/combiner — runs the strategy in isolation
on each watchlist symbol day-by-day, prints signal histogram.
"""

import asyncio
import sys
import functools

print = functools.partial(print, flush=True)

sys.path.insert(0, ".")

import yfinance as yf
import logging
logging.basicConfig(level=logging.WARNING)
for n in ("yfinance", "peewee", "urllib3", "httpx"):
    logging.getLogger(n).setLevel(logging.ERROR)

from data.indicator_service import IndicatorService
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from core.enums import SignalType


SYMBOLS = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "AVGO", "TSLA",
           "JPM", "BAC", "UNH", "LLY", "XOM", "CVX", "WMT", "COST",
           "JNJ", "MRK", "PG", "KO"]


async def main():
    indsvc = IndicatorService()
    strat = CrossSectionalMomentumStrategy()

    print(f"Strategy params: {strat.get_params()}")
    print(f"min_candles_required: {strat.min_candles_required}")
    print()

    histo = {"BUY": 0, "SELL": 0, "HOLD": 0}
    reason_counts: dict[str, int] = {}
    per_symbol = {}

    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d")
        except Exception as e:
            print(f"  {sym}: load failed: {e}")
            continue
        if df.empty:
            continue
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = indsvc.add_all_indicators(df)

        sym_buys = sym_sells = sym_holds = 0
        first_active_idx = None

        # Walk forward day by day
        for i in range(50, len(df)):
            window = df.iloc[:i]
            sig = await strat.analyze(window, sym)
            t = sig.signal_type.name if hasattr(sig.signal_type, "name") else str(sig.signal_type)
            histo[t] = histo.get(t, 0) + 1
            reason_counts[sig.reason or ""] = reason_counts.get(sig.reason or "", 0) + 1
            if t == "BUY":
                sym_buys += 1
                if first_active_idx is None:
                    first_active_idx = i
            elif t == "SELL":
                sym_sells += 1
            else:
                sym_holds += 1
        per_symbol[sym] = (sym_buys, sym_sells, sym_holds, first_active_idx, len(df))

    print("Aggregate signal histogram:")
    total = sum(histo.values())
    for t, n in histo.items():
        print(f"  {t:<6} {n:>6} ({n/max(total,1)*100:.1f}%)")

    print(f"\nPer-symbol BUY/SELL/HOLD (first BUY day idx, total bars):")
    for sym, (b, s, h, fb, n) in per_symbol.items():
        print(f"  {sym:<6} BUY={b:>4} SELL={s:>4} HOLD={h:>4} firstBUY={fb} bars={n}")

    print(f"\nTop 10 HOLD reasons:")
    sorted_r = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    for reason, n in sorted_r[:10]:
        print(f"  {n:>6}  {reason[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
