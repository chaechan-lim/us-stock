"""KR 1-min intraday backfill — bulk-fetch + parquet cache.

Fetches 90 days × 79 KR symbols of 1-min OHLCV via KIS
inquire-time-itemchartprice. Persists to:
  data/kr_intraday/{symbol}_{YYYYMMDD}.parquet

Each file has 390 rows max (09:00-15:30 KST) with columns:
  ts (int YYYYMMDDHHMM), open, high, low, close, volume.

Rate limit budget:
  79 symbols × 14 calls/day × 90 days = ~99,540 calls
  KIS real: 20 req/sec → 83 minutes sustained
  paper:    5 req/sec → 5.5 hours

Usage:
  cd /home/chans/us-stock
  source .env && set -a && export $(cat .env | xargs) && set +a
  cd backend
  python scripts/backfill_kr_intraday.py [--days 90] [--symbols 005930,000660] [--start 20260301]

Idempotent: skips files that already exist. Use --force to refetch.
"""

import argparse
import asyncio
import functools
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

print = functools.partial(print, flush=True)
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
for n in ("urllib3", "httpx", "aiohttp"):
    logging.getLogger(n).setLevel(logging.WARNING)

CACHE_DIR = Path("data/kr_intraday")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _trading_days(start_date: date, end_date: date) -> list[str]:
    """Return YYYYMMDD strings for Mon-Fri days in range.

    KR holiday calendar is non-trivial; this is approximate. The KIS
    endpoint silently returns empty output2 for non-trading days, so
    we tolerate the over-fetching cost (no harm beyond rate).
    """
    out = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:    # Mon-Fri
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _symbols_from_seed() -> list[tuple[str, str]]:
    """Return list of (symbol, exchange) tuples from kr_screener seed."""
    from scanner.kr_screener import _KR_UNIVERSE
    return [(s[0], s[1]) for s in _KR_UNIVERSE]


def _output_path(symbol: str, date_str: str) -> Path:
    return CACHE_DIR / f"{symbol}_{date_str}.parquet"


async def _fetch_one(
    adapter, symbol: str, exchange: str, date_str: str,
) -> int:
    """Fetch one (symbol, date) → parquet file. Returns row count."""
    candles = await adapter.fetch_intraday_session(
        symbol, date_str, exchange=exchange,
    )
    if not candles:
        return 0
    df = pd.DataFrame([
        {
            "ts": c.timestamp,
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        }
        for c in candles
    ])
    df.to_parquet(_output_path(symbol, date_str), index=False)
    return len(df)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                        help="lookback days (default 90)")
    parser.add_argument("--start", type=str, default=None,
                        help="start date YYYYMMDD (overrides --days)")
    parser.add_argument("--end", type=str, default=None,
                        help="end date YYYYMMDD (default today)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="comma-separated subset (default full _KR_UNIVERSE)")
    parser.add_argument("--force", action="store_true",
                        help="refetch even if parquet exists")
    args = parser.parse_args()

    # Resolve date range
    end = (
        datetime.strptime(args.end, "%Y%m%d").date()
        if args.end else date.today()
    )
    if args.start:
        start = datetime.strptime(args.start, "%Y%m%d").date()
    else:
        start = end - timedelta(days=args.days)
    dates = _trading_days(start, end)

    # Resolve symbols
    if args.symbols:
        wanted = set(args.symbols.split(","))
        symbols = [(s, e) for (s, e) in _symbols_from_seed() if s in wanted]
    else:
        symbols = _symbols_from_seed()

    print(f"Backfill: {len(symbols)} symbols × {len(dates)} days = "
          f"{len(symbols) * len(dates)} (symbol, date) pairs")
    print(f"  Cache: {CACHE_DIR.resolve()}")

    # Initialize adapter
    from config import AppConfig
    from exchange.kis_auth import KISAuth
    from exchange.kis_kr_adapter import KISKRAdapter

    config = AppConfig()
    auth = KISAuth(
        app_key=config.kis.app_key,
        app_secret=config.kis.app_secret,
        base_url=config.kis.base_url,
    )
    adapter = KISKRAdapter(config.kis, auth)
    await adapter.initialize()

    fetched = 0
    skipped = 0
    empty = 0
    errors = 0
    for sym, exch in symbols:
        for date_str in dates:
            out_path = _output_path(sym, date_str)
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                n = await _fetch_one(adapter, sym, exch, date_str)
                if n == 0:
                    empty += 1
                else:
                    fetched += 1
                    if fetched % 20 == 0:
                        print(f"  fetched={fetched} skipped={skipped} "
                              f"empty={empty} err={errors}")
            except Exception as e:
                errors += 1
                print(f"  ERROR {sym} {date_str}: {e}")

    print(f"\nDONE: fetched={fetched} skipped={skipped} empty={empty} errors={errors}")


if __name__ == "__main__":
    asyncio.run(main())
