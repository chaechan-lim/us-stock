"""Universe diagnostic — compare backtest universe vs live trading reality.

Answers: is the backtest testing what live actually trades?

For each market (US, KR):
1. BACKTEST universe = backend/backtest/full_pipeline.py constants
   (DEFAULT_UNIVERSE + WIDE_UNIVERSE for US, DEFAULT_KR_UNIVERSE for KR)
2. LIVE universe = union of:
     - active DB watchlist entries
     - symbols with any Order (buy or sell) in the last N days

3. Report gaps both directions:
   - Backtest-only symbols: noise (backtest trades things live never touches)
   - Live-only symbols: blind spot (backtest can't tell us how these perform)

No side effects — purely read-only diagnostic.

Usage:
    cd backend && python scripts/compare_universe.py [--days 90]
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from backtest.full_pipeline import (
    DEFAULT_KR_UNIVERSE,
    DEFAULT_UNIVERSE,
    WIDE_UNIVERSE,
)
from db.session import get_session_factory
from db.trade_repository import TradeRepository


# yfinance KR symbols carry a `.KS` suffix; the live DB stores bare codes.
def _strip_kr_suffix(sym: str) -> str:
    return sym.removesuffix(".KS").removesuffix(".KQ")


async def _live_symbols(market: str, days: int) -> tuple[set[str], set[str]]:
    """Return (watchlist_symbols, traded_symbols) for the given market."""
    SessionFactory = get_session_factory()
    cutoff = datetime.utcnow() - timedelta(days=days)

    async with SessionFactory() as session:
        repo = TradeRepository(session)

        watchlist_rows = await repo.get_watchlist(active_only=True, market=market)
        watchlist = {w.symbol for w in watchlist_rows}

        # Large limit — we post-filter by timestamp
        trades = await repo.get_trade_history(limit=5000, exclude_paper=True)
        traded = {
            t.symbol for t in trades
            if t.market == market and t.created_at and t.created_at >= cutoff
        }

    return watchlist, traded


def _backtest_symbols(market: str) -> tuple[set[str], set[str]]:
    """Return (narrow, wide) backtest universes, yfinance-suffixes stripped."""
    if market == "KR":
        kr = {_strip_kr_suffix(s) for s in DEFAULT_KR_UNIVERSE}
        return kr, kr  # KR has only one universe
    narrow = set(DEFAULT_UNIVERSE)
    wide = set(WIDE_UNIVERSE)
    return narrow, wide


def _print_report(
    market: str,
    narrow_bt: set[str],
    wide_bt: set[str],
    watchlist: set[str],
    traded: set[str],
    days: int,
) -> None:
    live = watchlist | traded
    print(f"\n{'=' * 72}")
    print(f"  {market} market — backtest vs live")
    print(f"{'=' * 72}")
    print(f"  Backtest narrow universe:  {len(narrow_bt):>4} symbols")
    if market == "US":
        print(f"  Backtest wide universe:    {len(wide_bt):>4} symbols")
    print(f"  Live watchlist (active):    {len(watchlist):>4} symbols")
    print(f"  Live traded ({days}d):        {len(traded):>4} symbols")
    print(f"  Live union:                 {len(live):>4} symbols")

    # Key diagnostic — using WIDE for US since that's the closer match
    compare_bt = wide_bt if market == "US" else narrow_bt

    overlap = compare_bt & live
    bt_only = compare_bt - live         # backtest noise
    live_only = live - compare_bt        # backtest blind spot
    traded_blind = traded - compare_bt   # the most damning gap

    if compare_bt:
        coverage = len(overlap) / len(compare_bt) * 100
        print(f"\n  Backtest → live coverage:   {len(overlap)}/{len(compare_bt)} ({coverage:.0f}%)")
    if live:
        live_covered = len(overlap) / len(live) * 100
        print(f"  Live → backtest coverage:   {len(overlap)}/{len(live)} ({live_covered:.0f}%)")

    print(f"\n  ✗ Backtest-only (noise — {len(bt_only)}):")
    if bt_only:
        for sym in sorted(bt_only)[:15]:
            print(f"      {sym}")
        if len(bt_only) > 15:
            print(f"      ... and {len(bt_only) - 15} more")

    print(f"\n  ✗ Live-only (blind spot — {len(live_only)}):")
    if live_only:
        # Highlight actually-traded blind spots first — they matter more than
        # watchlist entries that may never fire a signal
        actually_traded = sorted(live_only & traded)
        watchlist_only = sorted(live_only - traded)
        for sym in actually_traded[:15]:
            print(f"      {sym}  [TRADED]")
        for sym in watchlist_only[:10]:
            print(f"      {sym}")
        remaining = len(actually_traded) + len(watchlist_only) - 25
        if remaining > 0:
            print(f"      ... and {remaining} more")

    if traded_blind:
        print(f"\n  ⚠ Symbols live traded but backtest can't evaluate: {len(traded_blind)}")
        print(f"    {sorted(traded_blind)[:20]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                        help="Count live trades from the last N days (default 90)")
    parser.add_argument("--markets", nargs="+", default=["US", "KR"],
                        help="Markets to compare (default: US KR)")
    args = parser.parse_args()

    for market in args.markets:
        narrow_bt, wide_bt = _backtest_symbols(market)
        watchlist, traded = await _live_symbols(market, args.days)
        _print_report(market, narrow_bt, wide_bt, watchlist, traded, args.days)


if __name__ == "__main__":
    asyncio.run(main())
