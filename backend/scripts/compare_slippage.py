"""Slippage calibration — compare live fill prices to daily reference.

Current backtest slippage is a fixed per-market constant
(KR 0.08%, US 0.05%). This script checks whether live reality matches.

For each filled Order in the last N days:
    signed_slippage = (filled_price - typical_price) / typical_price
where typical_price = (high + low + close) / 3 from yfinance daily bars.

Interpretation:
    BUY orders:  positive slippage = bad (paid more than typical)
    SELL orders: negative slippage = bad (received less than typical)
    Symmetric signed slippage (ignoring side): |effective execution gap|

Reports median and p95 per market. The daily-bar reference is crude
(actual fills happen intraday, not at the day's typical price), so use
the output for directional calibration — not tight precision.

Usage:
    cd backend && python scripts/compare_slippage.py [--days 90]
"""

import argparse
import asyncio
import logging
import math
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd  # noqa
import yfinance as yf  # noqa
from db.session import get_session_factory  # noqa
from db.trade_repository import TradeRepository  # noqa


# Current backtest settings — anchor points we're trying to validate
CURRENT_SLIPPAGE_PCT = {
    "US": 0.05,
    "KR": 0.08,
}


def _yf_symbol(symbol: str, market: str) -> str:
    """Map internal symbol → yfinance ticker."""
    if market != "KR":
        return symbol
    # KR tickers on yfinance have .KS (KOSPI) or .KQ (KOSDAQ) suffix.
    # The DB symbol is bare; default to KS and fall back to KQ.
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        return symbol
    return f"{symbol}.KS"


async def _collect_fills(market: str, days: int) -> list[dict]:
    """Pull filled orders with price data from the last N days."""
    SessionFactory = get_session_factory()
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with SessionFactory() as session:
        repo = TradeRepository(session)
        trades = await repo.get_trade_history(limit=5000, exclude_paper=True)

    fills: list[dict] = []
    for t in trades:
        if t.market != market:
            continue
        if t.status != "filled":
            continue
        if not t.filled_price or t.filled_price <= 0:
            continue
        if not t.created_at or t.created_at < cutoff:
            continue
        fills.append({
            "symbol": t.symbol,
            "side": (t.side or "").upper(),
            "filled_price": float(t.filled_price),
            "date": t.created_at.date(),
        })
    return fills


def _fetch_reference_prices(
    symbols: set[str], market: str, date_start: datetime, date_end: datetime,
) -> dict[tuple[str, str], dict[str, float]]:
    """Batch-fetch yfinance daily OHLC keyed by (symbol, yyyy-mm-dd)."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not symbols:
        return out
    yf_tickers = [_yf_symbol(s, market) for s in symbols]
    # yf.download accepts comma-separated but batching avoids sparse 1-symbol calls
    try:
        df = yf.download(
            " ".join(yf_tickers),
            start=date_start.strftime("%Y-%m-%d"),
            end=(date_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        logger.warning("yfinance batch fetch failed: %s", e)
        return out

    # Multi-ticker returns column MultiIndex; single-ticker returns flat columns
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
    else:
        tickers = [yf_tickers[0]] if yf_tickers else []

    for yt in tickers:
        # Back-map yfinance ticker to internal symbol
        base = yt.replace(".KS", "").replace(".KQ", "")
        tdf = df[yt] if isinstance(df.columns, pd.MultiIndex) else df
        for idx, row in tdf.iterrows():
            try:
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) and v > 0 for v in (h, l, c)):
                continue
            key = (base, idx.strftime("%Y-%m-%d"))
            out[key] = {
                "high": h,
                "low": l,
                "close": c,
                "typical": (h + l + c) / 3,
            }
    return out


def _signed_slippage(fill: dict, reference: dict) -> float:
    """Signed slippage where positive = worse-than-reference."""
    ref = reference["typical"]
    if ref <= 0:
        return 0.0
    raw = (fill["filled_price"] - ref) / ref
    # Flip sign for SELL so "positive = bad" holds across both sides
    return raw if fill["side"] == "BUY" else -raw


def _p(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = int(round((len(sorted_v) - 1) * pct))
    return sorted_v[idx]


def _report(market: str, fills: list[dict], refs: dict) -> None:
    matched: list[tuple[float, dict]] = []
    for f in fills:
        key = (f["symbol"], f["date"].strftime("%Y-%m-%d"))
        ref = refs.get(key)
        if not ref:
            continue
        matched.append((_signed_slippage(f, ref), f))

    print(f"\n{'=' * 72}")
    print(f"  {market} slippage calibration")
    print(f"{'=' * 72}")
    print(f"  Fills collected:   {len(fills)}")
    print(f"  Matched to ref:    {len(matched)}")
    print(f"  Backtest setting:  {CURRENT_SLIPPAGE_PCT[market]:.2f}%")

    if not matched:
        print("  (no overlapping price data — skip)")
        return

    slippages_pct = [s * 100 for s, _ in matched]
    abs_pct = [abs(x) for x in slippages_pct]
    buy_pct = [s * 100 for s, f in matched if f["side"] == "BUY"]
    sell_pct = [s * 100 for s, f in matched if f["side"] == "SELL"]

    def _fmt(vals: list[float], label: str) -> None:
        if not vals:
            print(f"  {label:<15} (no data)")
            return
        print(
            f"  {label:<15} median={statistics.median(vals):+.3f}% "
            f"mean={statistics.mean(vals):+.3f}% "
            f"p95={_p([abs(v) for v in vals], 0.95):+.3f}% (|abs|) "
            f"n={len(vals)}"
        )

    print("")
    _fmt(slippages_pct, "All (signed)")
    _fmt(buy_pct, "BUY (signed)")
    _fmt(sell_pct, "SELL (signed)")
    _fmt(abs_pct, "|abs| all")

    # Recommendation: use p75 of |abs| as a realistic worst-case fill cost.
    # Median alone understates; p95 inflates from single-fill outliers.
    p75 = _p(abs_pct, 0.75)
    current = CURRENT_SLIPPAGE_PCT[market]
    gap = p75 - current
    print(
        f"\n  Observed p75 |slippage| = {p75:.3f}%"
        f" → current setting {current:.2f}% "
        f"({'close' if abs(gap) < 0.05 else ('too low' if gap > 0 else 'too high')}, "
        f"Δ {gap:+.2f}pp)"
    )

    # Worst fills — surface any blowup cases
    worst = sorted(matched, key=lambda m: abs(m[0]), reverse=True)[:5]
    print("\n  Top |slippage| fills:")
    for slip, f in worst:
        print(
            f"    {f['date']} {f['symbol']:>8} {f['side']:<4} "
            f"@{f['filled_price']:<10.2f} slip={slip * 100:+.2f}%"
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                        help="Look back N days (default 90)")
    parser.add_argument("--markets", nargs="+", default=["US", "KR"])
    args = parser.parse_args()

    for market in args.markets:
        print(f"\n▶ Collecting {market} fills (last {args.days}d)...")
        fills = await _collect_fills(market, args.days)
        print(f"  {len(fills)} filled orders")
        if not fills:
            continue

        dates = [f["date"] for f in fills]
        date_start = datetime.combine(min(dates), datetime.min.time()) - timedelta(days=1)
        date_end = datetime.combine(max(dates), datetime.min.time())
        symbols = {f["symbol"] for f in fills}
        print(f"  Fetching yfinance reference for {len(symbols)} symbols...")
        refs = _fetch_reference_prices(symbols, market, date_start, date_end)

        _report(market, fills, refs)


if __name__ == "__main__":
    asyncio.run(main())
