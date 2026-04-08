"""Snapshot the live UniverseExpander output for use in backtests.

The live trading engine builds its universe dynamically each day from
yfinance screeners + KIS ranking APIs + sector ETF holdings. Backtests
that use the static DEFAULT_UNIVERSE miss the small/mid-cap segment the
live system actually trades, which biases their alpha estimates.

This script runs UniverseExpander.expand() once and writes the result to
data/universe_snapshot.json (or a custom path) so backtests can load it
via PipelineConfig.universe_path.

Usage:
    cd backend && ../venv/bin/python scripts/snapshot_universe.py
    # → data/universe_snapshot_us.json (US, default)

    cd backend && ../venv/bin/python scripts/snapshot_universe.py --market KR
    # → data/universe_snapshot_kr.json

Survivorship caveat: snapshots are point-in-time. A 2-year backtest using
today's snapshot still has survivorship bias — the symbols in today's
"day_gainers" screener may not have existed or been ranked highly two
years ago. This is a fundamental limitation of any static universe;
mitigate by re-snapshotting periodically and re-running backtests.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from scanner.universe_expander import UniverseExpander, KRUniverseExpander
from scanner.etf_universe import ETFUniverse
from scanner.sector_analyzer import SectorAnalyzer


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data"


async def snapshot_us(out_path: Path) -> int:
    expander = UniverseExpander(
        etf_universe=ETFUniverse(),
        sector_analyzer=SectorAnalyzer(),
        # Live engine uses kis_adapter + rate_limiter; snapshot runs without
        # them so KIS ranking sources will be skipped (yfinance + sectors only).
        kis_adapter=None,
        rate_limiter=None,
        max_per_screener=15,
        max_total=120,
    )
    result = await expander.expand()
    payload = {
        "market": "US",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "max_total": expander._max_total,
        "total_discovered": result.total_discovered,
        "symbols": result.symbols,
        "etf_symbols": result.etf_symbols,
        "sources": {k: v[:50] for k, v in result.sources.items()},  # truncate per-source for readability
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return len(result.symbols)


async def snapshot_kr(out_path: Path) -> int:
    expander = KRUniverseExpander(
        kis_adapter=None,
        rate_limiter=None,
    )
    result = await expander.expand_kr()
    payload = {
        "market": "KR",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_discovered": result.total_discovered,
        "symbols": result.symbols,
        "sources": {k: v[:50] for k, v in result.sources.items()},
        "exchange_map": result.exchange_map,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return len(result.symbols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["US", "KR"], default="US")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: data/universe_snapshot_<market>.json)")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = DEFAULT_OUT_DIR / f"universe_snapshot_{args.market.lower()}.json"

    if args.market == "US":
        n = asyncio.run(snapshot_us(out_path))
    else:
        n = asyncio.run(snapshot_kr(out_path))

    print(f"✓ Wrote {n} symbols to {out_path}")


if __name__ == "__main__":
    main()
