"""Snapshot SignalQualityTracker state from the live trades DB.

The live SignalQualityTracker is purely in-memory (no DB persistence), so
on every backend restart it loses all accumulated trade history. Backtests
that use a fresh tracker therefore start cold while live runs against
6+ months of accumulated history — leading to different gating decisions
and Kelly sizing.

This script reads completed sells from the trades table, derives
(strategy, symbol, return_pct, timestamp) records, and writes a JSON
snapshot that backtests can load via
``PipelineConfig.signal_quality_seed_path``.

Usage:
    cd backend && ../venv/bin/python scripts/snapshot_signal_quality.py
    # → data/signal_quality_snapshot.json
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from db.session import get_session_factory
from db.trade_repository import TradeRepository
from analytics.signal_quality import SignalQualityTracker


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "signal_quality_snapshot.json"


async def build_snapshot(out_path: Path, max_history: int = 5000) -> int:
    """Read closed sells from the trades table and serialize a tracker.

    Args:
        out_path: Where to write the JSON snapshot.
        max_history: Max trades to fetch (most recent first).

    Returns:
        Total trade records ingested into the tracker.
    """
    tracker = SignalQualityTracker()
    SessionFactory = get_session_factory()

    async with SessionFactory() as session:
        repo = TradeRepository(session)
        # Pull a large recent window — repository sorts desc by created_at.
        orders = await repo.get_trade_history(limit=max_history, exclude_paper=True)

    # Filter to filled SELLs that have a pnl_pct (= a complete trade)
    records: list[dict] = []
    for o in orders:
        side = (o.side or "").upper()
        if side != "SELL":
            continue
        if o.status != "filled":
            continue
        if o.pnl_pct is None:
            continue
        # Attribution: prefer the original BUY strategy if present, else
        # fall back to the SELL strategy_name.
        strategy = o.buy_strategy or o.strategy_name
        if not strategy:
            continue
        # Strip role suffixes the SELL side appends (e.g. supertrend:profit_taking)
        strategy = strategy.split(":")[0]
        ts = o.created_at.timestamp() if o.created_at else 0.0
        records.append({
            "strategy": strategy,
            "symbol": o.symbol,
            "return_pct": float(o.pnl_pct) / 100.0,  # DB stores % (5.2 = 5.2%)
            "timestamp": ts,
        })

    n = tracker.seed_from_trades(records)

    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "source": "trades_db",
        "total_records": n,
        "tracker": tracker.to_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    # Print per-strategy summary so the operator can sanity-check
    print(f"\n✓ Wrote {n} trade records to {out_path}\n")
    print(f"{'strategy':<28} {'trades':>7} {'WR%':>6} {'PF':>6} {'avg_win':>9} {'avg_loss':>9}")
    print("-" * 75)
    for name in sorted(tracker._trades.keys()):
        m = tracker.get_metrics(name)
        print(f"{name:<28} {m.total_trades:>7d} {m.win_rate*100:>6.1f} "
              f"{m.profit_factor:>6.2f} {m.avg_win*100:>+8.2f}% {m.avg_loss*100:>+8.2f}%")
    return n


def main():
    out_path = DEFAULT_OUT
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    asyncio.run(build_snapshot(out_path))


if __name__ == "__main__":
    main()
