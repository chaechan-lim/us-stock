"""Seed a SignalQualityTracker from the live trades DB.

The tracker is in-memory only — without seeding, every backend restart
starts Kelly sizing and gating decisions blind. This helper reads filled
SELLs from the orders table, attributes each to the entry strategy
(buy_strategy first, falling back to the SELL's strategy_name), and
appends the records to a tracker.

Called during engine init in main.py so both US and KR evaluation loops
get a tracker carrying live history at startup.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from analytics.signal_quality import SignalQualityTracker
from db.trade_repository import TradeRepository

logger = logging.getLogger(__name__)


async def seed_tracker_from_db(
    tracker: SignalQualityTracker,
    session_factory: Callable[[], Awaitable],
    *,
    market: str | None = None,
    max_history: int = 5000,
    min_trades_per_strategy: int = 30,
) -> int:
    """Seed `tracker` from filled SELLs in the orders table.

    Args:
        tracker: SignalQualityTracker to mutate (records are appended).
        session_factory: async sessionmaker (callable returning a context
            manager that yields a session).
        market: Optional 'US' or 'KR' filter — when set, only orders for
            that market are seeded.
        max_history: Cap on rows pulled from DB (most recent first).
        min_trades_per_strategy: Strategies with fewer than this many
            qualifying SELLs are skipped — Kelly sizing on small samples
            is noise (e.g. PF 10 from 3 trades). Default 30 was chosen
            after the A/B backtest at compare_signal_quality_seed.py
            showed that seeding all strategies regressed 2y Ret by 2.9pp
            because tiny-sample strategies got over-sized.

    Returns:
        Count of records ingested. Zero on empty DB or no closed SELLs.
    """
    async with session_factory() as session:
        repo = TradeRepository(session)
        orders = await repo.get_trade_history(limit=max_history, exclude_paper=True)

    records: list[dict] = []
    for o in orders:
        if (o.side or "").upper() != "SELL":
            continue
        if o.status != "filled":
            continue
        if o.pnl_pct is None:
            continue
        if market and o.market != market:
            continue
        # Attribute to the BUY-side strategy when present (correct for
        # round-trip P&L); fall back to the SELL's own strategy_name.
        strategy = o.buy_strategy or o.strategy_name
        if not strategy:
            continue
        # Strip role suffix the SELL appends (e.g. "supertrend:profit_taking")
        strategy = strategy.split(":")[0]
        ts = o.created_at.timestamp() if o.created_at else 0.0
        records.append({
            "strategy": strategy,
            "symbol": o.symbol,
            # DB stores pnl_pct as percent (5.2 = 5.2%); tracker wants fraction.
            "return_pct": float(o.pnl_pct) / 100.0,
            "timestamp": ts,
        })

    # Filter: drop strategies below the min-sample threshold. Their Kelly
    # stats would be noise (e.g. 3 wins/0 losses → PF=∞ → over-sized).
    if min_trades_per_strategy > 1:
        counts: dict[str, int] = {}
        for r in records:
            counts[r["strategy"]] = counts.get(r["strategy"], 0) + 1
        dropped = {s for s, c in counts.items() if c < min_trades_per_strategy}
        if dropped:
            records = [r for r in records if r["strategy"] not in dropped]
            logger.info(
                "SignalQualityTracker seed: dropped %d under-sampled strategies "
                "(<%d trades): %s",
                len(dropped), min_trades_per_strategy, sorted(dropped),
            )

    n = tracker.seed_from_trades(records)
    if n > 0:
        logger.info(
            "SignalQualityTracker seeded with %d records "
            "(market=%s) across %d strategies",
            n, market or "ALL", len(tracker._trades),
        )
    else:
        logger.info(
            "SignalQualityTracker seed empty (market=%s) — DB has no closed SELLs "
            "with ≥%d trades per strategy",
            market or "ALL", min_trades_per_strategy,
        )
    return n
