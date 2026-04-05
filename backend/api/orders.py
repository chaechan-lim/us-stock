"""Orders API endpoint — account-scoped order/trade history from DB.

GET /orders?account_id=X&market=ALL&limit=50&offset=0

- account_id: optional; if provided must match a configured account (→ 404 if not)
- market: "US" | "KR" | "ALL" (default "ALL")
- limit/offset: pagination (default 50, max 200)

Reads directly from the DB via TradeRepository. Borrows the session factory
from api.trades (set by main.py lifespan via init_trades()).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.accounts import validate_account_id_or_404
from api.trades import _order_to_dict
from db.trade_repository import TradeRepository

router = APIRouter(prefix="/orders", tags=["orders"])
logger = logging.getLogger(__name__)


@router.get("/")
async def get_orders(
    account_id: Optional[str] = Depends(validate_account_id_or_404),
    market: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """Return order history from DB, optionally filtered by account and market.

    account_id is validated against configured accounts; unknown IDs return 404.
    market filter applies post-query (Python-level) to avoid DB schema assumptions.
    """
    from api.trades import _session_factory

    if not _session_factory:
        return []

    try:
        async with _session_factory() as session:
            repo = TradeRepository(session)
            orders = await repo.get_trade_history(
                limit=limit,
                offset=offset,
                account_id=account_id,
            )

        # Apply optional market filter
        if market and market != "ALL":
            orders = [o for o in orders if getattr(o, "market", "US") == market]

        return [_order_to_dict(o) for o in orders]
    except Exception as e:
        logger.warning("Failed to fetch orders from DB: %s", e)
        return []
