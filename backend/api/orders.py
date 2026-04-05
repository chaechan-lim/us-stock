"""Orders API endpoint — account-scoped order/trade history from DB.

GET /orders?account_id=X&market=ALL&limit=50&offset=0

- account_id: optional; if provided must match a configured account (→ 404 if not)
- market: "US" | "KR" | "ALL" (default "ALL")
- limit/offset: pagination (default 50, max 200)

Reads directly from the DB via TradeRepository. Borrows the session factory
from api.trades (set by main.py lifespan via init_trades()).
"""

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query

from api.accounts import validate_account_id_or_404
from api.trades import order_to_dict
from db.trade_repository import TradeRepository

router = APIRouter(prefix="/orders", tags=["orders"])
logger = logging.getLogger(__name__)


@router.get("/")
async def get_orders(
    account_id: Optional[str] = Depends(validate_account_id_or_404),
    market: Optional[Literal["US", "KR", "ALL"]] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """Return order history from DB, optionally filtered by account and market.

    account_id is validated against configured accounts; unknown IDs return 404.
    market filter applies post-query (Python-level) on dict representation.
    """
    from api.trades import _session_factory

    if not _session_factory:
        logger.warning(
            "GET /orders called but _session_factory is not initialised "
            "(init_trades() not yet called); returning empty list"
        )
        return []

    try:
        async with _session_factory() as session:
            repo = TradeRepository(session)
            orders = await repo.get_trade_history(
                limit=limit,
                offset=offset,
                account_id=account_id,
            )
            # Materialise to plain dicts while the session is still open.
            # SQLAlchemy async ORM objects become detached after session close;
            # accessing attributes outside the session raises MissingGreenlet
            # for any lazily-loaded column or relationship.
            order_dicts = [order_to_dict(o) for o in orders]

        # Apply optional market filter on plain dicts (safe after session close)
        if market and market != "ALL":
            order_dicts = [o for o in order_dicts if o.get("market") == market]

        return order_dicts
    except Exception as e:
        logger.warning("Failed to fetch orders from DB: %s", e)
        return []
