"""Positions API endpoint — account-scoped live positions.

GET /positions?account_id=X&market=ALL

- account_id: optional; if provided must match a configured account (→ 404 if not)
- market: "US" | "KR" | "ALL" (default "ALL")

For the current single-adapter deployment, all configured accounts share the
same market-data adapters, so the live position data returned is identical
regardless of account_id.  The parameter is validated for correctness and
kept for future per-account adapter routing.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request

from api.accounts import validate_account_id_or_404
from api.portfolio import _enrich_positions, _get_market_data

router = APIRouter(prefix="/positions", tags=["positions"])
logger = logging.getLogger(__name__)


@router.get("/")
async def get_positions(
    request: Request,
    market: str = "ALL",
    account_id: Optional[str] = Depends(validate_account_id_or_404),
) -> list[dict]:
    """Return live positions, optionally filtered by account and market.

    account_id is validated against configured accounts; unknown IDs return 404.
    market=ALL returns combined US + KR positions.
    """
    if market == "ALL":
        results: list[dict] = []
        for m in ("US", "KR"):
            md = _get_market_data(request, m)
            if not md:
                continue
            try:
                positions = await md.get_positions()
                results.extend(await _enrich_positions(positions, m, request))
            except Exception as e:
                logger.warning("Position fetch failed for %s market: %s", m, e)
        return results

    md = _get_market_data(request, market)
    if not md:
        return []

    try:
        positions = await md.get_positions()
        return await _enrich_positions(positions, market, request)
    except Exception as e:
        logger.warning("Position fetch failed for %s market: %s", market, e)
        return []
