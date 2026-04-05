"""Accounts API endpoint — list configured trading accounts.

Exposes a simple GET /accounts endpoint that returns account metadata
(excluding sensitive credentials like app_key/app_secret).

Also provides shared helpers used by positions.py, orders.py, and
portfolio.py to validate account_id parameters.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from config.accounts import load_accounts

router = APIRouter(prefix="/accounts", tags=["accounts"])
logger = logging.getLogger(__name__)

# Module-level cache — accounts config is static for the lifetime of the process.
_accounts_cache: list[dict] | None = None


def _get_accounts() -> list[dict]:
    """Return cached account list (excludes sensitive fields)."""
    global _accounts_cache
    if _accounts_cache is None:
        raw = load_accounts()
        _accounts_cache = [
            {
                "account_id": a.account_id,
                "name": a.name,
                "markets": a.markets,
                "is_paper": a.is_paper,
            }
            for a in raw
        ]
    return _accounts_cache


def is_valid_account_id(account_id: str) -> bool:
    """Return True when *account_id* matches a configured account."""
    return any(a["account_id"] == account_id for a in _get_accounts())


async def validate_account_id_or_404(
    account_id: Optional[str] = None,
) -> Optional[str]:
    """FastAPI dependency: validate account_id query param.

    Raises HTTP 404 when an unknown account_id is provided.
    Returns the account_id unchanged (or None when omitted).
    """
    if account_id is not None and not is_valid_account_id(account_id):
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' not found",
        )
    return account_id


@router.get("/")
async def list_accounts() -> list[dict]:
    """Return configured accounts (no sensitive credentials)."""
    return _get_accounts()
