"""News sentiment API endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(prefix="/news", tags=["news"])
logger = logging.getLogger(__name__)

# Cached sentiment state — updated by scheduler task
_last_summary: dict | None = None
_last_signals: list[dict] = []
_last_updated: str | None = None

# KR cached sentiment state
_kr_last_summary: dict | None = None
_kr_last_signals: list[dict] = []
_kr_last_updated: str | None = None

_empty_summary = {
    "symbol_sentiments": {},
    "sector_sentiments": {},
    "market_sentiment": 0.0,
    "actionable_count": 0,
    "analyzed_count": 0,
}


def update_sentiment_cache(
    summary_dict: dict,
    signals: list[dict],
) -> None:
    """Called from scheduler task to cache latest US sentiment results."""
    global _last_summary, _last_signals, _last_updated
    _last_summary = summary_dict
    _last_signals = signals
    _last_updated = datetime.now(timezone.utc).isoformat()


def update_kr_sentiment_cache(
    summary_dict: dict,
    signals: list[dict],
) -> None:
    """Called from scheduler task to cache latest KR sentiment results."""
    global _kr_last_summary, _kr_last_signals, _kr_last_updated
    _kr_last_summary = summary_dict
    _kr_last_signals = signals
    _kr_last_updated = datetime.now(timezone.utc).isoformat()


@router.get("/sentiment")
async def get_sentiment():
    """Get latest news sentiment analysis results (US + KR)."""
    return {
        "summary": _last_summary or _empty_summary,
        "signals": _last_signals,
        "updated_at": _last_updated,
        "kr": {
            "summary": _kr_last_summary or _empty_summary,
            "signals": _kr_last_signals,
            "updated_at": _kr_last_updated,
        },
    }
