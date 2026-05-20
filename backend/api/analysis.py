"""Daily analysis API — serves artifacts written by
scripts/daily_post_market_analysis.py.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Repo root resolved relative to this file: backend/api/ → repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACT_DIR = _REPO_ROOT / "data" / "daily_analyses"


def _load_artifact(d: date) -> dict | None:
    path = _ARTIFACT_DIR / f"{d}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("daily analysis read failed for %s: %s", d, e)
        return None


@router.get("/daily")
async def daily_analysis(days: int = 7):
    """Return up to `days` most recent daily analysis artifacts.

    Newest first. Empty list when no artifacts exist (e.g. first run not
    yet completed).
    """
    if not _ARTIFACT_DIR.exists():
        return {"artifacts": [], "count": 0}

    if days < 1:
        days = 1
    if days > 90:
        days = 90

    today = date.today()
    out = []
    # Walk backwards up to days*2 (skipping weekends/missing days)
    for back in range(days * 2):
        d = today - timedelta(days=back)
        art = _load_artifact(d)
        if art is not None:
            out.append(art)
            if len(out) >= days:
                break
    return {"artifacts": out, "count": len(out)}


@router.get("/daily/latest")
async def daily_analysis_latest():
    """Return the single most recent artifact (or 404 if none)."""
    if not _ARTIFACT_DIR.exists():
        raise HTTPException(status_code=404, detail="no analysis artifacts yet")
    today = date.today()
    for back in range(14):
        d = today - timedelta(days=back)
        art = _load_artifact(d)
        if art is not None:
            return art
    raise HTTPException(status_code=404, detail="no analysis artifacts yet")
