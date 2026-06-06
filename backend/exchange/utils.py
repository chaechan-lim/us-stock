"""Shared utilities for exchange adapters."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# RISK-H15 (2026-06-06): count silent coercion failures so silent
# data loss is at least visible from /health. Each failure used to
# return 0.0 with no log, feeding RiskManager equity / exposure /
# sizing math zeros it couldn't distinguish from genuine zeros.
_coerce_failures: int = 0


def coerce_failure_count() -> int:
    """Return the number of safe_float / safe_int coercion failures
    since process start. Drop on the /health endpoint to surface
    silent data corruption."""
    return _coerce_failures


def _record_failure(raw: Any, field: str | None) -> None:
    global _coerce_failures
    _coerce_failures += 1
    logger.warning(
        "safe_float: failed to coerce %r%s — using 0.0 fallback "
        "(this is now visible via /health.coerce_failures)",
        raw,
        f" (field={field})" if field else "",
    )


def safe_float(val: Any, field: str | None = None) -> float:
    """Safely convert a KIS API value to float.

    KIS endpoints can return "N/A", "-", or empty strings for
    fields during trading halts or special states.

    RISK-H15: silently returning 0.0 used to feed corrupted equity /
    exposure numbers into RiskManager. We still return 0.0 (changing
    the contract would ripple too far) but now log a warning and
    increment a counter that /health surfaces.
    """
    if val is None or val == "":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        _record_failure(val, field)
        return 0.0


def safe_int(val: Any, field: str | None = None) -> int:
    """Companion to safe_float for integer KIS fields (quantities)."""
    if val is None or val == "":
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        _record_failure(val, field)
        return 0
