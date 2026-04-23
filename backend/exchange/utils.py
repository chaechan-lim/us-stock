"""Shared utilities for exchange adapters."""

from typing import Any


def safe_float(val: Any) -> float:
    """Safely convert a KIS API value to float.

    KIS endpoints can return "N/A", "-", or empty strings for
    fields during trading halts or special states.
    """
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0
