"""UTC datetime helpers.

`datetime.utcnow()` is deprecated since Python 3.12. The recommended
replacement `datetime.now(timezone.utc)` returns a timezone-AWARE
datetime, which breaks comparisons with our naive DB columns
(`Column(DateTime)` is timezone-naive by default).

`now_utc_naive()` returns the same naive-UTC value the legacy code
expects without the deprecation warning. Use this in production code
that writes / compares datetimes stored in the DB.

Use `now_utc()` for new code that wants the aware variant (e.g.
external-API payloads, log records).
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["now_utc", "now_utc_naive"]


def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def now_utc_naive() -> datetime:
    """Current time as a naive datetime expressing UTC.

    Matches the naive-UTC convention used by `Column(DateTime,
    default=datetime.utcnow)` throughout `core.models`.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
