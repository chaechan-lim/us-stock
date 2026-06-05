"""Tests for core.timeutil — UTC helpers replacing datetime.utcnow()."""

from datetime import datetime, timezone

import pytest

from core.timeutil import now_utc, now_utc_naive


def test_now_utc_returns_aware_datetime():
    ts = now_utc()
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None
    assert ts.utcoffset() == timezone.utc.utcoffset(ts)


def test_now_utc_naive_returns_naive_datetime():
    ts = now_utc_naive()
    assert isinstance(ts, datetime)
    assert ts.tzinfo is None


def test_now_utc_and_naive_are_consistent():
    """Aware and naive helpers must agree on the wall-clock time."""
    aware = now_utc()
    naive = now_utc_naive()
    # Strip tz from aware so they compare apples-to-apples.
    aware_naive = aware.replace(tzinfo=None)
    # Within a millisecond — the two helpers each call datetime.now() so
    # there's a tiny gap. 1ms is plenty of headroom.
    assert abs((aware_naive - naive).total_seconds()) < 0.001


def test_naive_matches_legacy_utcnow_semantics():
    """now_utc_naive() must be a drop-in for datetime.utcnow().

    The deprecated utcnow() returned a naive datetime whose wall-clock
    components were UTC. Anything written to our DB columns
    (Column(DateTime, default=now_utc_naive)) must round-trip identically
    to what datetime.utcnow() would have produced.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = datetime.utcnow()
    naive = now_utc_naive()
    assert legacy.tzinfo == naive.tzinfo  # both None
    # Same wall-clock time (within tiny scheduling gap)
    assert abs((legacy - naive).total_seconds()) < 0.001


def test_now_utc_naive_used_as_sqlalchemy_default():
    """Verify the helper is callable (SQLAlchemy passes the callable as
    default and invokes it per-row). A non-callable would silently make
    every row share one timestamp."""
    assert callable(now_utc_naive)
    a = now_utc_naive()
    # Second call must not return the same object reference.
    b = now_utc_naive()
    assert a is not b
