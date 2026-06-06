"""M13 (2026-06-07): TTL-based log dedup for warning storms.

Hot-loop warnings ("Strategy X failed on Y", "Real-time price fetch
failed", etc.) fire per-symbol-per-cycle. With 17 strategies × 20
symbols × per-minute eval, a single sustained outage can produce
~20k duplicate lines per hour and bury everything else in journal.

This is a tiny in-process TTL cache: same `key` within `seconds`
collapses to a single emission; once the window expires the next
call logs normally.

Process-local on purpose — multi-process dedup would need Redis and
the trade-off (network call inside every warning path) isn't worth
it for an ergonomic improvement.
"""

from __future__ import annotations

import logging
import time
from threading import Lock

_LOCK = Lock()
_last_seen: dict[str, float] = {}


def warn_once_per(
    log: logging.Logger,
    key: str,
    seconds: float,
    msg: str,
    *args,
    **kwargs,
) -> bool:
    """logger.warning(msg, *args) iff `key` hasn't been seen in `seconds`.

    Returns True when the line was actually emitted, False when
    suppressed. `exc_info=True` and other kwargs forward to .warning().
    """
    now = time.monotonic()
    with _LOCK:
        prev = _last_seen.get(key)
        if prev is not None and (now - prev) < seconds:
            return False
        _last_seen[key] = now
    log.warning(msg, *args, **kwargs)
    return True


def reset() -> None:
    """Drop the dedup cache. Used by tests and process restarts."""
    with _LOCK:
        _last_seen.clear()


def cache_size() -> int:
    with _LOCK:
        return len(_last_seen)
