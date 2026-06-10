"""Background asyncio task helpers.

RES-H7 (2026-06-06): every site doing `asyncio.create_task(coro)`
without retaining a reference is at risk of two failure modes:

1. PEP-correctness — without a strong ref the GC can cancel the
   task mid-await. _persist_funnel_event writes one DB row per
   rejected signal; a tick rejecting 40 signals spawns 40 detached
   tasks and the GC can drop any of them between yield points.
2. Silent exceptions — when a fire-and-forget task raises, the
   exception lands in the GC's `_GeneratorExit` and is reported
   only at interpreter shutdown.

`spawn(coro, name=...)` retains the ref in a module-level set and
attaches a done-callback that logs exceptions. Drop-in replacement
for `asyncio.create_task` in the engine, services, and main.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

_tasks: set[asyncio.Task[Any]] = set()


def _on_done(task: asyncio.Task[Any]) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    logger.error(
        "Background task %s raised: %s",
        task.get_name(),
        exc,
        exc_info=exc,
    )


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
    """Spawn a tracked background task. The task is retained in a
    module-level set until completion and any exception is logged.

    Use this instead of `asyncio.create_task(coro)` for fire-and-
    forget work.
    """
    task = asyncio.create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_on_done)
    return task


def pending_count() -> int:
    """Number of currently tracked background tasks. Useful for
    health endpoints + shutdown observation."""
    return len(_tasks)


async def drain(timeout: float = 15.0) -> int:
    """Wait for tracked tasks to finish on shutdown. Returns the
    number that completed within the timeout."""
    if not _tasks:
        return 0
    pending = list(_tasks)
    done, _still = await asyncio.wait(pending, timeout=timeout)
    return len(done)
