"""Tests for TradingScheduler."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from engine.scheduler import (
    MarketPhase,
    TaskEntry,
    TradingScheduler,
    get_kr_market_phase,
    get_market_phase,
    is_market_open,
)

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


def _make_dt(hour, minute=0, weekday=1):
    """Create a datetime for testing. weekday: 0=Mon, 6=Sun."""
    # We create a Tuesday (weekday=1) in 2024 for testing
    # Jan 2, 2024 is a Tuesday
    from datetime import timedelta
    base = datetime(2024, 1, 2, hour, minute, tzinfo=ET)  # Tuesday
    # Adjust to desired weekday
    diff = weekday - base.weekday()
    return base + timedelta(days=diff)


def test_market_phase_premarket():
    dt = _make_dt(7, 0, weekday=1)  # 7:00 AM ET, Tuesday
    assert get_market_phase(dt) == MarketPhase.PRE_MARKET


def test_market_phase_regular():
    dt = _make_dt(10, 0, weekday=1)  # 10:00 AM ET, Tuesday
    assert get_market_phase(dt) == MarketPhase.REGULAR


def test_market_phase_regular_open():
    dt = _make_dt(9, 30, weekday=1)  # 9:30 AM ET exactly
    assert get_market_phase(dt) == MarketPhase.REGULAR


def test_market_phase_after_hours():
    dt = _make_dt(17, 0, weekday=1)  # 5:00 PM ET
    assert get_market_phase(dt) == MarketPhase.AFTER_HOURS


def test_market_phase_closed_night():
    dt = _make_dt(22, 0, weekday=1)  # 10:00 PM ET
    assert get_market_phase(dt) == MarketPhase.CLOSED


def test_market_phase_weekend():
    dt = _make_dt(10, 0, weekday=5)  # Saturday 10 AM
    assert get_market_phase(dt) == MarketPhase.CLOSED


def test_is_market_open():
    open_dt = _make_dt(12, 0, weekday=1)
    closed_dt = _make_dt(22, 0, weekday=1)
    assert is_market_open(open_dt) is True
    assert is_market_open(closed_dt) is False


# ── KR market phase tests ──────────────────────────────────────────

def _make_kst(hour, minute=0, weekday=1):
    """Create a KST datetime for testing."""
    from datetime import timedelta
    base = datetime(2024, 1, 2, hour, minute, tzinfo=KST)  # Tuesday
    diff = weekday - base.weekday()
    return base + timedelta(days=diff)


def test_kr_market_phase_premarket():
    dt = _make_kst(8, 30, weekday=1)  # 8:30 KST
    assert get_kr_market_phase(dt) == MarketPhase.PRE_MARKET


def test_kr_market_phase_regular_open():
    dt = _make_kst(9, 0, weekday=1)  # 9:00 KST exactly
    assert get_kr_market_phase(dt) == MarketPhase.REGULAR


def test_kr_market_phase_regular_mid():
    dt = _make_kst(12, 0, weekday=1)  # 12:00 KST
    assert get_kr_market_phase(dt) == MarketPhase.REGULAR


def test_kr_market_phase_regular_close():
    dt = _make_kst(15, 29, weekday=1)  # 15:29 KST
    assert get_kr_market_phase(dt) == MarketPhase.REGULAR


def test_kr_market_phase_after_hours():
    dt = _make_kst(16, 0, weekday=1)  # 16:00 KST
    assert get_kr_market_phase(dt) == MarketPhase.AFTER_HOURS


def test_kr_market_phase_closed_night():
    dt = _make_kst(20, 0, weekday=1)  # 20:00 KST
    assert get_kr_market_phase(dt) == MarketPhase.CLOSED


def test_kr_market_phase_closed_early():
    dt = _make_kst(7, 0, weekday=1)  # 7:00 KST (before pre-market)
    assert get_kr_market_phase(dt) == MarketPhase.CLOSED


def test_kr_market_phase_weekend():
    dt = _make_kst(10, 0, weekday=5)  # Saturday
    assert get_kr_market_phase(dt) == MarketPhase.CLOSED


def test_task_entry_should_run():
    task = TaskEntry("test", AsyncMock(), interval_sec=60, phases=None)
    now = _make_dt(10, 0)
    assert task.should_run(now) is True

    task.last_run = now
    assert task.should_run(now) is False


def test_task_entry_phase_filter():
    task = TaskEntry(
        "regular_only", AsyncMock(), interval_sec=60,
        phases=[MarketPhase.REGULAR],
    )
    regular = _make_dt(10, 0, weekday=1)
    closed = _make_dt(22, 0, weekday=1)

    assert task.should_run(regular) is True
    assert task.should_run(closed) is False


def test_task_entry_kr_market_phase():
    """KR tasks use KR market phase detection."""
    task = TaskEntry(
        "kr_task", AsyncMock(), interval_sec=60,
        phases=[MarketPhase.REGULAR], market="KR",
    )
    # KST 10:00 = KR REGULAR (09:00-15:30)
    kr_regular = _make_kst(10, 0, weekday=1)
    assert task.should_run(kr_regular) is True

    # KST 20:00 = KR CLOSED
    kr_closed = _make_kst(20, 0, weekday=1)
    assert task.should_run(kr_closed) is False


def test_scheduler_add_task():
    scheduler = TradingScheduler()
    scheduler.add_task("test", AsyncMock(), interval_sec=60)
    assert "test" in scheduler.task_names
    assert len(scheduler.task_names) == 1


def test_scheduler_status():
    scheduler = TradingScheduler()
    scheduler.add_task("task1", AsyncMock(), 60, [MarketPhase.REGULAR])
    scheduler.add_task("task2", AsyncMock(), 120)

    status = scheduler.get_status()
    assert status["running"] is False
    assert "market_phase" in status
    assert len(status["tasks"]) == 2


@pytest.mark.asyncio
async def test_scheduler_start_stop():
    scheduler = TradingScheduler()
    scheduler._tick_interval = 0.1

    fn = AsyncMock()
    scheduler.add_task("always", fn, interval_sec=0)  # runs every tick

    task = asyncio.create_task(scheduler.start())
    await asyncio.sleep(0.3)
    assert scheduler.running is True

    await scheduler.stop()
    await asyncio.sleep(0.2)
    assert scheduler.running is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert fn.call_count >= 1


# ── Dynamic task registration/deregistration (Phase 4: STOCK-85) ─────────


def test_remove_task_existing():
    """remove_task() removes a registered task and the name is no longer listed."""
    scheduler = TradingScheduler()
    scheduler.add_task("task_a", AsyncMock(), interval_sec=60)
    scheduler.add_task("task_b", AsyncMock(), interval_sec=60)

    scheduler.remove_task("task_a")

    assert "task_a" not in scheduler.task_names
    assert "task_b" in scheduler.task_names


def test_remove_task_returns_true_when_found():
    """remove_task() returns True for an existing task."""
    scheduler = TradingScheduler()
    scheduler.add_task("task_x", AsyncMock(), interval_sec=60)
    assert scheduler.remove_task("task_x") is True


def test_remove_task_returns_false_when_not_found():
    """remove_task() returns False (and does not raise) for unknown task names."""
    scheduler = TradingScheduler()
    assert scheduler.remove_task("no_such_task") is False


def test_remove_task_does_not_affect_other_tasks():
    """Removing one task leaves all other registered tasks intact."""
    scheduler = TradingScheduler()
    for name in ("alpha", "beta", "gamma"):
        scheduler.add_task(name, AsyncMock(), interval_sec=60)

    scheduler.remove_task("beta")

    assert scheduler.task_names == ["alpha", "gamma"]


def test_remove_tasks_by_prefix_removes_matching():
    """remove_tasks_by_prefix() removes all tasks with the given prefix."""
    scheduler = TradingScheduler()
    scheduler.add_task("ACC001:US:evaluation_loop", AsyncMock(), interval_sec=300)
    scheduler.add_task("ACC001:US:position_check", AsyncMock(), interval_sec=60)
    scheduler.add_task("ACC001:KR:evaluation_loop", AsyncMock(), interval_sec=300, market="KR")
    scheduler.add_task("health_check", AsyncMock(), interval_sec=120)

    removed = scheduler.remove_tasks_by_prefix("ACC001:US:")

    assert removed == 2
    remaining = scheduler.task_names
    assert "ACC001:US:evaluation_loop" not in remaining
    assert "ACC001:US:position_check" not in remaining
    # Other markets and global tasks must survive
    assert "ACC001:KR:evaluation_loop" in remaining
    assert "health_check" in remaining


def test_remove_tasks_by_prefix_returns_zero_on_no_match(caplog):
    """remove_tasks_by_prefix() returns 0 and logs a warning when nothing matches."""
    import logging

    scheduler = TradingScheduler()
    scheduler.add_task("health_check", AsyncMock(), interval_sec=120)

    with caplog.at_level(logging.WARNING):
        removed = scheduler.remove_tasks_by_prefix("ACC999:")

    assert removed == 0
    assert scheduler.task_names == ["health_check"]
    assert "ACC999:" in caplog.text


def test_remove_tasks_by_prefix_raises_on_empty_prefix():
    """remove_tasks_by_prefix() raises ValueError for an empty prefix.

    An empty prefix matches every task name and would silently wipe the
    entire task list — a serious operational risk on a live trading system.
    """
    scheduler = TradingScheduler()
    scheduler.add_task("health_check", AsyncMock(), interval_sec=120)
    scheduler.add_task("evaluation_loop", AsyncMock(), interval_sec=300)

    with pytest.raises(ValueError, match="empty"):
        scheduler.remove_tasks_by_prefix("")

    # The task list must be completely intact after the failed call
    assert scheduler.task_count == 2


def test_task_count_property():
    """task_count reflects the current number of registered tasks."""
    scheduler = TradingScheduler()
    assert scheduler.task_count == 0

    scheduler.add_task("t1", AsyncMock(), interval_sec=60)
    assert scheduler.task_count == 1

    scheduler.add_task("t2", AsyncMock(), interval_sec=60)
    assert scheduler.task_count == 2

    scheduler.remove_task("t1")
    assert scheduler.task_count == 1


@pytest.mark.asyncio
async def test_remove_task_while_running_is_safe():
    """Removing a task while the scheduler is running does not crash the loop.

    The removed task must not be called after removal; the remaining task
    continues to execute normally.
    """
    scheduler = TradingScheduler()
    scheduler._tick_interval = 0.05  # fast for testing

    fn_keep = AsyncMock()
    fn_remove = AsyncMock()

    scheduler.add_task("keep", fn_keep, interval_sec=0)
    scheduler.add_task("remove_me", fn_remove, interval_sec=0)

    sched_task = asyncio.create_task(scheduler.start())

    # Let both tasks execute for a couple of ticks
    await asyncio.sleep(0.15)
    calls_before = fn_remove.call_count
    assert calls_before >= 1

    # Remove one task while running
    assert scheduler.remove_task("remove_me") is True

    # Let the scheduler run for a few more ticks
    await asyncio.sleep(0.15)

    # fn_remove must not have been called after removal
    assert fn_remove.call_count == calls_before
    # fn_keep continues to be called
    assert fn_keep.call_count > calls_before

    await scheduler.stop()
    sched_task.cancel()
    try:
        await sched_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_add_task_while_running_is_picked_up():
    """A task added while the scheduler is running is picked up on the next tick."""
    scheduler = TradingScheduler()
    scheduler._tick_interval = 0.05

    fn_late = AsyncMock()

    sched_task = asyncio.create_task(scheduler.start())

    # Start with zero tasks, let the scheduler spin for a tick
    await asyncio.sleep(0.08)
    assert fn_late.call_count == 0

    # Dynamically register a task
    scheduler.add_task("late_arrival", fn_late, interval_sec=0)

    # Give the scheduler a couple of ticks to pick it up
    await asyncio.sleep(0.15)
    assert fn_late.call_count >= 1

    await scheduler.stop()
    sched_task.cancel()
    try:
        await sched_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_multi_account_tasks_isolated():
    """Tasks for multiple accounts run independently.

    Simulate two-account concurrent execution: each account's tasks execute
    without interfering with the other's task set.
    """
    scheduler = TradingScheduler()
    scheduler._tick_interval = 0.05

    fn_acc1 = AsyncMock()
    fn_acc2 = AsyncMock()

    # Register tasks using account-prefixed names
    scheduler.add_task("ACC001:US:evaluation_loop", fn_acc1, interval_sec=0)
    scheduler.add_task("ACC002:US:evaluation_loop", fn_acc2, interval_sec=0)

    sched_task = asyncio.create_task(scheduler.start())
    await asyncio.sleep(0.2)

    # Both account tasks must have executed
    assert fn_acc1.call_count >= 1
    assert fn_acc2.call_count >= 1

    # Remove ACC001 tasks — ACC002 must continue running
    removed = scheduler.remove_tasks_by_prefix("ACC001:")
    assert removed == 1

    calls_acc1_before = fn_acc1.call_count
    await asyncio.sleep(0.2)

    assert fn_acc1.call_count == calls_acc1_before  # ACC001 stopped
    assert fn_acc2.call_count > calls_acc1_before    # ACC002 continues

    await scheduler.stop()
    sched_task.cancel()
    try:
        await sched_task
    except asyncio.CancelledError:
        pass
