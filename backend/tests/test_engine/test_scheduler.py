"""Tests for TradingScheduler."""

import pytest
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock

from engine.scheduler import (
    TradingScheduler,
    MarketPhase,
    get_market_phase,
    get_kr_market_phase,
    is_market_open,
    TaskEntry,
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
