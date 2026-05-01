"""Tests for HolidayCalendar — KR via KIS, US hardcoded NYSE."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from services.holiday_calendar import (
    HolidayCalendar,
    holiday_calendar,
    is_kr_holiday,
    is_us_holiday,
    next_kr_trading_day,
)


def test_kr_fallback_includes_labor_day_2026():
    assert is_kr_holiday(date(2026, 5, 1))


def test_kr_fallback_lunar_new_year():
    assert is_kr_holiday(date(2026, 2, 17))


def test_kr_normal_weekday_not_holiday():
    # 2026-05-04 Monday — regular trading day.
    assert not is_kr_holiday(date(2026, 5, 4))


def test_us_nyse_2026_holidays():
    assert is_us_holiday(date(2026, 1, 1))    # New Year
    assert is_us_holiday(date(2026, 5, 25))   # Memorial Day
    assert is_us_holiday(date(2026, 12, 25))  # Christmas


def test_us_nyse_2026_normal_day():
    assert not is_us_holiday(date(2026, 5, 4))


def test_us_juneteenth_observed_2027():
    # 2027-06-19 = Saturday → observed Friday 2027-06-18.
    assert is_us_holiday(date(2027, 6, 18))


def test_next_kr_trading_day_skips_holiday():
    # 2026-04-30 (Thu) → next is 2026-05-04 (Mon) since 5/1 is 근로자의 날
    # and 5/2-5/3 is weekend.
    assert next_kr_trading_day(date(2026, 4, 30)) == date(2026, 5, 4)


@pytest.mark.asyncio
async def test_refresh_kr_merges_with_fallback():
    cal = HolidayCalendar()
    fake_adapter = AsyncMock()
    # Simulate KIS returning a date the fallback doesn't have (future year).
    fake_adapter.fetch_kr_holidays.return_value = [date(2028, 1, 1)]

    n = await cal.refresh_kr(fake_adapter)
    assert n == 1
    assert cal.is_kr_holiday(date(2028, 1, 1))    # New from KIS
    assert cal.is_kr_holiday(date(2026, 5, 1))    # Fallback preserved


@pytest.mark.asyncio
async def test_refresh_kr_keeps_cache_on_failure():
    cal = HolidayCalendar()
    fake_adapter = AsyncMock()
    fake_adapter.fetch_kr_holidays.side_effect = RuntimeError("network down")

    n = await cal.refresh_kr(fake_adapter)
    assert n == 0
    # Fallback intact — refresh failure does not blank the cache.
    assert cal.is_kr_holiday(date(2026, 5, 1))


@pytest.mark.asyncio
async def test_refresh_kr_keeps_cache_on_empty():
    cal = HolidayCalendar()
    fake_adapter = AsyncMock()
    fake_adapter.fetch_kr_holidays.return_value = []

    n = await cal.refresh_kr(fake_adapter)
    assert n == 0
    assert cal.is_kr_holiday(date(2026, 5, 1))


def test_module_singleton_is_shared():
    # Sanity: the module-level helper hits the singleton, not a fresh one.
    assert holiday_calendar.is_kr_holiday(date(2026, 5, 1))
