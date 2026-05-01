"""Market holiday calendar — KR via KIS API, US via hardcoded NYSE list.

The scheduler used to mark every weekday 09:00–15:30 KST as REGULAR, which
caused KR tasks to fire on 근로자의 날 / 추석 / 신정 etc. Engine wasted API
calls and log noise; orders would have been rejected by KIS anyway, but the
loop tried.

Design:
- Module-level `HolidayCalendar` singleton holds two `set[date]`s.
- Sync `is_kr_holiday(d)` / `is_us_holiday(d)` for the scheduler's hot path
  (called every loop tick — must not block on network).
- Async `refresh_kr(adapter)` pulls KIS 국내휴장일조회 (TR_ID CTCA0903R) and
  swaps the set in place. Called at startup + weekly from the scheduler.
- Hardcoded fallback bootstraps both sets so a cold start with no network
  still gets the right answer for known dates.
- US: hardcoded NYSE list. KIS has no clean overseas holiday endpoint.
  NYSE publishes years in advance; update yearly.

Early-close days (e.g. day after Thanksgiving 1pm close) are NOT modeled —
the market is still REGULAR for the morning, so the existing time-based
phase logic is correct enough.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


# --- Hardcoded fallbacks ----------------------------------------------------
# Updated 2026-05-01. Verify each Jan; KIS API refresh keeps KR honest at
# runtime, US must be updated manually.

_KR_HOLIDAYS_FALLBACK: set[date] = {
    date(2026, 1, 1),    # 신정
    date(2026, 2, 16),   # 설날 연휴
    date(2026, 2, 17),   # 설날
    date(2026, 2, 18),   # 설날 연휴
    date(2026, 3, 2),    # 삼일절 대체공휴일 (3/1 일)
    date(2026, 5, 1),    # 근로자의 날
    date(2026, 5, 5),    # 어린이날
    date(2026, 5, 25),   # 부처님오신날 대체공휴일 (5/24 일)
    date(2026, 8, 17),   # 광복절 대체공휴일 (8/15 토)
    date(2026, 9, 24),   # 추석 연휴
    date(2026, 9, 25),   # 추석
    date(2026, 10, 5),   # 개천절 대체공휴일 (10/3 토)
    date(2026, 10, 9),   # 한글날
    date(2026, 12, 25),  # 성탄절
    date(2026, 12, 31),  # 연말휴장
    # 2027 — extend yearly
    date(2027, 1, 1),
}

_US_HOLIDAYS: set[date] = {
    # NYSE 2026 — https://www.nyse.com/markets/hours-calendars
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day observed (7/4 = Sat)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # NYSE 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth observed (6/19 = Sat)
    date(2027, 7, 5),    # Independence Day observed (7/4 = Sun)
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # Christmas observed (12/25 = Sat)
}


class HolidayCalendar:
    """Singleton holding KR + US holiday sets. Sync read, async refresh."""

    def __init__(self) -> None:
        self._kr: set[date] = set(_KR_HOLIDAYS_FALLBACK)
        self._us: set[date] = set(_US_HOLIDAYS)
        self._kr_last_refresh: datetime | None = None

    def is_kr_holiday(self, d: date) -> bool:
        return d in self._kr

    def is_us_holiday(self, d: date) -> bool:
        return d in self._us

    @property
    def kr_holidays(self) -> set[date]:
        return set(self._kr)

    @property
    def us_holidays(self) -> set[date]:
        return set(self._us)

    @property
    def kr_last_refresh(self) -> datetime | None:
        return self._kr_last_refresh

    async def refresh_kr(self, adapter) -> int:
        """Pull KR holidays from KIS API and merge into the cache.

        Returns the number of holiday dates fetched. On any failure, leaves
        the existing set intact and logs a warning — the hardcoded fallback
        keeps the engine correct for known dates even if KIS is unreachable.
        """
        try:
            holidays = await adapter.fetch_kr_holidays()
        except Exception as e:
            logger.warning("KR holiday refresh failed (keeping cache): %s", e)
            return 0

        if not holidays:
            logger.warning("KR holiday refresh returned empty — keeping cache")
            return 0

        # Merge with hardcoded fallback so we never lose dates KIS may omit
        # for an off-by-one cycle (e.g. fetched window doesn't cover Jan 1).
        merged = set(_KR_HOLIDAYS_FALLBACK) | set(holidays)
        self._kr = merged
        self._kr_last_refresh = datetime.now()
        logger.info(
            "KR holiday cache refreshed: %d dates (%d from KIS)",
            len(merged), len(holidays),
        )
        return len(holidays)


# Module-level singleton — scheduler imports `holiday_calendar` directly.
holiday_calendar = HolidayCalendar()


def is_kr_holiday(d: date) -> bool:
    return holiday_calendar.is_kr_holiday(d)


def is_us_holiday(d: date) -> bool:
    return holiday_calendar.is_us_holiday(d)


def next_kr_trading_day(start: date) -> date:
    """Next KR weekday that is not a holiday. Cheap helper for logging/tests."""
    d = start
    for _ in range(14):
        d = d + timedelta(days=1)
        if d.weekday() < 5 and not is_kr_holiday(d):
            return d
    return d  # 14-day window exhausted; return whatever we have
