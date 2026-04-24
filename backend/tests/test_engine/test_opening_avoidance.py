"""Unit tests for engine.scheduler.is_opening_minutes.

The BUY suppression during the first N minutes after regular open
(2026-04-24 live-only tweak) depends on this helper being correct
across DST boundaries, weekends, and both markets.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from engine.scheduler import is_opening_minutes

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


class TestUSOpeningMinutes:
    def test_open_plus_5_min_true(self):
        assert is_opening_minutes("US", 30, datetime(2026, 4, 24, 9, 35, tzinfo=ET))

    def test_exact_open_true(self):
        assert is_opening_minutes("US", 30, datetime(2026, 4, 24, 9, 30, tzinfo=ET))

    def test_open_plus_29_min_true(self):
        assert is_opening_minutes("US", 30, datetime(2026, 4, 24, 9, 59, tzinfo=ET))

    def test_open_plus_30_min_false(self):
        assert not is_opening_minutes("US", 30, datetime(2026, 4, 24, 10, 0, tzinfo=ET))

    def test_pre_market_false(self):
        assert not is_opening_minutes("US", 30, datetime(2026, 4, 24, 9, 15, tzinfo=ET))

    def test_custom_window_10_min(self):
        at_5m = datetime(2026, 4, 24, 9, 35, tzinfo=ET)
        at_15m = datetime(2026, 4, 24, 9, 45, tzinfo=ET)
        assert is_opening_minutes("US", 10, at_5m)
        assert not is_opening_minutes("US", 10, at_15m)

    def test_weekend_false(self):
        assert not is_opening_minutes("US", 30, datetime(2026, 4, 25, 9, 35, tzinfo=ET))


class TestKROpeningMinutes:
    def test_open_plus_15_min_true(self):
        assert is_opening_minutes("KR", 30, datetime(2026, 4, 24, 9, 15, tzinfo=KST))

    def test_open_plus_45_min_false(self):
        assert not is_opening_minutes("KR", 30, datetime(2026, 4, 24, 9, 45, tzinfo=KST))

    def test_pre_market_false(self):
        assert not is_opening_minutes("KR", 30, datetime(2026, 4, 24, 8, 30, tzinfo=KST))


class TestZeroWindowDisabled:
    def test_minutes_zero_always_false(self):
        """opening_avoidance_minutes=0 in yaml should disable the check entirely."""
        # inside regular session
        assert not is_opening_minutes(
            "US", 0, datetime(2026, 4, 24, 9, 35, tzinfo=ET)
        )
