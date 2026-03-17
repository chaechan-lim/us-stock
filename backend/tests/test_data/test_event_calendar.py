"""Tests for unified event calendar service."""

import pytest
from datetime import date, timedelta

from data.earnings_service import EarningsCalendarService, EarningsEvent
from data.macro_calendar import MacroCalendarService
from data.insider_service import InsiderTradingService, InsiderTransaction
from data.event_calendar import EventCalendarService


@pytest.fixture
def event_calendar():
    earnings = EarningsCalendarService(api_key="test")
    macro = MacroCalendarService()
    insider = InsiderTradingService(api_key="test")
    return EventCalendarService(earnings, macro, insider)


class TestShouldSkipBuy:
    def test_skip_on_fomc(self, event_calendar):
        skip, reason = event_calendar.should_skip_buy.__wrapped__(event_calendar, "AAPL") if hasattr(event_calendar.should_skip_buy, '__wrapped__') else (False, "")
        # Test with known FOMC date
        event_calendar.macro.should_block_buys = lambda t=None: (True, "FOMC Decision Day")
        skip, reason = event_calendar.should_skip_buy("AAPL")
        assert skip is True
        assert "FOMC" in reason

    def test_skip_on_earnings(self, event_calendar):
        # Mock macro to not block — isolate earnings logic from real calendar
        event_calendar.macro.should_block_buys = lambda t=None: (False, "")
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        event_calendar.earnings._cache = {
            "AAPL": [EarningsEvent(symbol="AAPL", date=tomorrow, hour="amc")],
        }
        skip, reason = event_calendar.should_skip_buy("AAPL")
        assert skip is True
        assert "earnings" in reason

    def test_no_skip_normal(self, event_calendar):
        # Override macro to not block
        event_calendar.macro.should_block_buys = lambda t=None: (False, "")
        skip, _ = event_calendar.should_skip_buy("MSFT")
        assert skip is False


class TestSizingMultiplier:
    def test_normal_day(self, event_calendar):
        event_calendar.macro.get_sizing_multiplier = lambda t=None: 1.0
        assert event_calendar.get_sizing_multiplier() == 1.0

    def test_cpi_day(self, event_calendar):
        event_calendar.macro.get_sizing_multiplier = lambda t=None: 0.5
        assert event_calendar.get_sizing_multiplier() == 0.5


class TestConfidenceAdjustment:
    def test_insider_boost(self, event_calendar):
        event_calendar.insider._cache = {
            "AAPL": [
                InsiderTransaction(
                    symbol="AAPL", name="CEO", share=100000,
                    change=5000, filing_date="2026-03-10",
                    transaction_date="2026-03-08",
                    transaction_code="P", transaction_price=150.0,
                ),
            ]
        }
        assert event_calendar.get_confidence_adjustment("AAPL") == 0.10

    def test_no_insider_data(self, event_calendar):
        assert event_calendar.get_confidence_adjustment("MSFT") == 0.0


class TestSlMultiplier:
    def test_earnings_near(self, event_calendar):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        event_calendar.earnings._cache = {
            "AAPL": [EarningsEvent(symbol="AAPL", date=tomorrow, hour="amc")],
        }
        assert event_calendar.get_sl_multiplier("AAPL") == 1.5

    def test_no_earnings(self, event_calendar):
        assert event_calendar.get_sl_multiplier("MSFT") is None


class TestToDict:
    def test_serialization(self, event_calendar):
        result = event_calendar.to_dict()
        assert "earnings" in result
        assert "macro" in result
        assert "insider" in result
