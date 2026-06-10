"""Tests for macro event calendar."""

from data.macro_calendar import MacroCalendarService, ALL_EVENTS


def test_fomc_count():
    # M17 (2026-06-07): 2026 + 2027 → 16 meetings × 2 days
    fomc = [e for e in ALL_EVENTS if e.event_type == "FOMC"]
    assert len(fomc) == 32


def test_cpi_count():
    # 12 (2026) + 12 (2027)
    cpi = [e for e in ALL_EVENTS if e.event_type == "CPI"]
    assert len(cpi) == 24


def test_jobs_count():
    jobs = [e for e in ALL_EVENTS if e.event_type == "JOBS"]
    assert len(jobs) == 24


def test_events_sorted():
    dates = [e.date for e in ALL_EVENTS]
    assert dates == sorted(dates)


class TestMacroCalendarService:
    def test_is_event_day_fomc(self):
        svc = MacroCalendarService()
        events = svc.is_event_day("2026-01-28")
        assert len(events) == 1
        assert events[0].event_type == "FOMC"

    def test_is_event_day_no_event(self):
        svc = MacroCalendarService()
        events = svc.is_event_day("2026-02-15")  # No event
        assert events == []

    def test_sizing_multiplier_fomc(self):
        svc = MacroCalendarService()
        assert svc.get_sizing_multiplier("2026-01-28") == 0.0

    def test_sizing_multiplier_cpi(self):
        svc = MacroCalendarService()
        assert svc.get_sizing_multiplier("2026-03-11") == 0.5

    def test_sizing_multiplier_jobs(self):
        svc = MacroCalendarService()
        assert svc.get_sizing_multiplier("2026-01-09") == 0.5

    def test_sizing_multiplier_normal(self):
        svc = MacroCalendarService()
        assert svc.get_sizing_multiplier("2026-02-15") == 1.0

    def test_should_block_buys_fomc(self):
        svc = MacroCalendarService()
        blocked, reason = svc.should_block_buys("2026-01-28")
        assert blocked is True
        assert "FOMC" in reason

    def test_should_block_buys_normal(self):
        svc = MacroCalendarService()
        blocked, _ = svc.should_block_buys("2026-02-15")
        assert blocked is False

    def test_should_block_buys_cpi(self):
        svc = MacroCalendarService()
        # CPI reduces sizing but doesn't block
        blocked, _ = svc.should_block_buys("2026-03-11")
        assert blocked is False

    def test_to_dict(self):
        svc = MacroCalendarService()
        result = svc.to_dict()
        assert isinstance(result, list)
        for item in result:
            assert "date" in item
            assert "event_type" in item
            assert "description" in item

    def test_custom_sizing_multiplier(self):
        svc = MacroCalendarService()
        svc.fomc_sizing_mult = 0.3  # Custom: reduced but not zero
        assert svc.get_sizing_multiplier("2026-01-28") == 0.3
