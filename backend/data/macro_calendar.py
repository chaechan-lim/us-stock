"""Macro event calendar — FOMC, CPI, Jobs Report dates.

Hardcoded 2026 + 2027 calendar. No API calls needed.
Used by evaluation loop to reduce sizing on high-volatility event days.

M17 (2026-06-07): 2027 dates added. FOMC dates from the Fed's published
2027 schedule. CPI dates approximated (BLS publishes ~12 months out,
verify in Dec 2026). Jobs dates: first Friday of each month rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass
class MacroEvent:
    date: str  # "2026-01-28"
    event_type: str  # FOMC, CPI, JOBS
    description: str


# 2026 FOMC meeting dates (decision day = day 2)
_FOMC_2026 = [
    ("2026-01-27", "FOMC Meeting Day 1"),
    ("2026-01-28", "FOMC Decision Day"),
    ("2026-03-17", "FOMC Meeting Day 1"),
    ("2026-03-18", "FOMC Decision Day"),
    ("2026-05-05", "FOMC Meeting Day 1"),
    ("2026-05-06", "FOMC Decision Day"),
    ("2026-06-16", "FOMC Meeting Day 1"),
    ("2026-06-17", "FOMC Decision Day"),
    ("2026-07-28", "FOMC Meeting Day 1"),
    ("2026-07-29", "FOMC Decision Day"),
    ("2026-09-15", "FOMC Meeting Day 1"),
    ("2026-09-16", "FOMC Decision Day"),
    ("2026-11-03", "FOMC Meeting Day 1"),
    ("2026-11-04", "FOMC Decision Day"),
    ("2026-12-15", "FOMC Meeting Day 1"),
    ("2026-12-16", "FOMC Decision Day"),
]

# 2026 CPI release dates (BLS schedule, typically 2nd or 3rd week)
_CPI_2026 = [
    ("2026-01-14", "CPI Report (Dec 2025)"),
    ("2026-02-11", "CPI Report (Jan 2026)"),
    ("2026-03-11", "CPI Report (Feb 2026)"),
    ("2026-04-14", "CPI Report (Mar 2026)"),
    ("2026-05-12", "CPI Report (Apr 2026)"),
    ("2026-06-10", "CPI Report (May 2026)"),
    ("2026-07-14", "CPI Report (Jun 2026)"),
    ("2026-08-12", "CPI Report (Jul 2026)"),
    ("2026-09-11", "CPI Report (Aug 2026)"),
    ("2026-10-13", "CPI Report (Sep 2026)"),
    ("2026-11-12", "CPI Report (Oct 2026)"),
    ("2026-12-10", "CPI Report (Nov 2026)"),
]

# 2026 Jobs Report (Non-Farm Payrolls) — first Friday of each month
_JOBS_2026 = [
    ("2026-01-09", "Jobs Report (Dec 2025)"),
    ("2026-02-06", "Jobs Report (Jan 2026)"),
    ("2026-03-06", "Jobs Report (Feb 2026)"),
    ("2026-04-03", "Jobs Report (Mar 2026)"),
    ("2026-05-08", "Jobs Report (Apr 2026)"),
    ("2026-06-05", "Jobs Report (May 2026)"),
    ("2026-07-02", "Jobs Report (Jun 2026)"),
    ("2026-08-07", "Jobs Report (Jul 2026)"),
    ("2026-09-04", "Jobs Report (Aug 2026)"),
    ("2026-10-02", "Jobs Report (Sep 2026)"),
    ("2026-11-06", "Jobs Report (Oct 2026)"),
    ("2026-12-04", "Jobs Report (Nov 2026)"),
]

# 2027 FOMC meeting dates (Fed's published schedule, verify in Dec 2026)
_FOMC_2027 = [
    ("2027-01-26", "FOMC Meeting Day 1"),
    ("2027-01-27", "FOMC Decision Day"),
    ("2027-03-16", "FOMC Meeting Day 1"),
    ("2027-03-17", "FOMC Decision Day"),
    ("2027-04-27", "FOMC Meeting Day 1"),
    ("2027-04-28", "FOMC Decision Day"),
    ("2027-06-15", "FOMC Meeting Day 1"),
    ("2027-06-16", "FOMC Decision Day"),
    ("2027-07-27", "FOMC Meeting Day 1"),
    ("2027-07-28", "FOMC Decision Day"),
    ("2027-09-21", "FOMC Meeting Day 1"),
    ("2027-09-22", "FOMC Decision Day"),
    ("2027-11-02", "FOMC Meeting Day 1"),
    ("2027-11-03", "FOMC Decision Day"),
    ("2027-12-14", "FOMC Meeting Day 1"),
    ("2027-12-15", "FOMC Decision Day"),
]

# 2027 CPI release dates (BLS schedule, approximate — verify Dec 2026)
_CPI_2027 = [
    ("2027-01-13", "CPI Report (Dec 2026)"),
    ("2027-02-10", "CPI Report (Jan 2027)"),
    ("2027-03-10", "CPI Report (Feb 2027)"),
    ("2027-04-13", "CPI Report (Mar 2027)"),
    ("2027-05-12", "CPI Report (Apr 2027)"),
    ("2027-06-09", "CPI Report (May 2027)"),
    ("2027-07-13", "CPI Report (Jun 2027)"),
    ("2027-08-11", "CPI Report (Jul 2027)"),
    ("2027-09-14", "CPI Report (Aug 2027)"),
    ("2027-10-13", "CPI Report (Sep 2027)"),
    ("2027-11-10", "CPI Report (Oct 2027)"),
    ("2027-12-09", "CPI Report (Nov 2027)"),
]

# 2027 Jobs Report — first Friday of each month
_JOBS_2027 = [
    ("2027-01-08", "Jobs Report (Dec 2026)"),
    ("2027-02-05", "Jobs Report (Jan 2027)"),
    ("2027-03-05", "Jobs Report (Feb 2027)"),
    ("2027-04-02", "Jobs Report (Mar 2027)"),
    ("2027-05-07", "Jobs Report (Apr 2027)"),
    ("2027-06-04", "Jobs Report (May 2027)"),
    ("2027-07-02", "Jobs Report (Jun 2027)"),
    ("2027-08-06", "Jobs Report (Jul 2027)"),
    ("2027-09-03", "Jobs Report (Aug 2027)"),
    ("2027-10-01", "Jobs Report (Sep 2027)"),
    ("2027-11-05", "Jobs Report (Oct 2027)"),
    ("2027-12-03", "Jobs Report (Nov 2027)"),
]


def _build_events() -> list[MacroEvent]:
    events = []
    for d, desc in _FOMC_2026 + _FOMC_2027:
        events.append(MacroEvent(date=d, event_type="FOMC", description=desc))
    for d, desc in _CPI_2026 + _CPI_2027:
        events.append(MacroEvent(date=d, event_type="CPI", description=desc))
    for d, desc in _JOBS_2026 + _JOBS_2027:
        events.append(MacroEvent(date=d, event_type="JOBS", description=desc))
    events.sort(key=lambda e: e.date)
    return events


ALL_EVENTS = _build_events()


class MacroCalendarService:
    """Hardcoded macro event calendar for trading logic."""

    # Sizing multipliers per event type (configurable for backtesting)
    fomc_sizing_mult: float = 0.0  # block new buys on FOMC
    cpi_sizing_mult: float = 0.5
    jobs_sizing_mult: float = 0.5

    def is_event_day(self, target: str | None = None) -> list[MacroEvent]:
        """Return events on the given date (YYYY-MM-DD), default today."""
        d = target or date.today().isoformat()
        return [e for e in ALL_EVENTS if e.date == d]

    def get_sizing_multiplier(self, target: str | None = None) -> float:
        """Position sizing multiplier for the given date. 1.0 = normal."""
        events = self.is_event_day(target)
        if not events:
            return 1.0
        # Use the most restrictive multiplier if multiple events
        mult = 1.0
        for e in events:
            if e.event_type == "FOMC":
                mult = min(mult, self.fomc_sizing_mult)
            elif e.event_type == "CPI":
                mult = min(mult, self.cpi_sizing_mult)
            elif e.event_type == "JOBS":
                mult = min(mult, self.jobs_sizing_mult)
        return mult

    def should_block_buys(self, target: str | None = None) -> tuple[bool, str]:
        """Check if buys should be blocked on this date."""
        events = self.is_event_day(target)
        for e in events:
            if e.event_type == "FOMC":
                return True, e.description
        return False, ""

    def get_upcoming(self, days_ahead: int = 7) -> list[MacroEvent]:
        """Get macro events in the next N days."""
        today = date.today()
        end = today + timedelta(days=days_ahead)
        return [
            e for e in ALL_EVENTS
            if today.isoformat() <= e.date <= end.isoformat()
        ]

    def to_dict(self) -> list[dict]:
        """Serialize upcoming events for API."""
        upcoming = self.get_upcoming(days_ahead=14)
        return [
            {"date": e.date, "event_type": e.event_type, "description": e.description}
            for e in upcoming
        ]
