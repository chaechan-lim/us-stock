"""Earnings calendar service — Finnhub earnings API.

Fetches upcoming earnings dates to avoid buying near earnings
and to widen stop-loss for held positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import aiohttp

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


@dataclass
class EarningsEvent:
    symbol: str
    date: str  # "2026-03-15"
    hour: str  # "bmo" / "amc" / ""
    eps_estimate: float | None = None
    eps_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    quarter: int = 0
    year: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "hour": self.hour,
            "eps_estimate": self.eps_estimate,
            "eps_actual": self.eps_actual,
            "revenue_estimate": self.revenue_estimate,
            "revenue_actual": self.revenue_actual,
        }


class EarningsCalendarService:
    """Fetch and cache earnings dates from Finnhub."""

    # Configurable params (for backtesting)
    buy_block_days: int = 3  # skip buy if earnings within N days
    sl_widen_days: int = 2  # widen SL if earnings within N days
    sl_widen_factor: float = 1.5  # SL multiplier near earnings

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        # symbol -> list of upcoming earnings
        self._cache: dict[str, list[EarningsEvent]] = {}

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def fetch_earnings(
        self, from_date: str, to_date: str,
    ) -> list[EarningsEvent]:
        """Fetch earnings calendar for a date range."""
        if not self.available:
            return []
        session = await self._get_session()
        url = f"{FINNHUB_BASE_URL}/calendar/earnings"
        params = {"from": from_date, "to": to_date, "token": self._api_key}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Finnhub earnings: HTTP %d", resp.status)
                    return []
                data = await resp.json()

            events = []
            for item in data.get("earningsCalendar", []):
                events.append(EarningsEvent(
                    symbol=item.get("symbol", ""),
                    date=item.get("date", ""),
                    hour=item.get("hour", ""),
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                    quarter=item.get("quarter", 0),
                    year=item.get("year", 0),
                ))
            return events
        except Exception as e:
            logger.warning("Finnhub earnings fetch failed: %s", e)
            return []

    async def refresh(self, symbols: list[str]) -> None:
        """Refresh cache for the given symbols (2-week lookahead)."""
        if not self.available:
            return

        today = date.today()
        from_date = today.isoformat()
        to_date = (today + timedelta(days=14)).isoformat()

        all_events = await self.fetch_earnings(from_date, to_date)
        watchlist_set = set(s.upper() for s in symbols)

        # Filter to watchlist symbols only
        self._cache.clear()
        for e in all_events:
            if e.symbol in watchlist_set:
                self._cache.setdefault(e.symbol, []).append(e)

        logger.info(
            "Earnings calendar: %d events for %d symbols (next 14 days)",
            sum(len(v) for v in self._cache.values()),
            len(self._cache),
        )

    def get_upcoming(self, symbol: str, days_ahead: int = 3) -> list[EarningsEvent]:
        """Get earnings within N days for a symbol.

        DT-H10 (2026-06-06): cached earnings dates are ET-anchored
        (Finnhub returns the symbol's local exchange date). Server is
        Asia/Seoul; `date.today()` rolls at 00:00 KST = 10:00 ET, so
        on the KST-morning of a US earnings date the local server
        already flipped to the day-after while the actual event is
        ~9h away. Buy-block lifted prematurely. Now we anchor on ET.
        """
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo

        today = _dt.now(ZoneInfo("America/New_York")).date()
        cutoff = (today + timedelta(days=days_ahead)).isoformat()
        return [
            e for e in self._cache.get(symbol, [])
            if today.isoformat() <= e.date <= cutoff
        ]

    def has_earnings_within(self, symbol: str, days: int | None = None) -> bool:
        """Check if symbol has earnings within N days."""
        return len(self.get_upcoming(symbol, days or self.buy_block_days)) > 0

    def get_sl_multiplier(self, symbol: str) -> float | None:
        """If earnings are near, return SL widen factor. None = no change."""
        if self.get_upcoming(symbol, self.sl_widen_days):
            return self.sl_widen_factor
        return None

    def to_dict(self) -> list[dict]:
        """Serialize all cached earnings for API."""
        result = []
        for events in self._cache.values():
            for e in events:
                result.append(e.to_dict())
        result.sort(key=lambda x: x["date"])
        return result

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
