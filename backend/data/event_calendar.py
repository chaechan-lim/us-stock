"""Unified event calendar service — combines earnings, macro, insider data.

Single entry point for evaluation loop and risk management integration.
"""

from __future__ import annotations

import logging
from datetime import date

from data.earnings_service import EarningsCalendarService
from data.macro_calendar import MacroCalendarService
from data.insider_service import InsiderTradingService

try:
    from data.dart_service import DARTInsiderService
except ImportError:    # service file is optional
    DARTInsiderService = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class EventCalendarService:
    """Facade combining earnings, macro events, and insider trading."""

    def __init__(
        self,
        earnings: EarningsCalendarService,
        macro: MacroCalendarService,
        insider: InsiderTradingService,
        kr_insider: "DARTInsiderService | None" = None,
    ):
        self.earnings = earnings
        self.macro = macro
        self.insider = insider     # US — Finnhub
        self.kr_insider = kr_insider  # KR — DART (optional, disabled by default)

    async def refresh_all(self, symbols: list[str]) -> None:
        """Refresh earnings + insider data for watchlist symbols.

        US symbols → Finnhub insider service.
        KR symbols (6-digit numeric codes) → DART, when configured.
        """
        await self.earnings.refresh(symbols)
        us_symbols = [s for s in symbols if not s.isdigit()]
        kr_symbols = [s for s in symbols if s.isdigit() and len(s) == 6]
        await self.insider.refresh(us_symbols)
        if self.kr_insider and self.kr_insider.enabled and kr_symbols:
            await self.kr_insider.refresh(kr_symbols)

    def should_skip_buy(self, symbol: str) -> tuple[bool, str]:
        """Check if buy should be blocked.

        Returns (should_skip, reason).
        """
        # FOMC day: block all buys
        blocked, reason = self.macro.should_block_buys()
        if blocked:
            return True, reason

        # Earnings within N days: block buy for this symbol
        if self.earnings.has_earnings_within(symbol):
            events = self.earnings.get_upcoming(symbol)
            dates = ", ".join(e.date for e in events)
            return True, f"earnings on {dates}"

        return False, ""

    def get_sizing_multiplier(self) -> float:
        """Position sizing multiplier for today (macro events)."""
        return self.macro.get_sizing_multiplier()

    def get_sl_multiplier(self, symbol: str) -> float | None:
        """SL widen factor if earnings are near. None = no change."""
        return self.earnings.get_sl_multiplier(symbol)

    def get_confidence_adjustment(self, symbol: str) -> float:
        """Insider trading confidence boost/penalty.

        Routes by symbol shape: 6-digit numeric → KR (DART), else US
        (Finnhub). Returns 0.0 when the relevant service is disabled.
        """
        if (
            self.kr_insider is not None
            and symbol.isdigit() and len(symbol) == 6
        ):
            return self.kr_insider.get_signal_adjustment(symbol)
        return self.insider.get_signal_adjustment(symbol)

    def to_dict(self) -> dict:
        """Full serialization for API response."""
        out = {
            "earnings": self.earnings.to_dict(),
            "macro": self.macro.to_dict(),
            "insider": self.insider.to_dict(),
        }
        if self.kr_insider is not None:
            out["kr_insider"] = self.kr_insider.to_dict()
        return out

    async def close(self) -> None:
        await self.earnings.close()
        await self.insider.close()
