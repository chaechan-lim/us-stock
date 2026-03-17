"""Tests for balance cache invalidation and exchange rate cache — STOCK-1.

Validates:
- invalidate_balance_cache() clears only balance/positions (not ticker/OHLCV)
- Exchange rate caching with 5-min TTL
- Exchange rate fallback chain
"""

import time
from unittest.mock import AsyncMock

import pytest

from data.market_data_service import (
    EXCHANGE_RATE_CACHE_TTL,
    MarketDataService,
)
from exchange.base import Balance, Position, Ticker
from services.rate_limiter import RateLimiter


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=Balance(currency="USD", total=100_000, available=80_000),
    )
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0),
        ],
    )
    adapter.fetch_ticker = AsyncMock(
        return_value=Ticker(symbol="AAPL", price=160.0, volume=1_000_000),
    )
    return adapter


@pytest.fixture
def service(mock_adapter):
    return MarketDataService(
        adapter=mock_adapter,
        rate_limiter=RateLimiter(max_per_second=100),
    )


class TestInvalidateBalanceCache:
    """Test targeted balance/positions cache invalidation."""

    async def test_invalidates_balance(self, service, mock_adapter):
        """invalidate_balance_cache clears balance cache."""
        await service.get_balance()
        assert mock_adapter.fetch_balance.call_count == 1

        # Without invalidation, should use cache
        await service.get_balance()
        assert mock_adapter.fetch_balance.call_count == 1

        # After invalidation, should re-fetch
        service.invalidate_balance_cache()
        await service.get_balance()
        assert mock_adapter.fetch_balance.call_count == 2

    async def test_invalidates_positions(self, service, mock_adapter):
        """invalidate_balance_cache clears positions cache."""
        await service.get_positions()
        assert mock_adapter.fetch_positions.call_count == 1

        await service.get_positions()
        assert mock_adapter.fetch_positions.call_count == 1

        service.invalidate_balance_cache()
        await service.get_positions()
        assert mock_adapter.fetch_positions.call_count == 2

    async def test_preserves_ticker_cache(self, service, mock_adapter):
        """invalidate_balance_cache does NOT clear ticker cache."""
        await service.get_ticker("AAPL")
        assert mock_adapter.fetch_ticker.call_count == 1

        service.invalidate_balance_cache()

        # Ticker should still be cached
        await service.get_ticker("AAPL")
        assert mock_adapter.fetch_ticker.call_count == 1

    async def test_multiple_invalidations(self, service, mock_adapter):
        """Multiple invalidations are safe (no crash on None cache)."""
        service.invalidate_balance_cache()
        service.invalidate_balance_cache()
        # No error — safe on empty cache

        await service.get_balance()
        service.invalidate_balance_cache()
        service.invalidate_balance_cache()
        assert mock_adapter.fetch_balance.call_count == 1


class TestExchangeRateCache:
    """Test exchange rate caching with separate TTL."""

    async def test_caches_rate(self, mock_adapter):
        """Exchange rate is cached after first fetch."""
        async def _rate():
            return 1395.0

        mock_adapter._fetch_exchange_rate = _rate
        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        rate1 = await svc.get_exchange_rate()
        assert rate1 == 1395.0

        # Second call should use cache (modify adapter to verify)
        async def _new_rate():
            return 9999.0  # should not be used

        mock_adapter._fetch_exchange_rate = _new_rate
        rate2 = await svc.get_exchange_rate()
        assert rate2 == 1395.0  # still cached

    async def test_fallback_to_last_exchange_rate(self, mock_adapter):
        """Falls back to _last_exchange_rate when _fetch_exchange_rate returns 0."""
        async def _rate():
            return 0.0

        mock_adapter._fetch_exchange_rate = _rate
        mock_adapter._last_exchange_rate = 1380.0

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        rate = await svc.get_exchange_rate()
        assert rate == 1380.0

    async def test_fallback_to_hard_default(self, mock_adapter):
        """Falls back to 1450 when no rate sources available."""
        # No _fetch_exchange_rate, no _last_exchange_rate
        adapter = AsyncMock(spec=[])  # empty spec — no attributes

        svc = MarketDataService(
            adapter=adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        rate = await svc.get_exchange_rate()
        assert rate == 1450.0

    async def test_exception_in_fetch_rate(self, mock_adapter):
        """Handles exception from _fetch_exchange_rate gracefully."""
        async def _rate():
            raise ConnectionError("API down")

        mock_adapter._fetch_exchange_rate = _rate
        mock_adapter._last_exchange_rate = 1390.0

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        rate = await svc.get_exchange_rate()
        assert rate == 1390.0

    async def test_cache_expired(self, mock_adapter):
        """Cache expires after EXCHANGE_RATE_CACHE_TTL."""
        call_count = 0

        async def _rate():
            nonlocal call_count
            call_count += 1
            return 1395.0 if call_count == 1 else 1400.0

        mock_adapter._fetch_exchange_rate = _rate

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        rate1 = await svc.get_exchange_rate()
        assert rate1 == 1395.0

        # Manually expire cache
        svc._exchange_rate_cache = (1395.0, time.time() - EXCHANGE_RATE_CACHE_TTL - 1)

        rate2 = await svc.get_exchange_rate()
        assert rate2 == 1400.0
