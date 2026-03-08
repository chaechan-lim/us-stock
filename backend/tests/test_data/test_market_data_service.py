"""Tests for MarketDataService with caching."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pandas as pd

from data.market_data_service import MarketDataService, TICKER_CACHE_TTL, OHLCV_CACHE_TTL
from exchange.base import Ticker, Candle, Balance, Position
from services.rate_limiter import RateLimiter


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.fetch_ticker = AsyncMock(return_value=Ticker(
        symbol="AAPL", price=150.0, volume=1_000_000,
    ))
    adapter.fetch_ohlcv = AsyncMock(return_value=[
        Candle(timestamp=1704067200, open=148.0, high=152.0, low=147.0, close=150.0, volume=500_000),
        Candle(timestamp=1704153600, open=150.0, high=155.0, low=149.0, close=154.0, volume=600_000),
    ])
    return adapter


@pytest.fixture
def rate_limiter():
    return RateLimiter(max_per_second=100)


@pytest.fixture
def service(mock_adapter, rate_limiter):
    return MarketDataService(adapter=mock_adapter, rate_limiter=rate_limiter)


class TestMarketDataService:
    async def test_get_ticker(self, service, mock_adapter):
        ticker = await service.get_ticker("AAPL")
        assert ticker.symbol == "AAPL"
        assert ticker.price == 150.0
        mock_adapter.fetch_ticker.assert_called_once_with("AAPL", "NASD")

    async def test_get_ticker_cache_hit(self, service, mock_adapter):
        await service.get_ticker("AAPL")
        await service.get_ticker("AAPL")
        # Should only call API once due to caching
        assert mock_adapter.fetch_ticker.call_count == 1

    async def test_get_ticker_cache_miss_different_symbol(self, service, mock_adapter):
        await service.get_ticker("AAPL")
        await service.get_ticker("TSLA")
        assert mock_adapter.fetch_ticker.call_count == 2

    async def test_get_ticker_custom_exchange(self, service, mock_adapter):
        await service.get_ticker("AAPL", exchange="NYSE")
        mock_adapter.fetch_ticker.assert_called_once_with("AAPL", "NYSE")

    @patch.object(MarketDataService, "_fetch_yfinance", return_value=pd.DataFrame())
    async def test_get_ohlcv(self, mock_yf, service, mock_adapter):
        df = await service.get_ohlcv("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 150.0

    @patch.object(MarketDataService, "_fetch_yfinance", return_value=pd.DataFrame())
    async def test_get_ohlcv_cache_hit(self, mock_yf, service, mock_adapter):
        await service.get_ohlcv("AAPL")
        await service.get_ohlcv("AAPL")
        assert mock_adapter.fetch_ohlcv.call_count == 1

    @patch.object(MarketDataService, "_fetch_yfinance", return_value=pd.DataFrame())
    async def test_get_ohlcv_empty(self, mock_yf, service, mock_adapter):
        mock_adapter.fetch_ohlcv.return_value = []
        df = await service.get_ohlcv("UNKNOWN")
        assert df.empty

    async def test_get_ohlcv_yfinance_first(self, service, mock_adapter):
        """yfinance is tried first; KIS adapter is not called if yfinance succeeds."""
        yf_df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000.0],
        })
        with patch.object(service, "_fetch_yfinance", return_value=yf_df):
            df = await service.get_ohlcv("AAPL")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 103.0
        mock_adapter.fetch_ohlcv.assert_not_called()

    async def test_get_balance_cached(self, mock_adapter):
        mock_adapter.fetch_balance = AsyncMock(return_value=Balance(
            currency="USD", total=100_000, available=80_000,
        ))
        limiter = RateLimiter(max_per_second=100)
        svc = MarketDataService(adapter=mock_adapter, rate_limiter=limiter)
        b1 = await svc.get_balance()
        b2 = await svc.get_balance()
        assert b1.total == 100_000
        assert mock_adapter.fetch_balance.call_count == 1  # cached

    async def test_get_positions_cached(self, mock_adapter):
        mock_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0),
        ])
        limiter = RateLimiter(max_per_second=100)
        svc = MarketDataService(adapter=mock_adapter, rate_limiter=limiter)
        p1 = await svc.get_positions()
        p2 = await svc.get_positions()
        assert len(p1) == 1
        assert mock_adapter.fetch_positions.call_count == 1  # cached

    async def test_get_price(self, service):
        price = await service.get_price("AAPL")
        assert price == 150.0

    async def test_get_multiple_tickers(self, service, mock_adapter):
        tickers = await service.get_multiple_tickers(["AAPL", "TSLA"])
        assert "AAPL" in tickers
        assert "TSLA" in tickers
        assert len(tickers) == 2

    async def test_get_multiple_tickers_partial_failure(self, service, mock_adapter):
        mock_adapter.fetch_ticker.side_effect = [
            Ticker(symbol="AAPL", price=150.0, volume=1_000_000),
            Exception("API error"),
        ]
        tickers = await service.get_multiple_tickers(["AAPL", "FAIL"])
        assert "AAPL" in tickers
        assert "FAIL" not in tickers

    async def test_invalidate_cache_specific(self, service, mock_adapter):
        await service.get_ticker("AAPL")
        service.invalidate_cache("AAPL")
        await service.get_ticker("AAPL")
        assert mock_adapter.fetch_ticker.call_count == 2

    async def test_invalidate_cache_all(self, service, mock_adapter):
        await service.get_ticker("AAPL")
        await service.get_ticker("TSLA")
        service.invalidate_cache()
        await service.get_ticker("AAPL")
        # 3 calls: AAPL, TSLA, AAPL again after invalidation
        assert mock_adapter.fetch_ticker.call_count == 3

    async def test_rate_limiter_called(self, mock_adapter):
        limiter = RateLimiter(max_per_second=100)
        limiter.acquire = AsyncMock()
        svc = MarketDataService(adapter=mock_adapter, rate_limiter=limiter)
        await svc.get_ticker("AAPL")
        limiter.acquire.assert_called_once()
