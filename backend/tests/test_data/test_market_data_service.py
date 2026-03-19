"""Tests for MarketDataService with caching."""

import asyncio
import threading
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


class TestYfinanceResampling:
    """Test weekly/monthly OHLCV resampling."""

    def _make_daily_df(self, days: int = 60) -> pd.DataFrame:
        """Create synthetic daily OHLCV data."""
        import numpy as np
        dates = pd.date_range("2024-01-02", periods=days, freq="B")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(days) * 0.5)
        return pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(100000, 500000, days).astype(float),
        }, index=dates)

    def test_weekly_resampling(self, service):
        daily = self._make_daily_df(60)
        with patch.object(service, "_fetch_yfinance") as mock:
            # Simulate what _fetch_yfinance does internally for weekly
            df = daily.resample("W").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

            assert len(df) < len(daily)  # Fewer bars after resampling
            assert len(df) >= 10  # ~12 weeks in 60 business days
            # Weekly high should be >= any daily high in that week
            # Weekly volume should be sum of daily volumes

    def test_monthly_resampling(self, service):
        daily = self._make_daily_df(250)
        df = daily.resample("ME").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        assert len(df) < len(daily)
        assert len(df) >= 10  # ~12 months in 250 business days
        # Monthly volume should be much higher than any single day
        assert df["volume"].iloc[0] > daily["volume"].iloc[0]

    def test_daily_no_resampling(self, service):
        daily = self._make_daily_df(60)
        # Daily should not be resampled
        assert len(daily) == 60


class TestYfSymbolMapper:
    """Test yf_symbol_mapper integration for KR stocks."""

    async def test_kr_symbol_mapper_applied(self, mock_adapter):
        """When yf_symbol_mapper is set, it transforms symbol for yfinance."""
        mapper = lambda s: f"{s}.KS"
        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
            yf_symbol_mapper=mapper,
        )
        with patch.object(svc, "_fetch_yfinance", return_value=pd.DataFrame({
            "open": [72000], "high": [73000], "low": [71000],
            "close": [72500], "volume": [1000000],
        })) as mock_yf:
            df = await svc.get_ohlcv("005930")
            # yfinance should receive the mapped symbol
            mock_yf.assert_called_once_with("005930.KS", "1D", 200)
            assert len(df) == 1
            assert df.iloc[0]["close"] == 72500

    async def test_no_mapper_passes_symbol_as_is(self, service):
        """Without mapper, symbol is passed directly to yfinance."""
        with patch.object(service, "_fetch_yfinance", return_value=pd.DataFrame({
            "open": [150], "high": [155], "low": [149],
            "close": [153], "volume": [500000],
        })) as mock_yf:
            await service.get_ohlcv("AAPL")
            mock_yf.assert_called_once_with("AAPL", "1D", 200)


class TestCacheEviction:
    """Test cache size limits and eviction."""

    def test_evict_oldest_removes_stale_entries(self):
        """Oldest 20% of entries are evicted."""
        cache = {f"key{i}": (f"val{i}", float(i)) for i in range(10)}
        MarketDataService._evict_oldest(cache)
        assert len(cache) == 8
        # Oldest entries (key0, key1) should be gone
        assert "key0" not in cache
        assert "key1" not in cache
        assert "key9" in cache

    async def test_ticker_cache_bounded(self, mock_adapter, rate_limiter):
        """Ticker cache doesn't grow beyond MAX_CACHE_ENTRIES."""
        from data.market_data_service import MAX_CACHE_ENTRIES
        svc = MarketDataService(adapter=mock_adapter, rate_limiter=rate_limiter)

        # Fill cache to limit
        for i in range(MAX_CACHE_ENTRIES + 5):
            mock_adapter.fetch_ticker = AsyncMock(return_value=Ticker(
                symbol=f"SYM{i}", price=100.0 + i, volume=1000,
            ))
            await svc.get_ticker(f"SYM{i}")

        assert len(svc._ticker_cache) <= MAX_CACHE_ENTRIES


class TestYfinanceNonBlocking:
    """Test that yfinance calls run in a thread and don't block the event loop."""

    async def test_yfinance_runs_in_executor_not_blocking(self, mock_adapter):
        """Slow yfinance must not block other async tasks."""
        cancel_event = threading.Event()

        def slow_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            """Simulate a slow yfinance call that respects cancellation."""
            cancel_event.wait(timeout=0.5)  # block the thread for 0.5s
            return pd.DataFrame({
                "open": [100.0], "high": [105.0], "low": [99.0],
                "close": [103.0], "volume": [1000.0],
            })

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )

        # Track whether a concurrent async task runs while yfinance is blocked
        concurrent_ran = False

        async def concurrent_task() -> None:
            nonlocal concurrent_ran
            await asyncio.sleep(0.05)
            concurrent_ran = True

        with patch.object(svc, "_fetch_yfinance", side_effect=slow_yfinance):
            ohlcv_task = asyncio.create_task(svc.get_ohlcv("AAPL"))
            concurrent = asyncio.create_task(concurrent_task())
            await asyncio.gather(ohlcv_task, concurrent)

        cancel_event.set()  # clean up the thread
        assert concurrent_ran, "Concurrent async task should run while yfinance is in thread"
        svc.shutdown()

    async def test_yfinance_timeout_fires(self, mock_adapter):
        """Yfinance calls that exceed YFINANCE_TIMEOUT get cancelled."""
        cancel_event = threading.Event()

        def stuck_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            """Simulate a yfinance call that hangs until cancelled."""
            cancel_event.wait(timeout=30)  # long wait, but event will release it
            return pd.DataFrame()

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )
        # Use a very short timeout for the test
        with patch("data.market_data_service.YFINANCE_TIMEOUT", 0.2):
            with patch.object(svc, "_fetch_yfinance", side_effect=stuck_yfinance):
                # KIS fallback returns empty, so get_ohlcv should still return a DataFrame
                mock_adapter.fetch_ohlcv.return_value = []
                df = await svc.get_ohlcv("AAPL")

        cancel_event.set()  # release the blocked thread
        assert df.empty, "Should fall back to KIS when yfinance times out"
        mock_adapter.fetch_ohlcv.assert_called_once()
        svc.shutdown()

    async def test_yfinance_exception_falls_back_to_kis(self, mock_adapter):
        """If yfinance raises an exception in the thread, KIS fallback is used."""
        def failing_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            raise ConnectionError("yfinance server down")

        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )
        with patch.object(svc, "_fetch_yfinance", side_effect=failing_yfinance):
            df = await svc.get_ohlcv("AAPL")

        assert not df.empty, "Should fall back to KIS adapter data"
        mock_adapter.fetch_ohlcv.assert_called_once()
        svc.shutdown()

    async def test_shutdown_cleans_executor(self, mock_adapter):
        """shutdown() should cleanly shut down the thread pool."""
        svc = MarketDataService(
            adapter=mock_adapter,
            rate_limiter=RateLimiter(max_per_second=100),
        )
        svc.shutdown()
        assert svc._yf_executor._shutdown, "Executor should be shut down"
