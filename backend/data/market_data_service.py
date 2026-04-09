"""Market data service with caching layer.

Provides OHLCV, tickers, balance, and positions with caching
to minimize KIS API calls within rate limits.

Data source strategy:
- OHLCV (daily): yfinance (no rate limit) with KIS fallback
- Ticker (real-time): KIS API with cache
- Balance/Positions: KIS API with short-TTL cache + rate limiter
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from exchange.base import ExchangeAdapter, Balance, Candle, Position, Ticker
from services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Cache TTLs
TICKER_CACHE_TTL = 10       # seconds
OHLCV_CACHE_TTL = 300       # 5 min (daily data doesn't change fast)
BALANCE_CACHE_TTL = 30      # seconds
POSITIONS_CACHE_TTL = 30    # seconds
EXCHANGE_RATE_CACHE_TTL = 300  # 5 min — rate doesn't change fast
API_CALL_TIMEOUT = 30       # seconds — max wait for a single adapter call
YFINANCE_TIMEOUT = 15       # seconds — max wait for yfinance (runs in thread)
_YF_HTTP_TIMEOUT = 10       # seconds — yfinance HTTP request timeout (passed to ticker.history)
MAX_CACHE_ENTRIES = 150     # max entries per cache dict (prevent unbounded growth)
# Thread pool for yfinance blocking I/O. Sized at 8 to provide headroom: if several
# yfinance calls hit sustained timeouts, cancelled futures still occupy threads until
# the underlying HTTP request completes (_YF_HTTP_TIMEOUT). With 8 workers, the pool
# can absorb a burst of slow calls without starving subsequent requests.
_YF_EXECUTOR_WORKERS = 8


class MarketDataService:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        rate_limiter: RateLimiter | None = None,
        yf_symbol_mapper: Any | None = None,
    ):
        self._adapter = adapter
        self._rate_limiter = rate_limiter or RateLimiter(max_per_second=20)
        self._yf_symbol_mapper = yf_symbol_mapper  # callable: symbol -> yfinance symbol
        self._ticker_cache: dict[str, tuple[Ticker, float]] = {}
        self._ohlcv_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._balance_cache: tuple[Balance, float] | None = None
        self._positions_cache: tuple[list[Position], float] | None = None
        self._exchange_rate_cache: tuple[float, float] | None = None  # (rate, timestamp)
        # Dedicated thread pool for yfinance blocking I/O — isolates from asyncio global pool
        self._yf_executor = ThreadPoolExecutor(
            max_workers=_YF_EXECUTOR_WORKERS, thread_name_prefix="yfinance"
        )

    # -- Market Data --

    async def get_ticker(self, symbol: str, exchange: str = "NASD") -> Ticker:
        """Get current ticker with caching."""
        cache_key = f"{exchange}:{symbol}"
        now = time.time()

        cached = self._ticker_cache.get(cache_key)
        if cached and (now - cached[1]) < TICKER_CACHE_TTL:
            self._ticker_cache[cache_key] = (cached[0], now)  # LRU: update access time
            return cached[0]

        await self._rate_limiter.acquire()
        ticker = await asyncio.wait_for(
            self._adapter.fetch_ticker(symbol, exchange),
            timeout=API_CALL_TIMEOUT,
        )
        if len(self._ticker_cache) >= MAX_CACHE_ENTRIES:
            self._evict_oldest(self._ticker_cache)
        self._ticker_cache[cache_key] = (ticker, now)
        return ticker

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 200,
        exchange: str = "NASD",
    ) -> pd.DataFrame:
        """Get OHLCV data — uses yfinance first, KIS as fallback."""
        cache_key = f"{exchange}:{symbol}:{timeframe}:{limit}"
        now = time.time()

        cached = self._ohlcv_cache.get(cache_key)
        if cached and (now - cached[1]) < OHLCV_CACHE_TTL:
            self._ohlcv_cache[cache_key] = (cached[0], now)  # LRU: update access time
            return cached[0]

        # Try yfinance first (no rate limit) — run in dedicated thread pool
        yf_sym = self._yf_symbol_mapper(symbol) if self._yf_symbol_mapper else symbol
        try:
            loop = asyncio.get_running_loop()
            df = await asyncio.wait_for(
                loop.run_in_executor(
                    self._yf_executor, self._fetch_yfinance, yf_sym, timeframe, limit
                ),
                timeout=YFINANCE_TIMEOUT,
            )
        except Exception as e:
            logger.warning("yfinance async fetch failed for %s: %s", yf_sym, e)
            df = pd.DataFrame()
        if not df.empty:
            if len(self._ohlcv_cache) >= MAX_CACHE_ENTRIES:
                self._evict_oldest(self._ohlcv_cache)
            self._ohlcv_cache[cache_key] = (df, now)
            return df

        # Fallback to KIS
        await self._rate_limiter.acquire()
        candles = await asyncio.wait_for(
            self._adapter.fetch_ohlcv(symbol, timeframe, limit, exchange),
            timeout=API_CALL_TIMEOUT,
        )

        if not candles:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            df = pd.DataFrame([
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ])

        if len(self._ohlcv_cache) >= MAX_CACHE_ENTRIES:
            self._evict_oldest(self._ohlcv_cache)
        self._ohlcv_cache[cache_key] = (df, now)
        return df

    def _fetch_yfinance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV from yfinance (no rate limit).

        Runs inside a dedicated thread pool — must NOT be called from the event loop directly.
        Uses _YF_HTTP_TIMEOUT to cap the underlying HTTP request so that threads are
        released promptly even when yfinance servers are unresponsive.
        """
        try:
            import yfinance as yf

            period_map = {"1D": "1y", "1W": "2y", "1M": "5y"}
            period = period_map.get(timeframe, "1y")

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", timeout=_YF_HTTP_TIMEOUT)
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    return pd.DataFrame()

            df = df[["open", "high", "low", "close", "volume"]]

            # Validate data quality: drop rows with zero/NaN prices
            for col in ["open", "high", "low", "close"]:
                df = df[df[col] > 0]
            df = df.dropna(subset=["open", "high", "low", "close"])
            if df.empty:
                return pd.DataFrame()

            # Resample to weekly or monthly if requested
            if timeframe == "1W":
                df = df.resample("W").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
            elif timeframe == "1M":
                df = df.resample("ME").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()

            return df.tail(limit)
        except Exception as e:
            logger.debug("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    # -- Account Data (cached + rate limited) --

    async def get_balance(self) -> Balance:
        """Get balance with caching and rate limiting."""
        now = time.time()
        if self._balance_cache and (now - self._balance_cache[1]) < BALANCE_CACHE_TTL:
            return self._balance_cache[0]

        await self._rate_limiter.acquire()
        balance = await asyncio.wait_for(
            self._adapter.fetch_balance(),
            timeout=API_CALL_TIMEOUT,
        )
        self._balance_cache = (balance, now)
        return balance

    async def get_positions(self) -> list[Position]:
        """Get positions with caching and rate limiting."""
        now = time.time()
        if self._positions_cache and (now - self._positions_cache[1]) < POSITIONS_CACHE_TTL:
            return self._positions_cache[0]

        await self._rate_limiter.acquire()
        positions = await asyncio.wait_for(
            self._adapter.fetch_positions(),
            timeout=API_CALL_TIMEOUT,
        )
        self._positions_cache = (positions, now)
        return positions

    async def get_exchange_rate(self) -> float:
        """Get USD/KRW exchange rate with caching (5-min TTL).

        Uses adapter's cached rate from CTRP6504R or dedicated rate API.
        Falls back to adapter._last_exchange_rate, then to 1450.
        """
        now = time.time()
        if (
            self._exchange_rate_cache
            and (now - self._exchange_rate_cache[1]) < EXCHANGE_RATE_CACHE_TTL
            and self._exchange_rate_cache[0] > 0
        ):
            return self._exchange_rate_cache[0]

        rate = 0.0
        # Try dedicated exchange rate fetch
        if hasattr(self._adapter, "_fetch_exchange_rate"):
            try:
                rate = await self._adapter._fetch_exchange_rate()
            except Exception as e:
                logger.warning("Exchange rate fetch failed, will use fallback: %s", e)

        # Fallback to cached rate from balance fetch
        if rate <= 0 and hasattr(self._adapter, "_last_exchange_rate"):
            rate = getattr(self._adapter, "_last_exchange_rate", 0.0)

        if rate <= 0:
            rate = 1450.0  # hard fallback

        self._exchange_rate_cache = (rate, now)
        return rate

    # -- Convenience --

    async def get_price(self, symbol: str, exchange: str = "NASD") -> float:
        """Get current price (convenience wrapper)."""
        ticker = await self.get_ticker(symbol, exchange)
        return ticker.price

    async def get_multiple_tickers(
        self, symbols: list[str], exchange: str = "NASD"
    ) -> dict[str, Ticker]:
        """Get tickers for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = await self.get_ticker(symbol, exchange)
            except Exception as e:
                logger.warning("Failed to fetch ticker for %s: %s", symbol, e)
        return result

    @staticmethod
    def _evict_oldest(cache: dict) -> None:
        """Remove the oldest 20% of cache entries by timestamp."""
        if not cache:
            return
        n_remove = max(1, len(cache) // 5)
        oldest = sorted(cache, key=lambda k: cache[k][1])[:n_remove]
        for k in oldest:
            del cache[k]

    def invalidate_cache(self, symbol: str | None = None) -> None:
        """Clear cache for a symbol or all."""
        if symbol:
            keys_to_remove = [k for k in self._ticker_cache if symbol in k]
            for k in keys_to_remove:
                del self._ticker_cache[k]
            keys_to_remove = [k for k in self._ohlcv_cache if symbol in k]
            for k in keys_to_remove:
                del self._ohlcv_cache[k]
        else:
            self._ticker_cache.clear()
            self._ohlcv_cache.clear()
        self._balance_cache = None
        self._positions_cache = None

    def invalidate_balance_cache(self) -> None:
        """Clear only balance and positions cache.

        More targeted than invalidate_cache() — preserves ticker/OHLCV caches.
        Use after order fills when only account state changed.
        """
        self._balance_cache = None
        self._positions_cache = None

    def shutdown(self) -> None:
        """Shut down the yfinance thread pool executor.

        Call during application shutdown for clean resource release.
        """
        self._yf_executor.shutdown(wait=False)

    def __del__(self) -> None:
        """Safety net: release thread pool if shutdown() was never called."""
        try:
            self._yf_executor.shutdown(wait=False)
        except Exception:
            pass

    async def __aenter__(self) -> "MarketDataService":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
