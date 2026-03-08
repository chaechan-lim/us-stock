"""Market data service with caching layer.

Provides OHLCV, tickers, balance, and positions with caching
to minimize KIS API calls within rate limits.

Data source strategy:
- OHLCV (daily): yfinance (no rate limit) with KIS fallback
- Ticker (real-time): KIS API with cache
- Balance/Positions: KIS API with short-TTL cache + rate limiter
"""

import time
import logging
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


class MarketDataService:
    def __init__(
        self,
        adapter: ExchangeAdapter,
        rate_limiter: RateLimiter | None = None,
    ):
        self._adapter = adapter
        self._rate_limiter = rate_limiter or RateLimiter(max_per_second=20)
        self._ticker_cache: dict[str, tuple[Ticker, float]] = {}
        self._ohlcv_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._balance_cache: tuple[Balance, float] | None = None
        self._positions_cache: tuple[list[Position], float] | None = None

    # -- Market Data --

    async def get_ticker(self, symbol: str, exchange: str = "NASD") -> Ticker:
        """Get current ticker with caching."""
        cache_key = f"{exchange}:{symbol}"
        now = time.time()

        cached = self._ticker_cache.get(cache_key)
        if cached and (now - cached[1]) < TICKER_CACHE_TTL:
            return cached[0]

        await self._rate_limiter.acquire()
        ticker = await self._adapter.fetch_ticker(symbol, exchange)
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
            return cached[0]

        # Try yfinance first (no rate limit)
        df = self._fetch_yfinance(symbol, timeframe, limit)
        if not df.empty:
            self._ohlcv_cache[cache_key] = (df, now)
            return df

        # Fallback to KIS
        await self._rate_limiter.acquire()
        candles = await self._adapter.fetch_ohlcv(symbol, timeframe, limit, exchange)

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

        self._ohlcv_cache[cache_key] = (df, now)
        return df

    def _fetch_yfinance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV from yfinance (no rate limit)."""
        try:
            import yfinance as yf

            period_map = {"1D": "1y", "1W": "2y", "1M": "5y"}
            period = period_map.get(timeframe, "1y")

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    return pd.DataFrame()

            df = df[["open", "high", "low", "close", "volume"]].tail(limit)
            return df
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
        balance = await self._adapter.fetch_balance()
        self._balance_cache = (balance, now)
        return balance

    async def get_positions(self) -> list[Position]:
        """Get positions with caching and rate limiting."""
        now = time.time()
        if self._positions_cache and (now - self._positions_cache[1]) < POSITIONS_CACHE_TTL:
            return self._positions_cache[0]

        await self._rate_limiter.acquire()
        positions = await self._adapter.fetch_positions()
        self._positions_cache = (positions, now)
        return positions

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
