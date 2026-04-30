"""Backtest data loader — yfinance with parquet cache for determinism.

Backtest results need to be deterministic across CI runs (otherwise the
gate keeps falsely failing PRs that don't touch the backtest path).
yfinance returns slightly different bars on different fetches (rounding,
adjustment timing, occasional missing days) — so we cache the raw OHLCV
to a git-tracked parquet file. Indicators are recomputed each load
because they're cheap and a pure function of OHLCV.

Cache flow:
    load(symbol, period='2y')
      → look for data/backtest_cache/{symbol}_{period}.parquet
      → if exists AND env CACHE_DISABLE not set: read it, recompute indicators
      → else: fetch yfinance, write cache, recompute indicators

`refresh_cache(symbols, period)` is the explicit "go fetch fresh data"
escape hatch — call from `ci_backtest_gate.py --update-baseline` so the
cache + baseline move together.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.indicator_service import IndicatorService

logger = logging.getLogger(__name__)


# Repo-root/data/backtest_cache — git-tracked so CI sees the same bars.
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_cache"


@dataclass
class BacktestData:
    symbol: str
    df: pd.DataFrame
    start_date: str
    end_date: str

    @property
    def trading_days(self) -> int:
        return len(self.df)


def _cache_path(symbol: str, period: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    # CSV not parquet — avoids the pyarrow dependency and CI install slowdown.
    # ~10KB per symbol × ~220 symbols ≈ 2MB total, easily git-trackable.
    return _CACHE_DIR / f"{safe}__{period}__{interval}.csv"


def _cache_disabled() -> bool:
    return os.environ.get("BACKTEST_CACHE_DISABLE", "").lower() in ("1", "true", "yes")


class BacktestDataLoader:
    """Load and prepare historical data for backtesting."""

    def __init__(self, indicator_service: IndicatorService | None = None):
        self._indicator_svc = indicator_service or IndicatorService()

    def load(
        self,
        symbol: str,
        period: str = "3y",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
    ) -> BacktestData:
        """Load historical data, preferring the parquet cache.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL')
            period: Data period ('1y', '3y', '5y', 'max')
            interval: Data interval ('1d', '1wk')
            start, end: Date range — overrides period, bypasses cache
                (custom ranges aren't worth caching for now).
            force_refresh: ignore cache, fetch fresh and rewrite.

        Returns:
            BacktestData with OHLCV + indicators
        """
        # Custom date ranges always go to yfinance — too many possible keys
        # to cache and they're rare in practice.
        if start:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            return self._build_data(symbol, df)

        cache_file = _cache_path(symbol, period, interval)
        if not force_refresh and not _cache_disabled() and cache_file.exists():
            try:
                raw = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not raw.empty:
                    return self._build_data(symbol, raw, _from_cache=True)
            except Exception as e:
                logger.warning("Cache read failed for %s (%s) — refetching", symbol, e)

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        # Persist OHLCV (NOT indicators — those are recomputed each load).
        try:
            ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ohlcv.to_csv(cache_file)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", symbol, e)

        return self._build_data(symbol, df)

    def _build_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        _from_cache: bool = False,
    ) -> BacktestData:
        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()
        df = self._indicator_svc.add_all_indicators(df)

        start_date = str(df.index[0].date()) if hasattr(df.index[0], "date") else str(df.index[0])
        end_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

        logger.info(
            "Loaded %d bars for %s (%s to %s)%s",
            len(df), symbol, start_date, end_date,
            " [cache]" if _from_cache else "",
        )

        return BacktestData(
            symbol=symbol,
            df=df,
            start_date=start_date,
            end_date=end_date,
        )

    def refresh_cache(
        self,
        symbols: list[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> dict[str, BacktestData]:
        """Force-refresh cache for the given symbols. Used by the CI gate's
        --update-baseline path so the snapshot + the cache move together."""
        result: dict[str, BacktestData] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(
                    symbol, period=period, interval=interval, force_refresh=True,
                )
            except Exception as e:
                logger.warning("Cache refresh failed for %s: %s", symbol, e)
        return result

    def load_multiple(
        self,
        symbols: list[str],
        period: str = "3y",
        interval: str = "1d",
    ) -> dict[str, BacktestData]:
        """Load data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, period=period, interval=interval)
            except Exception as e:
                logger.warning("Failed to load data for %s: %s", symbol, e)
        return result
