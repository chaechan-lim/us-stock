"""Stock classifier - categorize stocks by technical characteristics.

Analyzes OHLCV data to classify stocks into categories:
- growth_momentum: High momentum, strong uptrend (e.g. NVDA, META)
- stable_large_cap: Low volatility, steady trend (e.g. AAPL, MSFT, JNJ)
- cyclical_value: Mean-reverting, sector-sensitive (e.g. JPM, XOM)
- high_volatility: High ATR, wide swings (e.g. meme stocks, biotech)
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockCategory(str, Enum):
    GROWTH_MOMENTUM = "growth_momentum"
    STABLE_LARGE_CAP = "stable_large_cap"
    CYCLICAL_VALUE = "cyclical_value"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class StockProfile:
    symbol: str
    category: StockCategory
    volatility: float  # annualized volatility
    trend_strength: float  # ADX or slope-based
    momentum_score: float  # ROC-based
    mean_reversion_score: float  # how mean-reverting


class StockClassifier:
    """Classify stocks into categories from OHLCV data."""

    # Thresholds for classification
    HIGH_VOL_THRESHOLD = 0.45  # annualized vol > 45%
    LOW_VOL_THRESHOLD = 0.25  # annualized vol < 25%
    STRONG_MOMENTUM_THRESHOLD = 0.6  # momentum score > 0.6
    MEAN_REVERSION_THRESHOLD = 0.5  # mean reversion score > 0.5

    def classify(self, df: pd.DataFrame, symbol: str) -> StockProfile:
        """Classify a stock based on its OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume], 100+ rows
            symbol: Stock ticker

        Returns:
            StockProfile with category and scores
        """
        if len(df) < 60:
            return StockProfile(
                symbol=symbol,
                category=StockCategory.STABLE_LARGE_CAP,
                volatility=0.0,
                trend_strength=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0,
            )

        close = df["close"].values
        returns = pd.Series(close).pct_change().dropna().values

        volatility = self._annualized_volatility(returns)
        trend_strength = self._trend_strength(close)
        momentum_score = self._momentum_score(close)
        mean_reversion_score = self._mean_reversion_score(close, returns)

        category = self._determine_category(
            volatility, trend_strength, momentum_score, mean_reversion_score
        )

        profile = StockProfile(
            symbol=symbol,
            category=category,
            volatility=round(volatility, 4),
            trend_strength=round(trend_strength, 4),
            momentum_score=round(momentum_score, 4),
            mean_reversion_score=round(mean_reversion_score, 4),
        )
        logger.debug(
            "Classified %s as %s (vol=%.2f, trend=%.2f, mom=%.2f, mr=%.2f)",
            symbol, category.value, volatility, trend_strength,
            momentum_score, mean_reversion_score,
        )
        return profile

    def _annualized_volatility(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns) * np.sqrt(252))

    def _trend_strength(self, close: np.ndarray) -> float:
        """Measure trend strength using linear regression R-squared."""
        n = min(len(close), 60)
        prices = close[-n:]
        x = np.arange(n)
        # R-squared of linear fit
        correlation = np.corrcoef(x, prices)[0, 1]
        r_squared = correlation ** 2
        # Directional: positive slope = positive score
        slope = np.polyfit(x, prices, 1)[0]
        direction = 1.0 if slope > 0 else -0.5
        return float(r_squared * direction)

    def _momentum_score(self, close: np.ndarray) -> float:
        """Momentum score based on multi-period rate of change."""
        if len(close) < 60:
            return 0.0
        roc_20 = (close[-1] / close[-20] - 1) if close[-20] > 0 else 0
        roc_60 = (close[-1] / close[-60] - 1) if close[-60] > 0 else 0
        # Normalize: 20% over 60 days = score 1.0
        score = (roc_20 * 2 + roc_60) / 3
        return float(np.clip(score / 0.15, -1, 1))

    def _mean_reversion_score(
        self, close: np.ndarray, returns: np.ndarray
    ) -> float:
        """How mean-reverting is the stock (negative autocorrelation)."""
        if len(returns) < 30:
            return 0.0
        # Lag-1 autocorrelation of returns
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        # Price relative to SMA20
        sma20 = np.mean(close[-20:])
        deviation_pct = abs(close[-1] / sma20 - 1)
        # Mean-reverting stocks have negative autocorrelation
        # and frequently revert to SMA
        mr_score = max(0, -autocorr) + min(deviation_pct * 5, 0.5)
        return float(np.clip(mr_score, 0, 1))

    def _determine_category(
        self,
        volatility: float,
        trend_strength: float,
        momentum_score: float,
        mean_reversion_score: float,
    ) -> StockCategory:
        # High volatility overrides other categories
        if volatility > self.HIGH_VOL_THRESHOLD:
            return StockCategory.HIGH_VOLATILITY

        # Strong momentum + strong trend = growth
        if momentum_score > self.STRONG_MOMENTUM_THRESHOLD and trend_strength > 0.3:
            return StockCategory.GROWTH_MOMENTUM

        # Mean-reverting + weaker trend = cyclical/value
        if mean_reversion_score > self.MEAN_REVERSION_THRESHOLD and trend_strength < 0.5:
            return StockCategory.CYCLICAL_VALUE

        # Low volatility + moderate trend = stable large cap
        if volatility < self.LOW_VOL_THRESHOLD:
            return StockCategory.STABLE_LARGE_CAP

        # Default: use momentum as tiebreaker
        if momentum_score > 0.3:
            return StockCategory.GROWTH_MOMENTUM
        if mean_reversion_score > 0.3:
            return StockCategory.CYCLICAL_VALUE
        return StockCategory.STABLE_LARGE_CAP
