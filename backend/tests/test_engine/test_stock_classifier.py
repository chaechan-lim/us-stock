"""Tests for StockClassifier."""

import numpy as np
import pandas as pd
import pytest

from engine.stock_classifier import StockClassifier, StockCategory


@pytest.fixture
def classifier():
    return StockClassifier()


def _make_df(prices: list[float]) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame from close prices."""
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * n,
    })


class TestStockClassifier:
    def test_insufficient_data_returns_stable(self, classifier):
        df = _make_df([100.0] * 30)
        profile = classifier.classify(df, "TEST")
        assert profile.category == StockCategory.STABLE_LARGE_CAP

    def test_strong_uptrend_classified_as_growth(self, classifier):
        # Strong uptrend: price doubles in 100 days
        prices = [100 + i * 1.5 for i in range(100)]
        df = _make_df(prices)
        profile = classifier.classify(df, "NVDA")
        assert profile.category == StockCategory.GROWTH_MOMENTUM
        assert profile.momentum_score > 0.3
        assert profile.trend_strength > 0.3

    def test_flat_low_vol_classified_as_stable(self, classifier):
        # Low volatility, slight uptrend
        np.random.seed(42)
        base = 150.0
        prices = [base + i * 0.05 + np.random.normal(0, 0.3) for i in range(100)]
        df = _make_df(prices)
        profile = classifier.classify(df, "JNJ")
        assert profile.category in (
            StockCategory.STABLE_LARGE_CAP,
            StockCategory.CYCLICAL_VALUE,
        )
        assert profile.volatility < 0.3

    def test_high_volatility_detected(self, classifier):
        # Very volatile stock
        np.random.seed(42)
        prices = [100.0]
        for _ in range(99):
            change = np.random.normal(0, 0.04)  # 4% daily std
            prices.append(prices[-1] * (1 + change))
        df = _make_df(prices)
        profile = classifier.classify(df, "MEME")
        assert profile.volatility > 0.4

    def test_profile_has_all_fields(self, classifier):
        prices = [100 + i * 0.5 for i in range(100)]
        df = _make_df(prices)
        profile = classifier.classify(df, "TEST")
        assert profile.symbol == "TEST"
        assert isinstance(profile.category, StockCategory)
        assert isinstance(profile.volatility, float)
        assert isinstance(profile.trend_strength, float)
        assert isinstance(profile.momentum_score, float)
        assert isinstance(profile.mean_reversion_score, float)
