"""Tests for ML Factor Selector (LightGBM-based feature importance).

Verifies:
1. Feature extraction from price data
2. Dataset building with sufficient stocks
3. Model training and feature importance extraction
4. Factor weight mapping from importances
5. Insufficient data fallback
6. Edge cases (zero prices, missing fundamentals)
"""

import numpy as np
import pandas as pd
import pytest

from analytics.ml_factor_selector import (
    FactorSelectionResult,
    FeatureImportance,
    MLFactorSelector,
    MIN_BARS,
    MIN_STOCKS,
)


def _make_stock_df(
    n: int = 300,
    base: float = 100.0,
    trend: float = 0.0005,
    noise: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    returns = 1 + trend + np.random.normal(0, noise, n)
    close = base * np.cumprod(returns)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    })


def _make_universe(n_stocks: int = 15, n_bars: int = 300) -> dict[str, pd.DataFrame]:
    return {
        f"SYM{i:02d}": _make_stock_df(n=n_bars, trend=0.0005 * (i - 7), seed=i)
        for i in range(n_stocks)
    }


@pytest.fixture
def selector():
    return MLFactorSelector(
        forward_days=10,
        n_estimators=20,
        max_depth=3,
    )


@pytest.fixture
def universe():
    return _make_universe(n_stocks=15, n_bars=300)


class TestInsufficientData:
    def test_too_few_stocks(self, selector):
        data = _make_universe(n_stocks=5, n_bars=300)
        result = selector.select_factors(data)
        assert not result.success

    def test_too_few_bars(self, selector):
        data = _make_universe(n_stocks=15, n_bars=50)
        result = selector.select_factors(data)
        assert not result.success

    def test_empty_data(self, selector):
        result = selector.select_factors({})
        assert not result.success


class TestDatasetBuilding:
    def test_build_dataset_returns_arrays(self, selector, universe):
        X, y, names = selector._build_dataset(universe, None)
        assert len(X) > 0
        assert len(y) == len(X)
        assert len(names) == 13  # 13 features

    def test_feature_names(self, selector, universe):
        _, _, names = selector._build_dataset(universe, None)
        assert "momentum_6m" in names
        assert "volatility" in names
        assert "trend_strength" in names
        assert "revenue_growth" in names

    def test_no_nan_in_features(self, selector, universe):
        X, y, _ = selector._build_dataset(universe, None)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_features_clipped(self, selector, universe):
        X, _, _ = selector._build_dataset(universe, None)
        assert np.all(X >= -10)
        assert np.all(X <= 10)


class TestFeatureExtraction:
    def test_extract_features_valid(self, selector):
        df = _make_stock_df(n=300)
        close = df["close"].values
        volume = df["volume"].values
        features = selector._extract_features(close, volume, 260, {})
        assert features is not None
        assert len(features) == 13

    def test_extract_features_too_early(self, selector):
        df = _make_stock_df(n=300)
        close = df["close"].values
        volume = df["volume"].values
        features = selector._extract_features(close, volume, 50, {})
        assert features is None  # t < 252

    def test_extract_features_zero_price(self, selector):
        close = np.zeros(300)
        volume = np.ones(300) * 1_000_000
        features = selector._extract_features(close, volume, 260, {})
        assert features is None  # close[t] <= 0

    def test_fundamental_features_included(self, selector):
        df = _make_stock_df(n=300)
        close = df["close"].values
        volume = df["volume"].values
        fund = {"revenueGrowth": 0.25, "profitMargins": 0.15, "returnOnEquity": 0.20}
        features = selector._extract_features(close, volume, 260, fund)
        assert features is not None
        # revenue_growth should be 0.25 (index 8)
        assert features[8] == pytest.approx(0.25, abs=0.01)


class TestModelTraining:
    def test_training_succeeds(self, selector, universe):
        result = selector.select_factors(universe)
        assert result.success is True
        assert result.n_stocks == 15
        assert result.n_samples > 0

    def test_features_have_importance(self, selector, universe):
        result = selector.select_factors(universe)
        assert len(result.features) == 13
        # At least some features should have non-zero importance
        non_zero = [f for f in result.features if f.importance > 0]
        assert len(non_zero) > 0

    def test_importances_sum_to_one(self, selector, universe):
        result = selector.select_factors(universe)
        total = sum(f.importance for f in result.features)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_features_ranked(self, selector, universe):
        result = selector.select_factors(universe)
        assert result.features[0].rank == 1
        assert result.features[-1].rank == 13

    def test_top_features_sorted(self, selector, universe):
        result = selector.select_factors(universe)
        top = result.top_features
        assert len(top) == 13
        # First feature should have highest importance
        imp_map = {f.name: f.importance for f in result.features}
        assert imp_map[top[0]] >= imp_map[top[-1]]


class TestFactorWeightMapping:
    def test_to_factor_weights_sums_to_one(self, selector, universe):
        result = selector.select_factors(universe)
        weights = result.to_factor_weights()
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_to_factor_weights_keys(self, selector, universe):
        result = selector.select_factors(universe)
        weights = result.to_factor_weights()
        assert "growth" in weights
        assert "profitability" in weights
        assert "garp" in weights
        assert "momentum" in weights

    def test_empty_result_returns_defaults(self):
        result = FactorSelectionResult()
        weights = result.to_factor_weights()
        assert weights == {"growth": 0.35, "profitability": 0.30,
                           "garp": 0.20, "momentum": 0.15}


class TestWithFundamentals:
    def test_fundamentals_improve_dataset(self, selector, universe):
        fund = {
            sym: {
                "revenueGrowth": np.random.uniform(0, 0.5),
                "profitMargins": np.random.uniform(0.05, 0.3),
                "returnOnEquity": np.random.uniform(0.05, 0.25),
                "forwardPE": np.random.uniform(10, 40),
            }
            for sym in universe
        }
        result = selector.select_factors(universe, fundamental_data=fund)
        assert result.success is True
        # Fundamental features should have some importance
        fund_features = [f for f in result.features
                         if f.name in ("revenue_growth", "earnings_growth",
                                       "profit_margin", "roe", "garp_ratio")]
        # At least one fundamental feature should have non-zero importance
        assert any(f.importance > 0 for f in fund_features)


class TestCustomParams:
    def test_custom_forward_days(self):
        selector = MLFactorSelector(forward_days=5, n_estimators=10, max_depth=3)
        universe = _make_universe(n_stocks=12, n_bars=300)
        result = selector.select_factors(universe)
        assert result.success is True

    def test_more_estimators(self):
        selector = MLFactorSelector(n_estimators=50, max_depth=4)
        universe = _make_universe(n_stocks=12, n_bars=300)
        result = selector.select_factors(universe)
        assert result.success is True


class TestEdgeCases:
    def test_all_same_prices(self, selector):
        """All stocks with constant price → training may fail gracefully."""
        universe = {}
        for i in range(15):
            df = pd.DataFrame({
                "open": [100.0] * 300,
                "high": [100.0] * 300,
                "low": [100.0] * 300,
                "close": [100.0] * 300,
                "volume": [1_000_000.0] * 300,
            })
            universe[f"SYM{i:02d}"] = df
        result = selector.select_factors(universe)
        # May or may not succeed, but should not crash
        assert isinstance(result, FactorSelectionResult)

    def test_mixed_length_data(self, selector):
        """Stocks with different history lengths."""
        universe = {}
        for i in range(15):
            n = 300 + i * 10
            universe[f"SYM{i:02d}"] = _make_stock_df(n=n, seed=i)
        result = selector.select_factors(universe)
        assert result.success is True
