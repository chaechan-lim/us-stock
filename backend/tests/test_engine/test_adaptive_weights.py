"""Tests for AdaptiveWeightManager."""

import pytest

from engine.adaptive_weights import AdaptiveWeightManager
from engine.stock_classifier import StockCategory


@pytest.fixture
def manager():
    return AdaptiveWeightManager(alpha=0.6, min_signals_for_adaptation=3)


class TestAdaptiveWeightManager:
    def test_no_category_returns_market_weights(self, manager):
        market_weights = {"trend_following": 0.5, "rsi_divergence": 0.5}
        result = manager.get_weights("AAPL", market_weights)
        assert result == market_weights

    def test_category_blends_with_market_weights(self, manager):
        manager.set_category("NVDA", StockCategory.GROWTH_MOMENTUM)
        market_weights = {"trend_following": 1.0}
        result = manager.get_weights("NVDA", market_weights)
        # Should have more strategies than just trend_following
        assert len(result) > 1
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)

    def test_adaptive_weights_after_enough_signals(self, manager):
        manager.set_category("AAPL", StockCategory.STABLE_LARGE_CAP)
        market_weights = {"trend_following": 0.5, "rsi_divergence": 0.5}

        # Record enough correct signals for trend_following
        for _ in range(5):
            manager.record_signal_result("AAPL", "trend_following", True)
        for _ in range(5):
            manager.record_signal_result("AAPL", "rsi_divergence", False)

        result = manager.get_weights("AAPL", market_weights)
        # trend_following should have higher weight (it was more accurate)
        assert result.get("trend_following", 0) > result.get("rsi_divergence", 0)

    def test_weights_always_sum_to_one(self, manager):
        manager.set_category("TEST", StockCategory.HIGH_VOLATILITY)
        market_weights = {"a": 0.3, "b": 0.7}
        result = manager.get_weights("TEST", market_weights)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)

    def test_performance_summary(self, manager):
        manager.record_signal_result("AAPL", "trend_following", True)
        manager.record_signal_result("AAPL", "trend_following", True)
        manager.record_signal_result("AAPL", "trend_following", False)

        summary = manager.get_performance_summary("AAPL")
        assert "trend_following" in summary
        assert summary["trend_following"]["total_signals"] == 3
        assert 0 < summary["trend_following"]["accuracy"] < 1

    def test_get_category(self, manager):
        assert manager.get_category("AAPL") is None
        manager.set_category("AAPL", StockCategory.STABLE_LARGE_CAP)
        assert manager.get_category("AAPL") == StockCategory.STABLE_LARGE_CAP

    def test_ema_decay(self, manager):
        """Verify EMA gives more weight to recent results."""
        # Start with all correct
        for _ in range(10):
            manager.record_signal_result("X", "s1", True)
        acc_high = manager.get_performance_summary("X")["s1"]["accuracy"]

        # Then all wrong
        for _ in range(10):
            manager.record_signal_result("X", "s1", False)
        acc_low = manager.get_performance_summary("X")["s1"]["accuracy"]

        assert acc_low < acc_high

    def test_all_summaries(self, manager):
        manager.record_signal_result("AAPL", "s1", True)
        manager.record_signal_result("NVDA", "s2", False)
        summaries = manager.get_all_summaries()
        assert "AAPL" in summaries
        assert "NVDA" in summaries
