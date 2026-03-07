"""Tests for AdaptiveBacktestEngine."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backtest.adaptive_backtest import (
    AdaptiveBacktestEngine,
    AdaptiveBacktestResult,
    WeightModeResult,
)
from backtest.simulator import SimConfig
from engine.stock_classifier import StockCategory, StockProfile


def _make_ohlcv(n=200, base=100.0, trend=0.001, seed=42):
    """Generate realistic OHLCV data."""
    np.random.seed(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = [base]
    for _ in range(n - 1):
        close.append(close[-1] * (1 + trend + np.random.normal(0, 0.015)))
    close = np.array(close)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(500000, 2000000, n),
        },
        index=dates,
    )


class TestAdaptiveBacktestEngine:
    def test_get_mode_weights_equal(self):
        engine = AdaptiveBacktestEngine()
        names = ["s1", "s2", "s3"]
        from engine.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()
        result = engine._get_mode_weights("equal", names, {}, mgr, "AAPL")
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 0.01
        for v in result.values():
            assert abs(v - 1/3) < 0.01

    def test_get_mode_weights_market_state(self):
        engine = AdaptiveBacktestEngine()
        names = ["s1", "s2"]
        market_weights = {"s1": 0.7, "s2": 0.3}
        from engine.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()
        result = engine._get_mode_weights(
            "market_state", names, market_weights, mgr, "AAPL"
        )
        assert result["s1"] > result["s2"]
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_get_mode_weights_category(self):
        engine = AdaptiveBacktestEngine()
        names = ["dual_momentum", "trend_following"]
        market_weights = {"dual_momentum": 0.5, "trend_following": 0.5}
        from engine.adaptive_weights import AdaptiveWeightManager
        mgr = AdaptiveWeightManager()
        mgr.set_category("NVDA", StockCategory.GROWTH_MOMENTUM)
        result = engine._get_mode_weights(
            "category", names, market_weights, mgr, "NVDA"
        )
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_aggregate_equity(self):
        engine = AdaptiveBacktestEngine()
        dates = pd.bdate_range("2021-01-01", periods=5)
        eq1 = pd.Series([100, 101, 102, 103, 104], index=dates)
        eq2 = pd.Series([100, 99, 101, 102, 105], index=dates)
        result = engine._aggregate_equity({"A": eq1, "B": eq2})
        assert len(result) == 5
        assert result.iloc[0] == 200
        assert result.iloc[-1] == 209


class TestAdaptiveBacktestResult:
    def test_to_dict(self):
        from backtest.metrics import BacktestMetrics
        m = BacktestMetrics(
            total_return_pct=15.0, cagr=0.12, sharpe_ratio=1.5,
            sortino_ratio=2.0, max_drawdown_pct=-10.0,
            total_trades=20, win_rate=55.0, profit_factor=1.8,
            final_equity=115000, start_date="2021-01-01", end_date="2024-01-01",
        )
        mode_result = WeightModeResult(
            mode="equal",
            metrics=m,
            per_symbol={"AAPL": m},
            categories={"AAPL": "stable_large_cap"},
            total_trades=20,
        )
        result = AdaptiveBacktestResult(
            symbols=["AAPL"],
            period="3y",
            strategies_used=["s1"],
            modes={"equal": mode_result},
        )
        d = result.to_dict()
        assert "modes" in d
        assert "equal" in d["modes"]
        assert d["modes"]["equal"]["cagr"] == 12.0
        assert d["modes"]["equal"]["per_symbol"]["AAPL"]["cagr"] == 12.0

    def test_to_dict_handles_inf(self):
        from backtest.metrics import BacktestMetrics
        m = BacktestMetrics(
            profit_factor=float("inf"), sharpe_ratio=float("nan"),
        )
        mode_result = WeightModeResult(mode="equal", metrics=m, total_trades=0)
        result = AdaptiveBacktestResult(
            symbols=[], period="3y", strategies_used=[],
            modes={"equal": mode_result},
        )
        d = result.to_dict()
        assert d["modes"]["equal"]["profit_factor"] == 0.0
        assert d["modes"]["equal"]["sharpe_ratio"] == 0.0

    def test_summary(self):
        from backtest.metrics import BacktestMetrics
        m = BacktestMetrics(
            cagr=0.15, sharpe_ratio=1.5, max_drawdown_pct=-12.0,
            win_rate=55.0, total_trades=30,
        )
        result = AdaptiveBacktestResult(
            symbols=["AAPL"], period="3y", strategies_used=["s1"],
            modes={"equal": WeightModeResult(mode="equal", metrics=m, total_trades=30)},
        )
        s = result.summary()
        assert "equal" in s
        assert "15.0%" in s
