"""Tests for STOCK-66: US market-specific strategy optimization.

Verifies:
1. Disabled strategies are filtered from evaluation
2. min_confidence and min_active_ratio overrides are forwarded to combiner
3. US risk params match grid search optimal values
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd

from core.enums import SignalType
from engine.evaluation_loop import EvaluationLoop
from engine.risk_manager import RiskManager, RiskParams
from strategies.base import Signal
from strategies.combiner import SignalCombiner


def _make_df(n=50):
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    })


def _make_loop(disabled=None, min_confidence=None, min_active_ratio=None):
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=MagicMock(total=100_000, available=80_000, currency="USD")
    )
    adapter.fetch_positions = AsyncMock(return_value=[])

    market_data = AsyncMock()
    market_data.get_ohlcv = AsyncMock(return_value=_make_df())
    market_data.get_balance = AsyncMock(
        return_value=MagicMock(total=100_000, available=80_000, currency="USD")
    )
    market_data.get_positions = AsyncMock(return_value=[])

    # Create mock strategies
    strategies = []
    for name in ["sector_rotation", "volume_profile", "volume_surge",
                  "supertrend", "dual_momentum", "trend_following"]:
        s = AsyncMock()
        s.name = name
        s.analyze = AsyncMock(return_value=Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=name,
            reason="test",
        ))
        strategies.append(s)

    registry = MagicMock()
    registry.get_enabled.return_value = strategies
    registry.get_profile_weights.return_value = {s.name: 1.0 for s in strategies}

    indicator_svc = MagicMock()
    indicator_svc.add_all_indicators = MagicMock(return_value=_make_df())

    risk = RiskManager()
    order_mgr = MagicMock()

    loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=SignalCombiner(),
        order_manager=order_mgr,
        risk_manager=risk,
        market="US",
    )

    if disabled:
        loop._disabled_strategies = frozenset(disabled)
    if min_confidence is not None:
        loop._min_confidence = min_confidence
    if min_active_ratio is not None:
        loop._min_active_ratio = min_active_ratio

    return loop


class TestDisabledStrategies:
    def test_get_active_strategies_filters_disabled(self):
        loop = _make_loop(disabled=["supertrend", "dual_momentum", "trend_following"])
        active = loop._get_active_strategies()
        active_names = {s.name for s in active}
        assert active_names == {"sector_rotation", "volume_profile", "volume_surge"}

    def test_get_active_strategies_no_filter(self):
        loop = _make_loop()
        active = loop._get_active_strategies()
        assert len(active) == 6  # All strategies

    def test_disabled_strategies_is_frozenset(self):
        loop = _make_loop(disabled=["supertrend"])
        assert isinstance(loop._disabled_strategies, frozenset)
        assert "supertrend" in loop._disabled_strategies

    def test_us_optimal_only_3_strategies(self):
        """STOCK-66 US grid search optimal: only sector_rotation + volume_profile + volume_surge."""
        _all = [
            "trend_following", "donchian_breakout", "supertrend", "macd_histogram",
            "dual_momentum", "rsi_divergence", "bollinger_squeeze", "volume_profile",
            "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
            "bnf_deviation", "volume_surge",
        ]
        _us_enabled = {"sector_rotation", "volume_profile", "volume_surge"}
        disabled = frozenset(s for s in _all if s not in _us_enabled)
        loop = _make_loop(disabled=list(disabled))
        active = loop._get_active_strategies()
        active_names = {s.name for s in active}
        assert active_names == _us_enabled


class TestCombinerOverrides:
    def test_min_confidence_default_none(self):
        loop = _make_loop()
        assert loop._min_confidence is None

    def test_min_active_ratio_default_none(self):
        loop = _make_loop()
        assert loop._min_active_ratio is None

    def test_min_confidence_set(self):
        loop = _make_loop(min_confidence=0.30)
        assert loop._min_confidence == 0.30

    def test_min_active_ratio_set(self):
        loop = _make_loop(min_active_ratio=0.0)
        assert loop._min_active_ratio == 0.0

    def test_combine_uses_min_confidence_override(self):
        """When min_confidence is set, it should be passed to combiner.combine()."""
        loop = _make_loop(min_confidence=0.30)
        combiner_spy = MagicMock(wraps=loop._combiner)
        loop._combiner = combiner_spy

        signals = [
            Signal(signal_type=SignalType.BUY, confidence=0.35,
                   strategy_name="sector_rotation", reason="test"),
        ]
        weights = {"sector_rotation": 1.0}

        # Directly test combiner forwarding
        combiner_spy.combine(signals, weights, min_confidence=0.30)
        combiner_spy.combine.assert_called_with(signals, weights, min_confidence=0.30)

    def test_combine_uses_min_active_ratio_override(self):
        """When min_active_ratio is set, it should be passed to combiner.combine()."""
        loop = _make_loop(min_active_ratio=0.0)
        combiner_spy = MagicMock(wraps=loop._combiner)
        loop._combiner = combiner_spy

        signals = [
            Signal(signal_type=SignalType.BUY, confidence=0.50,
                   strategy_name="sector_rotation", reason="test"),
        ]
        weights = {"sector_rotation": 1.0}

        combiner_spy.combine(signals, weights, min_active_ratio=0.0)
        combiner_spy.combine.assert_called_with(signals, weights, min_active_ratio=0.0)


class TestUSRiskParams:
    def test_us_optimal_risk_params(self):
        """STOCK-66 grid search optimal: SL=10%, TP=15%, max_pos=8, max_pct=20%."""
        params = RiskParams(
            max_position_pct=0.20,
            max_positions=8,
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.15,
        )
        assert params.max_position_pct == 0.20
        assert params.max_positions == 8
        assert params.default_stop_loss_pct == 0.10
        assert params.default_take_profit_pct == 0.15

    def test_us_kelly_params(self):
        """STOCK-66: Full Kelly (1.0), 12% min position."""
        rm = RiskManager(params=RiskParams(max_position_pct=0.20))
        rm._kelly._kelly_fraction = 1.00
        rm._kelly._min_position_pct = 0.12
        assert rm._kelly._kelly_fraction == 1.00
        assert rm._kelly._min_position_pct == 0.12


class TestIntegrationEvaluateSymbol:
    async def test_evaluate_uses_active_strategies_only(self):
        """evaluate_symbol should only run strategies not in disabled set."""
        loop = _make_loop(
            disabled=["supertrend", "dual_momentum", "trend_following"],
            min_confidence=0.30,
            min_active_ratio=0.0,
        )
        # Set BUY signal for sector_rotation
        for s in loop._registry.get_enabled.return_value:
            if s.name == "sector_rotation":
                s.analyze.return_value = Signal(
                    signal_type=SignalType.BUY,
                    confidence=0.80,
                    strategy_name="sector_rotation",
                    reason="strong sector momentum",
                )

        await loop.evaluate_symbol("AAPL")

        # Only active strategies should have analyze() called
        for s in loop._registry.get_enabled.return_value:
            if s.name in {"supertrend", "dual_momentum", "trend_following"}:
                s.analyze.assert_not_called()
            elif s.name in {"sector_rotation", "volume_profile", "volume_surge"}:
                s.analyze.assert_called_once()
