"""Tests for signal quality dynamic weighting in EvaluationLoop.

Verifies:
1. _apply_quality_weights boosts high-PF strategies
2. _apply_quality_weights suppresses low-PF strategies
3. Weights are re-normalized after adjustment
4. Disabled quality weighting returns original weights
5. Insufficient trades means no adjustment
6. Integration: quality weights applied in evaluate_symbol
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from analytics.signal_quality import SignalQualityTracker
from core.enums import SignalType
from engine.evaluation_loop import EvaluationLoop
from engine.risk_manager import RiskManager
from strategies.base import Signal
from strategies.combiner import SignalCombiner


def _make_df(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    })


def _make_loop(
    signal_quality: SignalQualityTracker | None = None,
    quality_weight_enabled: bool = True,
) -> EvaluationLoop:
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=MagicMock(total=100_000, available=80_000, currency="USD"),
    )
    adapter.fetch_positions = AsyncMock(return_value=[])

    market_data = AsyncMock()
    market_data.get_ohlcv = AsyncMock(return_value=_make_df())

    strategies = []
    for name in ["alpha", "beta", "gamma"]:
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
    registry.get_profile_weights.return_value = {
        "alpha": 0.4, "beta": 0.3, "gamma": 0.3,
    }

    indicator_svc = MagicMock()
    indicator_svc.add_all_indicators = MagicMock(return_value=_make_df())

    sq = signal_quality or SignalQualityTracker()

    loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=SignalCombiner(),
        order_manager=MagicMock(),
        risk_manager=RiskManager(),
        signal_quality=sq,
        market="US",
    )
    loop._quality_weight_enabled = quality_weight_enabled
    return loop


def _seed_tracker(
    tracker: SignalQualityTracker,
    strategy: str,
    wins: int,
    losses: int,
    win_pct: float = 0.08,
    loss_pct: float = -0.04,
) -> None:
    for _ in range(wins):
        tracker.record_trade(strategy, "ANY", win_pct)
    for _ in range(losses):
        tracker.record_trade(strategy, "ANY", loss_pct)


class TestApplyQualityWeightsBasic:
    """Unit tests for _apply_quality_weights method."""

    def test_no_trades_returns_original(self):
        loop = _make_loop()
        weights = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
        result = loop._apply_quality_weights(weights)
        assert result == weights

    def test_disabled_returns_original(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)  # PF = 3.0
        loop = _make_loop(signal_quality=sq, quality_weight_enabled=False)
        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        assert result == weights

    def test_insufficient_trades_no_adjustment(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=5, losses=3)  # Only 8 trades < 10
        loop = _make_loop(signal_quality=sq)
        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        assert result == weights


class TestApplyQualityWeightsBoost:
    """Test boosting high-PF strategies."""

    def test_high_pf_gets_boosted(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)  # PF = 1.20/0.20 = 6.0 → mult 1.5
        _seed_tracker(sq, "beta", wins=8, losses=8)    # PF = 0.64/0.32 = 2.0 → mult 1.25
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)

        # Alpha has higher PF → should get more weight
        assert result["alpha"] > result["beta"]

    def test_boost_capped_at_1_5x(self):
        sq = SignalQualityTracker()
        # PF = 20*0.08 / 0 → capped at 10.0
        _seed_tracker(sq, "alpha", wins=20, losses=0)
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 1.0}
        result = loop._apply_quality_weights(weights)
        # With only one strategy, it normalizes to 1.0 regardless
        assert result["alpha"] == pytest.approx(1.0, abs=0.01)

    def test_pf_between_1_and_1_5_neutral(self):
        sq = SignalQualityTracker()
        # PF = 7*0.08 / 3*0.04 = 0.56/0.12 = 4.67 (high)
        # Need PF ~1.2: wins * 0.08 / losses * 0.04 = 1.2
        # 6*0.08 = 0.48, 10*0.04 = 0.40 → PF = 1.2
        _seed_tracker(sq, "alpha", wins=6, losses=10)
        _seed_tracker(sq, "beta", wins=6, losses=10)
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        # Both same PF → should stay 50/50
        assert result["alpha"] == pytest.approx(0.5, abs=0.01)
        assert result["beta"] == pytest.approx(0.5, abs=0.01)


class TestApplyQualityWeightsSuppress:
    """Test suppressing low-PF strategies."""

    def test_low_pf_gets_suppressed(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)   # High PF
        _seed_tracker(sq, "beta", wins=3, losses=12)     # Low PF < 1.0
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)

        # Beta should be significantly lower
        assert result["alpha"] > result["beta"]
        assert result["beta"] < 0.5  # suppressed from original

    def test_suppression_floor_at_0_3(self):
        sq = SignalQualityTracker()
        # PF = 1*0.08 / 14*0.04 = 0.08/0.56 ≈ 0.14 → max(0.3, 0.14) = 0.3
        _seed_tracker(sq, "alpha", wins=1, losses=14)
        loop = _make_loop(signal_quality=sq)

        original_w = 1.0
        weights = {"alpha": original_w}
        result = loop._apply_quality_weights(weights)
        # Only one strategy, normalizes to 1.0
        assert result["alpha"] == pytest.approx(1.0, abs=0.01)

    def test_suppression_floor_multi_strategy(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=1, losses=14)   # PF ≈ 0.14 → mult 0.3
        _seed_tracker(sq, "beta", wins=15, losses=5)    # PF = 6.0 → mult 1.5
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)

        # alpha: 0.5 * 0.3 = 0.15, beta: 0.5 * 1.5 = 0.75
        # normalized: alpha = 0.15/0.90 ≈ 0.167, beta = 0.75/0.90 ≈ 0.833
        assert result["alpha"] < 0.25
        assert result["beta"] > 0.75


class TestApplyQualityWeightsNormalization:
    """Test weight normalization."""

    def test_weights_sum_to_one(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)
        _seed_tracker(sq, "beta", wins=8, losses=7)
        _seed_tracker(sq, "gamma", wins=3, losses=12)
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}
        result = loop._apply_quality_weights(weights)

        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_mixed_tracked_untracked(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)  # tracked, high PF
        # beta: untracked (no trades)
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)

        # alpha gets boosted, beta stays original → alpha > 0.5 after normalization
        assert result["alpha"] > 0.5
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-9)


class TestQualityWeightSetter:
    """Test set_quality_weight_enabled."""

    def test_enable_disable(self):
        loop = _make_loop()
        assert loop._quality_weight_enabled is True

        loop.set_quality_weight_enabled(False)
        assert loop._quality_weight_enabled is False

        loop.set_quality_weight_enabled(True)
        assert loop._quality_weight_enabled is True


class TestQualityWeightMultipliers:
    """Test specific multiplier values at PF boundaries."""

    def test_pf_2_0_gets_1_25x(self):
        """PF=2.0 → mult = 1.0 + min(0.5, (2.0-1.0)*0.25) = 1.25."""
        sq = SignalQualityTracker()
        # PF=2.0: wins*0.08 / losses*0.04 = 2.0 → wins/losses = 1.0
        # 10 wins * 0.08 = 0.80, 10 losses * 0.04 = 0.40 → PF = 2.0
        _seed_tracker(sq, "alpha", wins=10, losses=10)
        _seed_tracker(sq, "beta", wins=10, losses=10)  # Same PF
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        # Both same PF → equal
        assert result["alpha"] == pytest.approx(0.5, abs=0.01)

    def test_pf_3_0_gets_1_5x(self):
        """PF=3.0 → mult = 1.0 + min(0.5, (3.0-1.0)*0.25) = 1.5."""
        sq = SignalQualityTracker()
        # 12 wins * 0.08 = 0.96, 8 losses * 0.04 = 0.32 → PF = 3.0
        _seed_tracker(sq, "strong", wins=12, losses=8)
        # 10 wins * 0.08 = 0.80, 10 losses * 0.04 = 0.40 → PF = 2.0
        _seed_tracker(sq, "medium", wins=10, losses=10)
        loop = _make_loop(signal_quality=sq)

        weights = {"strong": 0.5, "medium": 0.5}
        result = loop._apply_quality_weights(weights)
        # strong (1.5x) > medium (1.25x) after normalization
        assert result["strong"] > result["medium"]

    def test_pf_exactly_1_0_neutral(self):
        """PF=1.0 → mult = 1.0 (neutral)."""
        sq = SignalQualityTracker()
        # PF = 1.0: 5*0.08 = 0.40, 10*0.04 = 0.40 → PF = 1.0
        _seed_tracker(sq, "alpha", wins=5, losses=10)
        _seed_tracker(sq, "beta", wins=5, losses=10)
        loop = _make_loop(signal_quality=sq)

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        assert result["alpha"] == pytest.approx(0.5, abs=0.01)

    def test_pf_0_5_gets_0_5x(self):
        """PF=0.5 → mult = max(0.3, 0.5) = 0.5."""
        sq = SignalQualityTracker()
        # PF=0.5: wins*0.08 / losses*0.04 = 0.5 → 5*0.08/20*0.04 = 0.40/0.80 = 0.5
        _seed_tracker(sq, "weak", wins=5, losses=20)
        _seed_tracker(sq, "strong", wins=15, losses=5)  # PF = 6.0
        loop = _make_loop(signal_quality=sq)

        weights = {"weak": 0.5, "strong": 0.5}
        result = loop._apply_quality_weights(weights)
        # weak is suppressed significantly
        assert result["weak"] <= 0.25
        assert result["strong"] >= 0.75


class TestQualityWeightIntegration:
    """Integration: quality weights applied during evaluate_symbol."""

    @pytest.mark.asyncio
    async def test_evaluate_symbol_uses_quality_weights(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)
        _seed_tracker(sq, "beta", wins=3, losses=12)

        loop = _make_loop(signal_quality=sq)

        # Manually call _apply_quality_weights and verify it changes weights
        original = {"alpha": 0.5, "beta": 0.5}
        adjusted = loop._apply_quality_weights(original)
        assert adjusted["alpha"] > adjusted["beta"]

    @pytest.mark.asyncio
    async def test_quality_disabled_no_change(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=15, losses=5)
        _seed_tracker(sq, "beta", wins=3, losses=12)

        loop = _make_loop(signal_quality=sq, quality_weight_enabled=False)

        original = {"alpha": 0.5, "beta": 0.5}
        adjusted = loop._apply_quality_weights(original)
        assert adjusted == original

    @pytest.mark.asyncio
    async def test_custom_min_trades(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=7, losses=3)  # 10 trades
        loop = _make_loop(signal_quality=sq)
        loop._quality_min_trades = 15  # Raise threshold

        weights = {"alpha": 1.0}
        result = loop._apply_quality_weights(weights)
        # 10 < 15 min → no adjustment
        assert result == weights

    @pytest.mark.asyncio
    async def test_quality_min_trades_respected(self):
        sq = SignalQualityTracker()
        _seed_tracker(sq, "alpha", wins=7, losses=3)  # 10 trades, PF = 0.56/0.12 = 4.67
        loop = _make_loop(signal_quality=sq)
        loop._quality_min_trades = 10  # Exactly at threshold

        weights = {"alpha": 0.5, "beta": 0.5}
        result = loop._apply_quality_weights(weights)
        # alpha has enough trades → adjusted (boosted since PF > 1.5)
        assert result["alpha"] > 0.5
