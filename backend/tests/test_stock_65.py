"""Tests for STOCK-65: KR market strategy optimization.

Covers:
- RiskParams: kelly_fraction, min_position_pct, dynamic_sl_tp fields
- RiskManager: KellyPositionSizer initialized with new params
- RiskManager.calculate_dynamic_sl_tp: static mode when dynamic_sl_tp=False
- StrategyConfigLoader: market-specific config methods
- EvaluationLoop.set_disabled_strategies: filters strategies by market
- EvaluationLoop._min_confidence: per-market min_confidence override
- _apply_kr_eval_overrides: YAML key → EvaluationLoop attribute mapping
- strategies.yaml: KR section presence and values
"""

import pytest

from engine.evaluation_loop import EvaluationLoop
from engine.risk_manager import RiskManager, RiskParams
from strategies.config_loader import StrategyConfigLoader

# ---------------------------------------------------------------------------
# RiskParams: new fields
# ---------------------------------------------------------------------------


class TestRiskParamsNewFields:
    def test_default_kelly_fraction(self):
        params = RiskParams()
        assert params.kelly_fraction == 0.40

    def test_default_min_position_pct(self):
        params = RiskParams()
        assert params.min_position_pct == 0.05

    def test_default_dynamic_sl_tp(self):
        params = RiskParams()
        assert params.dynamic_sl_tp is True

    def test_custom_kelly_fraction(self):
        params = RiskParams(kelly_fraction=0.50)
        assert params.kelly_fraction == 0.50

    def test_custom_min_position_pct(self):
        params = RiskParams(min_position_pct=0.12)
        assert params.min_position_pct == 0.12

    def test_disable_dynamic_sl_tp(self):
        params = RiskParams(dynamic_sl_tp=False)
        assert params.dynamic_sl_tp is False


# ---------------------------------------------------------------------------
# RiskManager: KellyPositionSizer receives new params
# ---------------------------------------------------------------------------


class TestRiskManagerKellyParams:
    """Behavioral tests: verify RiskParams propagate to calculate_kelly_position_size output."""

    def test_kelly_fraction_affects_position_size(self):
        """Higher kelly_fraction produces a larger allocation when there is positive edge."""
        params_low = RiskParams(kelly_fraction=0.25, max_position_pct=0.20)
        params_high = RiskParams(kelly_fraction=0.50, max_position_pct=0.20)
        rm_low = RiskManager(params=params_low)
        rm_high = RiskManager(params=params_high)

        kwargs = dict(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000.0,
            cash_available=100_000.0,
            current_positions=0,
            win_rate=0.60,
            avg_win=0.12,
            avg_loss=0.06,
            signal_confidence=0.80,
        )
        result_low = rm_low.calculate_kelly_position_size(**kwargs)
        result_high = rm_high.calculate_kelly_position_size(**kwargs)

        assert result_low.allowed
        assert result_high.allowed
        assert result_high.allocation_usd > result_low.allocation_usd

    def test_min_position_pct_sets_allocation_floor(self):
        """min_position_pct clamps the Kelly allocation upward when edge is marginal."""
        params = RiskParams(kelly_fraction=0.40, min_position_pct=0.12, max_position_pct=0.20)
        rm = RiskManager(params=params)

        # Marginal edge + low confidence → raw Kelly would be below 0.12 floor
        result = rm.calculate_kelly_position_size(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000.0,
            cash_available=100_000.0,
            current_positions=0,
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.07,
            signal_confidence=0.35,
        )

        assert result.allowed
        assert result.allocation_usd >= 100_000.0 * 0.12

    def test_max_position_pct_caps_allocation(self):
        """max_position_pct is the hard cap on Kelly allocation."""
        params = RiskParams(kelly_fraction=0.50, max_position_pct=0.20)
        rm = RiskManager(params=params)

        # Strong edge would exceed 0.20 without the cap
        result = rm.calculate_kelly_position_size(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000.0,
            cash_available=100_000.0,
            current_positions=0,
            win_rate=0.70,
            avg_win=0.20,
            avg_loss=0.05,
            signal_confidence=0.90,
        )

        assert result.allowed
        assert result.allocation_usd <= 100_000.0 * 0.20


# ---------------------------------------------------------------------------
# RiskManager.calculate_dynamic_sl_tp: static mode
# ---------------------------------------------------------------------------


class TestDynamicSlTpStaticMode:
    def test_static_mode_returns_defaults(self):
        """When dynamic_sl_tp=False, always return configured defaults."""
        params = RiskParams(
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.15,
        )
        rm = RiskManager(params=params)
        # Even with valid ATR, should return defaults
        sl, tp = rm.calculate_dynamic_sl_tp(50000.0, 500.0, market="KR")
        assert sl == 0.10
        assert tp == 0.15

    def test_static_mode_ignores_atr(self):
        """Static mode ignores ATR entirely, not just zero-ATR fallback."""
        params = RiskParams(
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.15,
        )
        rm = RiskManager(params=params)
        # High ATR that would normally give wider SL/TP
        sl_high_atr, tp_high_atr = rm.calculate_dynamic_sl_tp(100.0, 20.0)
        # Low ATR
        sl_low_atr, tp_low_atr = rm.calculate_dynamic_sl_tp(100.0, 0.1)
        # Both should give same defaults
        assert sl_high_atr == sl_low_atr == 0.10
        assert tp_high_atr == tp_low_atr == 0.15

    def test_dynamic_mode_still_uses_atr(self):
        """When dynamic_sl_tp=True (default), ATR-based calculation proceeds."""
        params = RiskParams(dynamic_sl_tp=True, default_stop_loss_pct=0.08)
        rm = RiskManager(params=params)
        # High ATR should widen SL beyond the configured default
        sl_high_atr, _ = rm.calculate_dynamic_sl_tp(100.0, 8.0)
        # Zero ATR falls back to the configured default (atr <= 0 guard)
        sl_zero_atr, _ = rm.calculate_dynamic_sl_tp(100.0, 0.0)
        assert sl_high_atr != pytest.approx(0.08), "Dynamic mode should widen SL vs default"
        assert sl_zero_atr == pytest.approx(0.08), "Zero-ATR should fall back to default"


# ---------------------------------------------------------------------------
# KR risk params match spec values (STOCK-65)
# ---------------------------------------------------------------------------


class TestKRRiskParamsSpec:
    """Verify the exact KR optimized risk params can be expressed in RiskParams."""

    def test_kr_optimized_params(self):
        params = RiskParams(
            kelly_fraction=0.50,
            max_position_pct=0.20,
            min_position_pct=0.12,
            max_positions=8,
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.15,
            dynamic_sl_tp=False,
        )
        assert params.kelly_fraction == 0.50
        assert params.max_position_pct == 0.20
        assert params.min_position_pct == 0.12
        assert params.max_positions == 8
        assert params.default_stop_loss_pct == 0.10
        assert params.default_take_profit_pct == 0.15
        assert params.dynamic_sl_tp is False

    def test_kr_static_sl_tp_returns_spec_values(self):
        params = RiskParams(
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.10,
            default_take_profit_pct=0.15,
        )
        rm = RiskManager(params=params)
        sl, tp = rm.calculate_dynamic_sl_tp(50000.0, 1000.0, market="KR")
        assert sl == 0.10
        assert tp == 0.15

    def test_kr_max_positions(self):
        params = RiskParams(max_positions=8)
        rm = RiskManager(params=params)
        # 8 positions filled → next buy rejected
        result = rm.calculate_position_size(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=8,
        )
        assert result.allowed is False
        assert "Max positions" in result.reason

    def test_kr_max_position_pct(self):
        params = RiskParams(max_positions=10, max_position_pct=0.20)
        rm = RiskManager(params=params)
        result = rm.calculate_position_size(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=2,
        )
        assert result.allowed is True
        # Should not exceed 20% of portfolio
        assert result.allocation_usd <= 10_000_000 * 0.20


# ---------------------------------------------------------------------------
# StrategyConfigLoader: market-specific methods
# ---------------------------------------------------------------------------


class TestStrategyConfigLoaderMarketMethods:
    def test_get_market_config_kr(self):
        loader = StrategyConfigLoader()
        # Use typed wrappers — _get_market_config is private
        disabled = loader.get_market_disabled_strategies("KR")
        risk = loader.get_market_risk_config("KR")
        assert isinstance(disabled, list)
        assert isinstance(risk, dict)
        assert len(disabled) > 0

    def test_get_market_disabled_strategies_kr(self):
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("KR")
        assert isinstance(disabled, list)
        assert len(disabled) > 0
        # All strategies except supertrend and dual_momentum should be disabled
        assert "supertrend" not in disabled
        assert "dual_momentum" not in disabled
        # These should be disabled
        assert "trend_following" in disabled
        assert "donchian_breakout" in disabled
        assert "macd_histogram" in disabled
        assert "rsi_divergence" in disabled
        assert "bollinger_squeeze" in disabled
        assert "volume_profile" in disabled
        assert "regime_switch" in disabled
        assert "sector_rotation" in disabled
        assert "cis_momentum" in disabled
        assert "larry_williams" in disabled
        assert "bnf_deviation" in disabled
        assert "volume_surge" in disabled

    def test_get_market_risk_config_kr(self):
        loader = StrategyConfigLoader()
        risk_cfg = loader.get_market_risk_config("KR")
        assert isinstance(risk_cfg, dict)
        assert risk_cfg.get("kelly_fraction") == pytest.approx(0.50)
        assert risk_cfg.get("max_position_pct") == pytest.approx(0.20)
        assert risk_cfg.get("min_position_pct") == pytest.approx(0.12)
        assert risk_cfg.get("max_positions") == 8
        assert risk_cfg.get("default_stop_loss_pct") == pytest.approx(0.10)
        assert risk_cfg.get("default_take_profit_pct") == pytest.approx(0.15)
        assert risk_cfg.get("dynamic_sl_tp") is False

    def test_get_market_evaluation_loop_config_kr(self):
        loader = StrategyConfigLoader()
        eval_cfg = loader.get_market_evaluation_loop_config("KR")
        assert isinstance(eval_cfg, dict)
        assert eval_cfg.get("min_confidence") == pytest.approx(0.30)
        assert eval_cfg.get("min_active_ratio") is None  # null in YAML = no override
        assert eval_cfg.get("sell_cooldown_days") == 1
        assert eval_cfg.get("whipsaw_max_losses") == 2
        assert eval_cfg.get("min_hold_days") == 1

    def test_get_market_config_us_returns_empty(self):
        loader = StrategyConfigLoader()
        # US market doesn't have overrides in the config — typed wrappers return empty
        assert loader.get_market_disabled_strategies("US") == []
        assert loader.get_market_risk_config("US") == {}
        assert loader.get_market_evaluation_loop_config("US") == {}

    def test_get_market_disabled_strategies_unknown_market(self):
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("UNKNOWN")
        assert disabled == []

    def test_get_market_risk_config_unknown_market(self):
        loader = StrategyConfigLoader()
        risk_cfg = loader.get_market_risk_config("UNKNOWN")
        assert risk_cfg == {}

    def test_get_market_evaluation_loop_config_unknown_market(self):
        loader = StrategyConfigLoader()
        eval_cfg = loader.get_market_evaluation_loop_config("UNKNOWN")
        assert eval_cfg == {}


# ---------------------------------------------------------------------------
# EvaluationLoop: disabled strategies filtering
# ---------------------------------------------------------------------------


class TestEvaluationLoopDisabledStrategies:
    def _make_evaluation_loop(self) -> EvaluationLoop:
        """Create a minimal EvaluationLoop for testing."""
        from unittest.mock import MagicMock

        from engine.risk_manager import RiskManager
        from strategies.combiner import SignalCombiner
        from strategies.registry import StrategyRegistry

        adapter = MagicMock()
        market_data = MagicMock()
        indicator_svc = MagicMock()
        registry = MagicMock(spec=StrategyRegistry)
        combiner = SignalCombiner()
        order_manager = MagicMock()
        risk_manager = RiskManager()

        loop = EvaluationLoop(
            adapter=adapter,
            market_data=market_data,
            indicator_svc=indicator_svc,
            registry=registry,
            combiner=combiner,
            order_manager=order_manager,
            risk_manager=risk_manager,
            market="KR",
        )
        return loop

    def test_initial_disabled_strategies_empty(self):
        loop = self._make_evaluation_loop()
        assert loop._disabled_strategies == frozenset()

    def test_set_disabled_strategies(self):
        loop = self._make_evaluation_loop()
        loop.set_disabled_strategies(["trend_following", "macd_histogram"])
        assert "trend_following" in loop._disabled_strategies
        assert "macd_histogram" in loop._disabled_strategies

    def test_set_disabled_strategies_is_frozenset(self):
        loop = self._make_evaluation_loop()
        loop.set_disabled_strategies(["supertrend", "cis_momentum"])
        assert isinstance(loop._disabled_strategies, frozenset)

    def test_set_disabled_strategies_empty_list(self):
        loop = self._make_evaluation_loop()
        loop.set_disabled_strategies(["trend_following"])
        loop.set_disabled_strategies([])
        assert loop._disabled_strategies == frozenset()

    def test_set_disabled_strategies_kr_spec(self):
        """Verify all 12 KR-disabled strategies can be set."""
        loop = self._make_evaluation_loop()
        kr_disabled = [
            "trend_following",
            "donchian_breakout",
            "macd_histogram",
            "rsi_divergence",
            "bollinger_squeeze",
            "volume_profile",
            "regime_switch",
            "sector_rotation",
            "cis_momentum",
            "larry_williams",
            "bnf_deviation",
            "volume_surge",
        ]
        loop.set_disabled_strategies(kr_disabled)
        assert loop._disabled_strategies == frozenset(kr_disabled)
        # supertrend and dual_momentum must NOT be in disabled set
        assert "supertrend" not in loop._disabled_strategies
        assert "dual_momentum" not in loop._disabled_strategies

    def test_min_confidence_initial_value(self):
        loop = self._make_evaluation_loop()
        assert loop._min_confidence is None

    def test_min_confidence_can_be_set(self):
        loop = self._make_evaluation_loop()
        loop.set_min_confidence(0.30)
        assert loop._min_confidence == pytest.approx(0.30)

    def test_set_min_confidence_none_clears_override(self):
        loop = self._make_evaluation_loop()
        loop.set_min_confidence(0.40)
        loop.set_min_confidence(None)
        assert loop._min_confidence is None

    def test_set_min_confidence_out_of_range_raises(self):
        loop = self._make_evaluation_loop()
        with pytest.raises(ValueError, match="min_confidence must be in"):
            loop.set_min_confidence(1.5)
        with pytest.raises(ValueError, match="min_confidence must be in"):
            loop.set_min_confidence(-0.1)

    def test_min_active_ratio_initial_value(self):
        loop = self._make_evaluation_loop()
        assert loop._min_active_ratio is None

    def test_set_min_active_ratio(self):
        loop = self._make_evaluation_loop()
        loop.set_min_active_ratio(0.0)
        assert loop._min_active_ratio == pytest.approx(0.0)

    def test_set_min_active_ratio_none_clears_override(self):
        loop = self._make_evaluation_loop()
        loop.set_min_active_ratio(0.10)
        loop.set_min_active_ratio(None)
        assert loop._min_active_ratio is None

    def test_set_min_active_ratio_out_of_range_raises(self):
        loop = self._make_evaluation_loop()
        with pytest.raises(ValueError, match="min_active_ratio must be in"):
            loop.set_min_active_ratio(1.5)
        with pytest.raises(ValueError, match="min_active_ratio must be in"):
            loop.set_min_active_ratio(-0.1)

    @pytest.mark.asyncio
    async def test_min_confidence_forwarded_to_combiner_combine(self):
        """Verify min_confidence is forwarded through the real evaluate_symbol path."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import pandas as pd

        loop = self._make_evaluation_loop()
        loop.set_min_confidence(0.30)
        loop._registry.get_enabled.return_value = []  # no strategies → HOLD signal

        captured_kwargs: dict = {}
        original_combine = loop._combiner.combine

        def spy_combine(signals, weights, **kwargs):
            captured_kwargs.update(kwargs)
            return original_combine(signals, weights, **kwargs)

        loop._combiner.combine = spy_combine  # type: ignore[method-assign]

        # Minimal non-empty DataFrame so evaluate_symbol doesn't short-circuit
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.RangeIndex(2))
        loop._market_data.get_ohlcv = AsyncMock(return_value=df)
        # Ensure add_all_indicators passes the real df through so evaluate_symbol
        # does not short-circuit on a MagicMock before reaching self._combiner.combine
        loop._indicator_svc.add_all_indicators = MagicMock(return_value=df)

        with patch.object(loop, "_maybe_classify"):
            await loop.evaluate_symbol("005930")

        # The real evaluate_symbol code path must have forwarded min_confidence
        assert captured_kwargs.get("min_confidence") == pytest.approx(0.30)

    @pytest.mark.asyncio
    async def test_min_active_ratio_override_forwarded_to_combiner_combine(self):
        """Verify min_active_ratio override is forwarded through evaluate_symbol (is_held=True)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import pandas as pd

        loop = self._make_evaluation_loop()
        loop.set_min_active_ratio(0.0)
        loop._registry.get_enabled.return_value = []

        captured_kwargs: dict = {}
        original_combine = loop._combiner.combine

        def spy_combine(signals, weights, **kwargs):
            captured_kwargs.update(kwargs)
            return original_combine(signals, weights, **kwargs)

        loop._combiner.combine = spy_combine  # type: ignore[method-assign]

        df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.RangeIndex(2))
        loop._market_data.get_ohlcv = AsyncMock(return_value=df)
        loop._indicator_svc.add_all_indicators = MagicMock(return_value=df)

        with patch.object(loop, "_maybe_classify"):
            await loop.evaluate_symbol("005930", is_held=True)

        # min_active_ratio override must take precedence over the held-position 0.15 default
        assert captured_kwargs.get("min_active_ratio") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EvaluationLoop: strategy filtering integration
# ---------------------------------------------------------------------------


class TestEvaluationLoopStrategyFiltering:
    """Test that disabled strategies are actually filtered from evaluation."""

    def _make_mock_strategy(self, name: str):
        from unittest.mock import MagicMock

        from strategies.base import BaseStrategy

        strat = MagicMock(spec=BaseStrategy)
        strat.name = name
        return strat

    def _make_loop_with_strategies(self, strategy_names: list[str]) -> EvaluationLoop:
        from unittest.mock import MagicMock

        from engine.risk_manager import RiskManager
        from strategies.combiner import SignalCombiner
        from strategies.registry import StrategyRegistry

        strategies = [self._make_mock_strategy(n) for n in strategy_names]

        registry = MagicMock(spec=StrategyRegistry)
        registry.get_enabled.return_value = strategies

        loop = EvaluationLoop(
            adapter=MagicMock(),
            market_data=MagicMock(),
            indicator_svc=MagicMock(),
            registry=registry,
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="KR",
        )
        return loop, strategies

    def test_no_disabled_returns_all_strategies(self):
        loop, strategies = self._make_loop_with_strategies(
            ["supertrend", "dual_momentum", "trend_following"]
        )
        active = loop._get_active_strategies()
        assert len(active) == 3

    def test_disabled_strategies_filtered_out(self):
        loop, strategies = self._make_loop_with_strategies(
            ["supertrend", "dual_momentum", "trend_following", "macd_histogram"]
        )
        loop.set_disabled_strategies(["trend_following", "macd_histogram"])
        active = loop._get_active_strategies()
        assert len(active) == 2
        names = [s.name for s in active]
        assert "supertrend" in names
        assert "dual_momentum" in names
        assert "trend_following" not in names
        assert "macd_histogram" not in names

    def test_kr_disabled_leaves_only_supertrend_and_dual_momentum(self):
        all_strategy_names = [
            "trend_following",
            "dual_momentum",
            "donchian_breakout",
            "supertrend",
            "macd_histogram",
            "rsi_divergence",
            "bollinger_squeeze",
            "volume_profile",
            "regime_switch",
            "sector_rotation",
            "cis_momentum",
            "larry_williams",
            "bnf_deviation",
            "volume_surge",
        ]
        loop, strategies = self._make_loop_with_strategies(all_strategy_names)
        kr_disabled = [
            "trend_following",
            "donchian_breakout",
            "macd_histogram",
            "rsi_divergence",
            "bollinger_squeeze",
            "volume_profile",
            "regime_switch",
            "sector_rotation",
            "cis_momentum",
            "larry_williams",
            "bnf_deviation",
            "volume_surge",
        ]
        loop.set_disabled_strategies(kr_disabled)
        active = loop._get_active_strategies()
        assert len(active) == 2
        assert {s.name for s in active} == {"supertrend", "dual_momentum"}


# ---------------------------------------------------------------------------
# YAML config validation: KR section correctness
# ---------------------------------------------------------------------------


class TestYAMLKRSection:
    """Verify strategies.yaml has the correct KR market section structure."""

    def test_yaml_has_markets_section(self):
        loader = StrategyConfigLoader()
        markets = loader._config.get("markets", {})
        assert "KR" in markets

    def test_yaml_kr_has_disabled_strategies(self):
        loader = StrategyConfigLoader()
        kr = loader._config["markets"]["KR"]
        assert "disabled_strategies" in kr
        assert isinstance(kr["disabled_strategies"], list)

    def test_yaml_kr_has_risk_section(self):
        loader = StrategyConfigLoader()
        kr = loader._config["markets"]["KR"]
        assert "risk" in kr
        risk = kr["risk"]
        assert "kelly_fraction" in risk
        assert "max_position_pct" in risk
        assert "min_position_pct" in risk
        assert "max_positions" in risk
        assert "default_stop_loss_pct" in risk
        assert "default_take_profit_pct" in risk
        assert "dynamic_sl_tp" in risk

    def test_yaml_kr_has_evaluation_loop_section(self):
        loader = StrategyConfigLoader()
        kr = loader._config["markets"]["KR"]
        assert "evaluation_loop" in kr
        ev = kr["evaluation_loop"]
        assert "min_confidence" in ev
        assert "min_active_ratio" in ev
        assert "sell_cooldown_days" in ev
        assert "whipsaw_max_losses" in ev
        assert "min_hold_days" in ev

    def test_yaml_kr_disabled_count(self):
        """Exactly 12 strategies disabled (14 total - 2 enabled = 12)."""
        loader = StrategyConfigLoader()
        disabled = loader._config["markets"]["KR"]["disabled_strategies"]
        assert len(disabled) == 12

    def test_yaml_kr_risk_values(self):
        loader = StrategyConfigLoader()
        risk = loader._config["markets"]["KR"]["risk"]
        assert risk["kelly_fraction"] == pytest.approx(0.50)
        assert risk["max_position_pct"] == pytest.approx(0.20)
        assert risk["min_position_pct"] == pytest.approx(0.12)
        assert risk["max_positions"] == 8
        assert risk["default_stop_loss_pct"] == pytest.approx(0.10)
        assert risk["default_take_profit_pct"] == pytest.approx(0.15)
        assert risk["dynamic_sl_tp"] is False

    def test_yaml_kr_eval_loop_values(self):
        loader = StrategyConfigLoader()
        ev = loader._config["markets"]["KR"]["evaluation_loop"]
        assert ev["min_confidence"] == pytest.approx(0.30)
        # null in YAML → None in Python; means 'no override, use per-call defaults'
        assert ev["min_active_ratio"] is None
        assert ev["sell_cooldown_days"] == 1
        assert ev["whipsaw_max_losses"] == 2
        assert ev["min_hold_days"] == 1


# ---------------------------------------------------------------------------
# _apply_kr_eval_overrides: YAML key → EvaluationLoop attribute mapping
# ---------------------------------------------------------------------------


class TestApplyKrEvalOverrides:
    """Unit-tests for _apply_kr_eval_overrides in main.py.

    Verifies that the function correctly maps YAML keys to EvaluationLoop
    attributes and performs unit conversions (days → seconds).
    """

    def _make_loop(self) -> EvaluationLoop:
        from unittest.mock import MagicMock

        from strategies.combiner import SignalCombiner
        from strategies.registry import StrategyRegistry

        return EvaluationLoop(
            adapter=MagicMock(),
            market_data=MagicMock(),
            indicator_svc=MagicMock(),
            registry=MagicMock(spec=StrategyRegistry),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="KR",
        )

    def test_sell_cooldown_days_converted_to_secs(self):
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {
            "sell_cooldown_days": 2,
        }
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)
        assert loop._sell_cooldown_secs == 2 * 86400

    def test_min_hold_days_converted_to_secs(self):
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {
            "min_hold_days": 1,
        }
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)
        assert loop._min_hold_secs == 86400

    def test_whipsaw_max_losses_set(self):
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {
            "whipsaw_max_losses": 3,
        }
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)
        assert loop._max_loss_sells == 3

    def test_disabled_strategies_applied(self):
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {}
        loader.get_market_disabled_strategies.return_value = [
            "trend_following",
            "macd_histogram",
        ]
        _apply_kr_eval_overrides(loop, loader)
        assert "trend_following" in loop._disabled_strategies
        assert "macd_histogram" in loop._disabled_strategies

    def test_min_confidence_forwarded_to_setter(self):
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {
            "min_confidence": 0.30,
        }
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)
        assert loop._min_confidence == pytest.approx(0.30)

    def test_min_active_ratio_null_sets_none(self):
        """YAML null (None) should call set_min_active_ratio(None), not float(None)."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {
            "min_active_ratio": None,
        }
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)
        assert loop._min_active_ratio is None

    def test_sell_cooldown_days_null_is_safe(self):
        """YAML null for sell_cooldown_days must not raise TypeError."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        original = loop._sell_cooldown_secs
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {"sell_cooldown_days": None}
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)  # must not raise
        assert loop._sell_cooldown_secs == original  # unchanged

    def test_whipsaw_max_losses_null_is_safe(self):
        """YAML null for whipsaw_max_losses must not raise TypeError."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        original = loop._max_loss_sells
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {"whipsaw_max_losses": None}
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)  # must not raise
        assert loop._max_loss_sells == original  # unchanged

    def test_min_hold_days_null_is_safe(self):
        """YAML null for min_hold_days must not raise TypeError."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        original = loop._min_hold_secs
        loader = MagicMock()
        loader.get_market_evaluation_loop_config.return_value = {"min_hold_days": None}
        loader.get_market_disabled_strategies.return_value = []
        _apply_kr_eval_overrides(loop, loader)  # must not raise
        assert loop._min_hold_secs == original  # unchanged


# ---------------------------------------------------------------------------
# Hot-reload integration: lambda closure wiring
# ---------------------------------------------------------------------------


class TestHotReloadAppliesKrOverrides:
    """Verify the app.state.apply_kr_eval_overrides lambda correctly re-applies
    overrides when called after config changes (simulates POST /reload behavior).
    """

    def _make_loop(self) -> EvaluationLoop:
        from unittest.mock import MagicMock

        from strategies.combiner import SignalCombiner
        from strategies.registry import StrategyRegistry

        return EvaluationLoop(
            adapter=MagicMock(),
            market_data=MagicMock(),
            indicator_svc=MagicMock(),
            registry=MagicMock(spec=StrategyRegistry),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="KR",
        )

    def test_lambda_applies_initial_overrides(self):
        """Calling the lambda applies config to the loop (simulates startup)."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        config_loader = MagicMock()
        config_loader.get_market_evaluation_loop_config.return_value = {
            "min_confidence": 0.30,
            "sell_cooldown_days": 1,
        }
        config_loader.get_market_disabled_strategies.return_value = ["trend_following"]

        def apply_kr_overrides():
            _apply_kr_eval_overrides(loop, config_loader)

        apply_kr_overrides()

        assert loop._min_confidence == pytest.approx(0.30)
        assert loop._sell_cooldown_secs == 86400
        assert "trend_following" in loop._disabled_strategies

    def test_lambda_re_applies_updated_config_on_reload(self):
        """After config changes, calling the lambda again propagates new values."""
        from unittest.mock import MagicMock

        from main import _apply_kr_eval_overrides

        loop = self._make_loop()
        config_loader = MagicMock()
        config_loader.get_market_evaluation_loop_config.return_value = {
            "min_confidence": 0.30,
            "sell_cooldown_days": 1,
        }
        config_loader.get_market_disabled_strategies.return_value = ["trend_following"]

        def apply_kr_overrides():
            _apply_kr_eval_overrides(loop, config_loader)

        apply_kr_overrides()

        # Simulate YAML update, then hot-reload
        config_loader.get_market_evaluation_loop_config.return_value = {
            "min_confidence": 0.40,
            "sell_cooldown_days": 2,
        }
        config_loader.get_market_disabled_strategies.return_value = ["macd_histogram"]

        apply_kr_overrides()

        assert loop._min_confidence == pytest.approx(0.40)
        assert loop._sell_cooldown_secs == 2 * 86400
        assert "macd_histogram" in loop._disabled_strategies
