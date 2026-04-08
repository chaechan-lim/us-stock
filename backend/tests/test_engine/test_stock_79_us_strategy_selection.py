"""Tests for US strategy selection from config/strategies.yaml.

Verifies:
1. US disabled_strategies are loaded from config/strategies.yaml (not hardcoded)
2. Walk-forward validated strategies are enabled for US
3. Overfit strategies remain disabled
4. EvaluationLoop reflects the config-driven disabled set

Also covers STOCK-81 self-review follow-up:
5. _warn_if_disabled_empty emits WARNING when disabled list is empty (US path)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# NOTE: main is imported at module level so that setup_logging() runs during
# test collection, before pytest installs per-test caplog handlers.  This
# prevents the root.handlers.clear() call in setup_logging() from removing
# caplog's handler mid-test when _warn_if_disabled_empty tests run.
import main as _main_module  # noqa: I001 — must stay below third-party block
from core.enums import SignalType
from engine.evaluation_loop import EvaluationLoop
from engine.risk_manager import RiskManager
from strategies.base import Signal
from strategies.combiner import SignalCombiner
from strategies.config_loader import StrategyConfigLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_STRATEGIES = [
    "trend_following", "donchian_breakout", "supertrend", "macd_histogram",
    "dual_momentum", "rsi_divergence", "bollinger_squeeze", "volume_profile",
    "regime_switch", "sector_rotation", "cis_momentum", "larry_williams",
    "bnf_deviation", "volume_surge", "cross_sectional_momentum", "quality_factor",
    "pead_drift",
]

_US_ENABLED = {
    # 2026-04-08: donchian_breakout removed — fresh 2y pipeline backtest
    # showed 77 trades, WR 43%, PnL -$1,856 (single largest US loser).
    # 2026-04-09: dual_momentum removed — 9d live data 10 trades 3W/7L,
    # avg -2.28%, 3 of 5 SMALL_WIN_BIG_LOSS cases (FANG/CVE/XOM).
    # Backtest 2y -$3,865 contribution. See docs/IMPROVEMENT_PLAN.md §1.
    "volume_surge", "trend_following",
    "supertrend", "macd_histogram", "rsi_divergence", "regime_switch",
    "sector_rotation", "cross_sectional_momentum", "quality_factor",
    "pead_drift",
}


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


def _make_loop(disabled: list[str] | None = None) -> EvaluationLoop:
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

    strategies = []
    for name in _ALL_STRATEGIES:
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

    loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=SignalCombiner(),
        order_manager=MagicMock(),
        risk_manager=RiskManager(),
        market="US",
    )
    if disabled is not None:
        loop.set_disabled_strategies(disabled)
    return loop


# ---------------------------------------------------------------------------
# Config-loader tests
# ---------------------------------------------------------------------------

class TestUSConfigLoader:
    def test_us_disabled_strategies_loaded_from_yaml(self):
        """US disabled_strategies come from config, not hardcoded in main.py."""
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        assert isinstance(disabled, list)
        assert len(disabled) > 0

    def test_dual_momentum_disabled_2026_04_09(self):
        """2026-04-09: dual_momentum disabled in US after 9-day live data
        analysis showed 10 trades 3W/7L avg -2.28%, 3 of 5
        SMALL_WIN_BIG_LOSS cases. See docs/IMPROVEMENT_PLAN.md §1.
        """
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        assert "dual_momentum" in disabled

    def test_volume_surge_not_disabled(self):
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        assert "volume_surge" not in disabled

    def test_volume_profile_disabled(self):
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        assert "volume_profile" in disabled

    def test_bollinger_squeeze_disabled(self):
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        assert "bollinger_squeeze" in disabled

    def test_us_enabled_set_matches_walk_forward_validated(self):
        """Walk-forward validated strategies survive the disabled filter."""
        loader = StrategyConfigLoader()
        disabled = set(loader.get_market_disabled_strategies("US"))
        enabled = {s for s in _ALL_STRATEGIES if s not in disabled}
        assert enabled == _US_ENABLED

    def test_us_no_risk_overrides(self):
        """US risk params are not overridden in config — use code defaults."""
        loader = StrategyConfigLoader()
        assert loader.get_market_risk_config("US") == {}

    def test_us_no_eval_loop_overrides(self):
        """US evaluation_loop params are not overridden in config."""
        loader = StrategyConfigLoader()
        assert loader.get_market_evaluation_loop_config("US") == {}


# ---------------------------------------------------------------------------
# EvaluationLoop integration
# ---------------------------------------------------------------------------

class TestUSEvaluationLoop:
    def test_config_driven_disabled_produces_correct_active_set(self):
        """Applying config disabled list gives exactly {dual_momentum, volume_surge}."""
        loader = StrategyConfigLoader()
        disabled = loader.get_market_disabled_strategies("US")
        loop = _make_loop(disabled=disabled)
        active_names = {s.name for s in loop._get_active_strategies()}
        assert active_names == _US_ENABLED

    def test_dual_momentum_is_inactive_2026_04_09(self):
        """dual_momentum should NOT be in active strategies after 4-09 disable."""
        loader = StrategyConfigLoader()
        loop = _make_loop(disabled=loader.get_market_disabled_strategies("US"))
        active_names = {s.name for s in loop._get_active_strategies()}
        assert "dual_momentum" not in active_names

    def test_volume_profile_is_inactive(self):
        loader = StrategyConfigLoader()
        loop = _make_loop(disabled=loader.get_market_disabled_strategies("US"))
        active_names = {s.name for s in loop._get_active_strategies()}
        assert "volume_profile" not in active_names

    def test_bollinger_squeeze_is_inactive(self):
        loader = StrategyConfigLoader()
        loop = _make_loop(disabled=loader.get_market_disabled_strategies("US"))
        active_names = {s.name for s in loop._get_active_strategies()}
        assert "bollinger_squeeze" not in active_names

    @pytest.mark.asyncio
    async def test_evaluate_symbol_calls_only_active_strategies(self):
        """evaluate_symbol should only run walk-forward validated strategies."""
        loader = StrategyConfigLoader()
        loop = _make_loop(
            disabled=loader.get_market_disabled_strategies("US"),
        )
        loop._min_confidence = 0.30
        loop._min_active_ratio = 0.0

        await loop.evaluate_symbol("AAPL")

        for s in loop._registry.get_enabled.return_value:
            if s.name in _US_ENABLED:
                s.analyze.assert_called_once()
            else:
                s.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# STOCK-81: _warn_if_disabled_empty — defensive WARNING for misconfiguration
# ---------------------------------------------------------------------------

class TestWarnIfDisabledEmpty:
    """Unit-tests for _warn_if_disabled_empty in main.py.

    Verifies that a WARNING is emitted when disabled_strategies is empty so
    that a YAML misconfiguration (e.g. merge conflict that drops the list)
    becomes visible in startup logs rather than silently activating all
    strategies.

    Uses patch.object on the module-level logger rather than caplog to avoid
    the setup_logging() / caplog handler-clearing interaction.
    """

    def test_empty_list_emits_warning(self) -> None:
        """Empty disabled list → logger.warning called for that market."""
        with patch.object(_main_module.logger, "warning") as mock_warn:
            _main_module._warn_if_disabled_empty("US", [])

        mock_warn.assert_called_once()
        call_args = mock_warn.call_args[0]  # positional args tuple
        assert "US" in call_args[0] or any("US" in str(a) for a in call_args), (
            f"WARNING message should reference 'US'. Call args: {call_args}"
        )

    def test_non_empty_list_no_warning(self) -> None:
        """Non-empty disabled list → logger.warning NOT called."""
        with patch.object(_main_module.logger, "warning") as mock_warn:
            _main_module._warn_if_disabled_empty("US", ["sector_rotation", "volume_profile"])

        mock_warn.assert_not_called()

    def test_warning_mentions_strategy_count(self) -> None:
        """WARNING message references all-strategies-active risk (17 strategies)."""
        with patch.object(_main_module.logger, "warning") as mock_warn:
            _main_module._warn_if_disabled_empty("US", [])

        mock_warn.assert_called_once()
        # The full formatted message (with % substitution) should mention 17
        call_args = mock_warn.call_args[0]
        full_msg = call_args[0] % tuple(call_args[1:]) if len(call_args) > 1 else call_args[0]
        assert "17" in full_msg, (
            f"WARNING should mention 17 strategies. Full message: {full_msg}"
        )

    def test_kr_market_empty_list_emits_warning(self) -> None:
        """Same warning fires for KR market when disabled list is empty."""
        with patch.object(_main_module.logger, "warning") as mock_warn:
            _main_module._warn_if_disabled_empty("KR", [])

        mock_warn.assert_called_once()
        call_args = mock_warn.call_args[0]
        assert "KR" in call_args[0] or any("KR" in str(a) for a in call_args), (
            f"WARNING message should reference 'KR'. Call args: {call_args}"
        )

    def test_actual_us_config_does_not_trigger_warning(self) -> None:
        """The real strategies.yaml US config has a non-empty list → no WARNING."""
        loader = StrategyConfigLoader()
        us_disabled = loader.get_market_disabled_strategies("US")

        with patch.object(_main_module.logger, "warning") as mock_warn:
            _main_module._warn_if_disabled_empty("US", us_disabled)

        mock_warn.assert_not_called()
