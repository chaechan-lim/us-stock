"""Tests for trailing stop config retrieval (STOCK-19).

Verifies that StrategyConfigLoader and StrategyRegistry correctly
retrieve trailing stop configurations from strategies.yaml.
"""

from pathlib import Path

import pytest
import yaml

from strategies.config_loader import StrategyConfigLoader
from strategies.registry import StrategyRegistry


@pytest.fixture
def config_with_trailing(tmp_path: Path) -> Path:
    """Write a config with trailing stop sections."""
    config = {
        "global": {"min_confidence": 0.50},
        "profiles": {"uptrend": {"trend_following": 0.20}},
        "strategies": {
            "trend_following": {
                "enabled": True,
                "timeframe": "1D",
                "params": {"ema_fast": 20, "ema_slow": 50},
                "stop_loss": {"type": "fixed_pct", "max_pct": 0.10},
                "trailing_stop": {
                    "enabled": True,
                    "activation_pct": 0.08,
                    "trail_pct": 0.05,
                },
            },
            "macd_histogram": {
                "enabled": True,
                "timeframe": "1D",
                "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "stop_loss": {"type": "fixed_pct", "max_pct": 0.05},
                # No trailing_stop section
            },
            "volume_surge": {
                "enabled": True,
                "timeframe": "1D",
                "params": {"volume_threshold": 1.8},
                "trailing_stop": {
                    "enabled": False,  # explicitly disabled
                    "activation_pct": 0.03,
                    "trail_pct": 0.02,
                },
            },
        },
    }
    config_path = tmp_path / "strategies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_get_trailing_stop_config_with_section(config_with_trailing: Path):
    """Config with trailing_stop section returns correct values."""
    loader = StrategyConfigLoader(config_with_trailing)
    trail = loader.get_trailing_stop_config("trend_following")

    assert trail["enabled"] is True
    assert trail["activation_pct"] == 0.08
    assert trail["trail_pct"] == 0.05


def test_get_trailing_stop_config_missing_section(config_with_trailing: Path):
    """Config without trailing_stop section returns empty dict."""
    loader = StrategyConfigLoader(config_with_trailing)
    trail = loader.get_trailing_stop_config("macd_histogram")

    assert trail == {}


def test_get_trailing_stop_config_disabled(config_with_trailing: Path):
    """Config with trailing_stop enabled=false returns the config."""
    loader = StrategyConfigLoader(config_with_trailing)
    trail = loader.get_trailing_stop_config("volume_surge")

    assert trail["enabled"] is False
    assert trail["activation_pct"] == 0.03


def test_get_trailing_stop_config_unknown_strategy(config_with_trailing: Path):
    """Unknown strategy returns empty dict."""
    loader = StrategyConfigLoader(config_with_trailing)
    trail = loader.get_trailing_stop_config("nonexistent_strategy")

    assert trail == {}


def test_registry_get_trailing_stop_config(config_with_trailing: Path):
    """StrategyRegistry delegates trailing_stop config to loader."""
    loader = StrategyConfigLoader(config_with_trailing)
    registry = StrategyRegistry(config_loader=loader)

    trail = registry.get_trailing_stop_config("trend_following")
    assert trail["enabled"] is True
    assert trail["activation_pct"] == 0.08

    # Strategy without trailing config
    trail2 = registry.get_trailing_stop_config("macd_histogram")
    assert trail2 == {}
