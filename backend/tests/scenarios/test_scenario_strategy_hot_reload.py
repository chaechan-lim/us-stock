"""Scenario 8: Strategy hot-reload.

1. Registry loads strategies from config
2. Parameters changed at runtime via set_params
3. reload_config() picks up new values
4. Newly enabled strategies appear
5. Disabled strategies removed
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from strategies.registry import StrategyRegistry, STRATEGY_CLASSES
from strategies.config_loader import StrategyConfigLoader
from strategies.trend_following import TrendFollowingStrategy


def _write_config(path: Path, config: dict):
    with open(path, "w") as f:
        yaml.dump(config, f)


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary strategies.yaml for testing."""
    config = {
        "global": {"min_confidence": 0.50},
        "profiles": {
            "uptrend": {"trend_following": 0.40, "donchian_breakout": 0.30},
        },
        "strategies": {
            "trend_following": {
                "enabled": True,
                "timeframe": "1D",
                "params": {"ema_fast": 20, "ema_slow": 50, "adx_threshold": 25},
            },
            "donchian_breakout": {
                "enabled": True,
                "timeframe": "1D",
                "params": {"entry_period": 20, "exit_period": 10},
            },
            "supertrend": {
                "enabled": False,
                "timeframe": "1D",
                "params": {"confirmation_bars": 2},
            },
            "macd_histogram": {
                "enabled": False,
                "timeframe": "1D",
                "params": {"min_histogram_change": 0.5},
            },
        },
    }
    config_path = tmp_path / "strategies.yaml"
    _write_config(config_path, config)
    return config_path


def test_initial_load(temp_config):
    """Registry loads only enabled strategies."""
    loader = StrategyConfigLoader(config_path=temp_config)
    registry = StrategyRegistry(config_loader=loader)

    names = registry.get_names()
    assert "trend_following" in names
    assert "donchian_breakout" in names
    assert "supertrend" not in names
    assert "macd_histogram" not in names


def test_params_update_at_runtime(temp_config):
    """set_params changes strategy behavior immediately."""
    loader = StrategyConfigLoader(config_path=temp_config)
    registry = StrategyRegistry(config_loader=loader)

    tf = registry.get("trend_following")
    assert tf is not None
    assert tf.get_params()["ema_fast"] == 20

    tf.set_params({"ema_fast": 10, "adx_threshold": 30})
    assert tf.get_params()["ema_fast"] == 10
    assert tf.get_params()["adx_threshold"] == 30


def test_hot_reload_params_change(temp_config):
    """reload_config() picks up YAML changes."""
    loader = StrategyConfigLoader(config_path=temp_config)
    registry = StrategyRegistry(config_loader=loader)

    # Modify YAML
    with open(temp_config) as f:
        config = yaml.safe_load(f)

    config["strategies"]["trend_following"]["params"]["ema_fast"] = 15
    config["strategies"]["trend_following"]["params"]["adx_threshold"] = 30
    _write_config(temp_config, config)

    # Reload
    registry.reload_config()

    tf = registry.get("trend_following")
    assert tf.get_params()["ema_fast"] == 15
    assert tf.get_params()["adx_threshold"] == 30


def test_hot_reload_enable_strategy(temp_config):
    """Enabling a strategy via YAML reload adds it to registry."""
    loader = StrategyConfigLoader(config_path=temp_config)
    registry = StrategyRegistry(config_loader=loader)

    assert "supertrend" not in registry.get_names()

    # Enable supertrend in YAML
    with open(temp_config) as f:
        config = yaml.safe_load(f)

    config["strategies"]["supertrend"]["enabled"] = True
    _write_config(temp_config, config)

    registry.reload_config()
    assert "supertrend" in registry.get_names()


def test_hot_reload_disable_strategy(temp_config):
    """Disabling a strategy via YAML reload removes it from registry."""
    loader = StrategyConfigLoader(config_path=temp_config)
    registry = StrategyRegistry(config_loader=loader)

    assert "donchian_breakout" in registry.get_names()

    # Disable donchian in YAML
    with open(temp_config) as f:
        config = yaml.safe_load(f)

    config["strategies"]["donchian_breakout"]["enabled"] = False
    _write_config(temp_config, config)

    registry.reload_config()
    assert "donchian_breakout" not in registry.get_names()


def test_profile_weights_change_on_reload(temp_config):
    """Weight profiles update after reload."""
    loader = StrategyConfigLoader(config_path=temp_config)

    original = loader.get_profile_weights("uptrend")
    assert original["trend_following"] == 0.40

    # Change weights
    with open(temp_config) as f:
        config = yaml.safe_load(f)

    config["profiles"]["uptrend"]["trend_following"] = 0.25
    _write_config(temp_config, config)

    loader.reload()
    updated = loader.get_profile_weights("uptrend")
    assert updated["trend_following"] == 0.25


def test_all_strategy_classes_registered():
    """Every strategy class in STRATEGY_CLASSES is importable."""
    assert len(STRATEGY_CLASSES) == 14
    for name, cls in STRATEGY_CLASSES.items():
        assert hasattr(cls, "name")
        assert hasattr(cls, "analyze")
        assert hasattr(cls, "get_params")
        assert hasattr(cls, "set_params")
