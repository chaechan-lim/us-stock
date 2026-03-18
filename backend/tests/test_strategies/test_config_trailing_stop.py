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


# ── Tiered trailing stop + Breakeven stop config (STOCK-24) ──────────


@pytest.fixture
def config_with_tiered_trailing(tmp_path: Path) -> Path:
    """Config with tiered trailing stop and breakeven stop."""
    config = {
        "global": {
            "min_confidence": 0.50,
            "tiered_trailing_stop": {
                "enabled": True,
                "tiers": [
                    {"gain_pct": 0.10, "trail_pct": 0.05},
                    {"gain_pct": 0.15, "trail_pct": 0.04},
                    {"gain_pct": 0.20, "trail_pct": 0.03},
                ],
            },
            "breakeven_stop": {
                "enabled": True,
                "activation_ratio": 0.50,
                "lock_ratio": 0.75,
                "lock_pct": 0.50,
            },
        },
        "profiles": {"uptrend": {"trend_following": 0.20}},
    }
    config_path = tmp_path / "strategies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def config_without_tiered(tmp_path: Path) -> Path:
    """Config without tiered/breakeven sections."""
    config = {
        "global": {"min_confidence": 0.50},
        "profiles": {"uptrend": {"trend_following": 0.20}},
    }
    config_path = tmp_path / "strategies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_get_tiered_trailing_config(config_with_tiered_trailing: Path):
    """Tiered trailing stop config is read correctly."""
    loader = StrategyConfigLoader(config_with_tiered_trailing)
    cfg = loader.get_tiered_trailing_stop_config()

    assert cfg["enabled"] is True
    assert len(cfg["tiers"]) == 3
    assert cfg["tiers"][0]["gain_pct"] == 0.10
    assert cfg["tiers"][0]["trail_pct"] == 0.05
    assert cfg["tiers"][2]["gain_pct"] == 0.20
    assert cfg["tiers"][2]["trail_pct"] == 0.03


def test_get_breakeven_config(config_with_tiered_trailing: Path):
    """Breakeven stop config is read correctly."""
    loader = StrategyConfigLoader(config_with_tiered_trailing)
    cfg = loader.get_breakeven_stop_config()

    assert cfg["enabled"] is True
    assert cfg["activation_ratio"] == 0.50
    assert cfg["lock_ratio"] == 0.75
    assert cfg["lock_pct"] == 0.50


def test_get_tiered_config_missing(config_without_tiered: Path):
    """Missing tiered section returns empty dict."""
    loader = StrategyConfigLoader(config_without_tiered)
    assert loader.get_tiered_trailing_stop_config() == {}


def test_get_breakeven_config_missing(config_without_tiered: Path):
    """Missing breakeven section returns empty dict."""
    loader = StrategyConfigLoader(config_without_tiered)
    assert loader.get_breakeven_stop_config() == {}


def test_tiered_config_disabled(tmp_path: Path):
    """Tiered trailing stop with enabled=false."""
    config = {
        "global": {
            "tiered_trailing_stop": {
                "enabled": False,
                "tiers": [{"gain_pct": 0.10, "trail_pct": 0.05}],
            },
        },
    }
    config_path = tmp_path / "strategies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    loader = StrategyConfigLoader(config_path)
    cfg = loader.get_tiered_trailing_stop_config()
    assert cfg["enabled"] is False
    assert len(cfg["tiers"]) == 1


def test_breakeven_config_disabled(tmp_path: Path):
    """Breakeven stop with enabled=false."""
    config = {
        "global": {
            "breakeven_stop": {
                "enabled": False,
                "activation_ratio": 0.60,
            },
        },
    }
    config_path = tmp_path / "strategies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    loader = StrategyConfigLoader(config_path)
    cfg = loader.get_breakeven_stop_config()
    assert cfg["enabled"] is False
    assert cfg["activation_ratio"] == 0.60


def test_tiered_tiers_parsing_to_tuples(config_with_tiered_trailing: Path):
    """Verify tiers can be parsed into (gain, trail) tuples as RiskParams expects."""
    loader = StrategyConfigLoader(config_with_tiered_trailing)
    cfg = loader.get_tiered_trailing_stop_config()

    tiers = [(t["gain_pct"], t["trail_pct"]) for t in cfg["tiers"]]
    assert tiers == [(0.10, 0.05), (0.15, 0.04), (0.20, 0.03)]


def test_reload_updates_tiered_config(config_with_tiered_trailing: Path):
    """Reloading config picks up changes."""
    loader = StrategyConfigLoader(config_with_tiered_trailing)
    assert loader.get_tiered_trailing_stop_config()["enabled"] is True

    # Update the file
    new_config = {
        "global": {
            "tiered_trailing_stop": {
                "enabled": False,
                "tiers": [],
            },
        },
    }
    with open(config_with_tiered_trailing, "w") as f:
        yaml.dump(new_config, f)

    loader.reload()
    assert loader.get_tiered_trailing_stop_config()["enabled"] is False


def test_real_strategies_yaml_has_tiered_config():
    """Production strategies.yaml should have tiered trailing stop config."""
    loader = StrategyConfigLoader()  # uses default path
    cfg = loader.get_tiered_trailing_stop_config()
    # Should exist and be enabled after STOCK-24
    assert cfg.get("enabled") is True
    assert len(cfg.get("tiers", [])) >= 3


def test_real_strategies_yaml_has_breakeven_config():
    """Production strategies.yaml should have breakeven stop config."""
    loader = StrategyConfigLoader()  # uses default path
    cfg = loader.get_breakeven_stop_config()
    assert cfg.get("enabled") is True
    assert cfg.get("activation_ratio") == 0.50
