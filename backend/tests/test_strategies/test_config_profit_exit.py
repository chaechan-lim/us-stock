"""Tests for profit_exit config loading and registry propagation."""

from pathlib import Path

import pytest
import yaml

from strategies.base import BaseStrategy
from strategies.config_loader import StrategyConfigLoader
from strategies.registry import StrategyRegistry


@pytest.fixture(autouse=True)
def reset_profit_exit_params():
    """Reset class-level params to hardcoded defaults before/after each test."""
    _DEFAULTS = {
        "sell_confidence_boost_min_pnl": 0.02,
        "sell_confidence_boost_factor": 0.5,
        "sell_confidence_boost_max": 0.15,
        "profit_take_min_pnl": 0.05,
        "profit_take_base_confidence": 0.65,
        "profit_take_weakness_required": 1,
        "rsi_overbought": 70.0,
        "volume_weakness_ratio": 0.8,
        "high_profit_auto_sell_pnl": 0.10,
        "high_profit_auto_sell_confidence": 0.70,
    }
    BaseStrategy._profit_exit_params = _DEFAULTS.copy()
    yield
    BaseStrategy._profit_exit_params = _DEFAULTS.copy()


class TestConfigLoaderProfitExit:
    def test_get_profit_exit_config_from_yaml(self, tmp_path: Path):
        config = {
            "global": {
                "profit_exit": {
                    "sell_confidence_boost_min_pnl": 0.03,
                    "profit_take_min_pnl": 0.10,
                    "rsi_overbought": 75.0,
                }
            },
            "strategies": {},
        }
        config_file = tmp_path / "strategies.yaml"
        config_file.write_text(yaml.dump(config))

        loader = StrategyConfigLoader(config_file)
        result = loader.get_profit_exit_config()

        assert result["sell_confidence_boost_min_pnl"] == 0.03
        assert result["profit_take_min_pnl"] == 0.10
        assert result["rsi_overbought"] == 75.0

    def test_get_profit_exit_config_empty_when_missing(self, tmp_path: Path):
        config = {"global": {}, "strategies": {}}
        config_file = tmp_path / "strategies.yaml"
        config_file.write_text(yaml.dump(config))

        loader = StrategyConfigLoader(config_file)
        result = loader.get_profit_exit_config()

        assert result == {}

    def test_get_profit_exit_config_reload(self, tmp_path: Path):
        config = {
            "global": {"profit_exit": {"rsi_overbought": 70.0}},
            "strategies": {},
        }
        config_file = tmp_path / "strategies.yaml"
        config_file.write_text(yaml.dump(config))

        loader = StrategyConfigLoader(config_file)
        assert loader.get_profit_exit_config()["rsi_overbought"] == 70.0

        # Update config and reload
        config["global"]["profit_exit"]["rsi_overbought"] = 80.0
        config_file.write_text(yaml.dump(config))
        loader.reload()

        assert loader.get_profit_exit_config()["rsi_overbought"] == 80.0


class TestRegistryPropagation:
    def test_registry_load_propagates_profit_exit(self, tmp_path: Path):
        config = {
            "global": {
                "profit_exit": {
                    "sell_confidence_boost_factor": 0.7,
                    "rsi_overbought": 80.0,
                }
            },
            "strategies": {},
        }
        config_file = tmp_path / "strategies.yaml"
        config_file.write_text(yaml.dump(config))

        loader = StrategyConfigLoader(config_file)
        _registry = StrategyRegistry(config_loader=loader)

        assert BaseStrategy._profit_exit_params["sell_confidence_boost_factor"] == 0.7
        assert BaseStrategy._profit_exit_params["rsi_overbought"] == 80.0

    def test_registry_reload_propagates_profit_exit(self, tmp_path: Path):
        config = {
            "global": {"profit_exit": {"rsi_overbought": 70.0}},
            "strategies": {},
        }
        config_file = tmp_path / "strategies.yaml"
        config_file.write_text(yaml.dump(config))

        loader = StrategyConfigLoader(config_file)
        registry = StrategyRegistry(config_loader=loader)

        assert BaseStrategy._profit_exit_params["rsi_overbought"] == 70.0

        # Update and reload
        config["global"]["profit_exit"]["rsi_overbought"] = 85.0
        config_file.write_text(yaml.dump(config))
        registry.reload_config()

        assert BaseStrategy._profit_exit_params["rsi_overbought"] == 85.0
