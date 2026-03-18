"""Strategy configuration loader from YAML."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # local: us-stock/
_APP_ROOT = Path(__file__).parent.parent  # docker: /app/

# Try Docker path first (/app/config/), then local project root
if (_APP_ROOT / "config" / "strategies.yaml").exists():
    DEFAULT_CONFIG_PATH = _APP_ROOT / "config" / "strategies.yaml"
else:
    DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "strategies.yaml"


class StrategyConfigLoader:
    """Load and provide strategy parameters from strategies.yaml."""

    def __init__(self, config_path: Path | str | None = None):
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config: dict = {}
        self.reload()

    def reload(self) -> None:
        """Reload configuration from YAML file."""
        try:
            with open(self._path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.info("Strategy config loaded from %s", self._path)
        except FileNotFoundError:
            logger.warning("Strategy config not found at %s, using defaults", self._path)
            self._config = {}

    @property
    def global_config(self) -> dict:
        return self._config.get("global", {})

    def get_strategy_config(self, strategy_name: str) -> dict:
        """Get full config block for a strategy."""
        return self._config.get("strategies", {}).get(strategy_name, {})

    def get_strategy_params(self, strategy_name: str) -> dict:
        """Get just the params section for a strategy."""
        return self.get_strategy_config(strategy_name).get("params", {})

    def is_enabled(self, strategy_name: str) -> bool:
        return self.get_strategy_config(strategy_name).get("enabled", False)

    def get_profile_weights(self, market_state: str) -> dict[str, float]:
        """Get strategy weights for a market state profile."""
        return self._config.get("profiles", {}).get(market_state, {})

    def get_stop_loss_config(self, strategy_name: str) -> dict:
        return self.get_strategy_config(strategy_name).get("stop_loss", {})

    def get_trailing_stop_config(self, strategy_name: str) -> dict:
        """Get trailing stop config for a strategy.

        Returns dict with 'enabled', 'activation_pct', 'trail_pct' keys,
        or empty dict if trailing stop is not configured for this strategy.
        """
        return self.get_strategy_config(strategy_name).get("trailing_stop", {})

    def get_stock_profiles(self) -> dict[str, dict[str, float]]:
        """Get stock category -> strategy weight profiles."""
        return self._config.get("stock_profiles", {})

    def get_adaptive_config(self) -> dict:
        """Get adaptive weight blending configuration."""
        return self._config.get("adaptive", {})

    def get_consensus_config(self) -> dict:
        """Get group consensus configuration for SignalCombiner."""
        return self._config.get("consensus", {})

    def get_profit_exit_config(self) -> dict:
        """Get strategy-level profit-exit evaluation parameters.

        Returns the global.profit_exit section from strategies.yaml,
        used by BaseStrategy.evaluate_exit() for profit-taking decisions.
        """
        return self.global_config.get("profit_exit", {})

    def get_screening_config(self) -> dict:
        return self._config.get("screening", {})
