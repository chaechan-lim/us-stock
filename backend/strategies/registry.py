"""Strategy Registry - manages strategy instances and config reloading.

Provides a central registry for all available strategies,
handles config loading and hot-reload from strategies.yaml.
"""

import logging

from strategies.base import BaseStrategy
from strategies.config_loader import StrategyConfigLoader
from strategies.trend_following import TrendFollowingStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.macd_histogram import MACDHistogramStrategy
from strategies.dual_momentum import DualMomentumStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.bollinger_squeeze import BollingerSqueezeStrategy
from strategies.volume_profile import VolumeProfileStrategy
from strategies.regime_switch import RegimeSwitchStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.cis_momentum import CISMomentumStrategy
from strategies.larry_williams import LarryWilliamsStrategy
from strategies.bnf_deviation import BNFDeviationStrategy
from strategies.volume_surge_strategy import VolumeSurgeStrategy
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.quality_factor import QualityFactorStrategy
from strategies.pead_drift import PEADDriftStrategy

logger = logging.getLogger(__name__)

# All available strategy classes
STRATEGY_CLASSES: dict[str, type[BaseStrategy]] = {
    "trend_following": TrendFollowingStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "supertrend": SupertrendStrategy,
    "macd_histogram": MACDHistogramStrategy,
    "dual_momentum": DualMomentumStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "bollinger_squeeze": BollingerSqueezeStrategy,
    "volume_profile": VolumeProfileStrategy,
    "regime_switch": RegimeSwitchStrategy,
    "sector_rotation": SectorRotationStrategy,
    "cis_momentum": CISMomentumStrategy,
    "larry_williams": LarryWilliamsStrategy,
    "bnf_deviation": BNFDeviationStrategy,
    "volume_surge": VolumeSurgeStrategy,
    "cross_sectional_momentum": CrossSectionalMomentumStrategy,
    "quality_factor": QualityFactorStrategy,
    "pead_drift": PEADDriftStrategy,
}


class StrategyRegistry:
    """Central registry for strategy instances."""

    def __init__(self, config_loader: StrategyConfigLoader | None = None):
        self._config_loader = config_loader or StrategyConfigLoader()
        self._strategies: dict[str, BaseStrategy] = {}
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Instantiate enabled strategies with config params."""
        # Propagate profit_exit config to BaseStrategy class-level params
        profit_exit_cfg = self._config_loader.get_profit_exit_config()
        if profit_exit_cfg:
            BaseStrategy.set_profit_exit_params(profit_exit_cfg)

        for name, cls in STRATEGY_CLASSES.items():
            if self._config_loader.is_enabled(name):
                params = self._config_loader.get_strategy_params(name)
                self._strategies[name] = cls(params=params)
                logger.info("Loaded strategy: %s", name)

    @property
    def config_loader(self) -> StrategyConfigLoader:
        """Public accessor for the config loader (avoids coupling callers to internal layout)."""
        return self._config_loader

    def get(self, name: str) -> BaseStrategy | None:
        return self._strategies.get(name)

    def get_all(self) -> dict[str, BaseStrategy]:
        return dict(self._strategies)

    def get_enabled(self) -> list[BaseStrategy]:
        return list(self._strategies.values())

    def get_names(self) -> list[str]:
        return list(self._strategies.keys())

    def reload_config(self) -> None:
        """Hot-reload strategy configuration from YAML."""
        self._config_loader.reload()

        # Propagate profit_exit config on reload
        profit_exit_cfg = self._config_loader.get_profit_exit_config()
        if profit_exit_cfg:
            BaseStrategy.set_profit_exit_params(profit_exit_cfg)

        for name, strategy in self._strategies.items():
            params = self._config_loader.get_strategy_params(name)
            strategy.set_params(params)
            logger.info("Reloaded params for %s", name)

        # Check for newly enabled/disabled strategies
        for name, cls in STRATEGY_CLASSES.items():
            if self._config_loader.is_enabled(name) and name not in self._strategies:
                params = self._config_loader.get_strategy_params(name)
                self._strategies[name] = cls(params=params)
                logger.info("Newly enabled strategy: %s", name)
            elif not self._config_loader.is_enabled(name) and name in self._strategies:
                del self._strategies[name]
                logger.info("Disabled strategy: %s", name)

    def get_profile_weights(self, market_state: str) -> dict[str, float]:
        """Get strategy weights for current market state."""
        return self._config_loader.get_profile_weights(market_state)

    def get_trailing_stop_config(self, strategy_name: str) -> dict:
        """Get trailing stop config for a strategy from YAML."""
        return self._config_loader.get_trailing_stop_config(strategy_name)

    def get_stop_loss_config(self, strategy_name: str) -> dict:
        """Get stop_loss config for a strategy from YAML.

        Returned dict shape (see config/strategies.yaml strategies.<name>.stop_loss):
            type: "fixed_pct" | "atr" | "supertrend"
            max_pct: float (for fixed_pct)
            atr_multiplier: float (for atr)
        Empty dict if not configured (caller falls back to ATR/default).
        """
        return self._config_loader.get_stop_loss_config(strategy_name)

    def get_take_profit_config(self, strategy_name: str) -> dict:
        """Get take_profit config for a strategy from YAML."""
        return self._config_loader.get_strategy_config(strategy_name).get("take_profit", {})
