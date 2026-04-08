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

    def get_tiered_trailing_stop_config(self) -> dict:
        """Get tiered trailing stop configuration.

        Returns dict with 'enabled' and 'tiers' keys.
        Each tier has 'gain_pct' and 'trail_pct'.
        """
        return self.global_config.get("tiered_trailing_stop", {})

    def get_breakeven_stop_config(self) -> dict:
        """Get breakeven stop configuration.

        Returns dict with 'enabled', 'activation_ratio', 'lock_ratio', 'lock_pct'.
        """
        return self.global_config.get("breakeven_stop", {})

    def get_hard_sl_pct(self) -> float:
        """Get hard stop-loss threshold that bypasses min_hold.

        Returns the hard_sl_pct from global config, or -0.15 as default.
        When P&L drops below this, position is sold immediately even if
        it hasn't been held for the minimum hold period (4 hours).
        """
        return float(self.global_config.get("hard_sl_pct", -0.15))

    def _get_market_config(self, market: str) -> dict:
        """Get raw market-specific override dict (e.g. for 'KR' or 'US').

        Internal helper — callers should use the typed wrapper methods below
        (get_market_disabled_strategies, get_market_risk_config,
        get_market_evaluation_loop_config) rather than this raw dict.
        """
        return self._config.get("markets", {}).get(market, {})

    def get_market_disabled_strategies(self, market: str) -> list[str]:
        """Get list of strategy names disabled for a specific market.

        STOCK-65: KR market only runs supertrend + dual_momentum.
        Returns empty list if no market-specific overrides are set.
        """
        return list(self._get_market_config(market).get("disabled_strategies", []))

    def get_market_risk_config(self, market: str) -> dict:
        """Get risk parameter overrides for a specific market.

        STOCK-65: KR market uses grid-search optimized risk params.
        Returns empty dict if no market-specific overrides are set.
        """
        return dict(self._get_market_config(market).get("risk", {}))

    def get_market_evaluation_loop_config(self, market: str) -> dict:
        """Get evaluation loop overrides for a specific market.

        STOCK-65: KR market uses optimized evaluation loop params.
        Returns empty dict if no market-specific overrides are set.
        """
        return dict(self._get_market_config(market).get("evaluation_loop", {}))

    def get_market_cash_parking_config(self, market: str) -> dict:
        """Get cash parking config for a specific market.

        Schema (under markets.<MARKET>.cash_parking):
            enabled: bool       — master switch (default False)
            symbol: str         — parking symbol (default SPY for US, 069500 for KR)
            threshold: float    — park when cash/equity > threshold (default 0.30)
            buffer: float       — fraction of equity to keep in cash (default 0.10)

        Returns empty dict if no overrides are set.
        """
        return dict(self._get_market_config(market).get("cash_parking", {}))

    def get_screening_config(self) -> dict:
        return self._config.get("screening", {})
