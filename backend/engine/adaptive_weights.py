"""Adaptive strategy weight manager.

Dynamically adjusts strategy weights per stock based on:
1. Stock category (from StockClassifier) -> base weights from config
2. Rolling performance tracking -> adaptive weights

Final weight = alpha * category_weight + (1 - alpha) * adaptive_weight

Performance tracking uses a simple EMA of signal accuracy
(did the strategy's signal predict correct direction?).
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from engine.stock_classifier import StockCategory

logger = logging.getLogger(__name__)

# Default category-based strategy weights
DEFAULT_STOCK_PROFILES: dict[str, dict[str, float]] = {
    "growth_momentum": {
        "dual_momentum": 0.20,
        "cis_momentum": 0.18,
        "trend_following": 0.15,
        "supertrend": 0.15,
        "sector_rotation": 0.12,
        "donchian_breakout": 0.10,
        "regime_switch": 0.05,
        "volume_profile": 0.05,
    },
    "stable_large_cap": {
        "trend_following": 0.18,
        "donchian_breakout": 0.15,
        "rsi_divergence": 0.15,
        "bnf_deviation": 0.12,
        "macd_histogram": 0.10,
        "larry_williams": 0.10,
        "bollinger_squeeze": 0.10,
        "volume_profile": 0.05,
        "dual_momentum": 0.05,
    },
    "cyclical_value": {
        "sector_rotation": 0.18,
        "regime_switch": 0.15,
        "bnf_deviation": 0.15,
        "dual_momentum": 0.12,
        "trend_following": 0.12,
        "rsi_divergence": 0.10,
        "donchian_breakout": 0.10,
        "volume_profile": 0.08,
    },
    "high_volatility": {
        "supertrend": 0.18,
        "rsi_divergence": 0.15,
        "bnf_deviation": 0.15,
        "donchian_breakout": 0.15,
        "cis_momentum": 0.12,
        "trend_following": 0.10,
        "bollinger_squeeze": 0.10,
        "macd_histogram": 0.05,
    },
}


@dataclass
class StrategyPerformance:
    """Rolling performance tracker for a single strategy on a single stock."""
    correct_signals: float = 0.0  # EMA of correct predictions
    total_signals: int = 0
    last_update: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.5  # neutral prior
        return self.correct_signals


class AdaptiveWeightManager:
    """Manage per-stock adaptive strategy weights."""

    def __init__(
        self,
        alpha: float = 0.6,
        ema_decay: float = 0.1,
        min_signals_for_adaptation: int = 5,
        stock_profiles: dict[str, dict[str, float]] | None = None,
    ):
        """
        Args:
            alpha: Blend ratio (1.0 = pure category, 0.0 = pure adaptive)
            ema_decay: EMA decay factor for performance tracking
            min_signals_for_adaptation: Min signals before adaptive weights kick in
            stock_profiles: Category -> strategy weights (overrides defaults)
        """
        self._alpha = alpha
        self._ema_decay = ema_decay
        self._min_signals = min_signals_for_adaptation
        self._stock_profiles = stock_profiles or DEFAULT_STOCK_PROFILES

        # Per-stock, per-strategy performance: {symbol: {strategy_name: StrategyPerformance}}
        self._performance: dict[str, dict[str, StrategyPerformance]] = defaultdict(
            lambda: defaultdict(StrategyPerformance)
        )

        # Cached category assignments: {symbol: StockCategory}
        self._categories: dict[str, StockCategory] = {}

    def set_category(self, symbol: str, category: StockCategory) -> None:
        """Set/update the category for a stock."""
        self._categories[symbol] = category

    def get_category(self, symbol: str) -> StockCategory | None:
        return self._categories.get(symbol)

    def get_weights(
        self,
        symbol: str,
        market_state_weights: dict[str, float],
    ) -> dict[str, float]:
        """Get blended strategy weights for a specific stock.

        Combines three sources:
        1. Market state weights (from profiles in config)
        2. Stock category weights
        3. Adaptive weights (from rolling performance)

        Args:
            symbol: Stock ticker
            market_state_weights: Weights from current market state profile

        Returns:
            Blended strategy weight dict
        """
        category = self._categories.get(symbol)
        if not category:
            # No classification yet, fall back to market state weights
            return market_state_weights

        # 1. Get category base weights
        category_weights = self._stock_profiles.get(
            category.value, {}
        )
        if not category_weights:
            return market_state_weights

        # 2. Blend market state weights with category weights
        # Market state gets 40%, category gets 60%
        base_weights = self._blend_dicts(
            market_state_weights, category_weights, 0.4, 0.6
        )

        # 3. Apply adaptive adjustment if enough data
        perf = self._performance.get(symbol, {})
        has_enough_data = any(
            p.total_signals >= self._min_signals for p in perf.values()
        )

        if not has_enough_data:
            return self._normalize(base_weights)

        # Calculate adaptive weights from performance
        adaptive_weights = self._compute_adaptive_weights(perf, base_weights)

        # Final blend: alpha * base + (1 - alpha) * adaptive
        final = self._blend_dicts(
            base_weights, adaptive_weights, self._alpha, 1 - self._alpha
        )
        return self._normalize(final)

    def record_signal_result(
        self,
        symbol: str,
        strategy_name: str,
        was_correct: bool,
    ) -> None:
        """Record whether a strategy's signal was correct.

        Args:
            symbol: Stock ticker
            strategy_name: Strategy that generated the signal
            was_correct: True if signal predicted correct direction
        """
        perf = self._performance[symbol][strategy_name]
        target = 1.0 if was_correct else 0.0

        if perf.total_signals == 0:
            perf.correct_signals = target
        else:
            # EMA update
            perf.correct_signals = (
                (1 - self._ema_decay) * perf.correct_signals
                + self._ema_decay * target
            )
        perf.total_signals += 1
        perf.last_update = time.time()

    def get_performance_summary(self, symbol: str) -> dict[str, dict]:
        """Get performance summary for a stock."""
        perf = self._performance.get(symbol, {})
        return {
            name: {
                "accuracy": round(p.accuracy, 3),
                "total_signals": p.total_signals,
            }
            for name, p in perf.items()
            if p.total_signals > 0
        }

    def get_all_summaries(self) -> dict[str, dict]:
        """Get performance summaries for all tracked stocks."""
        return {
            symbol: self.get_performance_summary(symbol)
            for symbol in self._performance
        }

    def _compute_adaptive_weights(
        self,
        perf: dict[str, StrategyPerformance],
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute weights from performance data.

        Strategies with higher accuracy get boosted,
        those with lower accuracy get suppressed.
        """
        weights = {}
        for name, base_w in base_weights.items():
            p = perf.get(name)
            if p and p.total_signals >= self._min_signals:
                # Accuracy 0.5 = neutral (1x), 0.7 = boost (1.4x), 0.3 = suppress (0.6x)
                multiplier = p.accuracy * 2
                weights[name] = base_w * max(multiplier, 0.1)
            else:
                weights[name] = base_w
        return weights

    @staticmethod
    def _blend_dicts(
        d1: dict[str, float],
        d2: dict[str, float],
        w1: float,
        w2: float,
    ) -> dict[str, float]:
        """Blend two weight dicts."""
        all_keys = set(d1) | set(d2)
        return {
            k: d1.get(k, 0) * w1 + d2.get(k, 0) * w2
            for k in all_keys
        }

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {k: v / total for k, v in weights.items()}
