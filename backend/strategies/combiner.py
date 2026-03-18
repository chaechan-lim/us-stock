"""Signal Combiner - weighted voting across multiple strategies.

Combines signals from multiple strategies using market-state-adaptive
weight profiles defined in config/strategies.yaml.

Includes group consensus mechanism: strategies are grouped (trend, mean_reversion)
and intra-group agreement/disagreement adjusts effective weights to break deadlocks.
"""

import logging

from strategies.base import Signal
from core.enums import SignalType

logger = logging.getLogger(__name__)


class SignalCombiner:
    """Combine multiple strategy signals using weighted voting."""

    def __init__(
        self,
        consensus_config: dict | None = None,
        min_active_ratio: float = 0.05,
    ):
        cfg = consensus_config or {}
        self._consensus_enabled = cfg.get("enabled", False)
        self._min_group_signals = cfg.get("min_group_signals", 2)
        self._consensus_boost = cfg.get("consensus_boost", 0.30)
        self._discord_penalty = cfg.get("discord_penalty", 0.40)
        self._groups: dict[str, list[str]] = cfg.get("groups", {})
        self._min_active_ratio = min_active_ratio

    def _apply_consensus(
        self, signals: list[Signal], weights: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Adjust weights based on intra-group agreement.

        Returns:
            (modified_weights, group_agreement_scores)
        """
        if not self._consensus_enabled or not self._groups:
            return weights, {}

        modified = dict(weights)
        agreement_scores: dict[str, float] = {}

        for group_name, members in self._groups.items():
            # Collect active (non-HOLD, weighted) signals in this group
            active = [
                s for s in signals
                if s.strategy_name in members
                and s.signal_type != SignalType.HOLD
                and weights.get(s.strategy_name, 0) > 0
            ]

            if len(active) < self._min_group_signals:
                continue

            n_buy = sum(1 for s in active if s.signal_type == SignalType.BUY)
            n_sell = sum(1 for s in active if s.signal_type == SignalType.SELL)
            n_total = len(active)
            agreement = max(n_buy, n_sell) / n_total  # 0.5 to 1.0
            agreement_scores[group_name] = agreement

            if n_buy == n_sell:
                # Perfect tie — reduce group influence
                factor = 1.0 - self._discord_penalty * 0.5
                for s in active:
                    modified[s.strategy_name] = weights[s.strategy_name] * factor
            else:
                majority = SignalType.BUY if n_buy > n_sell else SignalType.SELL
                for s in active:
                    if s.signal_type == majority:
                        boost = 1.0 + self._consensus_boost * agreement
                        modified[s.strategy_name] = weights[s.strategy_name] * boost
                    else:
                        penalty = 1.0 - self._discord_penalty * (1.0 - agreement)
                        modified[s.strategy_name] = weights[s.strategy_name] * penalty

        return modified, agreement_scores

    def combine(
        self,
        signals: list[Signal],
        weights: dict[str, float],
        min_confidence: float = 0.35,
        held_sell_bias: float = 0.0,
    ) -> Signal:
        """Combine signals using weighted voting.

        Args:
            signals: List of signals from individual strategies
            weights: Strategy name -> weight mapping from profile
            min_confidence: Minimum combined confidence to act
            held_sell_bias: Extra SELL confidence boost for held positions.
                When > 0 and at least one strategy votes SELL, sell_norm
                is increased by this amount, making exits easier.

        Returns:
            Combined signal
        """
        if not signals:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy_name="combiner",
                reason="No signals",
            )

        # Apply group consensus weight adjustment
        effective_weights, agreement_scores = self._apply_consensus(signals, weights)

        buy_score = 0.0
        sell_score = 0.0
        active_weight = 0.0
        total_weight = 0.0
        reasons = []
        all_indicators = {}
        # Track top contributing strategy per side
        top_buy: tuple[float, str] = (0.0, "")   # (weighted_conf, name)
        top_sell: tuple[float, str] = (0.0, "")

        for signal in signals:
            w = effective_weights.get(signal.strategy_name, 0.0)
            if w <= 0:
                continue

            total_weight += w

            if signal.signal_type == SignalType.BUY:
                weighted_conf = signal.confidence * w
                buy_score += weighted_conf
                active_weight += w
                reasons.append(f"+{signal.strategy_name}({signal.confidence:.0%})")
                if weighted_conf > top_buy[0]:
                    top_buy = (weighted_conf, signal.strategy_name)
            elif signal.signal_type == SignalType.SELL:
                weighted_conf = signal.confidence * w
                sell_score += weighted_conf
                active_weight += w
                reasons.append(f"-{signal.strategy_name}({signal.confidence:.0%})")
                if weighted_conf > top_sell[0]:
                    top_sell = (weighted_conf, signal.strategy_name)

            # Collect indicators
            for k, v in signal.indicators.items():
                all_indicators[f"{signal.strategy_name}.{k}"] = v

        # Add consensus metadata
        for group_name, score in agreement_scores.items():
            all_indicators[f"combiner.{group_name}_agreement"] = score

        if active_weight == 0 or total_weight == 0:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy_name="combiner",
                reason="No active signals",
            )

        # Min active ratio: at least N% of strategies must be active (BUY/SELL)
        active_ratio = active_weight / total_weight
        if active_ratio < self._min_active_ratio:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=max(buy_score, sell_score) / active_weight,
                strategy_name="combiner",
                reason=f"Active ratio too low ({active_ratio:.0%} < {self._min_active_ratio:.0%})",
            )

        # Normalize by active weight only (HOLD signals excluded from denominator)
        buy_norm = buy_score / active_weight
        sell_norm = sell_score / active_weight

        # Held-position bias: boost SELL norm when evaluating held stocks.
        # Only activates when at least one strategy already votes SELL,
        # preventing phantom sells on stocks with no bearish signals.
        if held_sell_bias > 0 and sell_score > 0:
            sell_norm += held_sell_bias

        if buy_norm > sell_norm and buy_norm >= min_confidence:
            attribution = top_buy[1]
            if not attribution:
                logger.warning(
                    "Combiner BUY with no top contributor (buy_score=%.4f, active_weight=%.4f, signals=%d)",
                    buy_score, active_weight, len(signals),
                )
                # Extract from first BUY reason as fallback
                for s in signals:
                    if s.signal_type == SignalType.BUY and effective_weights.get(s.strategy_name, 0) > 0:
                        attribution = s.strategy_name
                        break
                attribution = attribution or "combiner"
            return Signal(
                signal_type=SignalType.BUY,
                confidence=buy_norm,
                strategy_name=attribution,
                reason=f"BUY consensus: {', '.join(reasons)}",
                indicators=all_indicators,
            )

        if sell_norm > buy_norm and sell_norm >= min_confidence:
            attribution = top_sell[1]
            if not attribution:
                logger.warning(
                    "Combiner SELL with no top contributor (sell_score=%.4f, active_weight=%.4f, signals=%d)",
                    sell_score, active_weight, len(signals),
                )
                for s in signals:
                    if s.signal_type == SignalType.SELL and effective_weights.get(s.strategy_name, 0) > 0:
                        attribution = s.strategy_name
                        break
                attribution = attribution or "combiner"
            return Signal(
                signal_type=SignalType.SELL,
                confidence=sell_norm,
                strategy_name=attribution,
                reason=f"SELL consensus: {', '.join(reasons)}",
                indicators=all_indicators,
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=max(buy_norm, sell_norm),
            strategy_name="combiner",
            reason=f"Below threshold ({max(buy_norm, sell_norm):.0%} < {min_confidence:.0%})",
            indicators=all_indicators,
        )
