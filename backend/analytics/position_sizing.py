"""Kelly Criterion Position Sizing.

Optimal bet sizing based on edge and odds:
  Kelly % = W - (1 - W) / R
  W = win probability
  R = win/loss ratio (avg win / avg loss)

Full Kelly is too aggressive — use fractional Kelly (default 0.25).

References:
- Kelly (1956) "A New Interpretation of Information Rate"
- Thorp (2006) "The Kelly Criterion in Blackjack"
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly Criterion calculation result."""
    kelly_fraction: float       # Raw Kelly fraction (can be >1 or <0)
    position_pct: float         # Fractional Kelly % of portfolio to allocate
    confidence_boost: float     # Multiplier from signal confidence
    factor_boost: float         # Multiplier from factor score
    final_allocation_pct: float # Final % of portfolio after all adjustments
    reason: str = ""


class KellyPositionSizer:
    """Position sizing using fractional Kelly Criterion.

    Combines:
    1. Kelly fraction from historical win rate and payoff ratio
    2. Signal confidence multiplier (high confidence → bigger position)
    3. Factor score multiplier (high factor score → bigger position)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.40,    # Use 40% Kelly (aggressive)
        max_position_pct: float = 0.15,  # Hard cap per position
        min_position_pct: float = 0.05,  # 5% minimum position
        confidence_exponent: float = 1.2, # Less confidence penalty
        factor_weight: float = 0.3,       # How much factor score affects sizing
    ):
        self._kelly_frac = kelly_fraction
        self._max_pct = max_position_pct
        self._min_pct = min_position_pct
        self._conf_exp = confidence_exponent
        self._factor_weight = factor_weight

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        signal_confidence: float = 0.5,
        factor_score: float = 0.0,
        portfolio_value: float = 100000,
    ) -> KellyResult:
        """Calculate optimal position size.

        Args:
            win_rate: Historical probability of winning (0-1).
            avg_win: Average winning trade return (e.g. 0.08 = 8%).
            avg_loss: Average losing trade return as positive (e.g. 0.04 = 4%).
            signal_confidence: Combined signal confidence (0-1).
            factor_score: Factor model composite z-score (-3 to +3 typical).
            portfolio_value: Total portfolio value in USD.

        Returns:
            KellyResult with optimal allocation percentage.
        """
        # Edge case: not enough data
        if avg_loss <= 0 or avg_win <= 0:
            return KellyResult(
                kelly_fraction=0, position_pct=0,
                confidence_boost=1, factor_boost=1,
                final_allocation_pct=0,
                reason="Insufficient trade history",
            )

        # Kelly formula: W - (1-W)/R where R = avg_win/avg_loss
        r = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / r

        # Negative Kelly = no edge, don't bet
        if kelly <= 0:
            return KellyResult(
                kelly_fraction=kelly, position_pct=0,
                confidence_boost=1, factor_boost=1,
                final_allocation_pct=0,
                reason=f"Negative Kelly ({kelly:.3f}): no edge",
            )

        # Fractional Kelly
        base_pct = kelly * self._kelly_frac

        # Signal confidence boost: higher confidence → larger position
        # conf^exponent scales 0.5→0.25, 0.7→0.49, 0.9→0.81
        confidence_boost = signal_confidence ** self._conf_exp

        # Factor score boost: positive factor → up to 1.5x, negative → down to 0.5x
        # Sigmoid-like mapping from z-score to multiplier
        factor_boost = 1.0 + self._factor_weight * np.tanh(factor_score)

        # Final allocation
        final_pct = base_pct * confidence_boost * factor_boost

        # Clamp
        final_pct = max(self._min_pct, min(final_pct, self._max_pct))

        return KellyResult(
            kelly_fraction=round(kelly, 4),
            position_pct=round(base_pct, 4),
            confidence_boost=round(confidence_boost, 4),
            factor_boost=round(factor_boost, 4),
            final_allocation_pct=round(final_pct, 4),
            reason="OK",
        )

    def calculate_quantity(
        self,
        result: KellyResult,
        portfolio_value: float,
        price: float,
    ) -> int:
        """Convert allocation percentage to share quantity."""
        if result.final_allocation_pct <= 0 or price <= 0:
            return 0
        allocation_usd = portfolio_value * result.final_allocation_pct
        return int(allocation_usd / price)
