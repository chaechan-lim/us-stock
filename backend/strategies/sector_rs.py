"""Sector Relative Strength (Sector-RS) strategy.

Per-symbol BUY signal that fires when:
  1. The symbol's sector is in the top-N by current relative-strength score
  2. The symbol itself has positive medium-term momentum within that sector

Differs from existing `sector_boost_weight` (a multiplier on other
strategies' confidence) by producing a standalone BUY of its own,
biased toward names that are *both* in strong sectors *and* showing
own-stock momentum.

Sector strength comes from a class-level snapshot set by the engine
(`set_sector_snapshot`) — analogous to `set_profit_exit_params` on
BaseStrategy. When unset, the strategy returns HOLD (no signal),
keeping it inert in environments that don't provide the snapshot.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class SectorRSStrategy(BaseStrategy):
    name = "sector_rs"
    display_name = "Sector Relative Strength"
    applicable_market_types = ["trending", "sideways"]
    required_timeframe = "1D"
    min_candles_required = 25  # for 20-day momentum

    # Class-level snapshot updated by the engine before each eval cycle.
    # {sector_name: strength_score_0_to_100} + {symbol: sector_name}.
    _sector_scores: dict[str, float] = {}
    _symbol_sector_map: dict[str, str] = {}

    @classmethod
    def set_sector_snapshot(
        cls,
        sector_scores: dict[str, float],
        symbol_sector_map: dict[str, str],
    ) -> None:
        """Engine call: refresh class-level sector context."""
        cls._sector_scores = dict(sector_scores or {})
        cls._symbol_sector_map = dict(symbol_sector_map or {})

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._top_n_sectors: int = p.get("top_n_sectors", 3)
        self._momentum_days: int = p.get("momentum_days", 20)
        self._min_own_momentum: float = p.get("min_own_momentum", 0.03)
        self._strength_pct_floor: float = p.get("strength_pct_floor", 60.0)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        # Strip exchange suffix for sector map lookup (mirrors SectorHistory)
        key = symbol
        if symbol.endswith(".KS") or symbol.endswith(".KQ"):
            key = symbol[:-3]
        sector = self._symbol_sector_map.get(key, "Unknown")
        if sector == "Unknown" or not self._sector_scores:
            return self._hold("No sector context")

        sector_score = self._sector_scores.get(sector, 0.0)
        # Top-N check: rank sectors by score, see if this sector is in top-N
        ranked = sorted(
            self._sector_scores.items(), key=lambda kv: kv[1], reverse=True,
        )
        top_n = {name for name, _ in ranked[: self._top_n_sectors]}

        # Compute own momentum (medium-term)
        close = df["close"]
        if len(close) <= self._momentum_days:
            return self._hold("Lookback too short")
        price = float(close.iloc[-1])
        past_price = float(close.iloc[-1 - self._momentum_days])
        if past_price <= 0:
            return self._hold("Invalid past price")
        own_momentum = (price - past_price) / past_price

        indicators = {
            "sector": sector,
            "sector_score": sector_score,
            "sector_rank": next(
                (i for i, (n, _) in enumerate(ranked, 1) if n == sector), -1,
            ),
            "own_momentum": own_momentum,
            "top_n": self._top_n_sectors,
        }

        # Buy conditions
        in_top = sector in top_n
        score_qualifies = sector_score >= self._strength_pct_floor
        momentum_qualifies = own_momentum >= self._min_own_momentum

        if not (in_top and score_qualifies and momentum_qualifies):
            bits = []
            if not in_top:
                bits.append(f"sector {sector} rank>{self._top_n_sectors}")
            if not score_qualifies:
                bits.append(
                    f"score {sector_score:.0f} < {self._strength_pct_floor:.0f}"
                )
            if not momentum_qualifies:
                bits.append(
                    f"own mom {own_momentum:+.1%} < "
                    f"{self._min_own_momentum:+.1%}"
                )
            return self._hold("; ".join(bits) or "no setup")

        # Confidence ramps with sector strength + own momentum
        confidence = 0.50
        if sector_score >= 70:
            confidence += 0.10
        if sector_score >= 80:
            confidence += 0.05
        if own_momentum >= 0.06:
            confidence += 0.10
        if own_momentum >= 0.10:
            confidence += 0.05
        confidence = min(0.85, confidence)

        return Signal(
            signal_type=SignalType.BUY,
            confidence=confidence,
            strategy_name=self.name,
            reason=(
                f"Sector {sector} score {sector_score:.0f} (top-"
                f"{self._top_n_sectors}), own mom {own_momentum:+.1%}"
            ),
            suggested_price=price,
            indicators=indicators,
        )

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "top_n_sectors": self._top_n_sectors,
            "momentum_days": self._momentum_days,
            "min_own_momentum": self._min_own_momentum,
            "strength_pct_floor": self._strength_pct_floor,
        }

    def set_params(self, params: dict) -> None:
        for key in (
            "top_n_sectors", "momentum_days",
            "min_own_momentum", "strength_pct_floor",
        ):
            if key in params:
                setattr(self, f"_{key}", params[key])
