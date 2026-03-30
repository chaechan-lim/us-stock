"""Sector Rotation Strategy.

Monthly rotation into strongest sectors based on relative strength.
BUY: Sector shows strong momentum vs SPY (top-N sectors).
SELL: Sector loses relative strength.

Volatility normalization: raw returns are divided by rolling
volatility so that low-vol sectors are not penalised relative to
high-vol ones (risk-adjusted scoring).

Relaxed sell: moderate underperformance no longer triggers an
aggressive sell — only severe weakness does.
"""

import numpy as np
import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class SectorRotationStrategy(BaseStrategy):
    name = "sector_rotation"
    display_name = "Sector Rotation"
    applicable_market_types = ["trending", "sideways"]
    required_timeframe = "1M"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._lookback_weeks = p.get("lookback_weeks", 12)
        self._min_strength_score = p.get(
            "min_strength_score", 60,
        )
        self._vol_window = p.get("vol_window", 20)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_volatility(
        closes: pd.Series, window: int,
    ) -> float:
        """Annualised volatility of daily log-returns."""
        if len(closes) < window + 1:
            return 0.0
        log_ret = np.log(
            closes.iloc[-window:]
            / closes.iloc[-window:].shift(1),
        ).dropna()
        if len(log_ret) < 2:
            return 0.0
        return float(log_ret.std() * np.sqrt(252))

    @staticmethod
    def _normalize_by_vol(
        raw_return: float, volatility: float,
    ) -> float:
        """Scale return by volatility (Sharpe-like)."""
        if volatility <= 0:
            return raw_return
        return raw_return / volatility

    # ------------------------------------------------------------------

    async def analyze(  # noqa: C901
        self, df: pd.DataFrame, symbol: str,
    ) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        price = float(row["close"])

        # --- momentum returns ------------------------------------
        lookback_days = self._lookback_weeks * 5
        lookback_idx = max(0, len(df) - lookback_days)
        past_price = float(df.iloc[lookback_idx]["close"])
        period_return = (
            (price - past_price) / past_price
            if past_price > 0 else 0
        )

        short_idx = max(0, len(df) - 20)
        short_price = float(df.iloc[short_idx]["close"])
        short_return = (
            (price - short_price) / short_price
            if short_price > 0 else 0
        )

        # --- volatility normalization ----------------------------
        volatility = self._compute_volatility(
            df["close"], self._vol_window,
        )
        norm_period = self._normalize_by_vol(
            period_return, volatility,
        )
        norm_short = self._normalize_by_vol(
            short_return, volatility,
        )

        # strength uses vol-normalised returns
        strength_score = (
            (norm_period * 60 + norm_short * 40) * 100
        )

        ema_20 = row.get("ema_20")
        ema_50 = row.get("ema_50")
        rsi = row.get("rsi")
        volume_ratio = row.get("volume_ratio")

        indicators = {
            "period_return": period_return,
            "short_return": short_return,
            "strength_score": strength_score,
            "volatility": volatility,
            "volatility_normalized_score": strength_score,
            "rsi": (
                float(rsi)
                if rsi is not None and not pd.isna(rsi)
                else 50
            ),
        }

        # --- BUY: strong sector momentum -------------------------
        if strength_score >= self._min_strength_score:
            confidence = 0.50
            if strength_score > self._min_strength_score * 1.5:
                confidence += 0.15
            elif (
                strength_score
                > self._min_strength_score * 1.2
            ):
                confidence += 0.10
            if (
                ema_20 is not None
                and ema_50 is not None
                and not pd.isna(ema_20)
                and not pd.isna(ema_50)
            ):
                if float(ema_20) > float(ema_50):
                    confidence += 0.10
            if (
                volume_ratio is not None
                and not pd.isna(volume_ratio)
                and float(volume_ratio) > 1.2
            ):
                confidence += 0.05
            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 0.95),
                strategy_name=self.name,
                reason=(
                    f"Sector strength {strength_score:.0f}"
                    f", return {period_return:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # --- SELL (relaxed) --------------------------------------
        # Severe weakness: both score deeply negative AND
        # short-term return well below threshold -> full sell.
        # Moderate weakness: score < 0 but short return only
        # mildly negative -> reduced-confidence sell so other
        # signals can override.
        sell_reason: str | None = None
        sell_confidence = 0.0

        if strength_score < 0 and short_return < -0.05:
            # severe
            sell_reason = "severe_weakness"
            sell_confidence = 0.55
            if strength_score < -self._min_strength_score:
                sell_confidence += 0.15
        elif strength_score < 0 and short_return < -0.02:
            # moderate — relaxed sell
            sell_reason = "moderate_weakness"
            sell_confidence = 0.35

        if sell_reason is not None:
            indicators["sell_reason"] = sell_reason
            return Signal(
                signal_type=SignalType.SELL,
                confidence=min(sell_confidence, 0.95),
                strategy_name=self.name,
                reason=(
                    f"Sector {sell_reason}"
                    f" {strength_score:.0f}"
                    f", short {short_return:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(
            f"Strength={strength_score:.0f},"
            f" below threshold",
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
            "lookback_weeks": self._lookback_weeks,
            "min_strength_score": self._min_strength_score,
            "vol_window": self._vol_window,
        }

    def set_params(self, params: dict) -> None:
        self._lookback_weeks = params.get(
            "lookback_weeks", self._lookback_weeks,
        )
        self._min_strength_score = params.get(
            "min_strength_score",
            self._min_strength_score,
        )
        self._vol_window = params.get(
            "vol_window", self._vol_window,
        )
