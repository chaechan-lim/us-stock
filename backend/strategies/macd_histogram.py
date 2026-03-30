"""MACD Histogram Strategy.

Detects momentum shifts via MACD histogram direction changes.

Buy: MACD histogram turns positive (crosses above zero) with increasing
     momentum, or bullish divergence detected.
Sell: MACD histogram turns negative (crosses below zero) with dynamic
      confidence based on histogram strength, or bearish divergence.
"""

import numpy as np
import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class MACDHistogramStrategy(BaseStrategy):
    name = "macd_histogram"
    display_name = "MACD Histogram"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 35

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._min_histogram_change = p.get(
            "min_histogram_change", 0.5
        )
        self._divergence_lookback = p.get(
            "divergence_lookback", 14
        )
        self._min_price_move_pct = p.get(
            "min_price_move_pct", 2.0
        )

    async def analyze(
        self, df: pd.DataFrame, symbol: str
    ) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])

        macd_hist = row.get("macd_histogram")
        prev_hist = prev.get("macd_histogram")
        macd = row.get("macd")
        macd_signal = row.get("macd_signal")
        rsi = row.get("rsi")

        if any(
            v is None or pd.isna(v)
            for v in [macd_hist, prev_hist]
        ):
            return self._hold("MACD not ready")

        hist_strength = self._histogram_strength(df)
        div_type = self._detect_divergence(df)

        indicators = {
            "macd": (
                float(macd)
                if macd is not None and not pd.isna(macd)
                else 0
            ),
            "macd_histogram": float(macd_hist),
            "macd_signal": (
                float(macd_signal)
                if macd_signal is not None
                and not pd.isna(macd_signal)
                else 0
            ),
            "rsi": (
                float(rsi)
                if rsi is not None and not pd.isna(rsi)
                else 50
            ),
            "divergence_type": div_type,
            "histogram_strength": hist_strength,
        }

        # Bullish divergence: price lower low, histogram higher
        if div_type == "bullish":
            confidence = self._calc_confidence(
                macd_hist, rsi, cross=True
            )
            confidence = min(confidence + 0.10, 0.90)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    "Bullish MACD divergence detected "
                    f"(hist={macd_hist:.2f})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # Bearish divergence: price higher high, histogram lower
        if div_type == "bearish":
            confidence = self._dynamic_sell_confidence(
                hist_strength
            )
            confidence = min(confidence + 0.10, 0.90)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    "Bearish MACD divergence detected "
                    f"(hist={macd_hist:.2f})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # BUY: histogram crosses above zero or accelerates
        hist_cross_up = prev_hist <= 0 and macd_hist > 0
        hist_accelerating = (
            macd_hist > 0
            and macd_hist > prev_hist
            and (macd_hist - prev_hist)
            > self._min_histogram_change
        )

        if hist_cross_up:
            confidence = self._calc_confidence(
                macd_hist, rsi, cross=True
            )
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    "MACD histogram crossed above zero "
                    f"({macd_hist:.2f})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        if (
            hist_accelerating
            and macd is not None
            and not pd.isna(macd)
            and macd > 0
        ):
            confidence = self._calc_confidence(
                macd_hist, rsi, cross=False
            )
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    "MACD histogram accelerating "
                    f"({prev_hist:.2f} -> {macd_hist:.2f})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: histogram crosses below zero (dynamic confidence)
        hist_cross_down = prev_hist >= 0 and macd_hist < 0

        if hist_cross_down:
            confidence = self._dynamic_sell_confidence(
                hist_strength
            )
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    "MACD histogram crossed below zero "
                    f"({macd_hist:.2f})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold("No MACD signal")

    def _histogram_strength(self, df: pd.DataFrame) -> float:
        """Compute histogram strength as ratio to recent range.

        Returns a value 0.0-1.0 where 1.0 means the current
        histogram is at the extreme of its recent range.
        """
        lookback = min(self._divergence_lookback, len(df))
        recent = df.iloc[-lookback:]
        hist_col = recent.get("macd_histogram")
        if hist_col is None or hist_col.isna().all():
            return 0.0

        values = hist_col.dropna().values.astype(float)
        if len(values) < 2:
            return 0.0

        hist_range = float(np.max(np.abs(values)))
        if hist_range == 0:
            return 0.0

        current = abs(float(values[-1]))
        return min(current / hist_range, 1.0)

    def _detect_divergence(
        self, df: pd.DataFrame
    ) -> str:
        """Detect divergence between price and MACD histogram.

        Compares price pivots with MACD histogram pivots over
        the lookback window, split into two halves.

        Returns:
            'bullish', 'bearish', or 'none'
        """
        lookback = min(
            self._divergence_lookback, len(df) - 1
        )
        if lookback < 4:
            return "none"

        recent = df.iloc[-lookback:]
        hist_col = recent.get("macd_histogram")
        if hist_col is None or hist_col.isna().any():
            return "none"

        prices = recent["close"].values.astype(float)
        hists = hist_col.values.astype(float)

        half = lookback // 2
        first_prices = prices[:half]
        second_prices = prices[half:]
        first_hists = hists[:half]
        second_hists = hists[half:]

        if len(first_prices) == 0 or len(second_prices) == 0:
            return "none"

        first_min_p = float(np.nanmin(first_prices))
        second_min_p = float(np.nanmin(second_prices))
        first_min_h = float(np.nanmin(first_hists))
        second_min_h = float(np.nanmin(second_hists))

        first_max_p = float(np.nanmax(first_prices))
        second_max_p = float(np.nanmax(second_prices))
        first_max_h = float(np.nanmax(first_hists))
        second_max_h = float(np.nanmax(second_hists))

        if first_min_p <= 0:
            return "none"

        # Bullish: price makes lower low, histogram higher
        price_drop_pct = (
            (first_min_p - second_min_p) / first_min_p * 100
        )
        if (
            price_drop_pct > self._min_price_move_pct
            and second_min_h > first_min_h
        ):
            return "bullish"

        # Bearish: price makes higher high, histogram lower
        price_rise_pct = (
            (second_max_p - first_max_p) / first_max_p * 100
        )
        if (
            price_rise_pct > self._min_price_move_pct
            and second_max_h < first_max_h
        ):
            return "bearish"

        return "none"

    def _dynamic_sell_confidence(
        self, hist_strength: float
    ) -> float:
        """Calculate sell confidence dynamically based on
        histogram strength.

        Stronger histogram decline -> higher sell confidence.
        Range: 0.50 (weak) to 0.80 (strong).
        """
        base = 0.50
        strength_bonus = hist_strength * 0.30
        return min(base + strength_bonus, 0.80)

    def _calc_confidence(
        self, hist, rsi, cross: bool
    ) -> float:
        conf = 0.60 if cross else 0.50
        if rsi is not None and not pd.isna(rsi):
            if 40 < rsi < 65:
                conf += 0.10
            elif rsi < 40:
                conf += 0.15  # Oversold bounce
        if abs(hist) > 1.0:
            conf += 0.10
        return min(conf, 0.90)

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "min_histogram_change": self._min_histogram_change,
            "divergence_lookback": self._divergence_lookback,
            "min_price_move_pct": self._min_price_move_pct,
        }

    def set_params(self, params: dict) -> None:
        self._min_histogram_change = params.get(
            "min_histogram_change",
            self._min_histogram_change,
        )
        self._divergence_lookback = params.get(
            "divergence_lookback",
            self._divergence_lookback,
        )
        self._min_price_move_pct = params.get(
            "min_price_move_pct",
            self._min_price_move_pct,
        )
