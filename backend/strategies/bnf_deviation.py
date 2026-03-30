"""BNF Deviation Strategy — Mean Reversion.

Adapted from coin project (4h crypto -> 1D stocks).
Deviation thresholds adjusted: crypto -10%/+5% -> stocks -5%/+3%.

Buy: price deviates below SMA by threshold (oversold)
Sell: price deviates above SMA by threshold (overbought)
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class BNFDeviationStrategy(BaseStrategy):
    name = "bnf_deviation"
    display_name = "BNF Deviation"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._sma_period = p.get("sma_period", 25)
        self._buy_deviation = p.get("buy_deviation", -5.0)
        self._sell_deviation = p.get("sell_deviation", 3.0)
        self._rsi_boost_threshold = p.get("rsi_boost_threshold", 35.0)
        self._trend_filter_enabled = p.get("trend_filter_enabled", True)
        self._trend_sma_period = p.get("trend_sma_period", 200)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        price = float(df["close"].iloc[-1])
        sma = self._get_sma(df)

        if sma is None or sma == 0:
            return self._hold("SMA not available")

        deviation = (price - sma) / sma * 100
        rsi = self._get_rsi(df)
        trend_bearish = self._is_trend_bearish(df, price)

        indicators = {
            "deviation_pct": round(deviation, 2),
            "sma": round(sma, 2),
            "rsi": round(rsi, 2) if rsi is not None else None,
            "price": price,
            "trend_bearish": trend_bearish,
        }

        # BUY: oversold deviation
        if deviation <= self._buy_deviation:
            # Block buy in confirmed downtrend (catching falling knife)
            if trend_bearish:
                return self._hold(
                    f"Oversold (dev={deviation:+.1f}%) but downtrend — buy suppressed"
                )

            if deviation <= -10:
                confidence = 0.85
            elif deviation <= -7:
                confidence = 0.70
            else:
                confidence = 0.55
            # RSI oversold bonus
            if rsi is not None and rsi < self._rsi_boost_threshold:
                confidence += 0.10
            return Signal(
                signal_type=SignalType.BUY,
                confidence=round(min(confidence, 0.95), 2),
                strategy_name=self.name,
                reason=f"Mean reversion buy: SMA{self._sma_period} deviation {deviation:+.1f}%",
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: overbought deviation
        if deviation >= self._sell_deviation:
            if deviation >= 8:
                confidence = 0.80
            elif deviation >= 5:
                confidence = 0.65
            else:
                confidence = 0.50
            return Signal(
                signal_type=SignalType.SELL,
                confidence=round(min(confidence, 0.95), 2),
                strategy_name=self.name,
                reason=f"Mean reversion sell: SMA{self._sma_period} deviation {deviation:+.1f}%",
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(f"Neutral: SMA{self._sma_period} deviation {deviation:+.1f}%")

    def _get_sma(self, df: pd.DataFrame) -> float | None:
        for col in [f"sma_{self._sma_period}", f"SMA_{self._sma_period}"]:
            if col in df.columns:
                val = df[col].iloc[-1]
                if not pd.isna(val):
                    return float(val)
        if len(df) >= self._sma_period:
            return float(df["close"].iloc[-self._sma_period:].mean())
        return None

    def _is_trend_bearish(self, df: pd.DataFrame, price: float) -> bool:
        """Check if price is in a confirmed downtrend using long-term SMA."""
        if not self._trend_filter_enabled:
            return False
        trend_sma = self._get_trend_sma(df)
        if trend_sma is None:
            return False  # No data → assume neutral
        return price < trend_sma

    def _get_trend_sma(self, df: pd.DataFrame) -> float | None:
        """Get long-term SMA (e.g. SMA-200) for trend determination."""
        for col in [f"sma_{self._trend_sma_period}", f"SMA_{self._trend_sma_period}"]:
            if col in df.columns:
                val = df[col].iloc[-1]
                if not pd.isna(val):
                    return float(val)
        if len(df) >= self._trend_sma_period:
            return float(df["close"].iloc[-self._trend_sma_period:].mean())
        return None

    def _get_rsi(self, df: pd.DataFrame) -> float | None:
        for col in ["rsi", "rsi_14", "RSI_14"]:
            if col in df.columns:
                val = df[col].iloc[-1]
                if not pd.isna(val):
                    return float(val)
        return None

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "sma_period": self._sma_period,
            "buy_deviation": self._buy_deviation,
            "sell_deviation": self._sell_deviation,
            "rsi_boost_threshold": self._rsi_boost_threshold,
            "trend_filter_enabled": self._trend_filter_enabled,
            "trend_sma_period": self._trend_sma_period,
        }

    def set_params(self, params: dict) -> None:
        for key in ["sma_period", "buy_deviation", "sell_deviation",
                     "rsi_boost_threshold", "trend_filter_enabled", "trend_sma_period"]:
            if key in params:
                setattr(self, f"_{key}", params[key])
