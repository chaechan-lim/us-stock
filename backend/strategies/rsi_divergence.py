"""RSI Divergence Strategy.

Detects bullish/bearish divergence between price and RSI using pivot points.
BUY: Price makes lower pivot low but RSI makes higher pivot low (bullish divergence).
SELL: Price makes higher pivot high but RSI makes lower pivot high (bearish divergence).
"""

import numpy as np
import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class RSIDivergenceStrategy(BaseStrategy):
    name = "rsi_divergence"
    display_name = "RSI Divergence"
    applicable_market_types = ["sideways", "downtrend"]
    required_timeframe = "1D"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._rsi_period = p.get("rsi_period", 14)
        self._overbought = p.get("overbought", 70)
        self._oversold = p.get("oversold", 30)
        self._divergence_lookback = p.get("divergence_lookback", 30)
        self._min_price_move_pct = p.get("min_price_move_pct", 1.0)
        self._pivot_order = p.get("pivot_order", 3)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        price = float(row["close"])
        rsi = row.get("rsi")

        if rsi is None or pd.isna(rsi):
            return self._hold("RSI not available")

        rsi = float(rsi)
        lookback = min(self._divergence_lookback, len(df))
        window = df.iloc[-lookback:]

        rsi_col = window.get("rsi")
        if rsi_col is None or rsi_col.isna().all():
            return self._hold("RSI data incomplete")

        prices = window["close"].values.astype(float)
        rsi_arr = rsi_col.values.astype(float)

        pivot_lows = self._find_pivot_lows(prices, rsi_arr)
        pivot_highs = self._find_pivot_highs(prices, rsi_arr)

        indicators: dict = {
            "rsi": rsi,
            "lookback": lookback,
            "pivot_lows": len(pivot_lows),
            "pivot_highs": len(pivot_highs),
            "divergence_type": None,
        }

        # Bullish divergence: price lower low, RSI higher low
        if len(pivot_lows) >= 2:
            prev_p, prev_r = pivot_lows[-2]
            curr_p, curr_r = pivot_lows[-1]
            if prev_p > 0:
                price_drop = (prev_p - curr_p) / prev_p * 100
                if price_drop >= self._min_price_move_pct and curr_r > prev_r:
                    indicators["divergence_type"] = "bullish"
                    if rsi <= self._oversold:
                        confidence = 0.70
                    elif rsi <= self._oversold + 10:
                        confidence = 0.55
                    else:
                        confidence = 0.45
                    return Signal(
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        strategy_name=self.name,
                        reason=f"Bullish divergence: lower low, higher RSI low={rsi:.0f}",
                        suggested_price=price,
                        indicators=indicators,
                    )

        # Bearish divergence: price higher high, RSI lower high
        if len(pivot_highs) >= 2:
            prev_p, prev_r = pivot_highs[-2]
            curr_p, curr_r = pivot_highs[-1]
            if prev_p > 0:
                price_rise = (curr_p - prev_p) / prev_p * 100
                if price_rise >= self._min_price_move_pct and curr_r < prev_r:
                    indicators["divergence_type"] = "bearish"
                    if rsi >= self._overbought:
                        confidence = 0.70
                    elif rsi >= self._overbought - 10:
                        confidence = 0.55
                    else:
                        confidence = 0.45
                    return Signal(
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        strategy_name=self.name,
                        reason=f"Bearish divergence: higher high, lower RSI high={rsi:.0f}",
                        suggested_price=price,
                        indicators=indicators,
                    )

        # Extreme RSI zones as secondary signals
        if rsi < self._oversold:
            return Signal(
                signal_type=SignalType.BUY,
                confidence=0.4,
                strategy_name=self.name,
                reason=f"RSI oversold at {rsi:.0f}",
                suggested_price=price,
                indicators=indicators,
            )

        if rsi > self._overbought:
            return Signal(
                signal_type=SignalType.SELL,
                confidence=0.4,
                strategy_name=self.name,
                reason=f"RSI overbought at {rsi:.0f}",
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(f"No divergence detected, RSI={rsi:.0f}")

    def _find_pivot_lows(
        self, prices: np.ndarray, rsi_arr: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Find pivot lows, deduplicating consecutive pivots at similar prices."""
        raw = self._raw_pivots(prices, rsi_arr, kind="low")
        return self._dedup_pivots(raw, keep="min")

    def _find_pivot_highs(
        self, prices: np.ndarray, rsi_arr: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Find pivot highs, deduplicating consecutive pivots at similar prices."""
        raw = self._raw_pivots(prices, rsi_arr, kind="high")
        return self._dedup_pivots(raw, keep="max")

    def _raw_pivots(
        self, prices: np.ndarray, rsi_arr: np.ndarray, kind: str,
    ) -> list[tuple[int, float, float]]:
        """Find raw pivot points with bar indices."""
        order = self._pivot_order
        pivots: list[tuple[int, float, float]] = []
        for i in range(order, len(prices) - order):
            if np.isnan(prices[i]) or np.isnan(rsi_arr[i]):
                continue
            left = prices[i - order : i]
            right = prices[i + 1 : i + order + 1]
            if np.any(np.isnan(left)) or np.any(np.isnan(right)):
                continue
            if kind == "low":
                if prices[i] <= np.min(left) and prices[i] <= np.min(right):
                    pivots.append((i, float(prices[i]), float(rsi_arr[i])))
            else:
                if prices[i] >= np.max(left) and prices[i] >= np.max(right):
                    pivots.append((i, float(prices[i]), float(rsi_arr[i])))
        return pivots

    @staticmethod
    def _dedup_pivots(
        raw: list[tuple[int, float, float]], keep: str,
    ) -> list[tuple[float, float]]:
        """Merge consecutive pivots within a cluster into a single representative."""
        if not raw:
            return []
        groups: list[list[tuple[int, float, float]]] = [[raw[0]]]
        for pivot in raw[1:]:
            # Consecutive bars or same price → same cluster
            if pivot[0] - groups[-1][-1][0] <= 2:
                groups[-1].append(pivot)
            else:
                groups.append([pivot])
        result: list[tuple[float, float]] = []
        for group in groups:
            if keep == "min":
                best = min(group, key=lambda x: x[1])
            else:
                best = max(group, key=lambda x: x[1])
            result.append((best[1], best[2]))
        return result

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "rsi_period": self._rsi_period,
            "overbought": self._overbought,
            "oversold": self._oversold,
            "divergence_lookback": self._divergence_lookback,
            "min_price_move_pct": self._min_price_move_pct,
            "pivot_order": self._pivot_order,
        }

    def set_params(self, params: dict) -> None:
        self._rsi_period = params.get("rsi_period", self._rsi_period)
        self._overbought = params.get("overbought", self._overbought)
        self._oversold = params.get("oversold", self._oversold)
        self._divergence_lookback = params.get("divergence_lookback", self._divergence_lookback)
        self._min_price_move_pct = params.get("min_price_move_pct", self._min_price_move_pct)
        self._pivot_order = params.get("pivot_order", self._pivot_order)
