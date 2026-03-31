"""Regime Switch Strategy.

SPY/VIX based bull/bear regime detection for leveraged ETF trading.
BUY: Bull regime -> leveraged long ETFs (TQQQ, SOXL).
SELL: Bear regime -> exit long, consider inverse ETFs.
Uses SMA-200 cross on SPY + VIX thresholds for confirmation.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class RegimeSwitchStrategy(BaseStrategy):
    name = "regime_switch"
    display_name = "Regime Switch"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 50

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._spy_sma_period = p.get("spy_sma_period", 200)
        self._vix_bull_threshold = p.get("vix_bull_threshold", 20)
        self._vix_bear_threshold = p.get("vix_bear_threshold", 25)
        self._confirmation_days = p.get("confirmation_days", 2)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        price = float(row["close"])

        # Use SMA-200 or SMA-50 as regime proxy
        sma_200 = row.get("sma_200")
        sma_50 = row.get("sma_50")
        ema_20 = row.get("ema_20")
        rsi = row.get("rsi")
        adx = row.get("adx")

        # VIX data (optional — graceful fallback if not present)
        vix_raw = row.get("vix")
        vix: float | None = None
        if vix_raw is not None and not pd.isna(vix_raw):
            vix = float(vix_raw)

        # Primary: price vs SMA_200
        if sma_200 is not None and not pd.isna(sma_200):
            regime_ma = float(sma_200)
        elif sma_50 is not None and not pd.isna(sma_50):
            regime_ma = float(sma_50)
        else:
            return self._hold("Moving averages not available")

        above_ma = price > regime_ma

        # Confirmation: count consecutive days above/below MA
        confirmed_days = 0
        for i in range(1, min(self._confirmation_days + 3, len(df))):
            prev = df.iloc[-1 - i]
            prev_price = float(prev["close"])
            if above_ma and prev_price > regime_ma:
                confirmed_days += 1
            elif not above_ma and prev_price < regime_ma:
                confirmed_days += 1
            else:
                break

        confirmed = confirmed_days >= self._confirmation_days

        # Determine regime label
        if above_ma and confirmed:
            regime = "bull"
        elif not above_ma and confirmed:
            regime = "bear"
        else:
            regime = "transition"

        indicators = {
            "price_vs_ma": "above" if above_ma else "below",
            "regime_ma": regime_ma,
            "confirmed_days": confirmed_days,
            "rsi": float(rsi) if rsi is not None and not pd.isna(rsi) else 50,
            "vix": vix,
            "regime": regime,
        }

        # Bull regime: price above MA, confirmed
        if above_ma and confirmed:
            # VIX-adjusted base confidence
            if vix is not None:
                if vix <= self._vix_bull_threshold:
                    confidence = 0.65  # Low VIX → safe bull
                elif vix <= self._vix_bear_threshold:
                    confidence = 0.55  # Moderate VIX → weaker bull
                else:
                    confidence = 0.40  # High VIX → entry suppression
            else:
                confidence = 0.55  # No VIX: preserve original behavior

            if adx is not None and not pd.isna(adx) and float(adx) > 25:
                confidence += 0.10
            if rsi is not None and not pd.isna(rsi) and 40 < float(rsi) < 70:
                confidence += 0.10
            if ema_20 is not None and not pd.isna(ema_20) and price > float(ema_20):
                confidence += 0.05
            vix_str = f", VIX={vix:.1f}" if vix is not None else ""
            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 0.95),
                strategy_name=self.name,
                reason=f"Bull regime: price above MA ({confirmed_days}d confirmed){vix_str}",
                suggested_price=price,
                indicators=indicators,
            )

        # Bear regime: price below MA, confirmed
        if not above_ma and confirmed:
            confidence = 0.55
            if rsi is not None and not pd.isna(rsi) and float(rsi) < 40:
                confidence += 0.10
            # High VIX confirms bear regime
            if vix is not None and vix >= self._vix_bear_threshold:
                confidence += 0.10
            vix_str = f", VIX={vix:.1f}" if vix is not None else ""
            return Signal(
                signal_type=SignalType.SELL,
                confidence=min(confidence, 0.95),
                strategy_name=self.name,
                reason=f"Bear regime: price below MA ({confirmed_days}d confirmed){vix_str}",
                suggested_price=price,
                indicators=indicators,
            )

        direction = "above" if above_ma else "below"
        return self._hold(f"Regime transition, {confirmed_days}d {direction} MA")

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "spy_sma_period": self._spy_sma_period,
            "vix_bull_threshold": self._vix_bull_threshold,
            "vix_bear_threshold": self._vix_bear_threshold,
            "confirmation_days": self._confirmation_days,
        }

    def set_params(self, params: dict) -> None:
        self._spy_sma_period = params.get("spy_sma_period", self._spy_sma_period)
        self._vix_bull_threshold = params.get("vix_bull_threshold", self._vix_bull_threshold)
        self._vix_bear_threshold = params.get("vix_bear_threshold", self._vix_bear_threshold)
        self._confirmation_days = params.get("confirmation_days", self._confirmation_days)
