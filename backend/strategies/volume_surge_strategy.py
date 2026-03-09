"""Volume Surge Strategy.

Detects abnormal volume spikes and determines buy/sell signals
based on price-volume confirmation patterns.

Buy: Volume surge (≥ threshold) + price rising + trend confirmation
Sell: Volume surge + price falling (distribution) OR held position weakening

Key insight: Volume precedes price. A surge with rising price = accumulation (buy),
a surge with falling price = distribution (sell).
"""

import pandas as pd

from strategies.base import BaseStrategy, Signal
from core.enums import SignalType


class VolumeSurgeStrategy(BaseStrategy):
    name = "volume_surge"
    display_name = "Volume Surge"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._volume_threshold = p.get("volume_threshold", 1.8)
        self._confirmation_bars = p.get("confirmation_bars", 1)
        self._min_price_change = p.get("min_price_change", 0.3)  # %

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])

        volume_ratio = row.get("volume_ratio")
        rsi = row.get("rsi")
        adx = row.get("adx")
        macd_hist = row.get("macd_histogram")
        obv = row.get("obv")
        ema_20 = row.get("ema_20")

        if volume_ratio is None or pd.isna(volume_ratio):
            return self._hold("Volume ratio not available")

        volume_ratio = float(volume_ratio)

        indicators = {
            "volume_ratio": volume_ratio,
            "rsi": float(rsi) if rsi is not None and not pd.isna(rsi) else 50,
            "adx": float(adx) if adx is not None and not pd.isna(adx) else 0,
            "macd_histogram": float(macd_hist) if macd_hist is not None and not pd.isna(macd_hist) else 0,
        }

        # Not a surge — hold
        if volume_ratio < self._volume_threshold:
            return self._hold(f"Volume ratio {volume_ratio:.1f}x below threshold")

        # Price change analysis
        prev_close = float(prev["close"])
        price_change_pct = (price - prev_close) / prev_close * 100

        # Multi-bar price momentum (3-bar)
        if len(df) >= 4:
            close_3 = float(df.iloc[-4]["close"])
            momentum_3d = (price - close_3) / close_3 * 100
        else:
            momentum_3d = price_change_pct

        # OBV trend (rising = accumulation, falling = distribution)
        obv_rising = False
        if obv is not None and not pd.isna(obv) and len(df) >= 6:
            obv_5 = df.iloc[-6:-1]["obv"].dropna()
            if len(obv_5) >= 3:
                obv_rising = float(obv) > float(obv_5.mean())

        # Trend filter: price above EMA20
        above_ema = True
        if ema_20 is not None and not pd.isna(ema_20):
            above_ema = price > float(ema_20)

        # Confirmation: check if previous bars also show volume interest
        confirmed = self._check_confirmation(df)

        # === BUY SIGNAL ===
        # Volume surge + price rising + trend/momentum confirmation
        if (
            price_change_pct > self._min_price_change
            and (momentum_3d > 0 or price_change_pct > 1.5)
            and (above_ema or volume_ratio >= self._volume_threshold * 1.5)
        ):
            confidence = self._calc_buy_confidence(
                volume_ratio, price_change_pct, rsi, adx, obv_rising,
            )

            # RSI overbought filter — reduce confidence, don't buy into extreme
            if rsi is not None and not pd.isna(rsi) and float(rsi) > 80:
                return self._hold("Volume surge but RSI overbought")

            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Volume surge {volume_ratio:.1f}x with "
                    f"price +{price_change_pct:.1f}% (accumulation)"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # === SELL SIGNAL ===
        # Volume surge + price falling = distribution pattern
        if (
            price_change_pct < -self._min_price_change
            and volume_ratio >= self._volume_threshold
        ):
            # Stronger sell if below EMA and MACD bearish
            bearish_macd = (
                macd_hist is not None
                and not pd.isna(macd_hist)
                and float(macd_hist) < 0
            )

            confidence = 0.55
            if not above_ema:
                confidence += 0.10
            if bearish_macd:
                confidence += 0.10
            if not obv_rising:
                confidence += 0.05
            if volume_ratio >= 4.0:
                confidence += 0.10
            confidence = min(confidence, 0.90)

            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Volume surge {volume_ratio:.1f}x with "
                    f"price {price_change_pct:.1f}% (distribution)"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(
            f"Volume surge {volume_ratio:.1f}x but no clear direction "
            f"(price {price_change_pct:+.1f}%)"
        )

    def _check_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume has been elevated for N consecutive bars."""
        if len(df) < self._confirmation_bars + 1:
            return False

        for i in range(1, self._confirmation_bars + 1):
            vr = df.iloc[-(i + 1)].get("volume_ratio")
            if vr is None or pd.isna(vr) or float(vr) < 1.5:
                return False
        return True

    def _calc_buy_confidence(
        self, volume_ratio: float, price_change: float,
        rsi, adx, obv_rising: bool,
    ) -> float:
        conf = 0.50

        # Volume strength
        if volume_ratio >= 5.0:
            conf += 0.15
        elif volume_ratio >= 3.5:
            conf += 0.10
        elif volume_ratio >= 2.5:
            conf += 0.05

        # Price momentum
        if price_change > 3.0:
            conf += 0.10
        elif price_change > 1.5:
            conf += 0.05

        # Trend strength (ADX)
        if adx is not None and not pd.isna(adx) and float(adx) > 25:
            conf += 0.10

        # RSI in healthy zone (not overbought)
        if rsi is not None and not pd.isna(rsi):
            rsi_val = float(rsi)
            if 45 < rsi_val < 65:
                conf += 0.05
            elif 65 <= rsi_val < 75:
                conf += 0.02

        # OBV confirmation
        if obv_rising:
            conf += 0.05

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
            "volume_threshold": self._volume_threshold,
            "confirmation_bars": self._confirmation_bars,
            "min_price_change": self._min_price_change,
        }

    def set_params(self, params: dict) -> None:
        self._volume_threshold = params.get("volume_threshold", self._volume_threshold)
        self._confirmation_bars = params.get("confirmation_bars", self._confirmation_bars)
        self._min_price_change = params.get("min_price_change", self._min_price_change)
