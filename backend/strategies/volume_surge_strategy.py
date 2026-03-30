"""Volume Surge Strategy.

Detects abnormal volume spikes and determines buy/sell signals
based on price-volume confirmation patterns.

Buy: Volume surge (>= threshold) + price rising + positive momentum
     + RSI probabilistic scoring (gradient, not binary)
Sell: Volume surge + price falling (distribution)

Key insight: Volume precedes price.
  Surge + rising price = accumulation (buy).
  Surge + falling price = distribution (sell).

STOCK-72 enhancements:
  - RSI probabilistic scoring: gradient contribution instead of
    binary overbought cutoff. RSI 30-50 is best zone (+0.10),
    50-65 neutral (+0.05), 65-75 slight penalty (+0.02),
    75-85 strong penalty (-0.05), 85+ severe penalty (-0.10).
  - Momentum AND condition: require both volume surge AND
    positive momentum (3-bar ROC > 0) for BUY signals.
  - New indicators: rsi_score, momentum_confirmed.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class VolumeSurgeStrategy(BaseStrategy):
    name = "volume_surge"
    display_name = "Volume Surge"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._volume_threshold = p.get(
            "volume_threshold", 1.8,
        )
        self._confirmation_bars = p.get(
            "confirmation_bars", 1,
        )
        self._min_price_change = p.get(
            "min_price_change", 0.3,
        )  # %

    async def analyze(
        self, df: pd.DataFrame, symbol: str,
    ) -> Signal:
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

        # RSI probabilistic score (STOCK-72)
        rsi_score = self._calc_rsi_score(rsi)

        # Price change analysis
        prev_close = float(prev["close"])
        price_change_pct = (
            (price - prev_close) / prev_close * 100
        )

        # Multi-bar price momentum — 3-bar ROC
        if len(df) >= 4:
            close_3 = float(df.iloc[-4]["close"])
            momentum_3d = (
                (price - close_3) / close_3 * 100
            )
        else:
            momentum_3d = price_change_pct

        # Momentum AND condition (STOCK-72):
        # Require positive 3-bar momentum for BUY signals
        momentum_confirmed = momentum_3d > 0

        indicators = {
            "volume_ratio": volume_ratio,
            "rsi": (
                float(rsi)
                if rsi is not None and not pd.isna(rsi)
                else 50
            ),
            "adx": (
                float(adx)
                if adx is not None and not pd.isna(adx)
                else 0
            ),
            "macd_histogram": (
                float(macd_hist)
                if macd_hist is not None
                and not pd.isna(macd_hist)
                else 0
            ),
            "rsi_score": rsi_score,
            "momentum_confirmed": momentum_confirmed,
        }

        # Not a surge — hold
        if volume_ratio < self._volume_threshold:
            return self._hold(
                f"Volume ratio {volume_ratio:.1f}x "
                f"below threshold"
            )

        # OBV trend (rising=accumulation, falling=distribution)
        obv_rising = False
        if (
            obv is not None
            and not pd.isna(obv)
            and len(df) >= 6
        ):
            obv_5 = df.iloc[-6:-1]["obv"].dropna()
            if len(obv_5) >= 3:
                obv_rising = float(obv) > float(
                    obv_5.mean()
                )

        # Trend filter: price above EMA20
        above_ema = True
        if ema_20 is not None and not pd.isna(ema_20):
            above_ema = price > float(ema_20)

        # === BUY SIGNAL ===
        # STOCK-72: Volume surge + price rising
        #   + momentum AND (3-bar ROC > 0)
        #   + trend confirmation
        if (
            price_change_pct > self._min_price_change
            and momentum_confirmed
            and (
                above_ema
                or volume_ratio
                >= self._volume_threshold * 1.5
            )
        ):
            confidence = self._calc_buy_confidence(
                volume_ratio,
                price_change_pct,
                rsi_score,
                adx,
                obv_rising,
            )

            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Volume surge {volume_ratio:.1f}x "
                    f"with price "
                    f"+{price_change_pct:.1f}% "
                    f"(accumulation)"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # === SELL SIGNAL ===
        # Volume surge + price falling = distribution
        if (
            price_change_pct < -self._min_price_change
            and volume_ratio >= self._volume_threshold
        ):
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
                    f"Volume surge {volume_ratio:.1f}x "
                    f"with price "
                    f"{price_change_pct:.1f}% "
                    f"(distribution)"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(
            f"Volume surge {volume_ratio:.1f}x but "
            f"no clear direction "
            f"(price {price_change_pct:+.1f}%)"
        )

    @staticmethod
    def _calc_rsi_score(rsi) -> float:
        """RSI probabilistic score (STOCK-72).

        Returns a gradient-based score reflecting how
        favorable the RSI zone is for buying:
          RSI < 30:  oversold, strong buy zone    → +0.10
          30-50:     accumulation zone             → +0.10
          50-65:     neutral/healthy               → +0.05
          65-75:     warming, slight caution       → +0.02
          75-85:     overbought, penalty           → -0.05
          85+:       extreme overbought, strong    → -0.10
          N/A:       neutral fallback              → 0.00
        """
        if rsi is None or pd.isna(rsi):
            return 0.0
        rsi_val = float(rsi)
        if rsi_val < 30:
            return 0.10
        if rsi_val < 50:
            return 0.10
        if rsi_val < 65:
            return 0.05
        if rsi_val < 75:
            return 0.02
        if rsi_val < 85:
            return -0.05
        return -0.10

    def _check_confirmation(
        self, df: pd.DataFrame,
    ) -> bool:
        """Check volume elevated for N consecutive bars."""
        if len(df) < self._confirmation_bars + 1:
            return False

        for i in range(1, self._confirmation_bars + 1):
            vr = df.iloc[-(i + 1)].get("volume_ratio")
            if (
                vr is None
                or pd.isna(vr)
                or float(vr) < 1.5
            ):
                return False
        return True

    def _calc_buy_confidence(
        self,
        volume_ratio: float,
        price_change: float,
        rsi_score: float,
        adx,
        obv_rising: bool,
    ) -> float:
        """Calculate buy confidence with RSI gradient.

        STOCK-72: Uses rsi_score (gradient) instead of
        binary RSI zone check. This allows the RSI
        contribution to smoothly scale confidence up
        or down based on the current RSI level.
        """
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
        if (
            adx is not None
            and not pd.isna(adx)
            and float(adx) > 25
        ):
            conf += 0.10

        # RSI probabilistic contribution (STOCK-72)
        conf += rsi_score

        # OBV confirmation
        if obv_rising:
            conf += 0.05

        return max(0.10, min(conf, 0.90))

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
        self._volume_threshold = params.get(
            "volume_threshold",
            self._volume_threshold,
        )
        self._confirmation_bars = params.get(
            "confirmation_bars",
            self._confirmation_bars,
        )
        self._min_price_change = params.get(
            "min_price_change",
            self._min_price_change,
        )
