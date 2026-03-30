"""Donchian Channel Breakout Strategy.

Enhanced turtle-style breakout strategy with ADX filter,
volume confirmation, and channel width confidence scaling.

Buy: Price breaks above N-period high (Donchian upper)
Sell: Price breaks below exit-period low (turtle exit),
      Donchian lower break, or ADX trend exhaustion
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class DonchianBreakoutStrategy(BaseStrategy):
    name = "donchian_breakout"
    display_name = "Donchian Breakout"
    applicable_market_types = ["trending"]
    required_timeframe = "1D"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._entry_period = p.get("entry_period", 20)
        self._exit_period = p.get("exit_period", 10)
        self._adx_threshold = p.get("adx_threshold", 25.0)
        self._volume_multiplier = p.get(
            "volume_multiplier", 1.5
        )
        self._adx_lookback = p.get("adx_lookback", 3)

    async def analyze(
        self, df: pd.DataFrame, symbol: str
    ) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])

        donchian_upper = row.get("donchian_upper")
        donchian_lower = row.get("donchian_lower")
        atr = row.get("atr")
        adx = row.get("adx")
        volume_ratio = row.get("volume_ratio")

        if any(
            v is None or pd.isna(v)
            for v in [donchian_upper, donchian_lower]
        ):
            return self._hold("Indicators not ready")

        donchian_upper = float(donchian_upper)
        donchian_lower = float(donchian_lower)

        # Channel width as percentage
        channel_width = 0.0
        if donchian_lower > 0:
            channel_width = (
                (donchian_upper - donchian_lower)
                / donchian_lower
                * 100
            )

        # Turtle exit levels (exclude current bar)
        exit_low, exit_high = self._calc_turtle_exits(df)

        # ADX trend exhaustion detection
        adx_declining = self._is_adx_declining(df)

        adx_val = (
            float(adx)
            if adx is not None and not pd.isna(adx)
            else 0
        )
        atr_val = (
            float(atr)
            if atr is not None and not pd.isna(atr)
            else 0
        )
        vr_val = (
            float(volume_ratio)
            if volume_ratio is not None
            and not pd.isna(volume_ratio)
            else 0
        )

        indicators = {
            "donchian_upper": donchian_upper,
            "donchian_lower": donchian_lower,
            "channel_width_pct": round(channel_width, 2),
            "atr": atr_val,
            "adx": adx_val,
            "volume_ratio": vr_val,
            "turtle_exit": {
                "exit_low": round(exit_low, 4),
                "exit_high": round(exit_high, 4),
                "period": self._exit_period,
            },
            "adx_declining": adx_declining,
        }

        # BUY: breakout above prior Donchian upper
        prev_upper = prev.get("donchian_upper")
        if (
            prev_upper is not None
            and not pd.isna(prev_upper)
        ):
            prev_upper = float(prev_upper)
        else:
            prev_upper = donchian_upper
        prev_close = float(prev["close"])
        if price > prev_upper and prev_close <= prev_upper:
            confidence = self._calc_buy_confidence(
                adx,
                volume_ratio,
                atr,
                price,
                prev_upper,
                channel_width,
            )
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Donchian breakout: price {price:.2f}"
                    f" > upper {prev_upper:.2f}"
                    f" (width={channel_width:.1f}%)"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: break below prior Donchian lower
        prev_lower = prev.get("donchian_lower")
        if (
            prev_lower is not None
            and not pd.isna(prev_lower)
        ):
            prev_lower = float(prev_lower)
        else:
            prev_lower = donchian_lower
        if price < prev_lower:
            width_bonus = min(channel_width / 20.0, 0.15)
            confidence = min(0.60 + width_bonus, 0.95)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Donchian lower break:"
                    f" price {price:.2f}"
                    f" < lower {prev_lower:.2f}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: turtle exit — break below exit-period low
        if price < exit_low:
            return Signal(
                signal_type=SignalType.SELL,
                confidence=0.55,
                strategy_name=self.name,
                reason=(
                    f"Turtle exit:"
                    f" price {price:.2f}"
                    f" < {self._exit_period}-day"
                    f" low {exit_low:.2f}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: ADX trend exhaustion — ADX was above threshold
        # and is now declining (trend losing momentum)
        if adx_declining and adx_val > self._adx_threshold:
            return Signal(
                signal_type=SignalType.SELL,
                confidence=0.50,
                strategy_name=self.name,
                reason=(
                    f"ADX trend exhaustion:"
                    f" ADX {adx_val:.1f} declining"
                    f" (threshold {self._adx_threshold})"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(
            "No breakout", indicators=indicators
        )

    def _calc_turtle_exits(
        self, df: pd.DataFrame
    ) -> tuple[float, float]:
        """Calculate turtle exit levels.

        Returns (exit_low, exit_high) using exit_period
        bars, excluding the current bar.
        - exit_low: N-day low for long exits
        - exit_high: N-day high for short exits
        """
        ep = self._exit_period
        if len(df) > ep + 1:
            window = df.iloc[-(ep + 1) : -1]
        else:
            window = df.iloc[:-1]

        exit_low = float(window["low"].min())
        exit_high = float(window["high"].max())
        return exit_low, exit_high

    def _is_adx_declining(
        self, df: pd.DataFrame
    ) -> bool:
        """Detect ADX trend exhaustion.

        ADX is considered declining when it has dropped
        for `adx_lookback` consecutive bars, indicating
        the trend is losing momentum after a peak.
        """
        lookback = self._adx_lookback
        needed = lookback + 1  # need lookback+1 valid ADX
        if len(df) < needed:
            return False

        recent = df["adx"].iloc[-needed:]
        if recent.isna().any():
            return False

        vals = [float(v) for v in recent]
        # Check each consecutive pair is declining
        for i in range(1, len(vals)):
            if vals[i] >= vals[i - 1]:
                return False
        return True

    def _calc_buy_confidence(
        self,
        adx,
        volume_ratio,
        atr,
        price,
        upper,
        channel_width,
    ) -> float:
        # Base confidence from channel width
        width_bonus = min(channel_width / 20.0, 0.15)
        conf = 0.55 + width_bonus

        # ADX bonus: strong trend confirmation
        if (
            adx
            and not pd.isna(adx)
            and float(adx) > self._adx_threshold
        ):
            conf += 0.10

        # Volume bonus: breakout with volume confirmation
        if (
            volume_ratio
            and not pd.isna(volume_ratio)
            and float(volume_ratio) > self._volume_multiplier
        ):
            conf += 0.10

        # Breakout strength: how far above channel
        if (
            atr
            and not pd.isna(atr)
            and float(atr) > 0
        ):
            breakout_atr = (price - upper) / float(atr)
            if breakout_atr > 0.5:
                conf += 0.05

        return min(round(conf, 2), 0.95)

    def _hold(
        self,
        reason: str,
        indicators: dict | None = None,
    ) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
            indicators=indicators or {},
        )

    def get_params(self) -> dict:
        return {
            "entry_period": self._entry_period,
            "exit_period": self._exit_period,
            "adx_threshold": self._adx_threshold,
            "volume_multiplier": self._volume_multiplier,
            "adx_lookback": self._adx_lookback,
        }

    def set_params(self, params: dict) -> None:
        self._entry_period = params.get(
            "entry_period", self._entry_period
        )
        self._exit_period = params.get(
            "exit_period", self._exit_period
        )
        self._adx_threshold = params.get(
            "adx_threshold", self._adx_threshold
        )
        self._volume_multiplier = params.get(
            "volume_multiplier", self._volume_multiplier
        )
        self._adx_lookback = params.get(
            "adx_lookback", self._adx_lookback
        )
