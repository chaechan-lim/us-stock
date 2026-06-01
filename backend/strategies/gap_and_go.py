"""Gap-and-Go strategy.

Hypothesis: On daily bars, a symbol that *gaps up* meaningfully at the
open AND holds the gap intraday with above-average volume tends to
continue for several days. Catches news-catalyst overnight reactions
that the trend/momentum strategies pick up too late.

Designed for KR sessions where overnight drivers are common (US session
close → KR pre-market reaction). Live entry should fire as soon as the
first 5-15 min of session confirm the gap; backtest approximates this
by reading at bar close (`close` of the gap day).

Signal conditions (BUY):
  gap_pct >= min_gap_pct         (default 3%)
  intraday_return >= 0           (close >= open, gap held)
  volume_ratio >= min_vol_ratio  (default 1.5x vs 20-day avg)

Confidence ramps up with gap size + intraday strength + volume.

NEVER produces SELL — exit logic comes from SL/TP/trailing. Pure entry.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class GapAndGoStrategy(BaseStrategy):
    name = "gap_and_go"
    display_name = "Gap-and-Go"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 25  # need 20-day vol avg + a few extras

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._min_gap_pct: float = p.get("min_gap_pct", 0.03)
        self._min_vol_ratio: float = p.get("min_vol_ratio", 1.5)
        self._min_intraday_return: float = p.get("min_intraday_return", 0.005)
        # Big-gap cap: filter pump-and-dump (e.g. 30% gaps often reverse)
        self._max_gap_pct: float = p.get("max_gap_pct", 0.15)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        prev_close = float(df.iloc[-2]["close"])
        today_open = float(df.iloc[-1]["open"])
        today_close = float(df.iloc[-1]["close"])
        today_high = float(df.iloc[-1]["high"])
        today_volume = float(df.iloc[-1]["volume"])

        # 20-day average volume (excluding today)
        vol_window = df["volume"].iloc[-21:-1]
        avg_volume = float(vol_window.mean()) if len(vol_window) > 0 else 0.0

        if prev_close <= 0 or today_open <= 0 or avg_volume <= 0:
            return self._hold("Invalid prices/volume")

        gap_pct = (today_open - prev_close) / prev_close
        intraday_return = (today_close - today_open) / today_open
        volume_ratio = today_volume / avg_volume

        indicators = {
            "gap_pct": gap_pct,
            "intraday_return": intraday_return,
            "volume_ratio": volume_ratio,
            "prev_close": prev_close,
            "today_open": today_open,
            "today_close": today_close,
        }

        # Conditions
        gap_qualifies = self._min_gap_pct <= gap_pct <= self._max_gap_pct
        intraday_holds = intraday_return >= self._min_intraday_return
        volume_qualifies = volume_ratio >= self._min_vol_ratio

        if not (gap_qualifies and intraday_holds and volume_qualifies):
            reason_bits = []
            if not gap_qualifies:
                reason_bits.append(
                    f"gap {gap_pct:.1%} out of [{self._min_gap_pct:.0%},"
                    f"{self._max_gap_pct:.0%}]"
                )
            if not intraday_holds:
                reason_bits.append(f"intraday {intraday_return:+.1%} weak")
            if not volume_qualifies:
                reason_bits.append(
                    f"vol {volume_ratio:.1f}x < {self._min_vol_ratio:.1f}x"
                )
            return self._hold("; ".join(reason_bits) or "no setup")

        # Confidence ramps: base + ramps capped at 0.85
        confidence = 0.50
        if gap_pct >= 0.05:
            confidence += 0.10
        if gap_pct >= 0.08:
            confidence += 0.05
        if intraday_return >= 0.02:
            confidence += 0.10
        if volume_ratio >= 2.0:
            confidence += 0.10
        if today_close >= today_high * 0.98:   # closes near high (strong)
            confidence += 0.05
        confidence = min(0.85, confidence)

        return Signal(
            signal_type=SignalType.BUY,
            confidence=confidence,
            strategy_name=self.name,
            reason=(
                f"Gap {gap_pct:+.1%}, intraday {intraday_return:+.1%}, "
                f"vol {volume_ratio:.1f}x"
            ),
            suggested_price=today_close,
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
            "min_gap_pct": self._min_gap_pct,
            "min_vol_ratio": self._min_vol_ratio,
            "min_intraday_return": self._min_intraday_return,
            "max_gap_pct": self._max_gap_pct,
        }

    def set_params(self, params: dict) -> None:
        for key in (
            "min_gap_pct", "min_vol_ratio",
            "min_intraday_return", "max_gap_pct",
        ):
            if key in params:
                setattr(self, f"_{key}", params[key])
