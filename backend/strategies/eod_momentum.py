"""End-of-Day Momentum strategy.

Hypothesis: Stocks closing at or near the 20-day high with above-average
volume tend to continue for several days. Catches breakout-on-strength
patterns that the trend strategies confirm too late.

Unlike Gap-and-Go (failed 2026-06-01) this does NOT require a gap up —
it's a clean N-day high breakout with volume confirmation. Different
signal mechanism, intended for stocks already in uptrend posture.

Signal conditions (BUY):
  close >= n_day_high                    (breakout of N-day range, default 20)
  close >= today_high * close_top_pct    (close in top 30% of day's range)
  volume_ratio >= min_vol_ratio          (default 1.5x vs 20-day avg)
  prev_close > prev_n_day_high * 0.95    (not a one-day spike — prior bar
                                          already near range top)

Confidence ramps with strength + volume + breakout magnitude.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class EODMomentumStrategy(BaseStrategy):
    name = "eod_momentum"
    display_name = "End-of-Day Momentum"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 25

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._lookback_days: int = p.get("lookback_days", 20)
        self._min_vol_ratio: float = p.get("min_vol_ratio", 1.5)
        self._close_top_pct: float = p.get("close_top_pct", 0.70)  # close >= high * 0.70
        # How far the prev close should already be from the prior N-day high
        # — filters one-day spikes that just barely poke above range
        self._prev_proximity: float = p.get("prev_proximity", 0.95)
        # Breakout must clear range by at least this much (filter noise)
        self._breakout_buffer: float = p.get("breakout_buffer", 0.005)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        lookback = self._lookback_days
        today_idx = len(df) - 1
        prev_idx = today_idx - 1

        today = df.iloc[today_idx]
        prev = df.iloc[prev_idx]

        close = float(today["close"])
        today_high = float(today["high"])
        today_low = float(today["low"])
        today_volume = float(today["volume"])
        prev_close = float(prev["close"])

        # N-day high excluding today
        prior_window = df.iloc[max(0, today_idx - lookback):today_idx]
        n_day_high = float(prior_window["high"].max()) if len(prior_window) else 0.0

        # Prior N-day high excluding both prev and today (for prev_proximity)
        prior_window_2 = df.iloc[max(0, today_idx - lookback - 1):prev_idx]
        prior_high = (
            float(prior_window_2["high"].max()) if len(prior_window_2) else 0.0
        )

        # 20-day avg volume excluding today
        vol_window = df["volume"].iloc[max(0, today_idx - 20):today_idx]
        avg_volume = float(vol_window.mean()) if len(vol_window) else 0.0

        if any(v <= 0 for v in (close, prev_close, n_day_high, avg_volume, today_high)):
            return self._hold("Invalid prices/volume")

        today_range = today_high - today_low
        if today_range <= 0:
            return self._hold("Zero intraday range")

        close_position_in_range = (close - today_low) / today_range
        breakout_pct = (close - n_day_high) / n_day_high
        volume_ratio = today_volume / avg_volume
        prev_proximity_ratio = prev_close / prior_high if prior_high > 0 else 0.0

        indicators = {
            "n_day_high": n_day_high,
            "breakout_pct": breakout_pct,
            "close_position_in_range": close_position_in_range,
            "volume_ratio": volume_ratio,
            "prev_proximity_ratio": prev_proximity_ratio,
        }

        # Conditions
        breakout_qualifies = breakout_pct >= self._breakout_buffer
        close_strong = close_position_in_range >= self._close_top_pct
        volume_qualifies = volume_ratio >= self._min_vol_ratio
        not_one_day_spike = prev_proximity_ratio >= self._prev_proximity

        if not (breakout_qualifies and close_strong and volume_qualifies
                and not_one_day_spike):
            reason_bits = []
            if not breakout_qualifies:
                reason_bits.append(f"breakout {breakout_pct:+.2%} below buffer")
            if not close_strong:
                reason_bits.append(
                    f"close pos {close_position_in_range:.0%} < "
                    f"{self._close_top_pct:.0%}"
                )
            if not volume_qualifies:
                reason_bits.append(
                    f"vol {volume_ratio:.1f}x < {self._min_vol_ratio:.1f}x"
                )
            if not not_one_day_spike:
                reason_bits.append(
                    f"one-day spike (prev {prev_proximity_ratio:.0%} of prior high)"
                )
            return self._hold("; ".join(reason_bits) or "no setup")

        # Confidence ramps
        confidence = 0.50
        if breakout_pct >= 0.02:
            confidence += 0.10
        if breakout_pct >= 0.04:
            confidence += 0.05
        if close_position_in_range >= 0.90:
            confidence += 0.10
        if volume_ratio >= 2.0:
            confidence += 0.10
        if prev_proximity_ratio >= 0.98:
            confidence += 0.05
        confidence = min(0.85, confidence)

        return Signal(
            signal_type=SignalType.BUY,
            confidence=confidence,
            strategy_name=self.name,
            reason=(
                f"Breakout {breakout_pct:+.2%}, "
                f"close@{close_position_in_range:.0%}, "
                f"vol {volume_ratio:.1f}x"
            ),
            suggested_price=close,
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
            "lookback_days": self._lookback_days,
            "min_vol_ratio": self._min_vol_ratio,
            "close_top_pct": self._close_top_pct,
            "prev_proximity": self._prev_proximity,
            "breakout_buffer": self._breakout_buffer,
        }

    def set_params(self, params: dict) -> None:
        for key in (
            "lookback_days", "min_vol_ratio", "close_top_pct",
            "prev_proximity", "breakout_buffer",
        ):
            if key in params:
                setattr(self, f"_{key}", params[key])
