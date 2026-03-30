"""Tests for Donchian Breakout Strategy.

Enhanced with ADX/volume/channel width, turtle exit,
and ADX trend exhaustion exit.
"""

import pandas as pd

from core.enums import SignalType
from strategies.donchian_breakout import (
    DonchianBreakoutStrategy,
)


def _make_df(
    n: int = 35,
    close: float = 110.0,
    donchian_upper: float = 105.0,
    donchian_lower: float = 90.0,
    prev_close: float = 104.0,
    prev_upper: float = 105.0,
    adx: float = 30.0,
    volume_ratio: float = 2.0,
    atr: float = 2.0,
    *,
    adx_series: list[float] | None = None,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
) -> pd.DataFrame:
    """Create DataFrame with breakout scenario.

    Args:
        adx_series: If provided, override ADX values for
            the last len(adx_series) bars. Useful for
            testing ADX declining detection.
        lows: If provided, override low values for
            historical bars (excluding prev and current).
        highs: If provided, override high values for
            historical bars (excluding prev and current).
    """
    data = []
    for i in range(n - 2):
        bar_low = 97.0
        bar_high = 103.0
        if lows and i < len(lows):
            bar_low = lows[i]
        if highs and i < len(highs):
            bar_high = highs[i]
        bar_adx = adx
        if adx_series:
            series_start = n - len(adx_series)
            if i >= series_start:
                idx = i - series_start
                bar_adx = adx_series[idx]
        data.append({
            "open": 100.0,
            "high": bar_high,
            "low": bar_low,
            "close": 100.0,
            "volume": 1_000_000.0,
            "donchian_upper": 105.0,
            "donchian_lower": 90.0,
            "donchian_mid": 97.5,
            "atr": atr,
            "adx": bar_adx,
            "volume_ratio": 1.2,
        })
    # Previous bar
    prev_adx = adx
    if adx_series and len(adx_series) >= 2:
        prev_adx = adx_series[-2]
    data.append({
        "open": 103.0,
        "high": 105.0,
        "low": 102.0,
        "close": prev_close,
        "volume": 1_200_000.0,
        "donchian_upper": prev_upper,
        "donchian_lower": 90.0,
        "donchian_mid": 97.5,
        "atr": atr,
        "adx": prev_adx,
        "volume_ratio": 1.5,
    })
    # Current bar
    cur_adx = adx
    if adx_series:
        cur_adx = adx_series[-1]
    data.append({
        "open": 105.0,
        "high": 112.0,
        "low": 104.0,
        "close": close,
        "volume": 2_000_000.0,
        "donchian_upper": donchian_upper,
        "donchian_lower": donchian_lower,
        "donchian_mid": 100.0,
        "atr": atr,
        "adx": cur_adx,
        "volume_ratio": volume_ratio,
    })
    return pd.DataFrame(data)


class TestDonchianBreakout:
    async def test_buy_on_breakout(self):
        strategy = DonchianBreakoutStrategy()
        df = _make_df(
            close=110.0,
            donchian_upper=105.0,
            prev_close=104.0,
            prev_upper=105.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "breakout" in signal.reason.lower()

    async def test_hold_no_breakout(self):
        strategy = DonchianBreakoutStrategy()
        df = _make_df(
            close=103.0, donchian_upper=105.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_sell_below_donchian_lower(self):
        strategy = DonchianBreakoutStrategy()
        df = _make_df(
            close=85.0,
            donchian_upper=105.0,
            donchian_lower=80.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "lower" in signal.reason.lower()

    async def test_sell_turtle_exit(self):
        """Price below exit-period low triggers turtle
        exit."""
        strategy = DonchianBreakoutStrategy()
        df = _make_df(
            close=91.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert (
            "turtle" in signal.reason.lower()
            or "exit" in signal.reason.lower()
        )

    async def test_insufficient_data(self):
        strategy = DonchianBreakoutStrategy()
        df = pd.DataFrame({"close": [100.0] * 5})
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_confidence_higher_with_adx(self):
        strategy = DonchianBreakoutStrategy()
        df_weak = _make_df(
            close=110.0, adx=15.0, volume_ratio=1.0
        )
        df_strong = _make_df(
            close=110.0, adx=35.0, volume_ratio=2.0
        )
        s_weak = await strategy.analyze(df_weak, "AAPL")
        s_strong = await strategy.analyze(
            df_strong, "AAPL"
        )
        assert s_strong.confidence > s_weak.confidence

    async def test_confidence_higher_with_wide_channel(
        self,
    ):
        strategy = DonchianBreakoutStrategy()
        df_narrow = _make_df(
            close=110.0,
            donchian_lower=104.0,
            adx=15.0,
            volume_ratio=1.0,
            atr=100.0,
        )
        df_wide = _make_df(
            close=110.0,
            donchian_lower=80.0,
            adx=15.0,
            volume_ratio=1.0,
            atr=100.0,
        )
        s_narrow = await strategy.analyze(
            df_narrow, "AAPL"
        )
        s_wide = await strategy.analyze(df_wide, "AAPL")
        assert s_wide.confidence > s_narrow.confidence

    async def test_channel_width_in_indicators(self):
        strategy = DonchianBreakoutStrategy()
        df = _make_df(close=110.0)
        signal = await strategy.analyze(df, "AAPL")
        assert "channel_width_pct" in signal.indicators

    async def test_get_set_params(self):
        strategy = DonchianBreakoutStrategy()
        params = strategy.get_params()
        assert params["entry_period"] == 20
        assert params["exit_period"] == 10
        assert params["adx_threshold"] == 25.0
        assert params["volume_multiplier"] == 1.5
        assert params["adx_lookback"] == 3

        strategy.set_params({
            "entry_period": 30,
            "exit_period": 15,
            "adx_threshold": 30.0,
        })
        assert strategy.get_params()["entry_period"] == 30
        assert (
            strategy.get_params()["adx_threshold"]
            == 30.0
        )

    async def test_custom_params(self):
        strategy = DonchianBreakoutStrategy(
            params={
                "adx_threshold": 20.0,
                "volume_multiplier": 1.2,
            }
        )
        assert (
            strategy.get_params()["adx_threshold"]
            == 20.0
        )
        assert (
            strategy.get_params()["volume_multiplier"]
            == 1.2
        )


class TestTurtleExit:
    """Tests for proper N-day trailing turtle exit."""

    async def test_turtle_exit_indicators_present(self):
        """turtle_exit indicator with exit_low,
        exit_high, period is always present."""
        strategy = DonchianBreakoutStrategy()
        df = _make_df(close=110.0)
        signal = await strategy.analyze(df, "AAPL")
        assert "turtle_exit" in signal.indicators
        te = signal.indicators["turtle_exit"]
        assert "exit_low" in te
        assert "exit_high" in te
        assert "period" in te
        assert te["period"] == 10

    async def test_turtle_exit_low_calculation(self):
        """exit_low is the minimum low over
        exit_period bars excluding current bar."""
        strategy = DonchianBreakoutStrategy(
            params={"exit_period": 5}
        )
        # Set specific lows for the last few hist bars
        # n=35, hist bars = 33 (indices 0..32)
        # exit_period=5, so window = iloc[-6:-1]
        # = bars at index 29..33 (prev bar index=33)
        # prev bar low = 102 (from _make_df)
        # bar indices 29..32 get custom lows
        lows = [97.0] * 28 + [95.0, 96.0, 98.0, 94.0]
        # last 4 hist + prev(102) = 5 bars window
        # min(95, 96, 98, 94, 102) = 94
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            lows=lows,
        )
        signal = await strategy.analyze(df, "AAPL")
        te = signal.indicators["turtle_exit"]
        assert te["exit_low"] == 94.0

    async def test_turtle_exit_high_calculation(self):
        """exit_high is the maximum high over
        exit_period bars excluding current bar."""
        strategy = DonchianBreakoutStrategy(
            params={"exit_period": 5}
        )
        highs = [103.0] * 28 + [
            108.0,
            106.0,
            107.0,
            109.0,
        ]
        # window: last 4 hist highs + prev(105) = 5 bars
        # max(108, 106, 107, 109, 105) = 109
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            highs=highs,
        )
        signal = await strategy.analyze(df, "AAPL")
        te = signal.indicators["turtle_exit"]
        assert te["exit_high"] == 109.0

    async def test_turtle_exit_sell_with_custom_lows(
        self,
    ):
        """Price below turtle exit_low triggers sell."""
        strategy = DonchianBreakoutStrategy(
            params={"exit_period": 5}
        )
        # All hist bars have low=99, prev has low=102
        # exit_low = min of last 5 bars excl current
        # = min(99, 99, 99, 99, 102) = 99
        # close=98 < 99 → turtle exit sell
        lows = [99.0] * 33
        df = _make_df(
            n=35,
            close=98.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            lows=lows,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "turtle" in signal.reason.lower()

    async def test_turtle_exit_period_in_params(self):
        """adx_lookback parameter round-trips."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_lookback": 5}
        )
        assert strategy.get_params()["adx_lookback"] == 5
        strategy.set_params({"adx_lookback": 2})
        assert strategy.get_params()["adx_lookback"] == 2


class TestADXTrendExhaustion:
    """Tests for ADX trend exhaustion exit signal."""

    async def test_adx_declining_indicator_present(self):
        """adx_declining indicator is always present."""
        strategy = DonchianBreakoutStrategy()
        df = _make_df(close=110.0)
        signal = await strategy.analyze(df, "AAPL")
        assert "adx_declining" in signal.indicators

    async def test_adx_declining_true(self):
        """ADX declining for 3 consecutive bars
        sets adx_declining=True."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_lookback": 3}
        )
        # Last 4 ADX values: 40, 38, 35, 32 (declining)
        # Need to fill all 35 bars
        adx_vals = [30.0] * 31 + [40.0, 38.0, 35.0, 32.0]
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.indicators["adx_declining"] is True

    async def test_adx_declining_false_when_rising(self):
        """ADX rising does not set adx_declining."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_lookback": 3}
        )
        adx_vals = [30.0] * 31 + [30.0, 32.0, 35.0, 38.0]
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert (
            signal.indicators["adx_declining"] is False
        )

    async def test_adx_declining_false_when_flat(self):
        """ADX flat (same value) does not trigger."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_lookback": 3}
        )
        adx_vals = [30.0] * 35
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert (
            signal.indicators["adx_declining"] is False
        )

    async def test_adx_exhaustion_sell_signal(self):
        """ADX declining + above threshold triggers
        sell with trend exhaustion reason."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_threshold": 25.0, "adx_lookback": 3}
        )
        # ADX declining from 40 → 38 → 35 → 32
        # All above threshold 25
        # Price 100.0 — no breakout, no donchian lower
        # break, no turtle exit (lows are 97)
        adx_vals = [30.0] * 31 + [40.0, 38.0, 35.0, 32.0]
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "adx" in signal.reason.lower()
        assert "exhaustion" in signal.reason.lower()
        assert signal.confidence == 0.50

    async def test_adx_exhaustion_no_sell_below_threshold(
        self,
    ):
        """ADX declining but below threshold does not
        trigger sell."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_threshold": 25.0, "adx_lookback": 3}
        )
        # ADX declining but below threshold: 20 → 18 → 15
        adx_vals = [20.0] * 31 + [24.0, 20.0, 18.0, 15.0]
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        # Should HOLD, not SELL (ADX below threshold)
        assert signal.signal_type == SignalType.HOLD

    async def test_adx_exhaustion_priority_after_turtle(
        self,
    ):
        """ADX exhaustion sell is checked after turtle
        exit (turtle has higher priority)."""
        strategy = DonchianBreakoutStrategy(
            params={"adx_threshold": 25.0, "adx_lookback": 3}
        )
        # Both conditions met: price < exit_low AND
        # ADX declining
        adx_vals = [30.0] * 31 + [40.0, 38.0, 35.0, 32.0]
        df = _make_df(
            n=35,
            close=91.0,  # below exit_low (97)
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        # Turtle exit takes priority
        assert "turtle" in signal.reason.lower()

    async def test_adx_nan_does_not_crash(self):
        """NaN in ADX series does not crash, returns
        adx_declining=False."""
        strategy = DonchianBreakoutStrategy()
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
        )
        # Inject NaN into ADX column
        df.loc[df.index[-2], "adx"] = float("nan")
        signal = await strategy.analyze(df, "AAPL")
        assert (
            signal.indicators["adx_declining"] is False
        )

    async def test_adx_lookback_custom(self):
        """Custom adx_lookback=2 detects shorter
        declining window."""
        strategy = DonchianBreakoutStrategy(
            params={
                "adx_threshold": 25.0,
                "adx_lookback": 2,
            }
        )
        # Only need 2 consecutive declines
        # (3 values: 40, 35, 30)
        adx_vals = [30.0] * 32 + [40.0, 35.0, 30.0]
        df = _make_df(
            n=35,
            close=100.0,
            donchian_upper=105.0,
            donchian_lower=85.0,
            adx_series=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "adx" in signal.reason.lower()
