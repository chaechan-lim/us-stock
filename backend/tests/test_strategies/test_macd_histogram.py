"""Tests for MACD Histogram Strategy."""

import pandas as pd

from core.enums import SignalType
from strategies.macd_histogram import MACDHistogramStrategy


def _make_df(
    n: int = 40,
    macd_hist_prev: float = -0.5,
    macd_hist_curr: float = 0.5,
    macd: float = 1.0,
    rsi: float = 55.0,
) -> pd.DataFrame:
    data = []
    for i in range(n - 2):
        data.append({
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "macd": 0.0,
            "macd_histogram": -1.0,
            "macd_signal": 0.0,
            "rsi": 50.0,
        })
    # Previous bar
    data.append({
        "open": 100.0,
        "high": 102.0,
        "low": 98.0,
        "close": 100.0,
        "volume": 1_000_000.0,
        "macd": macd - 0.2,
        "macd_histogram": macd_hist_prev,
        "macd_signal": 0.0,
        "rsi": rsi,
    })
    # Current bar
    data.append({
        "open": 101.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1_200_000.0,
        "macd": macd,
        "macd_histogram": macd_hist_curr,
        "macd_signal": 0.0,
        "rsi": rsi,
    })
    return pd.DataFrame(data)


def _make_divergence_df(
    divergence_type: str = "bullish",
    n: int = 40,
    lookback: int = 14,
) -> pd.DataFrame:
    """Build a DataFrame with a clear divergence pattern.

    For bullish: price makes lower lows, histogram makes
    higher lows across the lookback window.
    For bearish: price makes higher highs, histogram makes
    lower highs across the lookback window.
    """
    data = []
    # Fill prefix bars
    prefix_count = n - lookback
    for _ in range(prefix_count):
        data.append({
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "macd": 0.0,
            "macd_histogram": -1.0,
            "macd_signal": 0.0,
            "rsi": 50.0,
        })

    half = lookback // 2

    if divergence_type == "bullish":
        # First half: higher price, lower histogram
        for i in range(half):
            data.append({
                "open": 95.0,
                "high": 97.0,
                "low": 93.0,
                "close": 95.0,
                "volume": 1_000_000.0,
                "macd": -1.0,
                "macd_histogram": -3.0,
                "macd_signal": 0.0,
                "rsi": 35.0,
            })
        # Second half: lower price, higher histogram
        for i in range(lookback - half):
            data.append({
                "open": 91.0,
                "high": 93.0,
                "low": 89.0,
                "close": 91.0,
                "volume": 1_000_000.0,
                "macd": -0.5,
                "macd_histogram": -1.0,
                "macd_signal": 0.0,
                "rsi": 30.0,
            })
    elif divergence_type == "bearish":
        # First half: lower price, higher histogram
        for i in range(half):
            data.append({
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "macd": 1.0,
                "macd_histogram": 3.0,
                "macd_signal": 0.0,
                "rsi": 65.0,
            })
        # Second half: higher price, lower histogram
        for i in range(lookback - half):
            data.append({
                "open": 105.0,
                "high": 107.0,
                "low": 103.0,
                "close": 105.0,
                "volume": 1_000_000.0,
                "macd": 0.5,
                "macd_histogram": 1.0,
                "macd_signal": 0.0,
                "rsi": 60.0,
            })

    return pd.DataFrame(data)


class TestMACDHistogram:
    async def test_buy_histogram_crosses_up(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=-0.5, macd_hist_curr=0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "crossed above zero" in signal.reason

    async def test_sell_histogram_crosses_down(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=0.5, macd_hist_curr=-0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL

    async def test_buy_histogram_accelerating(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=0.5,
            macd_hist_curr=1.5,
            macd=2.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "accelerating" in signal.reason

    async def test_hold_no_crossover(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=1.0, macd_hist_curr=1.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_hold_negative_not_crossing(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=-1.0, macd_hist_curr=-0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_insufficient_data(self):
        strategy = MACDHistogramStrategy()
        df = pd.DataFrame({"close": [100.0] * 10})
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_get_set_params(self):
        strategy = MACDHistogramStrategy()
        p = strategy.get_params()
        assert p["min_histogram_change"] == 0.5
        assert p["divergence_lookback"] == 14
        assert p["min_price_move_pct"] == 2.0
        strategy.set_params({
            "min_histogram_change": 1.0,
            "divergence_lookback": 20,
            "min_price_move_pct": 3.0,
        })
        p2 = strategy.get_params()
        assert p2["min_histogram_change"] == 1.0
        assert p2["divergence_lookback"] == 20
        assert p2["min_price_move_pct"] == 3.0


class TestMACDDivergence:
    """Tests for MACD divergence detection."""

    async def test_bullish_divergence_detected(self):
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bullish")
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "Bullish MACD divergence" in signal.reason
        assert signal.indicators["divergence_type"] == (
            "bullish"
        )

    async def test_bearish_divergence_detected(self):
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bearish")
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "Bearish MACD divergence" in signal.reason
        assert signal.indicators["divergence_type"] == (
            "bearish"
        )

    async def test_no_divergence_flat_data(self):
        """No divergence when price and histogram flat."""
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=1.0, macd_hist_curr=1.0
        )
        result = strategy._detect_divergence(df)
        assert result == "none"

    async def test_divergence_type_in_indicators(self):
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=-0.5, macd_hist_curr=0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "divergence_type" in signal.indicators

    async def test_bullish_divergence_confidence_boost(
        self,
    ):
        """Bullish divergence should have boosted conf."""
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bullish")
        signal = await strategy.analyze(df, "AAPL")
        # Base cross confidence 0.60 + oversold 0.15
        # + divergence 0.10 = 0.85
        assert signal.confidence >= 0.70

    async def test_bearish_divergence_confidence(self):
        """Bearish divergence sell confidence boosted."""
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bearish")
        signal = await strategy.analyze(df, "AAPL")
        # Dynamic sell + divergence boost
        assert signal.confidence >= 0.60

    async def test_divergence_lookback_param(self):
        """Custom lookback parameter is respected."""
        strategy = MACDHistogramStrategy(
            {"divergence_lookback": 8}
        )
        assert strategy.get_params()["divergence_lookback"] == 8

    async def test_min_price_move_pct_param(self):
        """Custom min_price_move_pct is respected."""
        strategy = MACDHistogramStrategy(
            {"min_price_move_pct": 5.0}
        )
        assert (
            strategy.get_params()["min_price_move_pct"] == 5.0
        )

    async def test_no_divergence_insufficient_lookback(
        self,
    ):
        """Very short data should not detect divergence."""
        strategy = MACDHistogramStrategy()
        df = _make_df(n=36)
        signal = await strategy.analyze(df, "AAPL")
        # Should still work, just won't detect divergence
        assert "divergence_type" in signal.indicators


class TestDynamicSellConfidence:
    """Tests for dynamic sell confidence."""

    async def test_sell_confidence_varies_with_strength(
        self,
    ):
        """Sell confidence should differ based on
        histogram strength."""
        strategy = MACDHistogramStrategy()
        # Weak sell: small histogram drop
        df_weak = _make_df(
            macd_hist_prev=0.1, macd_hist_curr=-0.1
        )
        signal_weak = await strategy.analyze(
            df_weak, "AAPL"
        )

        # Strong sell: large histogram drop
        df_strong = _make_df(
            macd_hist_prev=0.1, macd_hist_curr=-3.0
        )
        signal_strong = await strategy.analyze(
            df_strong, "AAPL"
        )

        assert signal_weak.signal_type == SignalType.SELL
        assert signal_strong.signal_type == SignalType.SELL
        assert (
            signal_strong.confidence
            >= signal_weak.confidence
        )

    async def test_sell_confidence_range(self):
        """Dynamic sell confidence should be in
        expected range [0.50, 0.80]."""
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=0.5, macd_hist_curr=-0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert 0.50 <= signal.confidence <= 0.80

    async def test_histogram_strength_in_indicators(self):
        """histogram_strength should always appear in
        indicators."""
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=-0.5, macd_hist_curr=0.5
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "histogram_strength" in signal.indicators
        strength = signal.indicators["histogram_strength"]
        assert 0.0 <= strength <= 1.0

    async def test_histogram_strength_zero_range(self):
        """Strength should be 0 when all histograms are 0."""
        strategy = MACDHistogramStrategy()
        data = []
        for _ in range(40):
            data.append({
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "macd": 0.0,
                "macd_histogram": 0.0,
                "macd_signal": 0.0,
                "rsi": 50.0,
            })
        df = pd.DataFrame(data)
        strength = strategy._histogram_strength(df)
        assert strength == 0.0

    async def test_dynamic_sell_confidence_method(self):
        """Unit test for _dynamic_sell_confidence method."""
        strategy = MACDHistogramStrategy()
        # Minimum strength
        assert strategy._dynamic_sell_confidence(0.0) == 0.50
        # Maximum strength
        assert strategy._dynamic_sell_confidence(1.0) == 0.80
        # Mid strength
        conf = strategy._dynamic_sell_confidence(0.5)
        assert 0.64 <= conf <= 0.66  # 0.50 + 0.15 = 0.65

    async def test_histogram_strength_method(self):
        """Unit test for _histogram_strength method."""
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=-2.0, macd_hist_curr=1.0
        )
        strength = strategy._histogram_strength(df)
        # Current |1.0|, max of recent abs values >= 2.0
        assert 0.0 < strength <= 1.0

    async def test_detect_divergence_none(self):
        """Unit test: no divergence on flat data."""
        strategy = MACDHistogramStrategy()
        df = _make_df(
            macd_hist_prev=1.0, macd_hist_curr=1.0
        )
        result = strategy._detect_divergence(df)
        assert result == "none"

    async def test_detect_divergence_bullish(self):
        """Unit test: bullish divergence detected."""
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bullish")
        result = strategy._detect_divergence(df)
        assert result == "bullish"

    async def test_detect_divergence_bearish(self):
        """Unit test: bearish divergence detected."""
        strategy = MACDHistogramStrategy()
        df = _make_divergence_df("bearish")
        result = strategy._detect_divergence(df)
        assert result == "bearish"

    async def test_divergence_short_df(self):
        """Divergence on very short df returns 'none'."""
        strategy = MACDHistogramStrategy(
            {"divergence_lookback": 100}
        )
        df = _make_df(n=36)
        result = strategy._detect_divergence(df)
        # lookback capped to len(df)-1=35, half=17
        # No divergence on flat data
        assert result == "none"

    async def test_histogram_strength_nan_handling(self):
        """Strength handles NaN gracefully."""
        strategy = MACDHistogramStrategy()
        data = []
        for _ in range(40):
            data.append({
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "macd": 0.0,
                "macd_histogram": float("nan"),
                "macd_signal": 0.0,
                "rsi": 50.0,
            })
        df = pd.DataFrame(data)
        strength = strategy._histogram_strength(df)
        assert strength == 0.0
