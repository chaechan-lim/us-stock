"""Tests for Supertrend Strategy."""

import pandas as pd

from core.enums import SignalType
from strategies.supertrend_strategy import SupertrendStrategy


def _make_df(
    n: int = 25,
    direction: float = 1.0,
    close: float = 150.0,
    supertrend: float = 140.0,
    adx: float = 30.0,
    rsi: float = 55.0,
    adx_values: list[float] | None = None,
) -> pd.DataFrame:
    data = []
    for i in range(n):
        adx_val = adx_values[i] if adx_values else adx
        data.append({
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": 1_000_000.0,
            "supertrend": supertrend,
            "supertrend_direction": direction,
            "adx": adx_val,
            "rsi": rsi,
        })
    return pd.DataFrame(data)


class TestSupertrend:
    async def test_buy_bullish_confirmed(self):
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0, close=150.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5

    async def test_sell_bearish(self):
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=-1.0, close=130.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL

    async def test_hold_not_confirmed(self):
        strategy = SupertrendStrategy(
            params={"confirmation_bars": 3}
        )
        data = []
        for i in range(25):
            d = -1.0 if i < 23 else 1.0
            data.append({
                "open": 149.0,
                "high": 152.0,
                "low": 148.0,
                "close": 150.0,
                "volume": 1_000_000.0,
                "supertrend": 140.0,
                "supertrend_direction": d,
                "adx": 30.0,
                "rsi": 55.0,
            })
        df = pd.DataFrame(data)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_hold_bullish_but_price_below_st(self):
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0, close=135.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_insufficient_data(self):
        strategy = SupertrendStrategy()
        df = pd.DataFrame({"close": [100.0] * 5})
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_get_set_params(self):
        strategy = SupertrendStrategy()
        params = strategy.get_params()
        assert params["confirmation_bars"] == 2
        assert params["rsi_overbought"] == 75
        assert params["adx_lookback"] == 3
        strategy.set_params({
            "confirmation_bars": 5,
            "rsi_overbought": 80,
            "adx_lookback": 5,
        })
        params = strategy.get_params()
        assert params["confirmation_bars"] == 5
        assert params["rsi_overbought"] == 80
        assert params["adx_lookback"] == 5


class TestEntryOffset:
    """G1 limit-at-line entry suggestion (2026-04-30)."""

    async def test_default_no_offset_uses_market_price(self):
        strategy = SupertrendStrategy()
        df = _make_df(direction=1.0, close=150.0, supertrend=140.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert signal.suggested_price == 150.0  # current close

    async def test_offset_set_close_above_target_returns_limit(self):
        # close 150, line 140 → line × 1.02 = 142.8 < 150 → use limit
        strategy = SupertrendStrategy(params={"entry_offset_pct": 0.02})
        df = _make_df(direction=1.0, close=150.0, supertrend=140.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert signal.suggested_price == 142.8  # 140 × 1.02

    async def test_offset_set_close_at_target_uses_market(self):
        # close 142, line 140 → limit 142.8, but 142.8 >= 142 means we
        # don't override. Use market (close).
        strategy = SupertrendStrategy(params={"entry_offset_pct": 0.02})
        df = _make_df(direction=1.0, close=142.0, supertrend=140.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.suggested_price == 142.0

    async def test_offset_zero_disables(self):
        strategy = SupertrendStrategy(params={"entry_offset_pct": None})
        df = _make_df(direction=1.0, close=150.0, supertrend=140.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.suggested_price == 150.0

    async def test_offset_reason_string_marks_limit(self):
        strategy = SupertrendStrategy(params={"entry_offset_pct": 0.02})
        df = _make_df(direction=1.0, close=150.0, supertrend=140.0)
        signal = await strategy.analyze(df, "AAPL")
        assert "limit @" in signal.reason


class TestDynamicSellConfidence:
    """Test dynamic sell confidence scaling."""

    async def test_sell_confidence_small_gap(self):
        """Small gap below supertrend = lower confidence."""
        strategy = SupertrendStrategy()
        # price=139, supertrend=140 => gap ~0.71%
        df = _make_df(
            direction=-1.0, close=139.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        # Small gap => confidence near base (0.55)
        assert 0.5 <= signal.confidence <= 0.7

    async def test_sell_confidence_large_gap(self):
        """Large gap below supertrend = higher confidence."""
        strategy = SupertrendStrategy()
        # price=129, supertrend=140 => gap ~7.86%
        df = _make_df(
            direction=-1.0, close=129.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence >= 0.75

    async def test_sell_confidence_scales_with_distance(
        self,
    ):
        """Larger gaps should produce higher confidence."""
        strategy = SupertrendStrategy()

        # Small gap: price=138, supertrend=140
        df_small = _make_df(
            direction=-1.0, close=138.0, supertrend=140.0
        )
        sig_small = await strategy.analyze(df_small, "AAPL")

        # Large gap: price=128, supertrend=140
        df_large = _make_df(
            direction=-1.0, close=128.0, supertrend=140.0
        )
        sig_large = await strategy.analyze(df_large, "AAPL")

        assert sig_large.confidence > sig_small.confidence

    async def test_sell_confidence_capped(self):
        """Sell confidence must not exceed 0.95."""
        strategy = SupertrendStrategy()
        # Extreme gap
        df = _make_df(
            direction=-1.0, close=100.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.confidence <= 0.95


class TestADXGradient:
    """Test ADX gradient calculations and adjustments."""

    async def test_adx_gradient_rising(self):
        """Rising ADX should produce positive gradient."""
        strategy = SupertrendStrategy()
        # ADX: 20 -> 22 -> 24 -> 26 -> ... (rising)
        adx_vals = [20.0 + i * 2 for i in range(25)]
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx_values=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.indicators["adx_gradient"] > 0

    async def test_adx_gradient_falling(self):
        """Falling ADX should produce negative gradient."""
        strategy = SupertrendStrategy()
        adx_vals = [40.0 - i * 1.5 for i in range(25)]
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx_values=adx_vals,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.indicators["adx_gradient"] < 0

    async def test_adx_gradient_flat(self):
        """Flat ADX should produce ~0 gradient."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx=30.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.indicators["adx_gradient"] == 0.0

    async def test_rising_adx_boosts_buy_confidence(self):
        """Rising ADX should boost buy confidence."""
        strategy = SupertrendStrategy()

        # Flat ADX
        df_flat = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx=30.0,
        )
        sig_flat = await strategy.analyze(df_flat, "AAPL")

        # Rising ADX (strong gradient > 0.5)
        adx_vals = [20.0 + i * 3 for i in range(25)]
        df_rising = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx_values=adx_vals,
            rsi=55.0,
        )
        sig_rising = await strategy.analyze(
            df_rising, "AAPL"
        )

        assert sig_rising.confidence >= sig_flat.confidence

    async def test_falling_adx_reduces_buy_confidence(
        self,
    ):
        """Falling ADX should reduce buy confidence."""
        strategy = SupertrendStrategy()

        # Flat ADX
        df_flat = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx=30.0,
        )
        sig_flat = await strategy.analyze(df_flat, "AAPL")

        # Falling ADX (strong gradient < -0.5)
        adx_vals = [50.0 - i * 3 for i in range(25)]
        df_falling = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            adx_values=adx_vals,
            rsi=55.0,
        )
        sig_falling = await strategy.analyze(
            df_falling, "AAPL"
        )

        assert sig_falling.confidence <= sig_flat.confidence

    async def test_rising_adx_boosts_sell_confidence(self):
        """Rising ADX in bearish trend = stronger sell."""
        strategy = SupertrendStrategy()

        # Flat ADX sell
        df_flat = _make_df(
            direction=-1.0,
            close=130.0,
            supertrend=140.0,
            adx=30.0,
        )
        sig_flat = await strategy.analyze(df_flat, "AAPL")

        # Rising ADX sell
        adx_vals = [20.0 + i * 3 for i in range(25)]
        df_rising = _make_df(
            direction=-1.0,
            close=130.0,
            supertrend=140.0,
            adx_values=adx_vals,
        )
        sig_rising = await strategy.analyze(
            df_rising, "AAPL"
        )

        assert sig_rising.confidence >= sig_flat.confidence

    async def test_adx_gradient_in_indicators(self):
        """adx_gradient should be in indicators dict."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0, close=150.0, supertrend=140.0
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "adx_gradient" in signal.indicators


class TestRSIOverboughtExit:
    """Test RSI overbought exit warning."""

    async def test_rsi_overbought_reduces_buy(self):
        """RSI > 75 should reduce buy confidence."""
        strategy = SupertrendStrategy()

        # Normal RSI
        df_normal = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=55.0,
        )
        sig_normal = await strategy.analyze(
            df_normal, "AAPL"
        )

        # Overbought RSI
        df_ob = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=80.0,
        )
        sig_ob = await strategy.analyze(df_ob, "AAPL")

        assert sig_ob.signal_type == SignalType.BUY
        assert sig_ob.confidence < sig_normal.confidence

    async def test_rsi_overbought_warning_in_reason(self):
        """RSI overbought should add warning to reason."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=80.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "RSI overbought warning" in signal.reason

    async def test_rsi_below_threshold_no_warning(self):
        """RSI below threshold should not add warning."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=60.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "RSI overbought" not in signal.reason

    async def test_rsi_overbought_in_indicators(self):
        """rsi_overbought should be in indicators dict."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=80.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert "rsi_overbought" in signal.indicators
        assert signal.indicators["rsi_overbought"] is True

    async def test_rsi_not_overbought_indicator_false(
        self,
    ):
        """rsi_overbought indicator should be False."""
        strategy = SupertrendStrategy()
        df = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=55.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert (
            signal.indicators["rsi_overbought"] is False
        )

    async def test_custom_rsi_overbought_threshold(self):
        """Custom RSI overbought threshold should work."""
        strategy = SupertrendStrategy(
            params={"rsi_overbought": 80}
        )

        # RSI=78: below custom threshold (80), no warning
        df_below = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=78.0,
        )
        sig_below = await strategy.analyze(
            df_below, "AAPL"
        )
        assert "RSI overbought" not in sig_below.reason

        # RSI=82: above custom threshold (80), warning
        df_above = _make_df(
            direction=1.0,
            close=150.0,
            supertrend=140.0,
            rsi=82.0,
        )
        sig_above = await strategy.analyze(
            df_above, "AAPL"
        )
        assert "RSI overbought" in sig_above.reason

    async def test_rsi_overbought_confidence_floor(self):
        """Confidence should not go below 0.1 after RSI
        overbought penalty."""
        strategy = SupertrendStrategy()
        # Low base confidence + overbought RSI
        df = _make_df(
            direction=1.0,
            close=141.0,
            supertrend=140.0,
            adx=15.0,
            rsi=80.0,
        )
        signal = await strategy.analyze(df, "AAPL")
        assert signal.confidence >= 0.1
