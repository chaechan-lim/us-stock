"""Tests for BNF Deviation (Mean Reversion) Strategy."""

import pandas as pd

from core.enums import SignalType
from strategies.bnf_deviation import BNFDeviationStrategy


def _make_df(n: int = 35, close: float = 100.0, sma_25: float = 100.0,
             rsi: float = 50.0) -> pd.DataFrame:
    """Create DataFrame with deviation scenario."""
    data = {
        "open": [close * 0.999] * n,
        "high": [close * 1.01] * n,
        "low": [close * 0.99] * n,
        "close": [close] * n,
        "volume": [1_000_000.0] * n,
        "sma_25": [sma_25] * n,
        "rsi": [rsi] * n,
    }
    return pd.DataFrame(data)


class TestBNFDeviation:
    async def test_buy_oversold(self):
        """Price -6% below SMA should trigger buy (threshold -5%)."""
        strategy = BNFDeviationStrategy()
        df = _make_df(close=94.0, sma_25=100.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "mean reversion buy" in signal.reason.lower()

    async def test_sell_overbought(self):
        """Price +4% above SMA should trigger sell (threshold +3%)."""
        strategy = BNFDeviationStrategy()
        df = _make_df(close=104.0, sma_25=100.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL
        assert "mean reversion sell" in signal.reason.lower()

    async def test_hold_neutral(self):
        """Price near SMA should hold."""
        strategy = BNFDeviationStrategy()
        df = _make_df(close=101.0, sma_25=100.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_confidence_scales_with_deviation(self):
        strategy = BNFDeviationStrategy()
        df_mild = _make_df(close=94.0, sma_25=100.0)    # -6%
        df_deep = _make_df(close=88.0, sma_25=100.0)    # -12%
        s_mild = await strategy.analyze(df_mild, "AAPL")
        s_deep = await strategy.analyze(df_deep, "AAPL")
        assert s_deep.confidence > s_mild.confidence

    async def test_rsi_boost(self):
        """RSI below threshold should boost confidence."""
        strategy = BNFDeviationStrategy()
        df_no_rsi = _make_df(close=94.0, sma_25=100.0, rsi=50.0)
        df_rsi = _make_df(close=94.0, sma_25=100.0, rsi=25.0)
        s_no = await strategy.analyze(df_no_rsi, "AAPL")
        s_rsi = await strategy.analyze(df_rsi, "AAPL")
        assert s_rsi.confidence > s_no.confidence

    async def test_insufficient_data(self):
        strategy = BNFDeviationStrategy()
        df = pd.DataFrame({"close": [100.0] * 5})
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD

    async def test_sma_calculated_if_missing(self):
        """SMA should be calculated from raw prices if column missing."""
        strategy = BNFDeviationStrategy()
        closes = [100.0] * 35
        closes[-1] = 90.0  # -10% deviation
        df = pd.DataFrame({
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1_000_000.0] * 35,
        })
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY

    async def test_sell_confidence_scales(self):
        strategy = BNFDeviationStrategy()
        df_mild = _make_df(close=104.0, sma_25=100.0)   # +4%
        df_strong = _make_df(close=110.0, sma_25=100.0)  # +10%
        s_mild = await strategy.analyze(df_mild, "AAPL")
        s_strong = await strategy.analyze(df_strong, "AAPL")
        assert s_strong.confidence > s_mild.confidence

    async def test_indicators_present(self):
        strategy = BNFDeviationStrategy()
        df = _make_df(close=94.0, sma_25=100.0)
        signal = await strategy.analyze(df, "AAPL")
        assert "deviation_pct" in signal.indicators
        assert "sma" in signal.indicators
        assert signal.indicators["deviation_pct"] == -6.0

    async def test_get_set_params(self):
        strategy = BNFDeviationStrategy()
        params = strategy.get_params()
        assert params["sma_period"] == 25
        assert params["buy_deviation"] == -5.0
        assert params["sell_deviation"] == 3.0
        assert params["rsi_boost_threshold"] == 35.0

        strategy.set_params({"buy_deviation": -7.0, "sell_deviation": 5.0})
        assert strategy.get_params()["buy_deviation"] == -7.0
        assert strategy.get_params()["sell_deviation"] == 5.0

    async def test_custom_params(self):
        """Custom thresholds should work."""
        strategy = BNFDeviationStrategy(params={"buy_deviation": -3.0, "sell_deviation": 2.0})
        df = _make_df(close=96.5, sma_25=100.0)  # -3.5%
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY


# ---------------------------------------------------------------------------
# Trend filter tests (STOCK-68)
# ---------------------------------------------------------------------------

def _make_df_with_trend(close: float, sma_25: float, sma_200: float | None = None,
                        rsi: float = 50.0, n: int = 35) -> pd.DataFrame:
    """Create DataFrame with SMA-200 trend column."""
    df = _make_df(n=n, close=close, sma_25=sma_25, rsi=rsi)
    if sma_200 is not None:
        df["sma_200"] = sma_200
    return df


class TestTrendFilter:
    async def test_buy_suppressed_in_downtrend(self):
        """Oversold deviation blocked when price below SMA-200."""
        strategy = BNFDeviationStrategy()
        # price=94, SMA25=100 → dev=-6% (buy), but SMA200=105 → downtrend
        df = _make_df_with_trend(close=94.0, sma_25=100.0, sma_200=105.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD
        assert "downtrend" in signal.reason.lower()

    async def test_buy_allowed_in_uptrend(self):
        """Oversold deviation allowed when price above SMA-200."""
        strategy = BNFDeviationStrategy()
        # price=94, SMA25=100 → dev=-6%, SMA200=90 → uptrend
        df = _make_df_with_trend(close=94.0, sma_25=100.0, sma_200=90.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY

    async def test_sell_not_affected_by_trend_filter(self):
        """Sell signals should not be affected by trend filter."""
        strategy = BNFDeviationStrategy()
        # price=104, SMA25=100 → dev=+4% (sell), SMA200=110 → downtrend
        df = _make_df_with_trend(close=104.0, sma_25=100.0, sma_200=110.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.SELL

    async def test_trend_filter_disabled(self):
        """Trend filter can be disabled via params."""
        strategy = BNFDeviationStrategy(params={"trend_filter_enabled": False})
        # Would be suppressed with filter on
        df = _make_df_with_trend(close=94.0, sma_25=100.0, sma_200=105.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY

    async def test_no_sma200_data_allows_buy(self):
        """Without SMA-200 data, trend filter is neutral (allows buy)."""
        strategy = BNFDeviationStrategy()
        df = _make_df_with_trend(close=94.0, sma_25=100.0, sma_200=None)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY

    async def test_trend_bearish_in_indicators(self):
        """trend_bearish should be included in indicators dict."""
        strategy = BNFDeviationStrategy()
        # Buy signal in uptrend — indicators populated
        df = _make_df_with_trend(close=94.0, sma_25=100.0, sma_200=90.0)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.BUY
        assert "trend_bearish" in signal.indicators
        assert signal.indicators["trend_bearish"] is False

        # Sell signal in downtrend — indicators populated
        df_sell = _make_df_with_trend(close=104.0, sma_25=100.0, sma_200=110.0)
        signal_sell = await strategy.analyze(df_sell, "AAPL")
        assert signal_sell.signal_type == SignalType.SELL
        assert signal_sell.indicators["trend_bearish"] is True

    async def test_trend_sma_period_configurable(self):
        """Custom trend SMA period should work."""
        strategy = BNFDeviationStrategy(params={"trend_sma_period": 50})
        assert strategy.get_params()["trend_sma_period"] == 50

    async def test_get_set_params_includes_trend(self):
        """get_params and set_params should include trend filter params."""
        strategy = BNFDeviationStrategy()
        params = strategy.get_params()
        assert "trend_filter_enabled" in params
        assert "trend_sma_period" in params
        assert params["trend_filter_enabled"] is True
        assert params["trend_sma_period"] == 200

        strategy.set_params({"trend_filter_enabled": False, "trend_sma_period": 100})
        assert strategy.get_params()["trend_filter_enabled"] is False
        assert strategy.get_params()["trend_sma_period"] == 100
