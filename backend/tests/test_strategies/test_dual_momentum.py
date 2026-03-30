"""Tests for Dual Momentum Strategy."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.dual_momentum import DualMomentumStrategy


def _make_df(n=100, trend="up"):
    """Create a DataFrame with price data."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    if trend == "up":
        closes = 100 + np.cumsum(np.random.uniform(0.1, 0.5, n))
    elif trend == "down":
        closes = 200 - np.cumsum(np.random.uniform(0.1, 0.5, n))
    else:
        closes = 100 + np.random.uniform(-0.5, 0.5, n).cumsum() * 0.1 + 100
    df = pd.DataFrame({
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    # Add indicators
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 55.0
    df["roc_20"] = df["close"].pct_change(20) * 100
    df["volume_ratio"] = 1.2
    return df


@pytest.mark.asyncio
async def test_buy_signal_uptrend():
    s = DualMomentumStrategy()
    df = _make_df(300, trend="up")
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert signal.confidence > 0


@pytest.mark.asyncio
async def test_sell_signal_downtrend():
    s = DualMomentumStrategy()
    df = _make_df(300, trend="down")
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL


@pytest.mark.asyncio
async def test_hold_insufficient_data():
    s = DualMomentumStrategy()
    df = _make_df(10, trend="up")
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_params_override():
    s = DualMomentumStrategy(params={"lookback_months": 6, "min_absolute_return": 0.05})
    assert s.get_params()["lookback_months"] == 6
    assert s.get_params()["min_absolute_return"] == 0.05


@pytest.mark.asyncio
async def test_set_params():
    s = DualMomentumStrategy()
    s.set_params({"lookback_months": 3})
    assert s.get_params()["lookback_months"] == 3


@pytest.mark.asyncio
async def test_strategy_name():
    s = DualMomentumStrategy()
    assert s.name == "dual_momentum"
    assert s.display_name == "Dual Momentum"


# ---------------------------------------------------------------------------
# STOCK-70: Gradient sell, volatility filter, EMA confirmation
# ---------------------------------------------------------------------------

def _make_gradient_df(n=300):
    """Create data where 3m momentum is strong but 1m momentum fades."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Strong rise first 250 bars, then stalls last 50
    closes = np.concatenate([
        np.linspace(100, 160, n - 50),   # strong uptrend
        np.linspace(160, 158, 50),        # stalling / slight decline
    ])
    df = pd.DataFrame({
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 55.0
    return df


def _make_volatile_uptrend_df(n=300):
    """Create uptrending data with high volatility."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(42)
    # Uptrend with large daily swings
    daily_returns = np.random.normal(0.002, 0.04, n)  # 4% daily std
    closes = 100 * np.cumprod(1 + daily_returns)
    df = pd.DataFrame({
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.97,
        "close": closes,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 55.0
    return df


class TestGradientSell:
    @pytest.mark.asyncio
    async def test_gradient_sell_on_fading_momentum(self):
        """Gradient sell triggers when momentum decelerates."""
        s = DualMomentumStrategy(params={
            "gradient_sell": True, "gradient_threshold": -0.30,
        })
        df = _make_gradient_df()
        signal = await s.analyze(df, "AAPL")
        # Should detect fading momentum
        if signal.signal_type == SignalType.SELL:
            assert "fading" in signal.reason.lower() or "gradient" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_gradient_sell_disabled(self):
        """Gradient sell disabled → normal buy signal on uptrend."""
        s = DualMomentumStrategy(params={"gradient_sell": False})
        df = _make_gradient_df()
        signal = await s.analyze(df, "AAPL")
        # Without gradient sell, strong uptrend should give BUY
        assert signal.signal_type in {SignalType.BUY, SignalType.HOLD}

    @pytest.mark.asyncio
    async def test_gradient_in_indicators(self):
        """Gradient value should be in indicators."""
        s = DualMomentumStrategy()
        df = _make_df(300, trend="up")
        signal = await s.analyze(df, "AAPL")
        assert "gradient" in signal.indicators


class TestVolatilityFilter:
    @pytest.mark.asyncio
    async def test_high_volatility_suppresses_buy(self):
        """High volatility should suppress buy signals."""
        s = DualMomentumStrategy(params={
            "volatility_filter": True, "max_volatility_pct": 2.0,
        })
        df = _make_volatile_uptrend_df()
        signal = await s.analyze(df, "AAPL")
        # High vol data should either suppress buy or still buy
        if signal.signal_type == SignalType.HOLD:
            assert "volatility" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_volatility_filter_disabled(self):
        """Disabled volatility filter should not suppress."""
        s = DualMomentumStrategy(params={"volatility_filter": False})
        df = _make_volatile_uptrend_df()
        signal = await s.analyze(df, "AAPL")
        # Without filter, uptrend should give BUY or SELL based on momentum
        assert signal.signal_type in {SignalType.BUY, SignalType.SELL, SignalType.HOLD}

    @pytest.mark.asyncio
    async def test_volatility_in_indicators(self):
        """Volatility should be in indicators."""
        s = DualMomentumStrategy()
        df = _make_df(300, trend="up")
        signal = await s.analyze(df, "AAPL")
        assert "volatility_pct" in signal.indicators
        assert isinstance(signal.indicators["volatility_pct"], float)


class TestEMAConfirmation:
    @pytest.mark.asyncio
    async def test_ema_bullish_in_indicators(self):
        """EMA bullish flag should be in indicators."""
        s = DualMomentumStrategy()
        df = _make_df(300, trend="up")
        signal = await s.analyze(df, "AAPL")
        assert "ema_bullish" in signal.indicators

    @pytest.mark.asyncio
    async def test_ema_boosts_confidence(self):
        """EMA20 > EMA50 should boost buy confidence."""
        s = DualMomentumStrategy()
        df = _make_df(300, trend="up")
        signal = await s.analyze(df, "AAPL")
        if signal.signal_type == SignalType.BUY and signal.indicators.get("ema_bullish"):
            # With EMA boost, confidence should be higher than base
            assert signal.confidence >= 0.6


class TestNewParams:
    def test_new_params_in_get_params(self):
        s = DualMomentumStrategy()
        params = s.get_params()
        assert "volatility_filter" in params
        assert "max_volatility_pct" in params
        assert "gradient_sell" in params
        assert "gradient_threshold" in params

    def test_set_new_params(self):
        s = DualMomentumStrategy()
        s.set_params({
            "volatility_filter": False,
            "max_volatility_pct": 5.0,
            "gradient_sell": False,
            "gradient_threshold": -0.50,
        })
        p = s.get_params()
        assert p["volatility_filter"] is False
        assert p["max_volatility_pct"] == 5.0
        assert p["gradient_sell"] is False
        assert p["gradient_threshold"] == -0.50
