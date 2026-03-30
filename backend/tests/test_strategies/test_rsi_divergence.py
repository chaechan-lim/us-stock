"""Tests for RSI Divergence Strategy — pivot-based divergence detection."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.rsi_divergence import RSIDivergenceStrategy


def _make_df(n=60, rsi_values=None, prices=None):
    """Create DataFrame with RSI data."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    if prices is None:
        prices = 100 + np.random.uniform(-1, 1, n).cumsum()
    df = pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    if rsi_values is not None:
        df["rsi"] = rsi_values
    else:
        df["rsi"] = 50.0
    return df


def _make_bullish_divergence_df(n=60, pivot_order=3):
    """Create price/RSI with clear bullish divergence (lower price low, higher RSI low).

    W-bottom pattern with two distinct lows where price makes a lower low
    but RSI makes a higher low.
    """
    prices = np.zeros(n)
    rsi_vals = np.zeros(n)
    seg = n // 5

    # Seg 0: stable start
    prices[:seg] = np.linspace(110, 108, seg)
    rsi_vals[:seg] = np.linspace(55, 50, seg)
    # Seg 1: drop to first low (price=92, RSI=22)
    prices[seg:2*seg] = np.linspace(108, 92, seg)
    rsi_vals[seg:2*seg] = np.linspace(50, 22, seg)
    # Seg 2: bounce up
    prices[2*seg:3*seg] = np.linspace(92, 105, seg)
    rsi_vals[2*seg:3*seg] = np.linspace(22, 50, seg)
    # Seg 3: drop to LOWER low (price=87 < 92) but HIGHER RSI low (28 > 22)
    prices[3*seg:4*seg] = np.linspace(105, 87, seg)
    rsi_vals[3*seg:4*seg] = np.linspace(50, 28, seg)
    # Seg 4: bounce up from second low
    prices[4*seg:] = np.linspace(87, 95, n - 4*seg)
    rsi_vals[4*seg:] = np.linspace(28, 35, n - 4*seg)

    rsi_vals[-1] = 28.0
    return _make_df(n, rsi_values=rsi_vals, prices=prices)


def _make_bearish_divergence_df(n=60, pivot_order=3):
    """Create price/RSI with clear bearish divergence (higher price high, lower RSI high).

    M-top pattern with two distinct highs where price makes a higher high
    but RSI makes a lower high.
    """
    prices = np.zeros(n)
    rsi_vals = np.zeros(n)
    seg = n // 5

    # Seg 0: stable start
    prices[:seg] = np.linspace(95, 97, seg)
    rsi_vals[:seg] = np.linspace(45, 48, seg)
    # Seg 1: rise to first high (price=110, RSI=78)
    prices[seg:2*seg] = np.linspace(97, 110, seg)
    rsi_vals[seg:2*seg] = np.linspace(48, 78, seg)
    # Seg 2: pullback
    prices[2*seg:3*seg] = np.linspace(110, 100, seg)
    rsi_vals[2*seg:3*seg] = np.linspace(78, 50, seg)
    # Seg 3: rise to HIGHER high (price=114 > 110) but LOWER RSI high (72 < 78)
    prices[3*seg:4*seg] = np.linspace(100, 114, seg)
    rsi_vals[3*seg:4*seg] = np.linspace(50, 72, seg)
    # Seg 4: decline from second high
    prices[4*seg:] = np.linspace(114, 107, n - 4*seg)
    rsi_vals[4*seg:] = np.linspace(72, 65, n - 4*seg)

    rsi_vals[-1] = 72.0
    return _make_df(n, rsi_values=rsi_vals, prices=prices)


@pytest.mark.asyncio
async def test_oversold_buy():
    s = RSIDivergenceStrategy()
    df = _make_df(60)
    df["rsi"] = 25.0
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert "oversold" in signal.reason


@pytest.mark.asyncio
async def test_overbought_sell():
    s = RSIDivergenceStrategy()
    df = _make_df(60)
    df["rsi"] = 75.0
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL
    assert "overbought" in signal.reason


@pytest.mark.asyncio
async def test_hold_neutral_rsi():
    s = RSIDivergenceStrategy()
    df = _make_df(60)
    df["rsi"] = 50.0
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_hold_insufficient_data():
    s = RSIDivergenceStrategy()
    df = _make_df(10)
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_hold_no_rsi():
    s = RSIDivergenceStrategy()
    df = _make_df(60)
    df["rsi"] = np.nan
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_bullish_divergence():
    """Price lower pivot low but RSI higher pivot low → BUY."""
    s = RSIDivergenceStrategy(params={
        "min_price_move_pct": 1.0, "divergence_lookback": 40, "pivot_order": 3,
    })
    df = _make_bullish_divergence_df()
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert "Bullish divergence" in signal.reason or "divergence" in signal.reason.lower()
    assert signal.indicators["divergence_type"] == "bullish"


@pytest.mark.asyncio
async def test_bearish_divergence():
    """Price higher pivot high but RSI lower pivot high → SELL."""
    s = RSIDivergenceStrategy(params={
        "min_price_move_pct": 1.0, "divergence_lookback": 40, "pivot_order": 3,
    })
    df = _make_bearish_divergence_df()
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL
    assert "Bearish divergence" in signal.reason or "divergence" in signal.reason.lower()
    assert signal.indicators["divergence_type"] == "bearish"


@pytest.mark.asyncio
async def test_bullish_confidence_by_rsi_zone():
    """Bullish divergence confidence varies by RSI level."""
    s = RSIDivergenceStrategy(params={
        "min_price_move_pct": 1.0, "divergence_lookback": 40, "pivot_order": 3,
    })
    # Deep oversold
    df = _make_bullish_divergence_df()
    df.iloc[-1, df.columns.get_loc("rsi")] = 25.0  # below oversold (30)
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert signal.confidence == 0.70

    # Near oversold
    df2 = _make_bullish_divergence_df()
    df2.iloc[-1, df2.columns.get_loc("rsi")] = 35.0  # oversold+5
    signal2 = await s.analyze(df2, "AAPL")
    assert signal2.signal_type == SignalType.BUY
    assert signal2.confidence == 0.55


@pytest.mark.asyncio
async def test_bearish_confidence_by_rsi_zone():
    """Bearish divergence confidence varies by RSI level."""
    s = RSIDivergenceStrategy(params={
        "min_price_move_pct": 1.0, "divergence_lookback": 40, "pivot_order": 3,
    })
    # Deep overbought
    df = _make_bearish_divergence_df()
    df.iloc[-1, df.columns.get_loc("rsi")] = 75.0  # above overbought (70)
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL
    assert signal.confidence == 0.70


@pytest.mark.asyncio
async def test_indicators_include_pivots():
    """indicators dict must include pivot counts and divergence_type."""
    s = RSIDivergenceStrategy(params={
        "min_price_move_pct": 1.0, "divergence_lookback": 40, "pivot_order": 3,
    })
    df = _make_bullish_divergence_df()
    signal = await s.analyze(df, "AAPL")
    assert "pivot_lows" in signal.indicators
    assert "pivot_highs" in signal.indicators
    assert "divergence_type" in signal.indicators
    assert signal.indicators["pivot_lows"] >= 2


@pytest.mark.asyncio
async def test_params():
    s = RSIDivergenceStrategy(params={"overbought": 80, "oversold": 20, "pivot_order": 5})
    assert s.get_params()["overbought"] == 80
    assert s.get_params()["oversold"] == 20
    assert s.get_params()["pivot_order"] == 5
    s.set_params({"overbought": 75, "pivot_order": 4})
    assert s.get_params()["overbought"] == 75
    assert s.get_params()["pivot_order"] == 4


@pytest.mark.asyncio
async def test_no_divergence_neutral_rsi():
    """No pivot divergence and neutral RSI → HOLD."""
    s = RSIDivergenceStrategy()
    df = _make_df(60)
    df["rsi"] = 50.0  # Neutral RSI, no pivots
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD
    assert "No divergence" in signal.reason


@pytest.mark.asyncio
async def test_pivot_order_configurable():
    """Different pivot_order should change sensitivity."""
    # With order=2 (more sensitive), same data should find more pivots
    s2 = RSIDivergenceStrategy(params={"pivot_order": 2, "divergence_lookback": 40})
    s5 = RSIDivergenceStrategy(params={"pivot_order": 5, "divergence_lookback": 40})

    # Use data with clear but narrow pivots
    df = _make_bullish_divergence_df(n=60, pivot_order=2)
    sig2 = await s2.analyze(df, "AAPL")
    sig5 = await s5.analyze(df, "AAPL")

    # order=2 should detect pivots, order=5 might not (narrower window)
    assert sig2.indicators.get("pivot_lows", 0) >= sig5.indicators.get("pivot_lows", 0)
