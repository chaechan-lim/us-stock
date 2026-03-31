"""Tests for Regime Switch Strategy."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.regime_switch import RegimeSwitchStrategy


def _make_df(n=250, trend="up"):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    if trend == "up":
        closes = 100 + np.arange(n) * 0.3
    elif trend == "down":
        closes = 200 - np.arange(n) * 0.3
    else:
        closes = 150 + np.random.uniform(-1, 1, n).cumsum() * 0.1
    df = pd.DataFrame({
        "open": closes * 0.99,
        "high": closes * 1.01,
        "low": closes * 0.98,
        "close": closes,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    df["sma_200"] = df["close"].rolling(200).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["adx"] = 30.0
    df["rsi"] = 55.0
    return df


@pytest.mark.asyncio
async def test_buy_bull_regime():
    s = RegimeSwitchStrategy(params={"confirmation_days": 2})
    df = _make_df(250, trend="up")
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert "Bull" in signal.reason


@pytest.mark.asyncio
async def test_sell_bear_regime():
    s = RegimeSwitchStrategy(params={"confirmation_days": 2})
    df = _make_df(250, trend="down")
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.SELL
    assert "Bear" in signal.reason


@pytest.mark.asyncio
async def test_hold_insufficient_data():
    s = RegimeSwitchStrategy()
    df = _make_df(20, trend="up")
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_hold_no_ma():
    s = RegimeSwitchStrategy()
    df = _make_df(60, trend="up")
    df.drop(columns=["sma_200", "sma_50"], inplace=True)
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_params():
    s = RegimeSwitchStrategy(params={"spy_sma_period": 100, "confirmation_days": 5})
    assert s.get_params()["spy_sma_period"] == 100
    assert s.get_params()["confirmation_days"] == 5
    s.set_params({"vix_bull_threshold": 18})
    assert s.get_params()["vix_bull_threshold"] == 18


@pytest.mark.asyncio
async def test_strategy_metadata():
    s = RegimeSwitchStrategy()
    assert s.name == "regime_switch"
    assert "all" in s.applicable_market_types


# ---------------------------------------------------------------------------
# VIX integration tests (STOCK-67)
# ---------------------------------------------------------------------------

def _make_df_with_vix(n: int = 250, trend: str = "up", vix: float | None = None) -> pd.DataFrame:
    """Build a DataFrame with optional VIX column."""
    df = _make_df(n=n, trend=trend)
    if vix is not None:
        df["vix"] = vix
    return df


@pytest.mark.asyncio
async def test_bull_low_vix_high_confidence():
    """Bull regime + VIX below bull threshold → confidence 0.65 base."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=15.0)
    df["adx"] = 20.0   # below 25 → no adx boost
    df["rsi"] = 75.0   # outside 40-70 → no rsi boost
    df["ema_20"] = df["close"] * 1.1  # price below ema_20 → no ema boost
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.65) < 1e-9
    assert "VIX=15.0" in signal.reason


@pytest.mark.asyncio
async def test_bull_moderate_vix_normal_confidence():
    """Bull regime + VIX between bull and bear thresholds → confidence 0.55 base."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=22.0)
    df["adx"] = 20.0
    df["rsi"] = 75.0
    df["ema_20"] = df["close"] * 1.1
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.55) < 1e-9


@pytest.mark.asyncio
async def test_bull_high_vix_entry_suppression():
    """Bull regime + VIX above bear threshold → confidence 0.40 (entry suppressed)."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=35.0)
    df["adx"] = 20.0
    df["rsi"] = 75.0
    df["ema_20"] = df["close"] * 1.1
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.40) < 1e-9


@pytest.mark.asyncio
async def test_bull_no_vix_fallback():
    """Bull regime with no VIX column → confidence 0.55 base (original behavior)."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=None)
    df["adx"] = 20.0
    df["rsi"] = 75.0
    df["ema_20"] = df["close"] * 1.1
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.55) < 1e-9
    assert "VIX=" not in signal.reason


@pytest.mark.asyncio
async def test_bull_nan_vix_fallback():
    """Bull regime with NaN VIX → treated as absent, confidence 0.55 base."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=None)
    df["vix"] = float("nan")
    df["adx"] = 20.0
    df["rsi"] = 75.0
    df["ema_20"] = df["close"] * 1.1
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.55) < 1e-9


@pytest.mark.asyncio
async def test_bear_high_vix_boosts_confidence():
    """Bear regime + high VIX → confidence boosted by 0.10."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="down", vix=30.0)
    df["rsi"] = 50.0  # above 40 → no rsi boost
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.SELL
    assert abs(signal.confidence - 0.65) < 1e-9


@pytest.mark.asyncio
async def test_bear_low_vix_no_boost():
    """Bear regime + low VIX → no VIX boost, base confidence 0.55."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="down", vix=18.0)
    df["rsi"] = 50.0
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.SELL
    assert abs(signal.confidence - 0.55) < 1e-9


@pytest.mark.asyncio
async def test_indicators_include_vix_and_regime():
    """indicators dict must include 'vix' and 'regime' keys."""
    s = RegimeSwitchStrategy(params={"confirmation_days": 2})

    # Bull with VIX
    df = _make_df_with_vix(250, trend="up", vix=16.0)
    signal = await s.analyze(df, "TQQQ")
    assert signal.indicators["vix"] == pytest.approx(16.0)
    assert signal.indicators["regime"] == "bull"

    # Bear with VIX
    df_bear = _make_df_with_vix(250, trend="down", vix=28.0)
    signal_bear = await s.analyze(df_bear, "SPY")
    assert signal_bear.indicators["regime"] == "bear"

    # No VIX → vix key is None
    df_no = _make_df_with_vix(250, trend="up", vix=None)
    signal_no = await s.analyze(df_no, "TQQQ")
    assert signal_no.indicators["vix"] is None


@pytest.mark.asyncio
async def test_bull_vix_at_bull_threshold():
    """VIX exactly at bull threshold → still treated as low VIX (<=)."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="up", vix=20.0)
    df["adx"] = 20.0
    df["rsi"] = 75.0
    df["ema_20"] = df["close"] * 1.1
    signal = await s.analyze(df, "TQQQ")
    assert signal.signal_type == SignalType.BUY
    assert abs(signal.confidence - 0.65) < 1e-9


@pytest.mark.asyncio
async def test_bear_vix_at_bear_threshold():
    """VIX exactly at bear threshold → confidence boost applied (>=)."""
    s = RegimeSwitchStrategy(params={
        "confirmation_days": 2,
        "vix_bull_threshold": 20,
        "vix_bear_threshold": 25,
    })
    df = _make_df_with_vix(250, trend="down", vix=25.0)
    df["rsi"] = 50.0
    signal = await s.analyze(df, "SPY")
    assert signal.signal_type == SignalType.SELL
    assert abs(signal.confidence - 0.65) < 1e-9
