"""Tests for Sector Rotation Strategy."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.sector_rotation import SectorRotationStrategy


def _make_df(n=100, trend="up"):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    if trend == "strong_up":
        closes = 100 + np.arange(n) * 1.0  # Strong upward
    elif trend == "up":
        closes = 100 + np.arange(n) * 0.3
    elif trend == "down":
        closes = 200 - np.arange(n) * 0.5
    elif trend == "mild_down":
        # Mild decline: enough for moderate sell, not severe
        closes = 150 - np.arange(n) * 0.15
    else:
        rng = np.random.default_rng(42)
        closes = (
            100 + rng.uniform(-0.5, 0.5, n).cumsum() * 0.1
        )
    df = pd.DataFrame(
        {
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.randint(
                1_000_000, 5_000_000, n,
            ),
        },
        index=dates,
    )
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = 55.0
    df["volume_ratio"] = 1.3
    return df


@pytest.mark.asyncio
async def test_buy_strong_sector():
    s = SectorRotationStrategy(
        params={"min_strength_score": 20},
    )
    df = _make_df(100, trend="strong_up")
    signal = await s.analyze(df, "XLK")
    assert signal.signal_type == SignalType.BUY
    assert "strength" in signal.reason.lower()


@pytest.mark.asyncio
async def test_sell_weak_sector():
    s = SectorRotationStrategy(
        params={"min_strength_score": 20},
    )
    df = _make_df(100, trend="down")
    signal = await s.analyze(df, "XLE")
    assert signal.signal_type == SignalType.SELL


@pytest.mark.asyncio
async def test_hold_insufficient_data():
    s = SectorRotationStrategy()
    df = _make_df(10)
    signal = await s.analyze(df, "XLK")
    assert signal.signal_type == SignalType.HOLD


@pytest.mark.asyncio
async def test_hold_below_threshold():
    """With an impossibly high threshold the signal must not
    be BUY.  Use a flat (noisy) series so vol-normalisation
    does not inflate the score."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 9999},
    )
    df = _make_df(100, trend="flat")
    signal = await s.analyze(df, "XLK")
    assert signal.signal_type in (
        SignalType.HOLD,
        SignalType.SELL,
    )


@pytest.mark.asyncio
async def test_params():
    s = SectorRotationStrategy(
        params={
            "lookback_weeks": 8,
            "min_strength_score": 50,
        },
    )
    assert s.get_params()["lookback_weeks"] == 8
    s.set_params({"min_strength_score": 70})
    assert s.get_params()["min_strength_score"] == 70


@pytest.mark.asyncio
async def test_strategy_metadata():
    s = SectorRotationStrategy()
    assert s.name == "sector_rotation"
    assert s.display_name == "Sector Rotation"
    assert "trending" in s.applicable_market_types


# ------------------------------------------------------------------
# Volatility normalization tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_volatility_normalized_score_in_indicators():
    """Signal indicators must contain the new vol fields."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 10},
    )
    df = _make_df(100, trend="strong_up")
    signal = await s.analyze(df, "XLK")
    ind = signal.indicators
    assert "volatility" in ind
    assert "volatility_normalized_score" in ind
    assert ind["volatility"] > 0, (
        "uptrend should have vol"
    )
    assert ind["volatility_normalized_score"] == ind[
        "strength_score"
    ]


@pytest.mark.asyncio
async def test_low_vol_gets_higher_normalized_score():
    """A low-vol series with same return should score higher
    than a high-vol series after normalization."""
    n = 100
    dates = pd.date_range(
        "2023-01-01", periods=n, freq="B",
    )

    # Low-vol: smooth uptrend
    low_vol_closes = pd.Series(
        100 + np.arange(n) * 0.5, index=dates,
    )
    # High-vol: same overall trend but noisy
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3, n).cumsum()
    noise -= noise[-1]  # same endpoint
    high_vol_closes = pd.Series(
        100 + np.arange(n) * 0.5 + noise, index=dates,
    )

    def _build(closes):
        df = pd.DataFrame(
            {
                "open": closes * 0.99,
                "high": closes * 1.01,
                "low": closes * 0.98,
                "close": closes,
                "volume": 2_000_000,
            },
            index=dates,
        )
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["rsi"] = 55.0
        df["volume_ratio"] = 1.0
        return df

    s = SectorRotationStrategy(
        params={"min_strength_score": 5},
    )
    sig_low = await s.analyze(
        _build(low_vol_closes), "LOW",
    )
    sig_high = await s.analyze(
        _build(high_vol_closes), "HIGH",
    )
    low_score = sig_low.indicators[
        "volatility_normalized_score"
    ]
    high_score = sig_high.indicators[
        "volatility_normalized_score"
    ]
    assert low_score > high_score, (
        "low-vol asset should score higher"
        " after normalisation"
    )


@pytest.mark.asyncio
async def test_compute_volatility_short_series():
    """_compute_volatility returns 0 for too-short series."""
    closes = pd.Series([100.0, 101.0, 102.0])
    vol = SectorRotationStrategy._compute_volatility(
        closes, window=20,
    )
    assert vol == 0.0


@pytest.mark.asyncio
async def test_normalize_by_vol_zero():
    """Zero or negative vol must return raw return."""
    assert SectorRotationStrategy._normalize_by_vol(
        0.05, 0.0,
    ) == 0.05
    assert SectorRotationStrategy._normalize_by_vol(
        0.05, -1.0,
    ) == 0.05


# ------------------------------------------------------------------
# Relaxed sell condition tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_severe_sell_has_high_confidence():
    """Strong downtrend triggers severe sell conf >= 0.50."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 20},
    )
    df = _make_df(100, trend="down")
    signal = await s.analyze(df, "XLE")
    assert signal.signal_type == SignalType.SELL
    assert signal.confidence >= 0.50
    assert signal.indicators.get("sell_reason") in (
        "severe_weakness",
        "moderate_weakness",
    )


@pytest.mark.asyncio
async def test_moderate_sell_has_lower_confidence():
    """Mild decline triggers moderate sell with reduced
    confidence."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 20},
    )
    df = _make_df(100, trend="mild_down")
    signal = await s.analyze(df, "XLE")
    # Should be either SELL with low conf or HOLD
    if signal.signal_type == SignalType.SELL:
        assert signal.confidence <= 0.45
        assert (
            signal.indicators["sell_reason"]
            == "moderate_weakness"
        )


@pytest.mark.asyncio
async def test_sell_reason_indicator_present():
    """sell_reason key must appear in indicators on SELL."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 20},
    )
    df = _make_df(100, trend="down")
    signal = await s.analyze(df, "XLE")
    assert signal.signal_type == SignalType.SELL
    assert "sell_reason" in signal.indicators


@pytest.mark.asyncio
async def test_hold_no_sell_reason():
    """HOLD signals should not contain sell_reason."""
    s = SectorRotationStrategy(
        params={"min_strength_score": 9999},
    )
    # flat series — not enough decline for sell
    df = _make_df(100, trend="flat")
    signal = await s.analyze(df, "XLK")
    if signal.signal_type == SignalType.HOLD:
        assert "sell_reason" not in signal.indicators


# ------------------------------------------------------------------
# Params round-trip with vol_window
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vol_window_param():
    s = SectorRotationStrategy(
        params={"vol_window": 30},
    )
    assert s.get_params()["vol_window"] == 30
    s.set_params({"vol_window": 10})
    assert s.get_params()["vol_window"] == 10
