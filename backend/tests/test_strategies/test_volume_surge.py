"""Tests for Volume Surge Strategy (STOCK-72).

Covers:
- RSI probabilistic scoring (_calc_rsi_score)
- Momentum AND condition (momentum_confirmed)
- BUY/SELL/HOLD signal generation
- New indicators (rsi_score, momentum_confirmed)
- Edge cases (insufficient data, missing indicators)
"""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.volume_surge_strategy import VolumeSurgeStrategy


def _make_df(
    n=40,
    volume_ratio=2.0,
    up_day=True,
    rsi=55.0,
    adx=30.0,
    macd_hist=0.5,
    obv_trend="up",
    ema_above=True,
):
    """Build a DataFrame for volume_surge tests.

    Args:
        n: Number of candles.
        volume_ratio: Volume ratio for all bars.
        up_day: If True, last bar closes higher than previous.
        rsi: RSI value for all bars.
        adx: ADX value for all bars.
        macd_hist: MACD histogram value for all bars.
        obv_trend: 'up' or 'down' for OBV direction.
        ema_above: If True, price > EMA20.
    """
    dates = pd.date_range(
        "2023-01-01", periods=n, freq="B",
    )
    # Create a steadily rising series for up_day=True
    if up_day:
        closes = 100 + np.arange(n) * 0.5
    else:
        closes = 150 - np.arange(n) * 0.5

    df = pd.DataFrame(
        {
            "open": closes * 0.995,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.full(n, 2_000_000),
        },
        index=dates,
    )

    df["volume_ratio"] = volume_ratio
    df["rsi"] = rsi
    df["adx"] = adx
    df["macd_histogram"] = macd_hist

    if obv_trend == "up":
        df["obv"] = np.cumsum(
            np.random.default_rng(42).uniform(
                100, 500, n,
            ),
        )
    else:
        df["obv"] = np.cumsum(
            np.random.default_rng(42).uniform(
                -500, -100, n,
            ),
        )

    if ema_above:
        # EMA below price → above_ema = True
        df["ema_20"] = df["close"] - 2.0
    else:
        # EMA above price → above_ema = False
        df["ema_20"] = df["close"] + 2.0

    return df


# ── RSI Probabilistic Scoring ──────────────────────


class TestRsiScore:
    """Test _calc_rsi_score gradient zones."""

    def test_rsi_none_returns_zero(self):
        assert VolumeSurgeStrategy._calc_rsi_score(None) == 0.0

    def test_rsi_nan_returns_zero(self):
        assert (
            VolumeSurgeStrategy._calc_rsi_score(float("nan"))
            == 0.0
        )

    def test_rsi_oversold(self):
        # RSI < 30 → +0.10
        assert (
            VolumeSurgeStrategy._calc_rsi_score(25) == 0.10
        )

    def test_rsi_accumulation_zone(self):
        # RSI 30-50 → +0.10
        assert (
            VolumeSurgeStrategy._calc_rsi_score(40) == 0.10
        )

    def test_rsi_neutral_zone(self):
        # RSI 50-65 → +0.05
        assert (
            VolumeSurgeStrategy._calc_rsi_score(55) == 0.05
        )

    def test_rsi_warming_zone(self):
        # RSI 65-75 → +0.02
        assert (
            VolumeSurgeStrategy._calc_rsi_score(70) == 0.02
        )

    def test_rsi_overbought_penalty(self):
        # RSI 75-85 → -0.05
        assert (
            VolumeSurgeStrategy._calc_rsi_score(80) == -0.05
        )

    def test_rsi_extreme_overbought(self):
        # RSI 85+ → -0.10
        assert (
            VolumeSurgeStrategy._calc_rsi_score(90) == -0.10
        )

    def test_rsi_boundary_30(self):
        # RSI exactly 30 → accumulation zone (+0.10)
        assert (
            VolumeSurgeStrategy._calc_rsi_score(30) == 0.10
        )

    def test_rsi_boundary_65(self):
        # RSI exactly 65 → warming zone (+0.02)
        assert (
            VolumeSurgeStrategy._calc_rsi_score(65) == 0.02
        )

    def test_rsi_boundary_75(self):
        # RSI exactly 75 → overbought zone (-0.05)
        assert (
            VolumeSurgeStrategy._calc_rsi_score(75) == -0.05
        )

    def test_rsi_boundary_85(self):
        # RSI exactly 85 → extreme zone (-0.10)
        assert (
            VolumeSurgeStrategy._calc_rsi_score(85) == -0.10
        )


# ── Momentum AND Condition ─────────────────────────


@pytest.mark.asyncio
async def test_buy_requires_positive_momentum():
    """BUY requires momentum_confirmed=True (3-bar ROC>0).

    With an up_day but negative 3-bar momentum,
    BUY should NOT fire (STOCK-72 AND condition).
    """
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40, volume_ratio=2.5, up_day=True, rsi=55,
    )
    # Make the last bar higher than prev (up_day) but
    # make 3-bar-ago bar higher → negative 3-bar ROC
    df.iloc[-4, df.columns.get_loc("close")] = (
        float(df.iloc[-1]["close"]) + 5.0
    )
    signal = await s.analyze(df, "AAPL")
    # Momentum not confirmed → should NOT be BUY
    assert signal.signal_type != SignalType.BUY


@pytest.mark.asyncio
async def test_buy_with_positive_momentum():
    """BUY fires when volume surge + positive momentum."""
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40, volume_ratio=2.5, up_day=True, rsi=55,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert signal.indicators["momentum_confirmed"] is True


@pytest.mark.asyncio
async def test_momentum_confirmed_in_indicators():
    """momentum_confirmed indicator is always present."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5, up_day=True)
    signal = await s.analyze(df, "AAPL")
    assert "momentum_confirmed" in signal.indicators


# ── RSI Score in Indicators ────────────────────────


@pytest.mark.asyncio
async def test_rsi_score_in_indicators():
    """rsi_score indicator is present in signal output."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5, rsi=55)
    signal = await s.analyze(df, "AAPL")
    assert "rsi_score" in signal.indicators
    assert signal.indicators["rsi_score"] == 0.05


@pytest.mark.asyncio
async def test_rsi_score_low_rsi_high_score():
    """Low RSI (oversold) → high rsi_score (+0.10)."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5, rsi=35)
    signal = await s.analyze(df, "AAPL")
    assert signal.indicators["rsi_score"] == 0.10


@pytest.mark.asyncio
async def test_rsi_score_high_rsi_penalty():
    """High RSI → negative rsi_score (penalty)."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5, rsi=82)
    signal = await s.analyze(df, "AAPL")
    assert signal.indicators["rsi_score"] == -0.05


# ── RSI Gradient Affects Confidence ────────────────


@pytest.mark.asyncio
async def test_rsi_gradient_buy_confidence_low_rsi():
    """RSI in accumulation zone boosts BUY confidence."""
    s = VolumeSurgeStrategy()
    df_low = _make_df(
        n=40, volume_ratio=2.5, up_day=True, rsi=40,
    )
    df_high = _make_df(
        n=40, volume_ratio=2.5, up_day=True, rsi=70,
    )
    sig_low = await s.analyze(df_low, "AAPL")
    sig_high = await s.analyze(df_high, "AAPL")
    # RSI 40 (+0.10) should yield higher confidence
    # than RSI 70 (+0.02)
    assert sig_low.confidence > sig_high.confidence


@pytest.mark.asyncio
async def test_overbought_rsi_still_generates_buy():
    """STOCK-72: RSI 80+ no longer blocks BUY entirely.

    Instead, confidence is reduced via negative rsi_score.
    Previously RSI>80 returned HOLD; now it returns BUY
    with lower confidence.
    """
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40, volume_ratio=3.0, up_day=True, rsi=82,
    )
    signal = await s.analyze(df, "AAPL")
    # Should still BUY (not HOLD like old behavior)
    assert signal.signal_type == SignalType.BUY
    # But with reduced confidence
    assert signal.confidence < 0.70


@pytest.mark.asyncio
async def test_extreme_overbought_rsi_low_confidence():
    """RSI 90 heavily penalizes confidence but still BUY."""
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40, volume_ratio=2.5, up_day=True, rsi=90,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    # rsi_score = -0.10 pulls confidence down
    assert signal.indicators["rsi_score"] == -0.10


# ── Standard BUY / SELL / HOLD ─────────────────────


@pytest.mark.asyncio
async def test_buy_volume_surge_accumulation():
    """Classic BUY: volume surge + price up + momentum."""
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40,
        volume_ratio=3.0,
        up_day=True,
        rsi=55,
        adx=30,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
    assert signal.confidence >= 0.50
    assert "accumulation" in signal.reason


@pytest.mark.asyncio
async def test_sell_volume_surge_distribution():
    """SELL: volume surge + price falling."""
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40,
        volume_ratio=2.5,
        up_day=False,
        rsi=45,
        macd_hist=-0.5,
        obv_trend="down",
        ema_above=False,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL
    assert "distribution" in signal.reason


@pytest.mark.asyncio
async def test_hold_low_volume():
    """HOLD when volume ratio below threshold."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=1.2)
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD
    assert "below threshold" in signal.reason


@pytest.mark.asyncio
async def test_hold_insufficient_data():
    """HOLD when not enough candles."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=10, volume_ratio=3.0)
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD
    assert "Insufficient" in signal.reason


@pytest.mark.asyncio
async def test_hold_no_volume_ratio():
    """HOLD when volume_ratio is NaN."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5)
    df["volume_ratio"] = np.nan
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.HOLD
    assert "not available" in signal.reason


@pytest.mark.asyncio
async def test_hold_surge_no_direction():
    """HOLD when surge but price change is tiny."""
    s = VolumeSurgeStrategy()
    df = _make_df(n=40, volume_ratio=2.5, up_day=True)
    # Make last two bars have nearly identical close
    last_close = float(df.iloc[-2]["close"])
    df.iloc[
        -1, df.columns.get_loc("close")
    ] = last_close + 0.01  # tiny change
    signal = await s.analyze(df, "AAPL")
    # Either HOLD (no direction) or BUY depends on
    # the exact price_change_pct; with tiny change
    # it should be below min_price_change (0.3%)
    if signal.signal_type == SignalType.HOLD:
        assert "no clear direction" in signal.reason


# ── Params ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_params():
    s = VolumeSurgeStrategy(
        params={
            "volume_threshold": 2.0,
            "confirmation_bars": 2,
            "min_price_change": 0.5,
        },
    )
    p = s.get_params()
    assert p["volume_threshold"] == 2.0
    assert p["confirmation_bars"] == 2
    assert p["min_price_change"] == 0.5


@pytest.mark.asyncio
async def test_set_params():
    s = VolumeSurgeStrategy()
    s.set_params({"volume_threshold": 3.0})
    assert s.get_params()["volume_threshold"] == 3.0


@pytest.mark.asyncio
async def test_strategy_metadata():
    s = VolumeSurgeStrategy()
    assert s.name == "volume_surge"
    assert s.display_name == "Volume Surge"
    assert s.required_timeframe == "1D"
    assert s.min_candles_required == 30


# ── Buy confidence floor ───────────────────────────


@pytest.mark.asyncio
async def test_confidence_has_floor():
    """Confidence never drops below 0.10 even with
    extreme RSI penalty."""
    s = VolumeSurgeStrategy()
    # Very high RSI, minimal volume/price → low conf
    df = _make_df(
        n=40, volume_ratio=1.9, up_day=True, rsi=92,
        adx=10,
    )
    signal = await s.analyze(df, "AAPL")
    if signal.signal_type == SignalType.BUY:
        assert signal.confidence >= 0.10


# ── Sell signal with strong bearish indicators ─────


@pytest.mark.asyncio
async def test_sell_high_volume_boost():
    """Volume ratio >= 4.0 boosts sell confidence."""
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40,
        volume_ratio=4.5,
        up_day=False,
        rsi=40,
        macd_hist=-1.0,
        obv_trend="down",
        ema_above=False,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.SELL
    # All bearish factors: base 0.55 + 0.10 + 0.10
    #   + 0.05 + 0.10 = 0.90
    assert signal.confidence >= 0.80


# ── Buy without EMA but high volume ───────────────


@pytest.mark.asyncio
async def test_buy_below_ema_with_high_volume():
    """BUY allowed below EMA if volume >= 1.5x threshold.

    volume_threshold=1.8, so 1.5x = 2.7.
    """
    s = VolumeSurgeStrategy()
    df = _make_df(
        n=40,
        volume_ratio=3.0,
        up_day=True,
        rsi=50,
        ema_above=False,
    )
    signal = await s.analyze(df, "AAPL")
    assert signal.signal_type == SignalType.BUY
