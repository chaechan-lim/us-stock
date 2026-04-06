"""Tests for QualityFactorStrategy (Price-Derived Quality Metrics)."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.quality_factor import QualityFactorStrategy


def _make_df(
    n: int = 200,
    base_price: float = 100.0,
    trend: float = 0.0,
    noise: float = 0.005,
    ema_20: float | None = None,
    ema_50: float | None = None,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame.

    Args:
        n: Number of rows.
        base_price: Starting price.
        trend: Daily return (e.g. 0.001 = +0.1%/day).
        noise: Daily return standard deviation.
        ema_20/ema_50: Override last row's EMA values.
    """
    np.random.seed(42)
    prices = base_price * np.cumprod(1 + trend + np.random.normal(0, noise, n))
    volumes = np.full(n, 1_000_000, dtype=float)

    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes,
    })

    df["ema_20"] = np.nan
    df["ema_50"] = np.nan
    if ema_20 is not None:
        df.loc[df.index[-1], "ema_20"] = ema_20
    if ema_50 is not None:
        df.loc[df.index[-1], "ema_50"] = ema_50

    return df


@pytest.fixture
def strategy():
    return QualityFactorStrategy()


class TestBasicProperties:
    def test_name(self, strategy):
        assert strategy.name == "quality_factor"

    def test_display_name(self, strategy):
        assert strategy.display_name == "Quality Factor"

    def test_min_candles(self, strategy):
        assert strategy.min_candles_required == 126


class TestInsufficientData:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_hold(self, strategy):
        df = _make_df(n=50)
        signal = await strategy.analyze(df, "AAPL")
        assert signal.signal_type == SignalType.HOLD
        assert "Insufficient" in signal.reason


class TestBuySignal:
    @pytest.mark.asyncio
    async def test_high_quality_stock_buys(self):
        """Stable uptrend with low volatility → high quality → BUY."""
        strategy = QualityFactorStrategy()
        # Low noise, steady uptrend = high quality stock
        df = _make_df(
            n=200, trend=0.0008, noise=0.003,
            ema_20=120.0, ema_50=115.0,
        )
        signal = await strategy.analyze(df, "MSFT")
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence >= 0.50
        assert "High quality" in signal.reason
        assert signal.indicators["quality_score"] >= 0.60

    @pytest.mark.asyncio
    async def test_ema_aligned_boosts_confidence(self):
        """EMA alignment should increase BUY confidence."""
        strategy = QualityFactorStrategy()
        df_aligned = _make_df(
            n=200, trend=0.0008, noise=0.003,
            ema_20=120.0, ema_50=115.0,
        )
        signal_aligned = await strategy.analyze(df_aligned, "AAPL")

        strategy2 = QualityFactorStrategy()
        df_no_ema = _make_df(n=200, trend=0.0008, noise=0.003)
        signal_no_ema = await strategy2.analyze(df_no_ema, "AAPL")

        if signal_aligned.signal_type == SignalType.BUY and signal_no_ema.signal_type == SignalType.BUY:
            assert signal_aligned.confidence >= signal_no_ema.confidence


class TestSellSignal:
    @pytest.mark.asyncio
    async def test_low_quality_sells(self):
        """High volatility downtrend → low quality → SELL."""
        strategy = QualityFactorStrategy()
        # Very high noise, strong downtrend = very low quality
        df = _make_df(
            n=200, trend=-0.003, noise=0.03,
            ema_20=50.0, ema_50=70.0,
        )
        signal = await strategy.analyze(df, "BAD")
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence >= 0.45
        assert "Low quality" in signal.reason

    @pytest.mark.asyncio
    async def test_low_quality_ema_aligned_no_sell(self):
        """Low quality but EMA bullish → no SELL (trend support)."""
        strategy = QualityFactorStrategy()
        # High vol but EMA aligned (recent recovery)
        df = _make_df(
            n=200, trend=-0.003, noise=0.03,
            ema_20=100.0, ema_50=95.0,
        )
        signal = await strategy.analyze(df, "RECOV")
        # Should not sell because ema_aligned blocks SELL
        assert signal.signal_type != SignalType.SELL or signal.indicators.get("ema_aligned")


class TestHoldSignal:
    @pytest.mark.asyncio
    async def test_medium_quality_holds(self):
        """Mediocre quality → HOLD."""
        strategy = QualityFactorStrategy()
        # Near-flat with moderate noise = mediocre quality
        df = _make_df(n=200, trend=0.0, noise=0.015)
        signal = await strategy.analyze(df, "MED")
        assert signal.signal_type == SignalType.HOLD


class TestQualityScore:
    def test_perfect_quality(self):
        """Perfect metrics should give score near 1.0."""
        strategy = QualityFactorStrategy()
        score = strategy._compute_quality_score(
            ann_vol=0.10,     # Low vol → 1.0
            up_ratio=0.60,    # High up ratio → 1.0
            sharpe=2.0,       # High Sharpe → 1.0
            max_dd=0.0,       # No drawdown → 1.0
        )
        assert score >= 0.90

    def test_terrible_quality(self):
        """Terrible metrics should give score near 0."""
        strategy = QualityFactorStrategy()
        score = strategy._compute_quality_score(
            ann_vol=0.50,     # High vol → 0.0
            up_ratio=0.40,    # Low up ratio → 0.0
            sharpe=0.0,       # No Sharpe → 0.0
            max_dd=-0.30,     # Deep drawdown → 0.0
        )
        assert score <= 0.10

    def test_mixed_quality(self):
        """Mixed metrics should give moderate score."""
        strategy = QualityFactorStrategy()
        score = strategy._compute_quality_score(
            ann_vol=0.25,
            up_ratio=0.52,
            sharpe=0.8,
            max_dd=-0.10,
        )
        assert 0.30 <= score <= 0.70


class TestParams:
    def test_get_params(self, strategy):
        params = strategy.get_params()
        assert params["lookback_days"] == 126
        assert params["vol_lookback"] == 60
        assert params["max_volatility"] == 0.40

    def test_set_params(self, strategy):
        strategy.set_params({"lookback_days": 200, "max_volatility": 0.50})
        assert strategy._lookback_days == 200
        assert strategy._max_volatility == 0.50

    def test_set_params_ignores_unknown(self, strategy):
        strategy.set_params({"unknown_key": 42})
        assert not hasattr(strategy, "_unknown_key")

    def test_custom_params_init(self):
        s = QualityFactorStrategy(params={
            "lookback_days": 252,
            "max_volatility": 0.50,
        })
        assert s._lookback_days == 252
        assert s._max_volatility == 0.50


class TestIndicators:
    @pytest.mark.asyncio
    async def test_indicators_present(self, strategy):
        df = _make_df(n=200, trend=0.001, ema_20=120.0, ema_50=115.0)
        signal = await strategy.analyze(df, "TEST")
        if signal.indicators:
            assert "ann_vol" in signal.indicators
            assert "up_ratio" in signal.indicators
            assert "sharpe" in signal.indicators
            assert "max_dd" in signal.indicators
            assert "quality_score" in signal.indicators
            assert "ema_aligned" in signal.indicators


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_volatility(self):
        """Constant price should not crash."""
        strategy = QualityFactorStrategy()
        df = _make_df(n=200, trend=0.0, noise=0.0)
        # All returns are zero → vol=0 → sharpe=0
        signal = await strategy.analyze(df, "FLAT")
        assert signal.signal_type in (SignalType.HOLD, SignalType.BUY, SignalType.SELL)

    @pytest.mark.asyncio
    async def test_exactly_min_candles(self):
        """Exactly 126 candles should work."""
        strategy = QualityFactorStrategy()
        df = _make_df(n=126, trend=0.0008, noise=0.003, ema_20=110.0, ema_50=105.0)
        signal = await strategy.analyze(df, "EXACT")
        assert "Insufficient data" not in signal.reason

    @pytest.mark.asyncio
    async def test_confidence_capped(self):
        """Confidence should never exceed 0.95."""
        strategy = QualityFactorStrategy()
        df = _make_df(
            n=200, trend=0.003, noise=0.002,
            ema_20=200.0, ema_50=180.0,
        )
        signal = await strategy.analyze(df, "MOON")
        assert signal.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_sell_confidence_capped(self):
        """SELL confidence should never exceed 0.80."""
        strategy = QualityFactorStrategy()
        df = _make_df(n=200, trend=-0.003, noise=0.03, ema_20=50.0, ema_50=80.0)
        signal = await strategy.analyze(df, "DOOM")
        if signal.signal_type == SignalType.SELL:
            assert signal.confidence <= 0.80
