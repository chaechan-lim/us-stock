"""Tests for Gap-and-Go strategy."""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.gap_and_go import GapAndGoStrategy


def _df(
    n: int = 30,
    gap_open_mult: float = 1.0,   # today.open vs prev.close
    intraday_mult: float = 1.0,   # today.close vs today.open
    vol_mult: float = 1.0,        # today.vol vs avg vol
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Construct a synthetic OHLCV with a deliberate gap setup at the end."""
    np.random.seed(7)
    returns = np.random.normal(0.0005, 0.005, n - 1)
    close = base_price * np.cumprod(1 + returns)
    close = np.append(close, close[-1])     # placeholder last bar; overridden below
    open_p = close * 0.999
    volume = np.random.randint(100_000, 200_000, n).astype(float)

    # Today's bar: deterministic
    prev_close = close[-2]
    today_open = prev_close * gap_open_mult
    today_close = today_open * intraday_mult
    open_p[-1] = today_open
    close[-1] = today_close
    volume[-1] = volume[-2] * vol_mult

    high = close * 1.005
    low = close * 0.995
    high[-1] = max(today_open, today_close) * 1.002
    low[-1] = min(today_open, today_close) * 0.998

    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low,
         "close": close, "volume": volume}
    )


class TestBuyConditions:
    async def test_buy_on_strong_gap(self):
        # 5% gap, +2% intraday, 2.5x volume → BUY
        s = GapAndGoStrategy()
        df = _df(gap_open_mult=1.05, intraday_mult=1.02, vol_mult=2.5)
        sig = await s.analyze(df, "TEST")
        assert sig.signal_type == SignalType.BUY
        assert 0.5 <= sig.confidence <= 0.85

    async def test_buy_confidence_scales_with_gap(self):
        s = GapAndGoStrategy()
        small_gap = await s.analyze(
            _df(gap_open_mult=1.035, intraday_mult=1.01, vol_mult=2.0), "T"
        )
        big_gap = await s.analyze(
            _df(gap_open_mult=1.08, intraday_mult=1.03, vol_mult=2.5), "T"
        )
        assert small_gap.signal_type == SignalType.BUY
        assert big_gap.signal_type == SignalType.BUY
        assert big_gap.confidence > small_gap.confidence


class TestHoldConditions:
    async def test_hold_when_gap_too_small(self):
        s = GapAndGoStrategy()
        # 1% gap < default 3% threshold
        sig = await s.analyze(_df(gap_open_mult=1.01, intraday_mult=1.01,
                                  vol_mult=2.0), "T")
        assert sig.signal_type == SignalType.HOLD
        assert "gap" in sig.reason.lower()

    async def test_hold_when_gap_too_big(self):
        s = GapAndGoStrategy()
        # 20% gap > default 15% cap (pump-and-dump filter)
        sig = await s.analyze(_df(gap_open_mult=1.20, intraday_mult=1.05,
                                  vol_mult=3.0), "T")
        assert sig.signal_type == SignalType.HOLD

    async def test_hold_when_gap_does_not_hold(self):
        s = GapAndGoStrategy()
        # 5% gap up but intraday closes -2% from open (gap fails to hold)
        sig = await s.analyze(_df(gap_open_mult=1.05, intraday_mult=0.98,
                                  vol_mult=2.0), "T")
        assert sig.signal_type == SignalType.HOLD
        assert "intraday" in sig.reason.lower()

    async def test_hold_when_volume_low(self):
        s = GapAndGoStrategy()
        # 5% gap + intraday OK but volume only 1.2x (default min 1.5x)
        sig = await s.analyze(_df(gap_open_mult=1.05, intraday_mult=1.02,
                                  vol_mult=1.2), "T")
        assert sig.signal_type == SignalType.HOLD
        assert "vol" in sig.reason.lower()

    async def test_hold_on_insufficient_data(self):
        s = GapAndGoStrategy()
        sig = await s.analyze(_df(n=10), "T")  # < 25 candles
        assert sig.signal_type == SignalType.HOLD
        assert "Insufficient" in sig.reason


class TestParams:
    def test_default_params(self):
        s = GapAndGoStrategy()
        p = s.get_params()
        assert p["min_gap_pct"] == 0.03
        assert p["max_gap_pct"] == 0.15
        assert p["min_intraday_return"] == 0.005
        assert p["min_vol_ratio"] == 1.5

    def test_set_params(self):
        s = GapAndGoStrategy()
        s.set_params({"min_gap_pct": 0.02, "min_vol_ratio": 2.0})
        p = s.get_params()
        assert p["min_gap_pct"] == 0.02
        assert p["min_vol_ratio"] == 2.0
        # Unchanged
        assert p["max_gap_pct"] == 0.15

    async def test_custom_threshold_lowers_bar(self):
        # Tight threshold should reject; loose threshold accepts
        df = _df(gap_open_mult=1.025, intraday_mult=1.01, vol_mult=2.0)
        s_default = GapAndGoStrategy()  # 3% threshold
        s_loose = GapAndGoStrategy({"min_gap_pct": 0.02})

        sig_default = await s_default.analyze(df, "T")
        sig_loose = await s_loose.analyze(df, "T")
        assert sig_default.signal_type == SignalType.HOLD
        assert sig_loose.signal_type == SignalType.BUY


class TestNeverSells:
    """Gap-and-Go is pure entry — SL/TP/trailing handle exits."""

    async def test_never_returns_sell(self):
        s = GapAndGoStrategy()
        # gap down (negative gap) → should not produce SELL
        df = _df(gap_open_mult=0.95, intraday_mult=0.98, vol_mult=2.0)
        sig = await s.analyze(df, "T")
        assert sig.signal_type != SignalType.SELL
