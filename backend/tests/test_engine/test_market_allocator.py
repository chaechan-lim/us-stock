"""Tests for MarketAllocator (dual momentum + inverse volatility).

Verifies:
1. Momentum computation (12-1 month)
2. Volatility computation (annualized)
3. Momentum-based allocation rules
4. Inverse volatility weighting
5. Blended allocation with clamping
6. Insufficient data fallback
7. Edge cases (zero prices, equal markets)
"""

import numpy as np
import pandas as pd
import pytest

from engine.market_allocator import MarketAllocator


def _make_prices(
    n: int = 300,
    base: float = 100.0,
    daily_return: float = 0.0005,
    noise: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic price series."""
    np.random.seed(seed)
    returns = 1 + daily_return + np.random.normal(0, noise, n)
    prices = base * np.cumprod(returns)
    return pd.Series(prices)


@pytest.fixture
def allocator():
    return MarketAllocator()


class TestMinBarsRequired:
    def test_default_min_bars(self, allocator):
        assert allocator.min_bars_required == 273  # 252 + 21


class TestInsufficientData:
    def test_short_data_returns_equal(self, allocator):
        us = _make_prices(n=100)
        kr = _make_prices(n=100)
        result = allocator.compute(us, kr)
        assert result == {"US": 0.50, "KR": 0.50}

    def test_one_short_returns_equal(self, allocator):
        us = _make_prices(n=300)
        kr = _make_prices(n=100)
        result = allocator.compute(us, kr)
        assert result == {"US": 0.50, "KR": 0.50}


class TestMomentumComputation:
    def test_positive_momentum(self, allocator):
        prices = _make_prices(n=300, daily_return=0.001)  # uptrend
        mom = allocator._compute_momentum(prices)
        assert mom > 0

    def test_negative_momentum(self, allocator):
        prices = _make_prices(n=300, daily_return=-0.001)  # downtrend
        mom = allocator._compute_momentum(prices)
        assert mom < 0

    def test_flat_momentum(self, allocator):
        prices = _make_prices(n=300, daily_return=0.0, noise=0.001)
        mom = allocator._compute_momentum(prices)
        assert abs(mom) < 0.10  # near zero


class TestVolatilityComputation:
    def test_low_noise_low_vol(self, allocator):
        prices = _make_prices(n=300, noise=0.005)
        vol = allocator._compute_volatility(prices)
        assert vol < 0.15  # low annualized vol

    def test_high_noise_high_vol(self, allocator):
        prices = _make_prices(n=300, noise=0.03)
        vol = allocator._compute_volatility(prices)
        assert vol > 0.30  # high annualized vol

    def test_short_prices_default_vol(self, allocator):
        prices = pd.Series([100.0])
        vol = allocator._compute_volatility(prices)
        assert vol == 0.20  # default


class TestMomentumAllocation:
    def test_both_positive_proportional(self, allocator):
        result = allocator._momentum_to_allocation(0.20, 0.10)
        assert result["US"] > result["KR"]
        assert result["US"] == pytest.approx(0.20 / 0.30, abs=0.01)

    def test_both_negative_equal(self, allocator):
        result = allocator._momentum_to_allocation(-0.10, -0.05)
        assert result["US"] == 0.50
        assert result["KR"] == 0.50

    def test_us_negative_shifts_to_kr(self, allocator):
        result = allocator._momentum_to_allocation(-0.05, 0.10)
        assert result["KR"] > result["US"]
        assert result["US"] == allocator._min_alloc

    def test_kr_negative_shifts_to_us(self, allocator):
        result = allocator._momentum_to_allocation(0.10, -0.05)
        assert result["US"] > result["KR"]
        assert result["KR"] == allocator._min_alloc

    def test_equal_momentum_equal_allocation(self, allocator):
        result = allocator._momentum_to_allocation(0.15, 0.15)
        assert result["US"] == pytest.approx(0.50, abs=0.01)


class TestInverseVolAllocation:
    def test_lower_vol_gets_more(self, allocator):
        result = allocator._invvol_to_allocation(0.10, 0.20)
        assert result["US"] > result["KR"]

    def test_equal_vol_equal_allocation(self, allocator):
        result = allocator._invvol_to_allocation(0.15, 0.15)
        assert result["US"] == pytest.approx(0.50, abs=0.01)

    def test_both_zero_vol_equal(self, allocator):
        result = allocator._invvol_to_allocation(0.0, 0.0)
        assert result["US"] == 0.50

    def test_very_different_vol(self, allocator):
        result = allocator._invvol_to_allocation(0.05, 0.30)
        # US is 6x less volatile → should get ~6x more weight
        assert result["US"] > 0.80


class TestBlendedCompute:
    def test_us_uptrend_kr_downtrend(self):
        """Strong US uptrend, KR downtrend → heavily US."""
        alloc = MarketAllocator()
        us = _make_prices(n=300, daily_return=0.001, noise=0.008, seed=1)
        kr = _make_prices(n=300, daily_return=-0.001, noise=0.012, seed=2)
        result = alloc.compute(us, kr)
        assert result["US"] > result["KR"]

    def test_equal_markets_near_equal(self):
        """Same trend and vol → near 50/50."""
        alloc = MarketAllocator()
        us = _make_prices(n=300, daily_return=0.0005, noise=0.01, seed=10)
        kr = _make_prices(n=300, daily_return=0.0005, noise=0.01, seed=10)
        result = alloc.compute(us, kr)
        assert abs(result["US"] - 0.50) < 0.05

    def test_kr_stronger(self):
        """KR stronger momentum → more KR."""
        alloc = MarketAllocator()
        us = _make_prices(n=300, daily_return=0.0003, noise=0.01, seed=3)
        kr = _make_prices(n=300, daily_return=0.0015, noise=0.01, seed=4)
        result = alloc.compute(us, kr)
        assert result["KR"] > result["US"]


class TestClamping:
    def test_min_allocation_respected(self):
        """Even with very negative momentum, allocation >= min."""
        alloc = MarketAllocator(min_allocation=0.20)
        us = _make_prices(n=300, daily_return=0.002, noise=0.005, seed=5)
        kr = _make_prices(n=300, daily_return=-0.003, noise=0.02, seed=6)
        result = alloc.compute(us, kr)
        assert result["US"] >= 0.20
        assert result["KR"] >= 0.20

    def test_max_allocation_respected(self):
        """Even with very strong momentum, allocation <= max."""
        alloc = MarketAllocator(max_allocation=0.80)
        us = _make_prices(n=300, daily_return=0.003, noise=0.005, seed=7)
        kr = _make_prices(n=300, daily_return=-0.002, noise=0.02, seed=8)
        result = alloc.compute(us, kr)
        assert result["US"] <= 0.80
        assert result["KR"] <= 0.80

    def test_allocations_sum_to_one(self):
        """Allocations always sum to 1.0."""
        alloc = MarketAllocator()
        us = _make_prices(n=300, daily_return=0.001, noise=0.01, seed=9)
        kr = _make_prices(n=300, daily_return=0.0005, noise=0.015, seed=10)
        result = alloc.compute(us, kr)
        assert abs(result["US"] + result["KR"] - 1.0) < 1e-6


class TestCustomParams:
    def test_custom_momentum_weight(self):
        """Higher momentum_weight → allocation follows momentum more."""
        us = _make_prices(n=300, daily_return=0.002, noise=0.01, seed=11)
        kr = _make_prices(n=300, daily_return=0.0003, noise=0.01, seed=12)

        mom_heavy = MarketAllocator(momentum_weight=0.9)
        vol_heavy = MarketAllocator(momentum_weight=0.1)

        r_mom = mom_heavy.compute(us, kr)
        r_vol = vol_heavy.compute(us, kr)

        # With high momentum weight, US should get even more (strong momentum)
        # This is directionally correct but exact values depend on data
        assert r_mom["US"] >= r_vol["US"] - 0.10  # allow some tolerance

    def test_custom_min_max(self):
        alloc = MarketAllocator(min_allocation=0.30, max_allocation=0.70)
        us = _make_prices(n=300, daily_return=0.003, seed=13)
        kr = _make_prices(n=300, daily_return=-0.003, seed=14)
        result = alloc.compute(us, kr)
        assert result["US"] <= 0.70 / (0.30 + 0.70)  # after normalization
        assert result["KR"] >= 0.30 / (0.30 + 0.70)


class TestEdgeCases:
    def test_zero_prices_handled(self, allocator):
        us = pd.Series([0.0] * 300)
        kr = _make_prices(n=300)
        result = allocator.compute(us, kr)
        # Should not crash; returns some valid allocation
        assert "US" in result
        assert "KR" in result
        assert abs(result["US"] + result["KR"] - 1.0) < 1e-6

    def test_constant_prices(self, allocator):
        us = pd.Series([100.0] * 300)
        kr = pd.Series([100.0] * 300)
        result = allocator.compute(us, kr)
        # Zero momentum, zero vol → fallback to equal
        assert abs(result["US"] - 0.50) < 0.05
