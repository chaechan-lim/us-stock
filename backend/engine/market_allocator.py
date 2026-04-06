"""Dynamic market allocation using dual momentum + inverse volatility.

Computes US/KR allocation weights based on:
1. Relative momentum: 12-1 month return of each market index
2. Absolute momentum: Only allocate to markets with positive momentum
3. Inverse volatility: Risk-parity weighting across markets
4. Regime awareness: Integrates with existing market state detection

Updates RiskManager.market_allocations dynamically instead of fixed 50/50.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketAllocator:
    """Compute dynamic US/KR allocation using dual momentum."""

    def __init__(
        self,
        momentum_lookback: int = 252,
        skip_recent: int = 21,
        vol_lookback: int = 60,
        momentum_weight: float = 0.6,
        min_allocation: float = 0.20,
        max_allocation: float = 0.80,
    ):
        """
        Args:
            momentum_lookback: Bars for 12-month momentum (252 trading days).
            skip_recent: Skip last N bars for reversal avoidance (21 = 1 month).
            vol_lookback: Bars for volatility calculation (60 = 3 months).
            momentum_weight: Blend ratio (momentum vs inverse vol). 0.6 = 60% momentum.
            min_allocation: Floor allocation per market.
            max_allocation: Ceiling allocation per market.
        """
        self._momentum_lookback = momentum_lookback
        self._skip_recent = skip_recent
        self._vol_lookback = vol_lookback
        self._momentum_weight = momentum_weight
        self._min_alloc = min_allocation
        self._max_alloc = max_allocation

    @property
    def min_bars_required(self) -> int:
        return self._momentum_lookback + self._skip_recent

    def compute(
        self,
        us_prices: pd.Series,
        kr_prices: pd.Series,
    ) -> dict[str, float]:
        """Compute target allocation for US and KR markets.

        Args:
            us_prices: US market index close prices (e.g. SPY).
            kr_prices: KR market index close prices (e.g. KODEX 200).

        Returns:
            {"US": float, "KR": float} summing to ~1.0, clamped to [min, max].
        """
        min_bars = self.min_bars_required
        if len(us_prices) < min_bars or len(kr_prices) < min_bars:
            logger.debug(
                "Insufficient data for allocation (US=%d, KR=%d, need=%d)",
                len(us_prices), len(kr_prices), min_bars,
            )
            return {"US": 0.50, "KR": 0.50}

        # 1. Momentum: 12-month return skipping last month
        us_mom = self._compute_momentum(us_prices)
        kr_mom = self._compute_momentum(kr_prices)

        # 2. Inverse volatility
        us_vol = self._compute_volatility(us_prices)
        kr_vol = self._compute_volatility(kr_prices)

        # 3. Momentum score → allocation weight
        mom_alloc = self._momentum_to_allocation(us_mom, kr_mom)

        # 4. Inverse vol → allocation weight
        vol_alloc = self._invvol_to_allocation(us_vol, kr_vol)

        # 5. Blend
        us_raw = (
            self._momentum_weight * mom_alloc["US"]
            + (1 - self._momentum_weight) * vol_alloc["US"]
        )
        kr_raw = (
            self._momentum_weight * mom_alloc["KR"]
            + (1 - self._momentum_weight) * vol_alloc["KR"]
        )

        # 6. Clamp and normalize
        us_clamped = max(self._min_alloc, min(self._max_alloc, us_raw))
        kr_clamped = max(self._min_alloc, min(self._max_alloc, kr_raw))
        total = us_clamped + kr_clamped
        us_final = us_clamped / total
        kr_final = kr_clamped / total

        logger.info(
            "MarketAllocator: US_mom=%.1f%% KR_mom=%.1f%% "
            "US_vol=%.1f%% KR_vol=%.1f%% → US=%.0f%% KR=%.0f%%",
            us_mom * 100, kr_mom * 100,
            us_vol * 100, kr_vol * 100,
            us_final * 100, kr_final * 100,
        )

        return {"US": round(us_final, 4), "KR": round(kr_final, 4)}

    def _compute_momentum(self, prices: pd.Series) -> float:
        """12-1 month momentum: return from T-252 to T-21."""
        end_idx = len(prices) - 1 - self._skip_recent
        start_idx = end_idx - self._momentum_lookback
        if start_idx < 0 or end_idx < 0:
            return 0.0
        p_start = float(prices.iloc[start_idx])
        p_end = float(prices.iloc[end_idx])
        if p_start <= 0:
            return 0.0
        return (p_end - p_start) / p_start

    def _compute_volatility(self, prices: pd.Series) -> float:
        """Annualized volatility from recent returns."""
        recent = prices.iloc[-self._vol_lookback:]
        if len(recent) < 2:
            return 0.20  # default 20%
        returns = recent.pct_change().dropna()
        if len(returns) < 2:
            return 0.20
        daily_vol = float(np.std(returns))
        return daily_vol * np.sqrt(252)

    def _momentum_to_allocation(
        self, us_mom: float, kr_mom: float,
    ) -> dict[str, float]:
        """Convert momentum scores to allocation weights.

        Rules:
        - Both positive: allocate proportionally to momentum
        - One negative: shift allocation toward the positive market
        - Both negative: equal allocation (cash-like, conservative)
        """
        if us_mom <= 0 and kr_mom <= 0:
            return {"US": 0.50, "KR": 0.50}

        if us_mom <= 0:
            return {"US": self._min_alloc, "KR": 1.0 - self._min_alloc}

        if kr_mom <= 0:
            return {"US": 1.0 - self._min_alloc, "KR": self._min_alloc}

        # Both positive: proportional allocation
        total_mom = us_mom + kr_mom
        return {
            "US": us_mom / total_mom,
            "KR": kr_mom / total_mom,
        }

    def _invvol_to_allocation(
        self, us_vol: float, kr_vol: float,
    ) -> dict[str, float]:
        """Inverse volatility weighting (risk parity)."""
        if us_vol <= 0 and kr_vol <= 0:
            return {"US": 0.50, "KR": 0.50}

        us_inv = 1.0 / max(us_vol, 0.01)
        kr_inv = 1.0 / max(kr_vol, 0.01)
        total_inv = us_inv + kr_inv

        return {
            "US": us_inv / total_inv,
            "KR": kr_inv / total_inv,
        }
