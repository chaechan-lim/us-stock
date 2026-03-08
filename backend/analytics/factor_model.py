"""Multi-Factor Scoring Model (Research-Backed).

Ranks stocks using empirically validated factors.
Based on 5-year factor research (60 stocks, quarterly IC analysis):

Top Predictive Factors:
1. Growth Factor (IC=0.22) — revenue & earnings growth
2. GARP Factor (IC=0.20) — growth at reasonable price
3. Profitability Factor (IC=0.18, IR=1.39) — margins & ROE (most consistent)
4. Quality Composite (IC=0.17, IR=1.21) — combined quality metrics
5. Momentum (IC=0.05-0.10) — weak alone, useful in combos

Removed (empirically harmful):
- Low Volatility (IC=-0.10) — high-vol stocks outperform
- Pure Value/cheap PE (IC=-0.08) — value trap
- Low Debt (IC=0.00) — no predictive power
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    """Per-stock factor scores (z-scores, higher = better)."""
    symbol: str
    growth: float = 0.0          # Revenue + earnings growth
    profitability: float = 0.0   # Profit margin + ROE
    garp: float = 0.0            # Growth at reasonable price
    momentum: float = 0.0        # Price momentum (6m)
    composite: float = 0.0
    rank: int = 0


@dataclass
class FactorWeights:
    """Weights for combining factors into composite score.

    Based on IC research results:
    - growth: IC=0.22, spread=+18%/6m
    - profitability: IC=0.18, IR=1.39 (most consistent)
    - garp: IC=0.20, IR=1.02
    - momentum: IC=0.10, useful as confirming signal
    """
    growth: float = 0.35
    profitability: float = 0.30
    garp: float = 0.20
    momentum: float = 0.15


class MultiFactorModel:
    """Cross-sectional multi-factor stock ranking model."""

    def __init__(self, weights: FactorWeights | None = None):
        self._weights = weights or FactorWeights()

    def score_universe(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: dict[str, dict] | None = None,
    ) -> list[FactorScores]:
        """Score and rank a universe of stocks by factor model.

        Args:
            price_data: {symbol: OHLCV DataFrame} with at least 126 bars.
            fundamental_data: {symbol: {revenueGrowth, profitMargins, ...}}.

        Returns:
            List of FactorScores sorted by composite score descending.
        """
        if not price_data:
            return []

        symbols = list(price_data.keys())
        fund = fundamental_data or {}

        # Compute raw factor values
        growth_raw = self._compute_growth(fund, symbols)
        profit_raw = self._compute_profitability(fund, symbols)
        garp_raw = self._compute_garp(fund, symbols)
        momentum_raw = self._compute_momentum(price_data)

        # Cross-sectional z-score normalization
        growth_z = self._zscore(growth_raw)
        profit_z = self._zscore(profit_raw)
        garp_z = self._zscore(garp_raw)
        momentum_z = self._zscore(momentum_raw)

        # Composite score
        w = self._weights
        results = []
        for sym in symbols:
            g = growth_z.get(sym, 0.0)
            p = profit_z.get(sym, 0.0)
            ga = garp_z.get(sym, 0.0)
            m = momentum_z.get(sym, 0.0)
            composite = (
                g * w.growth + p * w.profitability + ga * w.garp + m * w.momentum
            )
            results.append(FactorScores(
                symbol=sym,
                growth=round(g, 3),
                profitability=round(p, 3),
                garp=round(ga, 3),
                momentum=round(m, 3),
                composite=round(composite, 3),
            ))

        # Rank by composite
        results.sort(key=lambda s: s.composite, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _compute_growth(
        self, fundamentals: dict[str, dict], symbols: list[str],
    ) -> dict[str, float]:
        """Growth factor: revenue growth + earnings growth.

        IC=0.23 (revenue), IC=0.19 (earnings) — strongest predictors.
        """
        scores = {}
        for sym in symbols:
            data = fundamentals.get(sym, {})
            rev = data.get("revenue_growth") or data.get("revenueGrowth")
            earn = data.get("earnings_growth") or data.get("earningsGrowth")

            components = []
            if rev is not None:
                components.append(min(float(rev), 2.0))  # Cap at 200%
            if earn is not None:
                components.append(min(float(earn), 2.0))

            scores[sym] = float(np.mean(components)) if components else 0.0

        return scores

    def _compute_profitability(
        self, fundamentals: dict[str, dict], symbols: list[str],
    ) -> dict[str, float]:
        """Profitability factor: profit margin + ROE.

        IC=0.18 (margin, IR=1.39 most consistent), IC=0.11 (ROE).
        """
        scores = {}
        for sym in symbols:
            data = fundamentals.get(sym, {})
            margin = data.get("profit_margin") or data.get("profitMargins")
            roe = data.get("roe") or data.get("returnOnEquity")

            components = []
            if margin is not None:
                components.append(min(float(margin), 0.8))  # Cap at 80%
            if roe is not None:
                components.append(min(float(roe), 0.8))  # Cap at 80%

            scores[sym] = float(np.mean(components)) if components else 0.0

        return scores

    def _compute_garp(
        self, fundamentals: dict[str, dict], symbols: list[str],
    ) -> dict[str, float]:
        """GARP factor: Growth at Reasonable Price (growth / PE).

        IC=0.20, IR=1.02 — high growth relative to valuation.
        """
        scores = {}
        for sym in symbols:
            data = fundamentals.get(sym, {})
            rev = data.get("revenue_growth") or data.get("revenueGrowth")
            pe = data.get("pe_ratio") or data.get("forwardPE") or data.get("trailingPE")

            if rev is not None and pe and float(pe) > 0:
                scores[sym] = float(rev) / float(pe)
            else:
                scores[sym] = 0.0

        return scores

    def _compute_momentum(self, price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Momentum factor: 6-month return.

        IC=0.10 — weak alone but useful as confirming signal.
        Uses 6m return (simpler, similar IC to 12-1 momentum).
        """
        scores = {}
        for sym, df in price_data.items():
            close = df["close"]
            current = float(close.iloc[-1])

            if len(close) >= 126:
                ret_6m = current / float(close.iloc[-126]) - 1
                scores[sym] = ret_6m
            elif len(close) >= 63:
                ret_3m = current / float(close.iloc[-63]) - 1
                scores[sym] = ret_3m
            else:
                scores[sym] = 0.0

        return scores

    @staticmethod
    def _zscore(values: dict[str, float]) -> dict[str, float]:
        """Cross-sectional z-score normalization."""
        if not values:
            return {}
        arr = np.array(list(values.values()))
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return {s: 0.0 for s in values}
        return {s: float((v - mean) / std) for s, v in values.items()}

    def get_top_n(
        self,
        scores: list[FactorScores],
        n: int = 10,
        min_composite: float = 0.0,
    ) -> list[FactorScores]:
        """Get top N stocks by composite score."""
        return [s for s in scores[:n] if s.composite >= min_composite]
