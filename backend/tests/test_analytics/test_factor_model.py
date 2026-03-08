"""Tests for Multi-Factor Scoring Model (Research-Backed)."""

import numpy as np
import pandas as pd
import pytest

from analytics.factor_model import FactorScores, FactorWeights, MultiFactorModel


@pytest.fixture
def model():
    return MultiFactorModel()


@pytest.fixture
def price_data():
    """Generate synthetic OHLCV data for 3 stocks with different profiles."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")

    def make_ohlcv(trend: float, volatility: float) -> pd.DataFrame:
        returns = np.random.normal(trend, volatility, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        return pd.DataFrame({
            "open": close * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            "high": close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        }, index=dates)

    return {
        "WINNER": make_ohlcv(0.001, 0.015),    # Strong uptrend, moderate vol
        "LOSER": make_ohlcv(-0.0005, 0.025),   # Slight downtrend, high vol
        "STEADY": make_ohlcv(0.0003, 0.008),   # Slight uptrend, low vol
    }


@pytest.fixture
def fundamental_data():
    return {
        "WINNER": {
            "revenueGrowth": 0.25, "earningsGrowth": 0.30,
            "profitMargins": 0.22, "returnOnEquity": 0.28,
            "forwardPE": 20,
        },
        "LOSER": {
            "revenueGrowth": -0.05, "earningsGrowth": -0.10,
            "profitMargins": 0.05, "returnOnEquity": 0.04,
            "forwardPE": 40,
        },
        "STEADY": {
            "revenueGrowth": 0.10, "earningsGrowth": 0.12,
            "profitMargins": 0.18, "returnOnEquity": 0.20,
            "forwardPE": 15,
        },
    }


class TestFactorScores:
    def test_default_values(self):
        fs = FactorScores(symbol="AAPL")
        assert fs.growth == 0.0
        assert fs.profitability == 0.0
        assert fs.garp == 0.0
        assert fs.momentum == 0.0
        assert fs.composite == 0.0
        assert fs.rank == 0


class TestFactorWeights:
    def test_default_sum_to_one(self):
        w = FactorWeights()
        total = w.growth + w.profitability + w.garp + w.momentum
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights(self):
        w = FactorWeights(growth=0.5, profitability=0.2, garp=0.2, momentum=0.1)
        assert w.growth == 0.5

    def test_research_backed_defaults(self):
        w = FactorWeights()
        assert w.growth == 0.35   # Highest IC
        assert w.profitability == 0.30  # Most consistent
        assert w.garp == 0.20
        assert w.momentum == 0.15  # Weakest predictor


class TestScoreUniverse:
    def test_returns_sorted_by_composite(self, model, price_data, fundamental_data):
        results = model.score_universe(price_data, fundamental_data)
        assert len(results) == 3
        composites = [r.composite for r in results]
        assert composites == sorted(composites, reverse=True)

    def test_ranks_assigned_correctly(self, model, price_data, fundamental_data):
        results = model.score_universe(price_data, fundamental_data)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_winner_ranks_higher_than_loser(self, model, price_data, fundamental_data):
        results = model.score_universe(price_data, fundamental_data)
        symbol_rank = {r.symbol: r.rank for r in results}
        assert symbol_rank["WINNER"] < symbol_rank["LOSER"]

    def test_empty_price_data(self, model):
        assert model.score_universe({}) == []

    def test_without_fundamentals(self, model, price_data):
        results = model.score_universe(price_data)
        assert len(results) == 3
        # Growth, profitability, GARP should be zero without fundamentals
        for r in results:
            assert r.growth == 0.0
            assert r.profitability == 0.0
            assert r.garp == 0.0

    def test_short_price_data_handled(self, model):
        short_df = pd.DataFrame({
            "close": [100 + i for i in range(50)],
        })
        results = model.score_universe({"SHORT": short_df})
        assert len(results) == 1
        assert results[0].momentum == 0.0


class TestGrowth:
    def test_high_growth_scores_higher(self, model, fundamental_data):
        symbols = list(fundamental_data.keys())
        scores = model._compute_growth(fundamental_data, symbols)
        assert scores["WINNER"] > scores["STEADY"]
        assert scores["STEADY"] > scores["LOSER"]

    def test_missing_fundamentals_return_zero(self, model):
        scores = model._compute_growth({}, ["AAPL"])
        assert scores["AAPL"] == 0.0

    def test_growth_capped_at_200pct(self, model):
        data = {"HYPER": {"revenueGrowth": 3.0, "earningsGrowth": 5.0}}
        scores = model._compute_growth(data, ["HYPER"])
        assert scores["HYPER"] == 2.0  # Capped at 200%

    def test_negative_growth(self, model):
        data = {"DECLINING": {"revenueGrowth": -0.15, "earningsGrowth": -0.20}}
        scores = model._compute_growth(data, ["DECLINING"])
        assert scores["DECLINING"] < 0


class TestProfitability:
    def test_high_margin_scores_higher(self, model, fundamental_data):
        symbols = list(fundamental_data.keys())
        scores = model._compute_profitability(fundamental_data, symbols)
        assert scores["WINNER"] > scores["LOSER"]

    def test_missing_fundamentals_return_zero(self, model):
        scores = model._compute_profitability({}, ["AAPL"])
        assert scores["AAPL"] == 0.0

    def test_margin_capped(self, model):
        data = {"RICH": {"profitMargins": 0.9, "returnOnEquity": 0.9}}
        scores = model._compute_profitability(data, ["RICH"])
        assert scores["RICH"] == 0.8  # Both capped at 0.8


class TestGARP:
    def test_high_growth_low_pe_wins(self, model, fundamental_data):
        symbols = list(fundamental_data.keys())
        scores = model._compute_garp(fundamental_data, symbols)
        # WINNER: 0.25/20=0.0125, STEADY: 0.10/15=0.0067, LOSER: -0.05/40=-0.00125
        assert scores["WINNER"] > scores["STEADY"]
        assert scores["STEADY"] > scores["LOSER"]

    def test_no_pe_returns_zero(self, model):
        data = {"NODATA": {"revenueGrowth": 0.2}}
        scores = model._compute_garp(data, ["NODATA"])
        assert scores["NODATA"] == 0.0


class TestMomentum:
    def test_positive_momentum_for_uptrend(self, model, price_data):
        scores = model._compute_momentum(price_data)
        assert scores["WINNER"] > scores["LOSER"]

    def test_returns_zero_for_short_data(self, model):
        short_df = pd.DataFrame({"close": [100 + i for i in range(50)]})
        scores = model._compute_momentum({"SHORT": short_df})
        assert scores["SHORT"] == 0.0

    def test_uses_3m_when_6m_unavailable(self, model):
        # 80 bars: enough for 3m (63) but not 6m (126)
        df = pd.DataFrame({"close": [100 + i * 0.5 for i in range(80)]})
        scores = model._compute_momentum({"MED": df})
        assert scores["MED"] > 0  # Uptrend detected


class TestZScore:
    def test_zscore_normalization(self):
        values = {"A": 10, "B": 20, "C": 30}
        z = MultiFactorModel._zscore(values)
        assert abs(z["B"]) < 0.01  # Middle value ~ 0
        assert z["C"] > 0
        assert z["A"] < 0

    def test_zscore_empty(self):
        assert MultiFactorModel._zscore({}) == {}

    def test_zscore_uniform(self):
        values = {"A": 5.0, "B": 5.0, "C": 5.0}
        z = MultiFactorModel._zscore(values)
        for v in z.values():
            assert v == 0.0


class TestGetTopN:
    def test_get_top_n(self, model, price_data, fundamental_data):
        scores = model.score_universe(price_data, fundamental_data)
        top2 = model.get_top_n(scores, n=2)
        assert len(top2) == 2
        assert top2[0].rank == 1

    def test_min_composite_filter(self, model, price_data, fundamental_data):
        scores = model.score_universe(price_data, fundamental_data)
        filtered = model.get_top_n(scores, n=10, min_composite=999.0)
        assert len(filtered) == 0


class TestCustomWeights:
    def test_growth_heavy_weights(self, price_data, fundamental_data):
        heavy = MultiFactorModel(weights=FactorWeights(
            growth=0.70, profitability=0.10, garp=0.10, momentum=0.10,
        ))
        results = heavy.score_universe(price_data, fundamental_data)
        assert results[0].symbol == "WINNER"

    def test_momentum_only(self, price_data, fundamental_data):
        mom_only = MultiFactorModel(weights=FactorWeights(
            growth=0.0, profitability=0.0, garp=0.0, momentum=1.0,
        ))
        results = mom_only.score_universe(price_data, fundamental_data)
        assert results[0].symbol == "WINNER"  # Best price momentum
