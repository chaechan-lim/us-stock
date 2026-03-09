"""Tests for News Enricher."""

import pytest

from scanner.news_enricher import NewsEnricher, MAX_NEWS_ADJUSTMENT, NEWS_WEIGHT
from agents.news_sentiment_agent import NewsSentimentSummary, SentimentResult


def _make_candidates(n: int = 3) -> list[dict]:
    """Create test candidates."""
    return [
        {
            "symbol": f"SYM{i}",
            "indicator_score": 70 + i,
            "combined_score": 65 + i * 5,
            "grade": "B",
        }
        for i in range(n)
    ]


class TestNewsEnricher:
    def test_no_sentiment_data(self):
        enricher = NewsEnricher()
        candidates = _make_candidates()
        result = enricher.enrich(candidates, NewsSentimentSummary())

        # Should pass through with news_impact=NONE
        for c in result:
            assert c["news_impact"] == "NONE"
            assert c["news_sentiment"] == 0.0
            assert c["news_events"] == []

    def test_positive_symbol_sentiment_boosts_score(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(1)
        original_score = candidates[0]["combined_score"]

        summary = NewsSentimentSummary(
            symbol_sentiments={"SYM0": 0.8},
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)

        assert result[0]["combined_score"] > original_score
        assert result[0]["news_sentiment"] > 0
        assert result[0]["news_impact"] in ("HIGH", "MEDIUM")

    def test_negative_sentiment_penalizes_score(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(1)
        original_score = candidates[0]["combined_score"]

        summary = NewsSentimentSummary(
            symbol_sentiments={"SYM0": -0.7},
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)

        assert result[0]["combined_score"] < original_score
        assert result[0]["news_sentiment"] < 0

    def test_market_sentiment_affects_all(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(2)
        original_by_sym = {c["symbol"]: c["combined_score"] for c in candidates}

        summary = NewsSentimentSummary(
            market_sentiment=0.5,
            analyzed_count=10,
        )
        result = enricher.enrich(candidates, summary)

        # Both should be slightly boosted (via market sentiment)
        for c in result:
            orig = original_by_sym[c["symbol"]]
            assert c["combined_score"] >= orig - 0.01

    def test_score_clamped_to_0_100(self):
        enricher = NewsEnricher()
        candidates = [{"symbol": "X", "combined_score": 98, "grade": "A"}]

        summary = NewsSentimentSummary(
            symbol_sentiments={"X": 1.0},
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)
        assert result[0]["combined_score"] <= 100

    def test_score_not_below_zero(self):
        enricher = NewsEnricher()
        candidates = [{"symbol": "X", "combined_score": 5, "grade": "F"}]

        summary = NewsSentimentSummary(
            symbol_sentiments={"X": -1.0},
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)
        assert result[0]["combined_score"] >= 0

    def test_actionable_events_included(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(1)

        summary = NewsSentimentSummary(
            symbol_sentiments={"SYM0": 0.6},
            actionable_signals=[
                SentimentResult(
                    symbol="SYM0", sentiment=0.8, impact="HIGH",
                    category="earnings", key_event="Beat Q3 estimates",
                    trading_signal="BULLISH",
                ),
            ],
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)
        assert "Beat Q3 estimates" in result[0]["news_events"]

    def test_results_resorted_by_score(self):
        enricher = NewsEnricher()
        # Close scores so news can flip ranking
        candidates = [
            {"symbol": "A", "combined_score": 72, "grade": "B"},
            {"symbol": "B", "combined_score": 70, "grade": "B"},
        ]

        # Strongly boost B, penalize A
        summary = NewsSentimentSummary(
            symbol_sentiments={"A": -0.9, "B": 0.9},
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)

        # B should now be ranked higher than A
        assert result[0]["symbol"] == "B"

    def test_impact_levels(self):
        enricher = NewsEnricher()

        summary_high = NewsSentimentSummary(
            symbol_sentiments={"X": 0.7}, analyzed_count=1,
        )
        summary_med = NewsSentimentSummary(
            symbol_sentiments={"X": 0.3}, analyzed_count=1,
        )
        summary_low = NewsSentimentSummary(
            symbol_sentiments={"X": 0.05}, analyzed_count=1,
        )

        c1 = [{"symbol": "X", "combined_score": 50, "grade": "C"}]
        enricher.enrich(c1, summary_high)
        assert c1[0]["news_impact"] == "HIGH"

        c2 = [{"symbol": "X", "combined_score": 50, "grade": "C"}]
        enricher.enrich(c2, summary_med)
        assert c2[0]["news_impact"] == "MEDIUM"

        c3 = [{"symbol": "X", "combined_score": 50, "grade": "C"}]
        enricher.enrich(c3, summary_low)
        assert c3[0]["news_impact"] == "LOW"

    def test_no_symbol_news_uses_market(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(1)

        # No symbol-specific news, only market sentiment
        summary = NewsSentimentSummary(
            market_sentiment=0.6,
            analyzed_count=10,
        )
        result = enricher.enrich(candidates, summary)

        # Should still adjust based on market sentiment
        assert result[0]["news_sentiment"] > 0

    def test_last_summary_stored(self):
        enricher = NewsEnricher()
        assert enricher.last_summary is None

        summary = NewsSentimentSummary(analyzed_count=5)
        enricher.enrich(_make_candidates(), summary)
        assert enricher.last_summary is summary

    def test_custom_news_weight(self):
        enricher_light = NewsEnricher(news_weight=0.05)
        enricher_heavy = NewsEnricher(news_weight=0.30)

        summary = NewsSentimentSummary(
            symbol_sentiments={"SYM0": 0.8},
            analyzed_count=5,
        )

        c1 = _make_candidates(1)
        c2 = _make_candidates(1)
        original = c1[0]["combined_score"]

        enricher_light.enrich(c1, summary)
        enricher_heavy.enrich(c2, summary)

        # Heavier weight should produce larger adjustment
        delta_light = abs(c1[0]["combined_score"] - original)
        delta_heavy = abs(c2[0]["combined_score"] - original)
        assert delta_heavy > delta_light

    def test_max_3_events(self):
        enricher = NewsEnricher()
        candidates = _make_candidates(1)

        summary = NewsSentimentSummary(
            symbol_sentiments={"SYM0": 0.5},
            actionable_signals=[
                SentimentResult("SYM0", 0.5, "HIGH", "earnings", key_event=f"Event {i}", trading_signal="BULLISH")
                for i in range(5)
            ],
            analyzed_count=5,
        )
        result = enricher.enrich(candidates, summary)
        assert len(result[0]["news_events"]) <= 3


class TestConstants:
    def test_max_adjustment(self):
        assert MAX_NEWS_ADJUSTMENT == 15

    def test_news_weight(self):
        assert 0 < NEWS_WEIGHT < 1
