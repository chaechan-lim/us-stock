"""Tests for News Sentiment Analysis Agent."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.news_sentiment_agent import (
    NewsSentimentAgent,
    SentimentResult,
    NewsSentimentSummary,
    SYSTEM_PROMPT,
)
from data.news_service import NewsArticle
from services.llm.providers import LLMResponse


def _make_articles(n: int = 3) -> list[NewsArticle]:
    """Create test news articles."""
    return [
        NewsArticle(
            headline=f"Test headline {i}",
            summary=f"Test summary {i}",
            source="TestSource",
            symbol=f"SYM{i}",
            published_at=datetime(2026, 3, 9),
        )
        for i in range(n)
    ]


def _make_llm_client(text: str, raise_error: Exception | None = None):
    client = MagicMock()
    if raise_error:
        client.generate = AsyncMock(side_effect=raise_error)
    else:
        client.generate = AsyncMock(return_value=LLMResponse(text=text, model="test"))
    return client


class TestSentimentResult:
    def test_is_actionable_true(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.8, impact="HIGH",
            category="earnings", trading_signal="BULLISH",
        )
        assert r.is_actionable is True

    def test_is_actionable_false_low_impact(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.8, impact="LOW",
            category="other", trading_signal="BULLISH",
        )
        assert r.is_actionable is False

    def test_is_actionable_false_neutral(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.8, impact="HIGH",
            category="earnings", trading_signal="NEUTRAL",
        )
        assert r.is_actionable is False

    def test_is_actionable_false_weak_sentiment(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.1, impact="HIGH",
            category="earnings", trading_signal="BULLISH",
        )
        assert r.is_actionable is False

    def test_to_dict(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.6, impact="HIGH",
            category="earnings", sector_impact=["Technology"],
            key_event="Beat Q3", trading_signal="BULLISH",
        )
        d = r.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["sentiment"] == 0.6
        assert d["is_actionable"] is True
        assert "Technology" in d["sector_impact"]


class TestNewsSentimentSummary:
    def test_get_symbol_score(self):
        s = NewsSentimentSummary(symbol_sentiments={"AAPL": 0.5, "MSFT": -0.3})
        assert s.get_symbol_score("AAPL") == 0.5
        assert s.get_symbol_score("TSLA") == 0.0

    def test_get_sector_score(self):
        s = NewsSentimentSummary(sector_sentiments={"Technology": 0.7})
        assert s.get_sector_score("Technology") == 0.7
        assert s.get_sector_score("Energy") == 0.0

    def test_to_dict(self):
        s = NewsSentimentSummary(
            symbol_sentiments={"AAPL": 0.5},
            market_sentiment=0.3,
            analyzed_count=10,
        )
        d = s.to_dict()
        assert d["market_sentiment"] == 0.3
        assert d["analyzed_count"] == 10
        assert d["actionable_count"] == 0


class TestNewsSentimentAgent:
    async def test_no_llm_client_returns_empty(self):
        agent = NewsSentimentAgent(llm_client=None)
        result = await agent.analyze_batch(_make_articles())
        assert result.analyzed_count == 0

    async def test_empty_articles_returns_empty(self):
        client = _make_llm_client("[]")
        agent = NewsSentimentAgent(llm_client=client)
        result = await agent.analyze_batch([])
        assert result.analyzed_count == 0

    async def test_analyze_batch_success(self):
        response_data = json.dumps([
            {
                "symbol": "SYM0",
                "sentiment": 0.7,
                "impact": "HIGH",
                "category": "earnings",
                "sector_impact": ["Technology"],
                "key_event": "Beat Q3 estimates",
                "trading_signal": "BULLISH",
                "time_sensitivity": "IMMEDIATE",
            },
            {
                "symbol": "SYM1",
                "sentiment": -0.5,
                "impact": "MEDIUM",
                "category": "legal",
                "sector_impact": [],
                "key_event": "Lawsuit filed",
                "trading_signal": "BEARISH",
                "time_sensitivity": "SHORT_TERM",
            },
        ])
        client = _make_llm_client(response_data)
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles())

        assert result.analyzed_count == 3
        assert "SYM0" in result.symbol_sentiments
        assert result.symbol_sentiments["SYM0"] > 0
        assert result.symbol_sentiments["SYM1"] < 0
        assert len(result.actionable_signals) >= 1

    async def test_analyze_batch_llm_error(self):
        client = _make_llm_client("", raise_error=RuntimeError("API error"))
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles())
        assert result.analyzed_count == 3
        assert len(result.symbol_sentiments) == 0

    async def test_analyze_batch_with_markdown_json(self):
        response = '```json\n[{"symbol": "AAPL", "sentiment": 0.5, "impact": "MEDIUM", "category": "product", "trading_signal": "BULLISH"}]\n```'
        client = _make_llm_client(response)
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles(1))
        assert "AAPL" in result.symbol_sentiments

    async def test_analyze_batch_invalid_json(self):
        client = _make_llm_client("Not valid JSON at all")
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles())
        assert result.analyzed_count == 3
        assert len(result.symbol_sentiments) == 0

    async def test_sentiment_clamped(self):
        response = json.dumps([
            {"symbol": "X", "sentiment": 5.0, "impact": "HIGH", "category": "other", "trading_signal": "BULLISH"},
            {"symbol": "Y", "sentiment": -5.0, "impact": "HIGH", "category": "other", "trading_signal": "BEARISH"},
        ])
        client = _make_llm_client(response)
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles(2))
        assert result.symbol_sentiments.get("X", 0) <= 1.0
        assert result.symbol_sentiments.get("Y", 0) >= -1.0

    async def test_market_sentiment_from_market_symbol(self):
        response = json.dumps([
            {"symbol": "MARKET", "sentiment": -0.3, "impact": "HIGH", "category": "macro", "trading_signal": "BEARISH"},
        ])
        client = _make_llm_client(response)
        agent = NewsSentimentAgent(llm_client=client)

        result = await agent.analyze_batch(_make_articles(1))
        assert result.market_sentiment < 0

    async def test_save_insights_called(self):
        response = json.dumps([
            {
                "symbol": "AAPL", "sentiment": 0.8, "impact": "HIGH",
                "category": "earnings", "key_event": "Beat Q3",
                "trading_signal": "BULLISH",
            },
        ])
        client = _make_llm_client(response)
        mock_ctx = AsyncMock()
        mock_ctx.build_context = AsyncMock(return_value="")
        mock_ctx.save = AsyncMock()
        agent = NewsSentimentAgent(llm_client=client, context_service=mock_ctx)

        await agent.analyze_batch(_make_articles(1))
        assert mock_ctx.save.call_count > 0

    async def test_large_batch_chunked(self):
        """Batches larger than chunk_size should be split."""
        response = json.dumps([
            {"symbol": "X", "sentiment": 0.1, "impact": "LOW", "category": "other", "trading_signal": "NEUTRAL"},
        ])
        client = _make_llm_client(response)
        agent = NewsSentimentAgent(llm_client=client)

        # 25 articles > chunk_size of 20
        articles = _make_articles(25)
        result = await agent.analyze_batch(articles)

        # Should have made 2 LLM calls (20 + 5)
        assert client.generate.call_count == 2
        assert result.analyzed_count == 25


class TestBuildPrompt:
    def test_prompt_contains_articles(self):
        agent = NewsSentimentAgent(llm_client=None)
        articles = [
            NewsArticle("AAPL beats Q3", "Strong earnings", "Reuters", "AAPL", datetime.now()),
        ]
        prompt = agent._build_prompt(articles)
        assert "AAPL beats Q3" in prompt
        assert "Article 1" in prompt
        assert "AAPL" in prompt

    def test_prompt_with_memory(self):
        agent = NewsSentimentAgent(llm_client=None)
        articles = _make_articles(1)
        prompt = agent._build_prompt(articles, memory_context="Past insight: tech bullish")
        assert "Past insight" in prompt


class TestParseResponse:
    def test_parse_array(self):
        agent = NewsSentimentAgent(llm_client=None)
        raw = json.dumps([
            {"symbol": "A", "sentiment": 0.5, "impact": "HIGH", "category": "earnings", "trading_signal": "BULLISH"},
        ])
        results = agent._parse_response(raw)
        assert len(results) == 1
        assert results[0].symbol == "A"
        assert results[0].sentiment == 0.5

    def test_parse_single_object(self):
        agent = NewsSentimentAgent(llm_client=None)
        raw = json.dumps(
            {"symbol": "B", "sentiment": -0.3, "impact": "MEDIUM", "category": "other", "trading_signal": "BEARISH"},
        )
        results = agent._parse_response(raw)
        assert len(results) == 1
        assert results[0].symbol == "B"

    def test_parse_invalid_json(self):
        agent = NewsSentimentAgent(llm_client=None)
        results = agent._parse_response("not json")
        assert results == []


class TestAggregate:
    def test_weighted_average(self):
        agent = NewsSentimentAgent(llm_client=None)
        results = [
            SentimentResult("AAPL", 0.8, "HIGH", "earnings", trading_signal="BULLISH"),
            SentimentResult("AAPL", 0.4, "LOW", "analyst", trading_signal="BULLISH"),
        ]
        summary = agent._aggregate(results, 2)
        # HIGH=3.0 weight, LOW=1.0 weight
        # weighted avg = (0.8*3 + 0.4*1) / 4 = 0.7
        assert summary.symbol_sentiments["AAPL"] == pytest.approx(0.7, abs=0.01)

    def test_market_symbol_separate(self):
        agent = NewsSentimentAgent(llm_client=None)
        results = [
            SentimentResult("MARKET", -0.5, "HIGH", "macro"),
            SentimentResult("AAPL", 0.3, "MEDIUM", "earnings"),
        ]
        summary = agent._aggregate(results, 2)
        assert summary.market_sentiment == -0.5
        assert "MARKET" not in summary.symbol_sentiments
        assert "AAPL" in summary.symbol_sentiments

    def test_sector_accumulation(self):
        agent = NewsSentimentAgent(llm_client=None)
        results = [
            SentimentResult("AAPL", 0.6, "HIGH", "earnings", sector_impact=["Technology"]),
            SentimentResult("MSFT", 0.4, "MEDIUM", "product", sector_impact=["Technology"]),
        ]
        summary = agent._aggregate(results, 2)
        assert "Technology" in summary.sector_sentiments
        assert summary.sector_sentiments["Technology"] == pytest.approx(0.5, abs=0.01)


class TestSystemPrompt:
    def test_prompt_contains_key_sections(self):
        assert "sentiment" in SYSTEM_PROMPT.lower()
        assert "JSON" in SYSTEM_PROMPT
        assert "trading_signal" in SYSTEM_PROMPT
