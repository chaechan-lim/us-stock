"""Tests for AI Market Analyst Agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.market_analyst import MarketAnalystAgent, AIRecommendation, SYSTEM_PROMPT
from services.llm.providers import LLMResponse


def _make_llm_client(text: str, raise_error: Exception | None = None):
    """Create a mock LLMClient returning a given text."""
    client = MagicMock()
    if raise_error:
        client.generate = AsyncMock(side_effect=raise_error)
    else:
        client.generate = AsyncMock(return_value=LLMResponse(text=text, model="test"))
    return client


class TestMarketAnalyst:
    async def test_no_llm_client_returns_default(self):
        agent = MarketAnalystAgent(llm_client=None)
        result = await agent.analyze(
            symbol="AAPL", indicator_score=75,
            fundamental_data={}, market_context={},
        )
        assert result.symbol == "AAPL"
        assert result.recommendation == "HOLD"

    async def test_analyze_success(self):
        response_data = json.dumps({
            "recommendation": "STRONG_BUY",
            "conviction": "HIGH",
            "score": 90,
            "entry_timing": "NOW",
            "target_price": 200.0,
            "stop_loss_price": 160.0,
            "position_size": "FULL",
            "time_horizon": "MEDIUM",
            "key_reasons": ["Bull trend"],
            "risks": ["Market risk"],
            "summary": "Buy now.",
        })
        client = _make_llm_client(response_data)
        agent = MarketAnalystAgent(llm_client=client)

        result = await agent.analyze(
            symbol="AAPL", indicator_score=80,
            fundamental_data={}, market_context={},
            current_price=175.0,
        )
        assert result.recommendation == "STRONG_BUY"
        assert result.score == 90
        client.generate.assert_called_once()

        # Verify call args
        call_kwargs = client.generate.call_args.kwargs
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert call_kwargs["max_tokens"] == 1024

    async def test_analyze_api_error_returns_default(self):
        client = _make_llm_client("", raise_error=RuntimeError("All providers failed"))
        agent = MarketAnalystAgent(llm_client=client)

        result = await agent.analyze(
            symbol="AAPL", indicator_score=80,
            fundamental_data={}, market_context={},
        )
        assert result.symbol == "AAPL"
        assert result.recommendation == "HOLD"

    async def test_parse_valid_json(self):
        agent = MarketAnalystAgent(llm_client=None)
        raw = json.dumps({
            "symbol": "AAPL",
            "recommendation": "BUY",
            "conviction": "HIGH",
            "score": 85,
            "entry_timing": "NOW",
            "target_price": 200.0,
            "stop_loss_price": 155.0,
            "position_size": "FULL",
            "time_horizon": "MEDIUM",
            "key_reasons": ["Strong trend", "Good fundamentals"],
            "risks": ["Overbought RSI"],
            "summary": "AAPL looks strong.",
        })
        result = agent._parse_response("AAPL", raw)
        assert result.recommendation == "BUY"
        assert result.conviction == "HIGH"
        assert result.score == 85
        assert result.target_price == 200.0
        assert len(result.key_reasons) == 2

    async def test_parse_json_in_markdown(self):
        agent = MarketAnalystAgent(llm_client=None)
        raw = '```json\n{"recommendation": "SELL", "conviction": "MEDIUM", "score": 30}\n```'
        result = agent._parse_response("AAPL", raw)
        assert result.recommendation == "SELL"
        assert result.score == 30

    async def test_parse_invalid_json(self):
        agent = MarketAnalystAgent(llm_client=None)
        result = agent._parse_response("AAPL", "This is not JSON at all")
        assert result.symbol == "AAPL"
        assert result.recommendation == "HOLD"
        assert "This is not JSON" in result.summary

    async def test_build_prompt(self):
        agent = MarketAnalystAgent(llm_client=None)
        prompt = agent._build_prompt(
            symbol="AAPL",
            indicator_score=80,
            fundamental_data={"pe_ratio": 28},
            market_context={"market_state": "uptrend"},
            current_price=175.0,
        )
        assert "AAPL" in prompt
        assert "$175.00" in prompt
        assert "80/100" in prompt


class TestAIRecommendation:
    def test_defaults(self):
        r = AIRecommendation(symbol="TEST")
        assert r.recommendation == "HOLD"
        assert r.conviction == "LOW"
        assert r.score == 50
        assert r.position_size == "SKIP"


class TestSystemPrompt:
    def test_prompt_contains_key_sections(self):
        assert "Technical Analysis" in SYSTEM_PROMPT
        assert "Fundamental Analysis" in SYSTEM_PROMPT
        assert "Risk Assessment" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT
