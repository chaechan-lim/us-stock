"""Tests for AI Trade Review Agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.trade_review import TradeReviewAgent, TradeReview, SYSTEM_PROMPT
from services.llm.providers import LLMResponse


def _make_llm_client(text: str, raise_error: Exception | None = None):
    """Create a mock LLMClient returning a given text."""
    client = MagicMock()
    if raise_error:
        client.generate = AsyncMock(side_effect=raise_error)
    else:
        client.generate = AsyncMock(return_value=LLMResponse(text=text, model="test"))
    return client


class TestTradeReview:
    def test_defaults(self):
        r = TradeReview(symbol="AAPL", trade_date="2026-03-07", side="buy")
        assert r.grade == "C"
        assert r.score == 50
        assert r.timing_assessment == "fair"
        assert r.lessons == []
        assert r.improvements == []

    def test_custom_values(self):
        r = TradeReview(
            symbol="TSLA", trade_date="2026-03-07", side="sell",
            grade="A", score=95, timing_assessment="excellent",
            entry_quality="Near support", exit_quality="At resistance",
            lessons=["Good patience"], improvements=[],
            summary="Well executed.",
        )
        assert r.grade == "A"
        assert r.score == 95
        assert r.entry_quality == "Near support"


class TestTradeReviewAgent:
    async def test_no_llm_client_returns_default(self):
        agent = TradeReviewAgent(llm_client=None)
        result = await agent.review_trade(
            symbol="AAPL", side="buy", entry_price=150.0,
            exit_price=165.0, quantity=10, strategy_name="momentum",
            pnl=150.0, holding_days=5, market_context={}, indicator_data={},
        )
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.grade == "C"
        assert result.score == 50

    async def test_review_trade_success(self):
        response_data = json.dumps({
            "grade": "B",
            "score": 80,
            "timing_assessment": "good",
            "entry_quality": "Solid entry near support",
            "exit_quality": "Good exit at target",
            "lessons": ["Followed the plan"],
            "improvements": ["Tighter stop"],
            "summary": "Good trade overall.",
        })
        client = _make_llm_client(response_data)
        agent = TradeReviewAgent(llm_client=client)

        result = await agent.review_trade(
            symbol="AAPL", side="buy", entry_price=150.0,
            exit_price=165.0, quantity=10, strategy_name="momentum",
            pnl=150.0, holding_days=5,
            market_context={"vix": 15}, indicator_data={"rsi": 65},
        )
        assert result.grade == "B"
        assert result.score == 80
        assert result.timing_assessment == "good"
        assert len(result.lessons) == 1
        client.generate.assert_called_once()

    async def test_review_trade_api_error_fallback(self):
        client = _make_llm_client("", raise_error=RuntimeError("API error"))
        agent = TradeReviewAgent(llm_client=client)

        result = await agent.review_trade(
            symbol="AAPL", side="buy", entry_price=150.0,
            exit_price=165.0, quantity=10, strategy_name="momentum",
            pnl=150.0, holding_days=5,
            market_context={}, indicator_data={},
        )
        # Should return default on error
        assert result.symbol == "AAPL"
        assert result.grade == "C"
        assert result.score == 50

    async def test_parse_valid_json(self):
        agent = TradeReviewAgent(llm_client=None)
        raw = json.dumps({
            "grade": "A",
            "score": 92,
            "timing_assessment": "excellent",
            "entry_quality": "Entered near 20-day EMA support",
            "exit_quality": "Exited at resistance with volume confirmation",
            "lessons": ["Patience on entry paid off", "Volume confirmed breakout"],
            "improvements": ["Could have used trailing stop"],
            "summary": "Textbook momentum trade with strong execution.",
        })
        result = agent._parse_response("AAPL", "buy", raw)
        assert result.grade == "A"
        assert result.score == 92
        assert result.timing_assessment == "excellent"
        assert result.entry_quality == "Entered near 20-day EMA support"
        assert len(result.lessons) == 2
        assert len(result.improvements) == 1

    async def test_parse_json_in_markdown(self):
        agent = TradeReviewAgent(llm_client=None)
        raw = '```json\n{"grade": "B", "score": 78, "timing_assessment": "good"}\n```'
        result = agent._parse_response("MSFT", "sell", raw)
        assert result.grade == "B"
        assert result.score == 78
        assert result.timing_assessment == "good"

    async def test_parse_invalid_json_fallback(self):
        agent = TradeReviewAgent(llm_client=None)
        result = agent._parse_response("AAPL", "buy", "This is not valid JSON")
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.grade == "C"
        assert "This is not valid JSON" in result.summary

    async def test_build_prompt_includes_all_data(self):
        agent = TradeReviewAgent(llm_client=None)
        prompt = agent._build_prompt(
            symbol="NVDA", side="sell",
            entry_price=400.0, exit_price=450.0,
            quantity=20, strategy_name="breakout",
            pnl=1000.0, holding_days=10,
            market_context={"vix": 18.5, "market_state": "uptrend"},
            indicator_data={"rsi": 72, "macd_histogram": 1.5},
        )
        assert "NVDA" in prompt
        assert "SELL" in prompt
        assert "$400.00" in prompt
        assert "$450.00" in prompt
        assert "20 shares" in prompt
        assert "breakout" in prompt
        assert "$1000.00" in prompt
        assert "10 days" in prompt
        assert "vix" in prompt
        assert "rsi" in prompt

    async def test_build_prompt_open_position(self):
        agent = TradeReviewAgent(llm_client=None)
        prompt = agent._build_prompt(
            symbol="AAPL", side="buy",
            entry_price=150.0, exit_price=None,
            quantity=10, strategy_name="momentum",
            pnl=0.0, holding_days=0,
            market_context={}, indicator_data={},
        )
        assert "N/A (open)" in prompt


class TestDailyReview:
    async def test_no_llm_client_returns_default(self):
        agent = TradeReviewAgent(llm_client=None)
        result = await agent.review_daily_trades(
            trades=[{"symbol": "AAPL", "side": "buy", "pnl": 100}],
            portfolio_summary={"total_value": 50000},
        )
        assert result["overall_grade"] == "C"
        assert result["total_trades"] == 1

    async def test_empty_trades_returns_default(self):
        client = _make_llm_client("should not be called")
        agent = TradeReviewAgent(llm_client=client)
        result = await agent.review_daily_trades(
            trades=[], portfolio_summary={"total_value": 50000},
        )
        assert result["total_trades"] == 0
        client.generate.assert_not_called()

    async def test_daily_review_success(self):
        response_data = json.dumps({
            "overall_grade": "B",
            "overall_score": 78,
            "total_trades": 3,
            "best_trade": "NVDA",
            "worst_trade": "TSLA",
            "patterns_identified": ["Overtrading in tech"],
            "daily_lessons": ["Stick to the plan"],
            "recommendations": ["Reduce position sizes"],
            "summary": "Decent day with room for improvement.",
        })
        client = _make_llm_client(response_data)
        agent = TradeReviewAgent(llm_client=client)

        trades = [
            {"symbol": "NVDA", "side": "buy", "pnl": 500},
            {"symbol": "AAPL", "side": "sell", "pnl": 200},
            {"symbol": "TSLA", "side": "buy", "pnl": -300},
        ]
        result = await agent.review_daily_trades(
            trades=trades,
            portfolio_summary={"total_value": 100000, "cash": 40000},
        )
        assert result["overall_grade"] == "B"
        assert result["overall_score"] == 78
        assert result["best_trade"] == "NVDA"
        assert result["worst_trade"] == "TSLA"
        assert len(result["patterns_identified"]) == 1
        client.generate.assert_called_once()

    async def test_daily_review_parse_invalid_json(self):
        client = _make_llm_client("Not valid JSON response")
        agent = TradeReviewAgent(llm_client=client)

        result = await agent.review_daily_trades(
            trades=[{"symbol": "AAPL", "pnl": 100}],
            portfolio_summary={"total_value": 50000},
        )
        # Falls back to defaults but includes raw text in summary
        assert result["overall_grade"] == "C"
        assert result["overall_score"] == 50
        assert "Not valid JSON" in result["summary"]


class TestSystemPrompt:
    def test_prompt_contains_key_sections(self):
        assert "Timing Assessment" in SYSTEM_PROMPT
        assert "Strategy Adherence" in SYSTEM_PROMPT
        assert "Risk Management" in SYSTEM_PROMPT
        assert "Entry Quality" in SYSTEM_PROMPT
        assert "Exit Quality" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT
