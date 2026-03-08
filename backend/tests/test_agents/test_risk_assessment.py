"""Tests for AI Risk Assessment Agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.risk_assessment import RiskAssessmentAgent, RiskAssessment, SYSTEM_PROMPT
from services.llm.providers import LLMResponse


def _make_llm_client(text: str, raise_error: Exception | None = None):
    """Create a mock LLMClient returning a given text."""
    client = MagicMock()
    if raise_error:
        client.generate = AsyncMock(side_effect=raise_error)
    else:
        client.generate = AsyncMock(return_value=LLMResponse(text=text, model="test"))
    return client


class TestRiskAssessment:
    def test_defaults(self):
        r = RiskAssessment()
        assert r.overall_risk_level == "MEDIUM"
        assert r.risk_score == 50
        assert r.warnings == []
        assert r.recommendations == []

    def test_custom_values(self):
        r = RiskAssessment(
            overall_risk_level="HIGH",
            risk_score=75,
            concentration_risk="50% in tech sector",
            correlation_risk="High correlation between AAPL and MSFT",
            market_risk="VIX elevated at 28",
            drawdown_risk="Max potential loss 12%",
            warnings=["Overweight tech"],
            recommendations=["Reduce AAPL position"],
            summary="Portfolio is concentrated in tech.",
        )
        assert r.overall_risk_level == "HIGH"
        assert r.risk_score == 75
        assert len(r.warnings) == 1


class TestRiskAssessmentAgent:
    async def test_no_llm_client_returns_default(self):
        agent = RiskAssessmentAgent(llm_client=None)
        result = await agent.assess_portfolio(
            positions=[{"symbol": "AAPL", "value": 10000}],
            portfolio_summary={"total_value": 50000},
            market_context={"vix": 15},
        )
        assert result.overall_risk_level == "MEDIUM"
        assert result.risk_score == 50

    async def test_assess_portfolio_success(self):
        response_data = json.dumps({
            "overall_risk_level": "HIGH",
            "risk_score": 68,
            "concentration_risk": "Heavy tech allocation",
            "correlation_risk": "Moderate correlation",
            "market_risk": "Elevated VIX",
            "drawdown_risk": "10% potential drawdown",
            "warnings": ["Reduce tech exposure"],
            "recommendations": ["Add defensive positions"],
            "summary": "Elevated risk from tech concentration.",
        })
        client = _make_llm_client(response_data)
        agent = RiskAssessmentAgent(llm_client=client)

        result = await agent.assess_portfolio(
            positions=[
                {"symbol": "AAPL", "value": 15000},
                {"symbol": "MSFT", "value": 12000},
            ],
            portfolio_summary={"total_value": 50000, "cash": 23000},
            market_context={"vix": 22},
            recent_trades=[{"symbol": "NVDA", "side": "buy"}],
        )
        assert result.overall_risk_level == "HIGH"
        assert result.risk_score == 68
        assert len(result.warnings) == 1
        client.generate.assert_called_once()

    async def test_assess_portfolio_api_error_fallback(self):
        client = _make_llm_client("", raise_error=RuntimeError("All providers failed"))
        agent = RiskAssessmentAgent(llm_client=client)

        result = await agent.assess_portfolio(
            positions=[{"symbol": "AAPL", "value": 10000}],
            portfolio_summary={"total_value": 50000},
            market_context={"vix": 15},
        )
        assert result.overall_risk_level == "MEDIUM"
        assert result.risk_score == 50

    async def test_parse_valid_json(self):
        agent = RiskAssessmentAgent(llm_client=None)
        raw = json.dumps({
            "overall_risk_level": "HIGH",
            "risk_score": 72,
            "concentration_risk": "60% in technology sector",
            "correlation_risk": "AAPL and MSFT highly correlated",
            "market_risk": "VIX at 25 indicates elevated volatility",
            "drawdown_risk": "Potential 15% portfolio drawdown",
            "warnings": ["Tech overweight", "VIX elevated"],
            "recommendations": ["Diversify into healthcare", "Add hedges"],
            "summary": "Portfolio has elevated risk due to tech concentration.",
        })
        result = agent._parse_response(raw)
        assert result.overall_risk_level == "HIGH"
        assert result.risk_score == 72
        assert result.concentration_risk == "60% in technology sector"
        assert len(result.warnings) == 2
        assert len(result.recommendations) == 2

    async def test_parse_json_in_markdown(self):
        agent = RiskAssessmentAgent(llm_client=None)
        raw = '```json\n{"overall_risk_level": "LOW", "risk_score": 20, "warnings": []}\n```'
        result = agent._parse_response(raw)
        assert result.overall_risk_level == "LOW"
        assert result.risk_score == 20

    async def test_parse_invalid_json_fallback(self):
        agent = RiskAssessmentAgent(llm_client=None)
        result = agent._parse_response("This is not valid JSON at all")
        assert result.overall_risk_level == "MEDIUM"
        assert result.risk_score == 50
        assert "This is not valid JSON" in result.summary

    async def test_build_portfolio_prompt_includes_all_data(self):
        agent = RiskAssessmentAgent(llm_client=None)
        positions = [
            {"symbol": "AAPL", "value": 10000, "sector": "Technology"},
            {"symbol": "JNJ", "value": 8000, "sector": "Healthcare"},
        ]
        portfolio_summary = {"total_value": 50000, "cash": 32000}
        market_context = {"vix": 18.5, "sp500_change": -0.5}
        recent_trades = [{"symbol": "NVDA", "side": "buy", "pnl": 200}]

        prompt = agent._build_portfolio_prompt(
            positions, portfolio_summary, market_context, recent_trades,
        )
        assert "AAPL" in prompt
        assert "JNJ" in prompt
        assert "50000" in prompt
        assert "18.5" in prompt
        assert "NVDA" in prompt
        assert "Recent Trades" in prompt

    async def test_build_portfolio_prompt_no_recent_trades(self):
        agent = RiskAssessmentAgent(llm_client=None)
        prompt = agent._build_portfolio_prompt(
            positions=[{"symbol": "AAPL"}],
            portfolio_summary={"total_value": 50000},
            market_context={"vix": 15},
            recent_trades=None,
        )
        assert "Recent Trades" not in prompt


class TestPreTradeAssessment:
    async def test_no_llm_client_returns_approved(self):
        agent = RiskAssessmentAgent(llm_client=None)
        result = await agent.assess_pre_trade(
            symbol="AAPL",
            proposed_size=5000.0,
            current_positions=[],
            portfolio_summary={"total_value": 50000},
        )
        assert result["approved"] is True
        assert result["risk_level"] == "MEDIUM"
        assert result["suggested_size"] == 5000.0

    async def test_pre_trade_success(self):
        response_data = json.dumps({
            "approved": False,
            "risk_level": "HIGH",
            "reason": "Position would create 40% tech concentration",
            "suggested_size": 2500.0,
            "warnings": ["Exceeds sector concentration limit"],
        })
        client = _make_llm_client(response_data)
        agent = RiskAssessmentAgent(llm_client=client)

        result = await agent.assess_pre_trade(
            symbol="NVDA",
            proposed_size=10000.0,
            current_positions=[
                {"symbol": "AAPL", "value": 15000, "sector": "Technology"},
            ],
            portfolio_summary={"total_value": 50000, "cash": 35000},
        )
        assert result["approved"] is False
        assert result["risk_level"] == "HIGH"
        assert result["suggested_size"] == 2500.0
        assert len(result["warnings"]) == 1
        client.generate.assert_called_once()

    async def test_pre_trade_invalid_json_fallback(self):
        client = _make_llm_client("Not JSON")
        agent = RiskAssessmentAgent(llm_client=client)

        result = await agent.assess_pre_trade(
            symbol="AAPL",
            proposed_size=5000.0,
            current_positions=[],
            portfolio_summary={"total_value": 50000},
        )
        # Fallback: approved with warnings
        assert result["approved"] is True
        assert result["suggested_size"] == 5000.0
        assert len(result["warnings"]) == 1

    async def test_pre_trade_api_error_fallback(self):
        client = _make_llm_client("", raise_error=RuntimeError("API error"))
        agent = RiskAssessmentAgent(llm_client=client)

        result = await agent.assess_pre_trade(
            symbol="AAPL",
            proposed_size=5000.0,
            current_positions=[],
            portfolio_summary={"total_value": 50000},
        )
        assert result["approved"] is True
        assert result["risk_level"] == "MEDIUM"


class TestSystemPrompt:
    def test_prompt_contains_key_sections(self):
        assert "Concentration Risk" in SYSTEM_PROMPT
        assert "Correlation Risk" in SYSTEM_PROMPT
        assert "Market Risk" in SYSTEM_PROMPT
        assert "Drawdown Risk" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_prompt_contains_risk_levels(self):
        assert "LOW" in SYSTEM_PROMPT
        assert "MEDIUM" in SYSTEM_PROMPT
        assert "HIGH" in SYSTEM_PROMPT
        assert "CRITICAL" in SYSTEM_PROMPT
