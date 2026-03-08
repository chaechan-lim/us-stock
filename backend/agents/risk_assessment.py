"""AI Risk Assessment Agent — evaluates portfolio risk.

Uses LLMClient to perform comprehensive portfolio risk analysis,
including concentration, correlation, market, and drawdown risks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional portfolio risk analyst for US equities.
You assess portfolio-level risk by analyzing position concentration, sector
correlation, market conditions, and drawdown potential.

Your risk assessment framework:
1. **Concentration Risk**: Over-allocation to single stocks or sectors
2. **Correlation Risk**: Positions that move together amplify losses
3. **Market Risk**: Exposure to broad market downturns, rate changes, geopolitical events
4. **Drawdown Risk**: Maximum potential loss based on volatility and position sizes
5. **Liquidity Risk**: Positions that may be hard to exit quickly
6. **Timing Risk**: Being overexposed during high-volatility periods

Risk levels:
- LOW (0-25): Well-diversified, conservative positioning
- MEDIUM (26-50): Acceptable risk for active trading
- HIGH (51-75): Elevated risk, consider reducing exposure
- CRITICAL (76-100): Immediate action needed to reduce risk

Output your analysis as JSON with this exact structure:
{
  "overall_risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "risk_score": 0-100,
  "concentration_risk": "description of concentration risk",
  "correlation_risk": "description of correlation risk",
  "market_risk": "description of market risk",
  "drawdown_risk": "description of drawdown risk",
  "warnings": ["warning1", "warning2"],
  "recommendations": ["recommendation1", "recommendation2"],
  "summary": "One paragraph risk summary"
}"""

PRE_TRADE_PROMPT = """You are a professional portfolio risk analyst for US equities.
Evaluate whether a proposed trade is acceptable from a risk management perspective.

Consider:
1. Position sizing relative to portfolio value
2. Sector concentration after the trade
3. Correlation with existing positions
4. Current market conditions

Output your analysis as JSON with this exact structure:
{
  "approved": true | false,
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "reason": "Explanation of the risk assessment",
  "suggested_size": 0.0,
  "warnings": ["warning1"]
}"""


@dataclass
class RiskAssessment:
    overall_risk_level: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_score: int = 50  # 0-100 (higher = riskier)
    concentration_risk: str = ""
    correlation_risk: str = ""
    market_risk: str = ""
    drawdown_risk: str = ""
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""


class RiskAssessmentAgent:
    """AI agent for portfolio risk assessment using LLMClient."""

    def __init__(self, llm_client: LLMClient | None = None):
        self._llm_client = llm_client

    async def assess_portfolio(
        self,
        positions: list[dict],
        portfolio_summary: dict,
        market_context: dict,
        recent_trades: list[dict] | None = None,
    ) -> RiskAssessment:
        """Comprehensive portfolio risk assessment."""
        if not self._llm_client:
            logger.warning("No LLM client configured, returning default risk assessment")
            return RiskAssessment()

        user_prompt = self._build_portfolio_prompt(
            positions, portfolio_summary, market_context, recent_trades,
        )

        try:
            response = await self._llm_client.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=SYSTEM_PROMPT,
                max_tokens=1024,
            )
            return self._parse_response(response.text or "")

        except Exception as e:
            logger.error("Risk assessment failed: %s", e)
            return RiskAssessment()

    async def assess_pre_trade(
        self,
        symbol: str,
        proposed_size: float,
        current_positions: list[dict],
        portfolio_summary: dict,
    ) -> dict:
        """Quick risk check before entering a new trade."""
        default_result = {
            "approved": True,
            "risk_level": "MEDIUM",
            "reason": "No LLM client configured, defaulting to approved",
            "suggested_size": proposed_size,
            "warnings": [],
        }

        if not self._llm_client:
            logger.warning("No LLM client configured, returning default pre-trade check")
            return default_result

        total_value = portfolio_summary.get("total_value", 0)
        pct_of_portfolio = (proposed_size / total_value * 100) if total_value else 0

        user_prompt = f"""Evaluate this proposed trade from a risk perspective:

## Proposed Trade:
- Symbol: {symbol}
- Proposed Position Size: ${proposed_size:,.2f} ({pct_of_portfolio:.1f}% of portfolio)

## Current Positions:
{json.dumps(current_positions, indent=2, default=str)}

## Portfolio Summary:
{json.dumps(portfolio_summary, indent=2, default=str)}

Should this trade be approved from a risk management perspective? Respond as JSON."""

        try:
            response = await self._llm_client.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=PRE_TRADE_PROMPT,
                max_tokens=512,
            )
            return self._parse_pre_trade_response(response.text or "", proposed_size)

        except Exception as e:
            logger.error("Pre-trade risk check failed for %s: %s", symbol, e)
            return default_result

    def _build_portfolio_prompt(
        self,
        positions: list[dict],
        portfolio_summary: dict,
        market_context: dict,
        recent_trades: list[dict] | None,
    ) -> str:
        parts = [
            "Assess the risk of this portfolio:\n",
            "## Current Positions:",
            json.dumps(positions, indent=2, default=str),
            "\n## Portfolio Summary:",
            json.dumps(portfolio_summary, indent=2, default=str),
            "\n## Market Context:",
            json.dumps(market_context, indent=2, default=str),
        ]

        if recent_trades:
            parts.append("\n## Recent Trades:")
            parts.append(json.dumps(recent_trades, indent=2, default=str))

        parts.append("\nProvide your comprehensive risk assessment as JSON.")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> RiskAssessment:
        """Parse LLM's JSON response into RiskAssessment."""
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            return RiskAssessment(
                overall_risk_level=data.get("overall_risk_level", "MEDIUM"),
                risk_score=int(data.get("risk_score", 50)),
                concentration_risk=data.get("concentration_risk", ""),
                correlation_risk=data.get("correlation_risk", ""),
                market_risk=data.get("market_risk", ""),
                drawdown_risk=data.get("drawdown_risk", ""),
                warnings=data.get("warnings", []),
                recommendations=data.get("recommendations", []),
                summary=data.get("summary", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse risk assessment: %s", e)
            return RiskAssessment(summary=text[:500])

    def _parse_pre_trade_response(self, text: str, proposed_size: float) -> dict:
        """Parse LLM's JSON response for pre-trade check."""
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            return {
                "approved": data.get("approved", True),
                "risk_level": data.get("risk_level", "MEDIUM"),
                "reason": data.get("reason", ""),
                "suggested_size": float(data.get("suggested_size", proposed_size)),
                "warnings": data.get("warnings", []),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse pre-trade response: %s", e)
            return {
                "approved": True,
                "risk_level": "MEDIUM",
                "reason": f"Parse error: {e}",
                "suggested_size": proposed_size,
                "warnings": ["Could not parse AI response"],
            }
