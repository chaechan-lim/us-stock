"""AI Trade Review Agent — reviews executed trades for learning.

Post-trade analysis using LLMClient to evaluate trade quality,
identify patterns, and suggest improvements for future trades.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.agent_context import AgentContextService
    from services.llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional trade review analyst for US equities.
You review executed trades to assess quality, identify mistakes, and extract
actionable lessons for improving future trading performance.

Your review framework:
1. **Timing Assessment**: Was the entry/exit timed well relative to price action?
2. **Strategy Adherence**: Did the trade follow the stated strategy rules?
3. **Risk Management**: Was position sizing and stop-loss appropriate?
4. **Market Context**: Did the trade align with broader market conditions?
5. **Entry Quality**: Was the entry near support, after confirmation, etc.?
6. **Exit Quality**: Was the exit at resistance, trailing stop, or panic sell?

Grading scale:
- A (90-100): Excellent execution, textbook trade
- B (75-89): Good trade with minor room for improvement
- C (60-74): Fair trade, notable issues but acceptable outcome
- D (40-59): Poor trade, significant execution problems
- F (0-39): Failed trade, major errors in judgment or execution

Output your analysis as JSON with this exact structure:
{
  "grade": "A" | "B" | "C" | "D" | "F",
  "score": 0-100,
  "timing_assessment": "excellent" | "good" | "fair" | "poor",
  "entry_quality": "description of entry point quality",
  "exit_quality": "description of exit point quality",
  "lessons": ["lesson1", "lesson2"],
  "improvements": ["improvement1", "improvement2"],
  "summary": "One paragraph summary of the trade review"
}"""

DAILY_REVIEW_PROMPT = """당신은 듀얼 마켓(US + KR) 자동 거래 시스템의 일일 거래를
리뷰합니다. 활성 전략은 dual_momentum, supertrend, trend_following이며
시장별로 다르게 enable됩니다. 시스템은 ATR 기반 SL/TP, default trailing
stop (US 6% / KR 8% activation), tiered trailing (5/10/15/20% gain tier)을
사용합니다.

운영자가 이번 주 안에 행동에 옮길 수 있는 **구체적 패턴**만 찾으세요.
"전략 점검", "리스크 관리" 같은 일반론은 쓰지 마세요. 다음 항목을 살펴봐:

1. **Whipsaw / cross-strategy 충돌** — 한 전략이 매수한 종목을 24시간 안에
   다른 전략이 매도, 또는 trailing이 진입가 근처에서 짧은 peak 후 exit한 경우.
2. **Cap binding** — Max positions / Price too high for allocation /
   Max exposure / daily_buy_limit 등으로 BUY가 자주 reject됐는지 (시장별
   파라미터가 너무 빡빡하거나 stale한 신호).
3. **전략별 imbalance** — 한 전략이 PnL 압도(±), 저WR 전략, 신호 안 내는 전략.
4. **사이징 이상** — 비싼 종목에 qty=1 (allocation < 1주 가격), 또는 잡주 1주씩
   누적되는 케이스.
5. **Exit 타이밍** — 평균 peak gain vs exit gain (trailing이 너무 일찍 발동?
   take_profit이 너무 일찍?).

JSON 출력 (모든 텍스트 필드는 한국어로):
{
  "overall_grade": "A"|"B"|"C"|"D"|"F",
  "overall_score": 0-100,
  "total_trades": <int>,
  "best_trade": "<종목 — 좋았던 이유 (한국어)>",
  "worst_trade": "<종목 — 잘못된 이유 (한국어)>",
  "patterns_identified": [
    "<오늘 거래에 매인 구체적 관찰 (한국어). 예: 'CRML 라운드트립: 13:30 supertrend BUY → 14:21 trailing SELL, 순익 $2'>",
    "..."
  ],
  "daily_lessons": [
    "<config/code 노브에 매인 행동 가능 교훈 (한국어). 예: '오픈 30분 이내 매수 4건 모두 적자 — opening_avoidance 더 길게'>",
    "..."
  ],
  "recommendations": [
    "<구체적인 config/code 변경 제안 + 근거 (한국어). 예: 'KR sell_cooldown 1→2일로 늘려 005935 dm-supertrend 핑퐁 차단'>",
    "..."
  ],
  "summary": "<오늘 거래 한 단락 요약 + 가장 우선 행동할 패턴 (한국어)>"
}

구체적 패턴이 없으면 빈 array를 반환. 일반론보다 침묵이 낫습니다."""


@dataclass
class TradeReview:
    symbol: str
    trade_date: str
    side: str  # "buy" or "sell"
    grade: str = "C"  # "A", "B", "C", "D", "F"
    score: int = 50  # 0-100
    timing_assessment: str = "fair"  # "excellent", "good", "fair", "poor"
    entry_quality: str = ""  # assessment of entry point
    exit_quality: str = ""  # assessment of exit point (for sells)
    lessons: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    summary: str = ""


AGENT_TYPE = "trade_review"


class TradeReviewAgent:
    """AI agent for post-trade review and learning using LLMClient."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        context_service: AgentContextService | None = None,
    ):
        self._llm_client = llm_client
        self._ctx = context_service

    async def review_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float | None,
        quantity: int,
        strategy_name: str,
        pnl: float,
        holding_days: int,
        market_context: dict,
        indicator_data: dict,
    ) -> TradeReview:
        """Review a single trade execution quality."""
        from datetime import date

        trade_date = date.today().isoformat()

        if not self._llm_client:
            logger.warning("No LLM client configured, returning default trade review")
            return TradeReview(symbol=symbol, trade_date=trade_date, side=side)

        # Load past lessons for this symbol/strategy
        memory_context = ""
        if self._ctx:
            try:
                memory_context = await self._ctx.build_context(
                    AGENT_TYPE, symbol=symbol, max_tokens=1000,
                )
            except Exception as e:
                logger.debug("Failed to load agent context: %s", e)

        user_prompt = self._build_prompt(
            symbol, side, entry_price, exit_price, quantity,
            strategy_name, pnl, holding_days, market_context, indicator_data,
            memory_context,
        )

        try:
            response = await self._llm_client.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=SYSTEM_PROMPT,
                max_tokens=1024,
            )
            result = self._parse_response(symbol, side, response.text or "")

            # Save lessons to memory
            if self._ctx and result.lessons:
                try:
                    importance = 6 if result.grade in ("A", "B") else 7
                    lesson_text = (
                        f"[{strategy_name}] {side} {symbol} grade={result.grade}: "
                        + "; ".join(result.lessons[:3])
                    )
                    await self._ctx.save(
                        AGENT_TYPE, "lesson", symbol, lesson_text[:300],
                        importance=importance, ttl_days=30,
                    )
                except Exception as e:
                    logger.debug("Failed to save trade lesson: %s", e)

            return result

        except Exception as e:
            logger.error("Trade review failed for %s: %s", symbol, e)
            return TradeReview(symbol=symbol, trade_date=trade_date, side=side)

    async def review_daily_trades(
        self, trades: list[dict], portfolio_summary: dict
    ) -> dict:
        """Review all trades from today and return summary."""
        default_result = {
            "overall_grade": "C",
            "overall_score": 50,
            "total_trades": len(trades),
            "best_trade": "",
            "worst_trade": "",
            "patterns_identified": [],
            "daily_lessons": [],
            "recommendations": [],
            "summary": "",
        }

        if not self._llm_client:
            logger.warning("No LLM client configured, returning default daily review")
            return default_result

        if not trades:
            return default_result

        user_prompt = f"""Review today's trades:

## Trades:
{json.dumps(trades, indent=2, default=str)}

## Portfolio Summary:
{json.dumps(portfolio_summary, indent=2, default=str)}

Provide your daily trade review as JSON."""

        try:
            response = await self._llm_client.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=DAILY_REVIEW_PROMPT,
                max_tokens=1024,
            )
            return self._parse_daily_response(response.text or "", len(trades))

        except Exception as e:
            logger.error("Daily trade review failed: %s", e)
            return default_result

    def _build_prompt(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float | None,
        quantity: int,
        strategy_name: str,
        pnl: float,
        holding_days: int,
        market_context: dict,
        indicator_data: dict,
        memory_context: str = "",
    ) -> str:
        exit_str = f"${exit_price:.2f}" if exit_price is not None else "N/A (open)"
        pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price * quantity else 0

        parts = [f"""Review this {side.upper()} trade for {symbol}:

## Trade Details:
- Side: {side}
- Entry Price: ${entry_price:.2f}
- Exit Price: {exit_str}
- Quantity: {quantity} shares
- Strategy: {strategy_name}
- P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)
- Holding Period: {holding_days} days

## Market Context:
{json.dumps(market_context, indent=2, default=str)}

## Technical Indicators at Trade Time:
{json.dumps(indicator_data, indent=2, default=str)}"""]

        if memory_context:
            parts.append(f"\n{memory_context}")

        parts.append("\nProvide your trade review as JSON.")
        return "\n".join(parts)

    def _parse_response(self, symbol: str, side: str, text: str) -> TradeReview:
        """Parse LLM's JSON response into TradeReview."""
        from datetime import date

        trade_date = date.today().isoformat()

        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            return TradeReview(
                symbol=symbol,
                trade_date=trade_date,
                side=side,
                grade=data.get("grade", "C"),
                score=int(data.get("score", 50)),
                timing_assessment=data.get("timing_assessment", "fair"),
                entry_quality=data.get("entry_quality", ""),
                exit_quality=data.get("exit_quality", ""),
                lessons=data.get("lessons", []),
                improvements=data.get("improvements", []),
                summary=data.get("summary", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse trade review for %s: %s | text=%s", symbol, e, text[:300])
            return TradeReview(
                symbol=symbol, trade_date=trade_date, side=side, summary=text[:500],
            )

    def _parse_daily_response(self, text: str, trade_count: int) -> dict:
        """Parse LLM's JSON response for daily review."""
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            return {
                "overall_grade": data.get("overall_grade", "C"),
                "overall_score": int(data.get("overall_score", 50)),
                "total_trades": data.get("total_trades", trade_count),
                "best_trade": data.get("best_trade", ""),
                "worst_trade": data.get("worst_trade", ""),
                "patterns_identified": data.get("patterns_identified", []),
                "daily_lessons": data.get("daily_lessons", []),
                "recommendations": data.get("recommendations", []),
                "summary": data.get("summary", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse daily review: %s | text=%s", e, text[:300])
            return {
                "overall_grade": "C",
                "overall_score": 50,
                "total_trades": trade_count,
                "best_trade": "",
                "worst_trade": "",
                "patterns_identified": [],
                "daily_lessons": [],
                "recommendations": [],
                "summary": text[:500],
            }
