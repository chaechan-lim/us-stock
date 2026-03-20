"""News Sentiment Analysis Agent.

Uses LLM (Claude/Gemini) to analyze financial news sentiment
and extract actionable trading signals per symbol and sector.

Designed as a batch processor: analyzes multiple news articles at once
to minimize LLM calls and cost.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.agent_context import AgentContextService
    from services.llm import LLMClient
    from data.news_service import NewsArticle

logger = logging.getLogger(__name__)

AGENT_TYPE = "news_sentiment"

SYSTEM_PROMPT = """You are a financial news sentiment analyst.
You analyze batches of news headlines and summaries to extract:
1. Per-stock sentiment and potential price impact
2. Sector-level themes and rotation signals
3. Market-wide sentiment and risk events

For each analyzed item, produce a JSON array of sentiment assessments.

Output format — a JSON array:
[
  {
    "symbol": "AAPL",
    "sentiment": -1.0 to 1.0 (negative to positive),
    "impact": "HIGH" | "MEDIUM" | "LOW",
    "category": "earnings" | "product" | "legal" | "macro" | "analyst" | "sector" | "m_and_a" | "other",
    "sector_impact": ["Technology"],
    "key_event": "Brief description of the key event",
    "trading_signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    "time_sensitivity": "IMMEDIATE" | "SHORT_TERM" | "LONG_TERM"
  }
]

Rules:
- Sentiment ranges: -1.0 (very negative) to 1.0 (very positive)
- HIGH impact = likely >2% price move, MEDIUM = 0.5-2%, LOW = <0.5%
- Consolidate multiple articles about the same event into one assessment
- If market-wide news, use symbol "MARKET"
- Focus on actionable information, not noise
- Be skeptical of analyst upgrades/downgrades — they often lag price action"""


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single symbol or event."""
    symbol: str
    sentiment: float  # -1.0 to 1.0
    impact: str  # HIGH, MEDIUM, LOW
    category: str
    sector_impact: list[str] = field(default_factory=list)
    key_event: str = ""
    trading_signal: str = "NEUTRAL"
    time_sensitivity: str = "SHORT_TERM"

    @property
    def is_actionable(self) -> bool:
        """Returns True if this sentiment is strong enough to act on."""
        return (
            self.impact in ("HIGH", "MEDIUM")
            and abs(self.sentiment) >= 0.3
            and self.trading_signal != "NEUTRAL"
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sentiment": self.sentiment,
            "impact": self.impact,
            "category": self.category,
            "sector_impact": self.sector_impact,
            "key_event": self.key_event,
            "trading_signal": self.trading_signal,
            "time_sensitivity": self.time_sensitivity,
            "is_actionable": self.is_actionable,
        }


@dataclass
class NewsSentimentSummary:
    """Aggregated sentiment summary for scanner integration."""
    symbol_sentiments: dict[str, float] = field(default_factory=dict)
    sector_sentiments: dict[str, float] = field(default_factory=dict)
    market_sentiment: float = 0.0
    actionable_signals: list[SentimentResult] = field(default_factory=list)
    analyzed_count: int = 0

    def get_symbol_score(self, symbol: str) -> float:
        """Get sentiment score for a symbol (-1.0 to 1.0), default 0."""
        return self.symbol_sentiments.get(symbol, 0.0)

    def get_sector_score(self, sector: str) -> float:
        """Get sentiment score for a sector (-1.0 to 1.0), default 0."""
        return self.sector_sentiments.get(sector, 0.0)

    def to_dict(self) -> dict:
        return {
            "symbol_sentiments": self.symbol_sentiments,
            "sector_sentiments": self.sector_sentiments,
            "market_sentiment": self.market_sentiment,
            "actionable_count": len(self.actionable_signals),
            "analyzed_count": self.analyzed_count,
        }


class NewsSentimentAgent:
    """LLM-based news sentiment analysis agent."""

    # Default articles per LLM call (25 reduces calls by ~60% vs old default of 10)
    DEFAULT_BATCH_SIZE = 25

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        context_service: AgentContextService | None = None,
        model_override: str | None = None,
        batch_size: int | None = None,
    ):
        self._llm_client = llm_client
        self._ctx = context_service
        self._model_override = model_override  # e.g. "gemini-2.5-flash"
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE

    async def analyze_batch(
        self, articles: list[NewsArticle],
        symbol_names: dict[str, str] | None = None,
    ) -> NewsSentimentSummary:
        """Analyze a batch of news articles using LLM.

        Processes articles in chunks to stay within token limits.
        Returns aggregated sentiment summary.

        Args:
            articles: News articles to analyze
            symbol_names: Optional symbol->name mapping for context
        """
        if not self._llm_client or not articles:
            return NewsSentimentSummary()

        # Chunk articles to stay within token limits (configurable, default 25)
        chunk_size = self._batch_size
        all_results: list[SentimentResult] = []

        for i in range(0, len(articles), chunk_size):
            chunk = articles[i:i + chunk_size]
            results = await self._analyze_chunk(chunk, symbol_names)
            all_results.extend(results)

        # Aggregate into summary
        summary = self._aggregate(all_results, len(articles))

        # Save key insights to memory
        if self._ctx:
            await self._save_insights(summary, all_results)

        return summary

    async def _analyze_chunk(
        self, articles: list[NewsArticle],
        symbol_names: dict[str, str] | None = None,
    ) -> list[SentimentResult]:
        """Analyze a chunk of articles via LLM call."""
        if not self._llm_client:
            return []

        # Build memory context
        memory_context = ""
        if self._ctx:
            try:
                memory_context = await self._ctx.build_context(
                    AGENT_TYPE, max_tokens=500,
                )
            except Exception as e:
                logger.warning("Failed to build agent memory context: %s", e)

        prompt = self._build_prompt(articles, memory_context, symbol_names)

        try:
            response = await self._llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                system=SYSTEM_PROMPT,
                max_tokens=4096,
                model=self._model_override,
            )
            return self._parse_response(response.text or "")

        except Exception as e:
            logger.error("News sentiment LLM call failed: %s", e)
            return []

    def _build_prompt(
        self,
        articles: list[NewsArticle],
        memory_context: str = "",
        symbol_names: dict[str, str] | None = None,
    ) -> str:
        parts = [
            f"Analyze the following {len(articles)} financial news articles "
            "and provide sentiment assessments.\n",
        ]

        # If KR stocks, provide symbol-name mapping for context
        if symbol_names:
            names_str = ", ".join(
                f"{sym}={name}" for sym, name in list(symbol_names.items())[:40]
            )
            parts.append(
                f"**Symbol reference (code=company):** {names_str}\n"
                "Use the numeric stock codes (e.g. 005930) as the symbol field "
                "in your response, NOT the company names.\n"
            )

        for i, article in enumerate(articles, 1):
            label = article.symbol
            if symbol_names and article.symbol in symbol_names:
                label = f"{article.symbol} ({symbol_names[article.symbol]})"
            parts.append(
                f"### Article {i} [{label}] "
                f"({article.source}, {article.published_at.strftime('%Y-%m-%d')})\n"
                f"**Headline:** {article.headline}\n"
                f"**Summary:** {article.summary[:300]}\n"
            )

        if memory_context:
            parts.append(f"\n{memory_context}")

        parts.append(
            "\nProvide your sentiment analysis as a JSON array. "
            "You MUST produce one entry per stock symbol that has news. "
            "Consolidate duplicate events. Focus on actionable signals."
        )
        return "\n".join(parts)

    @staticmethod
    def _try_parse_json(text: str) -> list | None:
        """Try to parse JSON, recovering from truncated output."""
        text = text.strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass

        # Try recovering truncated JSON array: find last complete object
        if text.startswith("["):
            # Find last '}' and close the array
            last_brace = text.rfind("}")
            if last_brace > 0:
                truncated = text[:last_brace + 1] + "]"
                try:
                    data = json.loads(truncated)
                    if isinstance(data, list):
                        logger.debug("Recovered %d items from truncated JSON", len(data))
                        return data
                except json.JSONDecodeError:
                    pass

        return None

    def _parse_response(self, text: str) -> list[SentimentResult]:
        """Parse LLM JSON array response into SentimentResult list."""
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = self._try_parse_json(json_str)
            if data is None:
                logger.warning("Failed to parse news sentiment response | text=%s", json_str[:300])
                return []

            results = []
            for item in data:
                try:
                    sentiment = float(item.get("sentiment", 0))
                    sentiment = max(-1.0, min(1.0, sentiment))

                    results.append(SentimentResult(
                        symbol=item.get("symbol", "UNKNOWN"),
                        sentiment=sentiment,
                        impact=item.get("impact", "LOW"),
                        category=item.get("category", "other"),
                        sector_impact=item.get("sector_impact") or [],
                        key_event=item.get("key_event", ""),
                        trading_signal=item.get("trading_signal", "NEUTRAL"),
                        time_sensitivity=item.get("time_sensitivity", "SHORT_TERM"),
                    ))
                except (ValueError, TypeError) as e:
                    logger.debug("Skipping malformed sentiment item: %s", e)

            return results

        except (IndexError, KeyError) as e:
            logger.warning("Failed to parse news sentiment response: %s", e)
            return []

    def _aggregate(
        self, results: list[SentimentResult], total_articles: int,
    ) -> NewsSentimentSummary:
        """Aggregate individual results into a summary."""
        summary = NewsSentimentSummary(analyzed_count=total_articles)

        # Per-symbol sentiment (weighted average by impact)
        symbol_scores: dict[str, list[tuple[float, float]]] = {}
        sector_scores: dict[str, list[float]] = {}
        market_scores: list[float] = []

        impact_weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

        for r in results:
            weight = impact_weights.get(r.impact, 1.0)

            if r.symbol == "MARKET":
                market_scores.append(r.sentiment)
            else:
                if r.symbol not in symbol_scores:
                    symbol_scores[r.symbol] = []
                symbol_scores[r.symbol].append((r.sentiment, weight))

            # Sector accumulation
            for sector in r.sector_impact:
                if sector not in sector_scores:
                    sector_scores[sector] = []
                sector_scores[sector].append(r.sentiment)

            # Collect actionable signals
            if r.is_actionable:
                summary.actionable_signals.append(r)

        # Compute weighted averages
        for symbol, scores_weights in symbol_scores.items():
            total_weight = sum(w for _, w in scores_weights)
            if total_weight > 0:
                weighted = sum(s * w for s, w in scores_weights) / total_weight
                summary.symbol_sentiments[symbol] = round(weighted, 3)

        for sector, scores in sector_scores.items():
            if scores:
                summary.sector_sentiments[sector] = round(
                    sum(scores) / len(scores), 3,
                )

        if market_scores:
            summary.market_sentiment = round(
                sum(market_scores) / len(market_scores), 3,
            )

        return summary

    async def _save_insights(
        self, summary: NewsSentimentSummary, results: list[SentimentResult],
    ) -> None:
        """Save key news insights to agent memory."""
        if not self._ctx:
            return

        try:
            # Save actionable signals (high importance)
            for signal in summary.actionable_signals[:5]:
                importance = 8 if signal.impact == "HIGH" else 6
                await self._ctx.save(
                    AGENT_TYPE, "symbol", signal.symbol,
                    f"[{signal.trading_signal}] {signal.key_event} "
                    f"(sentiment={signal.sentiment:+.2f}, impact={signal.impact})",
                    importance=importance,
                    ttl_days=3,
                )

            # Save sector-level summary
            for sector, score in summary.sector_sentiments.items():
                if abs(score) >= 0.3:
                    direction = "positive" if score > 0 else "negative"
                    await self._ctx.save(
                        AGENT_TYPE, "sector", sector,
                        f"News sentiment {direction} ({score:+.2f}) for {sector}",
                        importance=5,
                        ttl_days=3,
                    )

            # Save market sentiment
            if abs(summary.market_sentiment) >= 0.2:
                direction = "positive" if summary.market_sentiment > 0 else "negative"
                await self._ctx.save(
                    AGENT_TYPE, "market", None,
                    f"Overall market news sentiment {direction} "
                    f"({summary.market_sentiment:+.2f}), "
                    f"{len(summary.actionable_signals)} actionable signals",
                    importance=6,
                    ttl_days=2,
                )

        except Exception as e:
            logger.debug("Failed to save news insights: %s", e)
