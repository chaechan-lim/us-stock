"""News Enricher — integrates news sentiment into the scanner pipeline.

Sits between Layer 2 (Fundamental) and Layer 3 (AI) in the pipeline.
Adjusts candidate scores based on recent news sentiment:
  - Positive news for a stock → boost score
  - Negative news → penalize score
  - Sector-level news → adjust all stocks in that sector

The enricher is optional — if news service is unavailable, pipeline
continues with original scores unchanged.
"""

import logging
from dataclasses import dataclass

from agents.news_sentiment_agent import NewsSentimentSummary

logger = logging.getLogger(__name__)

# How much news sentiment can adjust the combined score (0-100 scale)
MAX_NEWS_ADJUSTMENT = 15  # ±15 points max
NEWS_WEIGHT = 0.15  # 15% weight in final blended score


@dataclass
class NewsEnrichedCandidate:
    """Candidate with news sentiment score appended."""
    symbol: str
    original_score: float
    news_sentiment: float  # -1.0 to 1.0
    news_impact: str  # HIGH, MEDIUM, LOW, NONE
    news_adjusted_score: float
    news_events: list[str]  # Key event descriptions


class NewsEnricher:
    """Enriches scanner candidates with news sentiment data."""

    def __init__(self, news_weight: float = NEWS_WEIGHT):
        self._news_weight = news_weight
        self._last_summary: NewsSentimentSummary | None = None

    @property
    def last_summary(self) -> NewsSentimentSummary | None:
        return self._last_summary

    def enrich(
        self,
        candidates: list[dict],
        sentiment_summary: NewsSentimentSummary,
    ) -> list[dict]:
        """Enrich candidates with news sentiment scores.

        Modifies candidates in-place (adds news_ fields) and returns them.

        Args:
            candidates: List of candidate dicts from pipeline Layer 2
            sentiment_summary: Pre-computed sentiment from NewsSentimentAgent

        Returns:
            Same candidates list with news fields added and scores adjusted
        """
        self._last_summary = sentiment_summary

        if not sentiment_summary or not sentiment_summary.analyzed_count:
            # No news data — pass through unchanged
            for c in candidates:
                c["news_sentiment"] = 0.0
                c["news_impact"] = "NONE"
                c["news_events"] = []
            return candidates

        for candidate in candidates:
            symbol = candidate["symbol"]
            original_score = candidate.get("combined_score", 50.0)

            # Get symbol-level sentiment
            symbol_sentiment = sentiment_summary.get_symbol_score(symbol)

            # Get sector-level sentiment (if we can determine sector)
            sector_sentiment = 0.0
            sector_count = 0
            for sector, score in sentiment_summary.sector_sentiments.items():
                # Sector sentiment affects all candidates slightly
                sector_sentiment += score
                sector_count += 1
            if sector_count > 0:
                sector_sentiment /= sector_count

            # Blend: 70% symbol-specific, 30% sector/market
            market_sentiment = sentiment_summary.market_sentiment
            if symbol_sentiment != 0:
                blended = (
                    0.70 * symbol_sentiment
                    + 0.20 * sector_sentiment
                    + 0.10 * market_sentiment
                )
            else:
                # No symbol-specific news — use sector/market more
                blended = (
                    0.50 * sector_sentiment
                    + 0.50 * market_sentiment
                )

            # Calculate score adjustment
            adjustment = blended * MAX_NEWS_ADJUSTMENT
            adjusted_score = max(0, min(100, original_score + adjustment))

            # Determine impact level
            impact = "NONE"
            if abs(symbol_sentiment) >= 0.5:
                impact = "HIGH"
            elif abs(symbol_sentiment) >= 0.2:
                impact = "MEDIUM"
            elif abs(symbol_sentiment) > 0:
                impact = "LOW"

            # Collect key events for this symbol
            events = []
            for signal in sentiment_summary.actionable_signals:
                if signal.symbol == symbol:
                    events.append(signal.key_event)

            # Apply weighted blend with original score
            # final = (1 - news_weight) * original + news_weight * adjusted
            final_score = (
                (1 - self._news_weight) * original_score
                + self._news_weight * adjusted_score
            )

            candidate["combined_score"] = round(final_score, 1)
            candidate["news_sentiment"] = round(blended, 3)
            candidate["news_impact"] = impact
            candidate["news_events"] = events[:3]  # Top 3 events

        # Re-sort by adjusted score
        candidates.sort(key=lambda c: c["combined_score"], reverse=True)

        enriched_count = sum(
            1 for c in candidates if c.get("news_impact", "NONE") != "NONE"
        )
        logger.info(
            "News enrichment: %d/%d candidates had news data "
            "(market_sentiment=%.2f)",
            enriched_count, len(candidates),
            sentiment_summary.market_sentiment,
        )

        return candidates
