"""Scanner Pipeline Orchestrator.

Chains the 4 screening layers into a single pipeline:
  Layer 1: IndicatorScreener   (technical scoring, yfinance data)
  Layer 2: FundamentalEnricher  (yfinance fundamentals)
  Layer 2.5: NewsEnricher       (Finnhub news sentiment, optional)
  Layer 3: MarketAnalystAgent   (Claude AI analysis, optional)

Uses yfinance for bulk screening (no rate limits).
KIS API is reserved for order execution and real-time quotes only.
"""

import logging
from dataclasses import asdict

import yfinance as yf
import pandas as pd

from data.market_data_service import MarketDataService
from data.indicator_service import IndicatorService
from scanner.indicator_screener import IndicatorScreener
from scanner.fundamental_enricher import FundamentalEnricher
from scanner.news_enricher import NewsEnricher
from agents.market_analyst import MarketAnalystAgent
from agents.news_sentiment_agent import NewsSentimentSummary

logger = logging.getLogger(__name__)


def _fetch_yfinance_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV from yfinance (no rate limit)."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        # Ensure standard column names
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                return pd.DataFrame()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.debug("yfinance fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()


class ScannerPipeline:
    """Orchestrate the 3-layer stock screening pipeline."""

    def __init__(
        self,
        market_data: MarketDataService,
        indicator_svc: IndicatorService,
        enricher: FundamentalEnricher,
        ai_agent: MarketAnalystAgent | None = None,
        news_enricher: NewsEnricher | None = None,
    ):
        self._market_data = market_data
        self._indicator_svc = indicator_svc
        self._enricher = enricher
        self._ai_agent = ai_agent
        self._news_enricher = news_enricher
        self._screener = IndicatorScreener()
        self._last_news_summary: NewsSentimentSummary | None = None

    def set_news_summary(self, summary: NewsSentimentSummary) -> None:
        """Set pre-fetched news sentiment for use in next scan."""
        self._last_news_summary = summary

    async def run_full_scan(
        self,
        symbols: list[str],
        min_grade: str = "B",
        max_candidates: int = 20,
        news_summary: NewsSentimentSummary | None = None,
    ) -> list[dict]:
        """Run all layers and return ranked candidates.

        Layer 1 uses yfinance for bulk data (no rate limits).
        Layer 2 uses yfinance for fundamentals.
        Layer 2.5 applies news sentiment (optional).
        KIS API is not used during scanning.

        Args:
            symbols: List of stock tickers to scan.
            min_grade: Minimum grade to pass Layer 1 (default "B").
            max_candidates: Max results to return from Layer 2.
            news_summary: Pre-computed news sentiment (optional).

        Returns:
            List of candidate dicts sorted by combined score descending.
        """
        if not symbols:
            logger.info("Scanner pipeline: no symbols provided")
            return []

        # Layer 1: Score all symbols with IndicatorScreener (yfinance data)
        logger.info("Layer 1: Screening %d symbols (yfinance)", len(symbols))
        screener_scores = []
        last_prices: dict[str, float] = {}  # Cache prices from Layer 1
        for symbol in symbols:
            try:
                df = _fetch_yfinance_ohlcv(symbol, period="1y")
                if df.empty or len(df) < 50:
                    continue
                last_prices[symbol] = float(df.iloc[-1]["close"])
                df = self._indicator_svc.add_all_indicators(df)
                score = self._screener.score(df, symbol)
                screener_scores.append(score)
            except Exception as e:
                logger.warning("Layer 1 failed for %s: %s", symbol, e)

        # Filter by grade (pass as parameter to avoid mutating screener state)
        filtered = self._screener.filter_candidates(
            screener_scores, max_candidates=max_candidates, min_grade=min_grade,
        )
        logger.info(
            "Layer 1 complete: %d/%d passed (min_grade=%s)",
            len(filtered), len(screener_scores), min_grade,
        )

        if not filtered:
            return []

        # Layer 2: Enrich top candidates with FundamentalEnricher (yfinance)
        logger.info("Layer 2: Enriching %d candidates", len(filtered))
        enrich_input = [
            (s.symbol, s.total_score, last_prices.get(s.symbol, 0.0))
            for s in filtered
        ]

        enriched = await self._enricher.enrich_batch(enrich_input)
        logger.info("Layer 2 complete: %d enriched candidates", len(enriched))

        # Build results
        results = []
        for candidate in enriched:
            result = {
                "symbol": candidate.symbol,
                "indicator_score": candidate.indicator_score,
                "consensus_score": candidate.consensus_score,
                "fundamental_score": candidate.fundamental_score,
                "smart_money_score": candidate.smart_money_score,
                "combined_score": candidate.combined_score,
                "grade": candidate.grade,
            }
            results.append(result)

        # Layer 2.5: (optional) News sentiment enrichment
        active_summary = news_summary or self._last_news_summary
        if self._news_enricher and active_summary:
            logger.info("Layer 2.5: News sentiment enrichment")
            results = self._news_enricher.enrich(results, active_summary)
            logger.info("Layer 2.5 complete")

        # Layer 3: (optional) AI analysis on top 5
        if self._ai_agent and results:
            # Build market context from SPY if available
            market_ctx = {}
            try:
                spy_df = _fetch_yfinance_ohlcv("SPY", period="5d")
                if not spy_df.empty:
                    spy_price = float(spy_df.iloc[-1]["close"])
                    spy_change = (spy_price / float(spy_df.iloc[-2]["close"]) - 1) * 100
                    market_ctx = {
                        "spy_price": spy_price,
                        "spy_1d_change": round(spy_change, 2),
                    }
            except Exception as e:
                logger.warning("SPY market context fetch failed: %s", e)

            top_n = min(3, len(results))
            logger.info("Layer 3: AI analysis on top %d candidates", top_n)
            for result in results[:top_n]:
                try:
                    recommendation = await self._ai_agent.analyze(
                        symbol=result["symbol"],
                        indicator_score=result["indicator_score"],
                        fundamental_data={
                            "consensus_score": result["consensus_score"],
                            "fundamental_score": result["fundamental_score"],
                            "smart_money_score": result["smart_money_score"],
                        },
                        market_context=market_ctx,
                        current_price=last_prices.get(result["symbol"], 0.0),
                    )
                    result["ai_recommendation"] = recommendation.recommendation
                    result["ai_score"] = recommendation.score
                    result["ai_conviction"] = recommendation.conviction
                    result["ai_summary"] = recommendation.summary
                except Exception as e:
                    logger.warning("Layer 3 AI analysis failed for %s: %s", result["symbol"], e)
            logger.info("Layer 3 complete")

        # Sort by combined_score descending
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results[:max_candidates]
