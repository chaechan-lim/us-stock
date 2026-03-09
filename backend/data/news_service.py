"""Finnhub News Service.

Fetches company and market news from Finnhub API for sentiment analysis.
Uses aiohttp for async HTTP calls (already a project dependency).

Finnhub free tier: 60 requests/min — sufficient for hourly batch scans.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


@dataclass
class NewsArticle:
    """Parsed news article from Finnhub."""
    headline: str
    summary: str
    source: str
    symbol: str  # related ticker
    published_at: datetime
    url: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "symbol": self.symbol,
            "published_at": self.published_at.isoformat(),
            "category": self.category,
        }


@dataclass
class NewsBatch:
    """Collection of news articles for a batch analysis."""
    articles: list[NewsArticle] = field(default_factory=list)
    fetched_at: str = ""

    @property
    def symbols_covered(self) -> set[str]:
        return {a.symbol for a in self.articles}

    def for_symbol(self, symbol: str) -> list[NewsArticle]:
        return [a for a in self.articles if a.symbol == symbol]

    def for_sector(self, sector_keywords: list[str]) -> list[NewsArticle]:
        """Filter articles mentioning sector-related keywords."""
        kw_lower = [k.lower() for k in sector_keywords]
        return [
            a for a in self.articles
            if any(
                k in a.headline.lower() or k in a.summary.lower()
                for k in kw_lower
            )
        ]


# Sector keyword mapping for sector-level news filtering
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Technology": ["tech", "software", "semiconductor", "ai", "cloud", "chip"],
    "Financials": ["bank", "financial", "insurance", "lending", "fintech"],
    "Energy": ["oil", "gas", "energy", "crude", "opec", "solar", "wind"],
    "Healthcare": ["pharma", "biotech", "drug", "fda", "healthcare", "medical"],
    "Consumer_Disc": ["retail", "consumer", "e-commerce", "luxury", "auto"],
    "Consumer_Staples": ["food", "beverage", "grocery", "consumer staple"],
    "Industrials": ["manufacturing", "industrial", "defense", "aerospace"],
    "Materials": ["mining", "steel", "chemical", "commodity", "metal"],
    "Utilities": ["utility", "electric", "power", "water", "nuclear"],
    "Real_Estate": ["real estate", "reit", "housing", "mortgage", "property"],
    "Communications": ["media", "telecom", "streaming", "social media", "advertising"],
}


class FinnhubNewsService:
    """Async Finnhub news client."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_company_news(
        self,
        symbol: str,
        days_back: int = 3,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Fetch recent news for a specific company.

        Args:
            symbol: Stock ticker (e.g. 'AAPL')
            days_back: How many days of news to fetch
            max_articles: Max articles to return

        Returns:
            List of NewsArticle sorted by date descending
        """
        if not self.available:
            return []

        today = datetime.now()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        try:
            session = await self._get_session()
            url = f"{FINNHUB_BASE_URL}/company-news"
            params = {
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": self._api_key,
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Finnhub company news %s: HTTP %d", symbol, resp.status,
                    )
                    return []

                data = await resp.json()
                if not isinstance(data, list):
                    return []

                articles = []
                for item in data[:max_articles]:
                    try:
                        articles.append(NewsArticle(
                            headline=item.get("headline", ""),
                            summary=item.get("summary", "")[:500],
                            source=item.get("source", ""),
                            symbol=symbol,
                            published_at=datetime.fromtimestamp(
                                item.get("datetime", 0),
                            ),
                            url=item.get("url", ""),
                            category=item.get("category", ""),
                        ))
                    except (ValueError, TypeError):
                        continue

                return articles

        except Exception as e:
            logger.warning("Finnhub company news failed for %s: %s", symbol, e)
            return []

    async def fetch_market_news(
        self, category: str = "general", max_articles: int = 20,
    ) -> list[NewsArticle]:
        """Fetch general market news.

        Args:
            category: 'general', 'forex', 'crypto', 'merger'
            max_articles: Max articles to return
        """
        if not self.available:
            return []

        try:
            session = await self._get_session()
            url = f"{FINNHUB_BASE_URL}/news"
            params = {"category": category, "token": self._api_key}

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Finnhub market news: HTTP %d", resp.status)
                    return []

                data = await resp.json()
                if not isinstance(data, list):
                    return []

                articles = []
                for item in data[:max_articles]:
                    try:
                        related = item.get("related", "")
                        articles.append(NewsArticle(
                            headline=item.get("headline", ""),
                            summary=item.get("summary", "")[:500],
                            source=item.get("source", ""),
                            symbol=related if related else "MARKET",
                            published_at=datetime.fromtimestamp(
                                item.get("datetime", 0),
                            ),
                            url=item.get("url", ""),
                            category=item.get("category", "general"),
                        ))
                    except (ValueError, TypeError):
                        continue

                return articles

        except Exception as e:
            logger.warning("Finnhub market news failed: %s", e)
            return []

    async def fetch_batch(
        self,
        symbols: list[str],
        days_back: int = 3,
        max_per_symbol: int = 5,
    ) -> NewsBatch:
        """Fetch news for multiple symbols + market news in one batch.

        Respects Finnhub rate limit (60/min) by keeping requests sequential.
        For a typical watchlist of 30-50 symbols, this takes ~30-50 seconds.

        Args:
            symbols: List of tickers to fetch news for
            days_back: Days of news history
            max_per_symbol: Max articles per symbol

        Returns:
            NewsBatch with all articles
        """
        import asyncio

        all_articles: list[NewsArticle] = []

        # Market-wide news first
        market_articles = await self.fetch_market_news(max_articles=15)
        all_articles.extend(market_articles)

        # Company news — sequential to respect rate limits
        for symbol in symbols:
            articles = await self.fetch_company_news(
                symbol, days_back=days_back, max_articles=max_per_symbol,
            )
            all_articles.extend(articles)
            # Brief pause to stay within rate limits
            await asyncio.sleep(0.2)

        logger.info(
            "Finnhub batch: %d articles for %d symbols + market",
            len(all_articles), len(symbols),
        )

        return NewsBatch(
            articles=all_articles,
            fetched_at=datetime.now().isoformat(),
        )
