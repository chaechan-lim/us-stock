"""Naver Finance News Service for Korean stocks.

Fetches company news from Naver Finance mobile API.
No API key required — public endpoints with reasonable rate limits.

Used as the KR equivalent of Finnhub for news sentiment analysis.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import aiohttp

from data.news_service import NewsArticle, NewsBatch

logger = logging.getLogger(__name__)

NAVER_STOCK_NEWS_URL = "https://m.stock.naver.com/api/news/stock/{code}"
NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://m.stock.naver.com/",
}


class NaverNewsService:
    """Async Naver Finance news client for KR stocks."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    @property
    def available(self) -> bool:
        return True  # No API key needed

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=NAVER_HEADERS,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_company_news(
        self,
        symbol: str,
        max_articles: int = 5,
    ) -> list[NewsArticle]:
        """Fetch recent news for a Korean stock from Naver Finance.

        Args:
            symbol: 6-digit KR stock code (e.g. '005930')
            max_articles: Max articles to return

        Returns:
            List of NewsArticle sorted by date descending
        """
        try:
            session = await self._get_session()
            url = NAVER_STOCK_NEWS_URL.format(code=symbol)
            params = {"pageSize": max_articles, "page": 1}

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Naver news %s: HTTP %d", symbol, resp.status,
                    )
                    return []

                data = await resp.json()

                # Naver API response structure: {"items": [...]}
                items = data if isinstance(data, list) else data.get("items", [])
                if not items:
                    return []

                articles = []
                for item in items[:max_articles]:
                    try:
                        # Parse datetime from various Naver formats
                        pub_date = self._parse_date(
                            item.get("datetime", "")
                            or item.get("date", "")
                            or item.get("publishedAt", "")
                        )
                        title = (
                            item.get("title", "")
                            .replace("<b>", "").replace("</b>", "")
                            .replace("&quot;", '"').replace("&amp;", "&")
                        )
                        body = (
                            item.get("body", "")
                            or item.get("summary", "")
                            or item.get("description", "")
                        )[:500]

                        articles.append(NewsArticle(
                            headline=title,
                            summary=body,
                            source=item.get("officeName", "Naver"),
                            symbol=symbol,
                            published_at=pub_date,
                            url=item.get("url", ""),
                            category="kr_stock",
                        ))
                    except (ValueError, TypeError):
                        continue

                return articles

        except Exception as e:
            logger.warning("Naver news failed for %s: %s", symbol, e)
            return []

    async def fetch_batch(
        self,
        symbols: list[str],
        max_per_symbol: int = 5,
    ) -> NewsBatch:
        """Fetch news for multiple KR symbols.

        Sequential requests with brief pauses to be polite to Naver.

        Args:
            symbols: List of 6-digit KR stock codes
            max_per_symbol: Max articles per symbol

        Returns:
            NewsBatch with all articles
        """
        all_articles: list[NewsArticle] = []

        for symbol in symbols:
            articles = await self.fetch_company_news(
                symbol, max_articles=max_per_symbol,
            )
            all_articles.extend(articles)
            await asyncio.sleep(0.3)  # Rate-limit courtesy

        logger.info(
            "Naver news batch: %d articles for %d KR symbols",
            len(all_articles), len(symbols),
        )

        return NewsBatch(
            articles=all_articles,
            fetched_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse various Naver date formats."""
        if not date_str:
            return datetime.now()

        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",      # ISO with TZ
            "%Y-%m-%dT%H:%M:%S",         # ISO without TZ
            "%Y-%m-%d %H:%M:%S",         # Standard datetime
            "%Y-%m-%d %H:%M",            # Without seconds
            "%Y.%m.%d %H:%M",            # Korean style
            "%Y.%m.%d",                   # Date only
            "%Y-%m-%d",                   # Date only ISO
        ):
            try:
                return datetime.strptime(date_str[:len(fmt) + 5], fmt)
            except (ValueError, IndexError):
                continue

        return datetime.now()
