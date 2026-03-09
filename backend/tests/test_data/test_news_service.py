"""Tests for Finnhub News Service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.news_service import (
    FinnhubNewsService,
    NewsArticle,
    NewsBatch,
    SECTOR_KEYWORDS,
)


class TestNewsArticle:
    def test_to_dict(self):
        article = NewsArticle(
            headline="AAPL beats earnings",
            summary="Apple reported strong Q3 results",
            source="Reuters",
            symbol="AAPL",
            published_at=datetime(2026, 3, 9, 10, 0),
        )
        d = article.to_dict()
        assert d["headline"] == "AAPL beats earnings"
        assert d["symbol"] == "AAPL"
        assert d["source"] == "Reuters"
        assert "2026-03-09" in d["published_at"]


class TestNewsBatch:
    def test_symbols_covered(self):
        batch = NewsBatch(articles=[
            NewsArticle("h1", "s1", "src", "AAPL", datetime.now()),
            NewsArticle("h2", "s2", "src", "MSFT", datetime.now()),
            NewsArticle("h3", "s3", "src", "AAPL", datetime.now()),
        ])
        assert batch.symbols_covered == {"AAPL", "MSFT"}

    def test_for_symbol(self):
        batch = NewsBatch(articles=[
            NewsArticle("h1", "s1", "src", "AAPL", datetime.now()),
            NewsArticle("h2", "s2", "src", "MSFT", datetime.now()),
            NewsArticle("h3", "s3", "src", "AAPL", datetime.now()),
        ])
        aapl_news = batch.for_symbol("AAPL")
        assert len(aapl_news) == 2

    def test_for_sector(self):
        batch = NewsBatch(articles=[
            NewsArticle("AI chip maker", "semiconductor growth", "src", "NVDA", datetime.now()),
            NewsArticle("Bank earnings", "financial results", "src", "JPM", datetime.now()),
        ])
        tech_news = batch.for_sector(["chip", "semiconductor"])
        assert len(tech_news) == 1
        assert tech_news[0].symbol == "NVDA"

    def test_for_sector_empty(self):
        batch = NewsBatch(articles=[
            NewsArticle("Weather report", "sunny day", "src", "X", datetime.now()),
        ])
        assert len(batch.for_sector(["chip"])) == 0


class TestFinnhubNewsService:
    def test_not_available_without_key(self):
        svc = FinnhubNewsService(api_key="")
        assert svc.available is False

    def test_available_with_key(self):
        svc = FinnhubNewsService(api_key="test_key")
        assert svc.available is True

    async def test_fetch_company_news_no_key(self):
        svc = FinnhubNewsService(api_key="")
        result = await svc.fetch_company_news("AAPL")
        assert result == []

    async def test_fetch_market_news_no_key(self):
        svc = FinnhubNewsService(api_key="")
        result = await svc.fetch_market_news()
        assert result == []

    async def test_fetch_company_news_success(self):
        svc = FinnhubNewsService(api_key="test")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {
                "headline": "AAPL beats Q3 estimates",
                "summary": "Apple reported strong earnings",
                "source": "Reuters",
                "datetime": 1709971200,
                "url": "https://example.com/1",
                "category": "company",
            },
            {
                "headline": "iPhone 16 pre-orders strong",
                "summary": "Record demand for new iPhone",
                "source": "CNBC",
                "datetime": 1709884800,
                "url": "https://example.com/2",
                "category": "company",
            },
        ])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        svc._session = mock_session

        articles = await svc.fetch_company_news("AAPL")
        assert len(articles) == 2
        assert articles[0].headline == "AAPL beats Q3 estimates"
        assert articles[0].symbol == "AAPL"

    async def test_fetch_company_news_http_error(self):
        svc = FinnhubNewsService(api_key="test")

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        svc._session = mock_session

        articles = await svc.fetch_company_news("AAPL")
        assert articles == []

    async def test_fetch_company_news_exception(self):
        svc = FinnhubNewsService(api_key="test")

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=Exception("Connection failed"))
        svc._session = mock_session

        articles = await svc.fetch_company_news("AAPL")
        assert articles == []

    async def test_fetch_market_news_success(self):
        svc = FinnhubNewsService(api_key="test")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {
                "headline": "Fed holds rates steady",
                "summary": "Federal Reserve maintains rate",
                "source": "AP",
                "datetime": 1709971200,
                "related": "",
                "category": "general",
            },
        ])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_response)
        svc._session = mock_session

        articles = await svc.fetch_market_news()
        assert len(articles) == 1
        assert articles[0].symbol == "MARKET"

    async def test_close(self):
        svc = FinnhubNewsService(api_key="test")
        mock_session = AsyncMock()
        mock_session.closed = False
        svc._session = mock_session

        await svc.close()
        mock_session.close.assert_called_once()

    async def test_close_no_session(self):
        svc = FinnhubNewsService(api_key="test")
        await svc.close()  # Should not raise


class TestSectorKeywords:
    def test_all_sectors_have_keywords(self):
        expected = {
            "Technology", "Financials", "Energy", "Healthcare",
            "Consumer_Disc", "Consumer_Staples", "Industrials",
            "Materials", "Utilities", "Real_Estate", "Communications",
        }
        assert set(SECTOR_KEYWORDS.keys()) == expected

    def test_keywords_are_lowercase(self):
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"{sector}: '{kw}' not lowercase"
