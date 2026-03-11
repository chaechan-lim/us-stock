"""Tests for Naver Finance news service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from data.naver_news_service import NaverNewsService


class TestNaverNewsService:
    def test_always_available(self):
        svc = NaverNewsService()
        assert svc.available is True

    @pytest.mark.asyncio
    async def test_fetch_company_news_success(self):
        svc = NaverNewsService()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[
            {
                "total": 1,
                "items": [{
                    "title": "삼성전자 반도체 투자 확대",
                    "body": "삼성전자가 반도체 신규 라인 투자를 발표했다.",
                    "officeName": "한국경제",
                    "datetime": "202603100930",
                    "url": "https://example.com/news/1",
                }],
            },
            {
                "total": 1,
                "items": [{
                    "title": "삼성전자 <b>실적</b> 전망",
                    "body": "애널리스트들이 삼성전자 1분기 실적을 긍정적으로 전망했다.",
                    "officeName": "매일경제",
                    "datetime": "202603091400",
                    "url": "https://example.com/news/2",
                }],
            },
        ])

        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        svc._session = mock_session

        result = await svc.fetch_company_news("005930")
        assert len(result) == 2
        assert result[0].symbol == "005930"
        assert result[0].source == "한국경제"
        # HTML tags stripped
        assert "<b>" not in result[1].headline
        assert result[1].headline == "삼성전자 실적 전망"

    @pytest.mark.asyncio
    async def test_fetch_company_news_dict_format(self):
        """Test when Naver returns single dict {'items': [...]} format."""
        svc = NaverNewsService()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "items": [
                {
                    "title": "SK하이닉스 HBM 수주",
                    "body": "HBM 공급 확대",
                    "officeName": "서울경제",
                    "datetime": "202603101000",
                },
            ]
        })

        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        svc._session = mock_session

        result = await svc.fetch_company_news("000660")
        assert len(result) == 1
        assert result[0].symbol == "000660"
        assert result[0].headline == "SK하이닉스 HBM 수주"

    @pytest.mark.asyncio
    async def test_fetch_company_news_http_error(self):
        svc = NaverNewsService()
        mock_resp = AsyncMock()
        mock_resp.status = 429

        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        svc._session = mock_session

        result = await svc.fetch_company_news("005930")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_company_news_empty(self):
        svc = NaverNewsService()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[])

        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        svc._session = mock_session

        result = await svc.fetch_company_news("005930")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_batch(self):
        svc = NaverNewsService()

        # Mock fetch_company_news to avoid real HTTP
        from data.news_service import NewsArticle

        call_count = 0

        async def mock_fetch(symbol, max_articles=5):
            nonlocal call_count
            call_count += 1
            return [
                NewsArticle(
                    headline=f"News for {symbol}",
                    summary="Test summary",
                    source="테스트",
                    symbol=symbol,
                    published_at=datetime.now(),
                )
            ]

        svc.fetch_company_news = mock_fetch

        batch = await svc.fetch_batch(["005930", "000660", "035420"])
        assert len(batch.articles) == 3
        assert call_count == 3
        assert batch.articles[0].symbol == "005930"
        assert batch.articles[2].symbol == "035420"

    def test_parse_date_compact(self):
        dt = NaverNewsService._parse_date("202603100930")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 10
        assert dt.hour == 9

    def test_parse_date_iso(self):
        dt = NaverNewsService._parse_date("2026-03-10T09:30:00")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 10

    def test_parse_date_korean(self):
        dt = NaverNewsService._parse_date("2026.03.10 09:30")
        assert dt.year == 2026
        assert dt.month == 3

    def test_parse_date_empty(self):
        dt = NaverNewsService._parse_date("")
        # Should return now() without error
        assert dt.year >= 2026

    def test_parse_date_date_only(self):
        dt = NaverNewsService._parse_date("2026-03-10")
        assert dt.year == 2026
        assert dt.day == 10
