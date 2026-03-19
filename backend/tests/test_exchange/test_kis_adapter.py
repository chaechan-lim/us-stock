"""Tests for KIS US stock adapter — order price handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from exchange.kis_adapter import TR_ID_LIVE, TR_ID_PAPER, KISAdapter


@pytest.fixture
def mock_auth():
    auth = AsyncMock()
    auth.ensure_valid_token = AsyncMock()
    auth.get_auth_headers = MagicMock(return_value={"Authorization": "Bearer test"})
    auth.get_hashkey = AsyncMock(return_value="test-hash")
    return auth


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.base_url = "https://openapivts.koreainvestment.com:29443"
    config.account_no = "12345678"
    config.account_product = "01"
    return config


@pytest.fixture
def adapter(mock_config, mock_auth):
    return KISAdapter(config=mock_config, auth=mock_auth)


def _mock_post_response(adapter, response_data):
    """Helper to mock a POST response on the adapter."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=response_data)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._session = MagicMock()
    adapter._session.post = MagicMock(return_value=ctx)


class TestInit:
    def test_paper_mode(self, adapter):
        assert adapter._is_paper is True
        assert adapter._tr == TR_ID_PAPER

    def test_live_mode(self, mock_auth):
        config = MagicMock()
        config.base_url = "https://openapi.koreainvestment.com:9443"
        a = KISAdapter(config=config, auth=mock_auth)
        assert a._is_paper is False
        assert a._tr == TR_ID_LIVE


class TestMarketOrderPrice:
    """Verify that market orders send OVRS_ORD_UNPR='0' per KIS API spec."""

    @pytest.mark.asyncio
    async def test_market_buy_sends_zero_price(self, adapter):
        """Market buy order must send OVRS_ORD_UNPR='0' even when price is provided."""
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234567"},
            },
        )

        result = await adapter.create_buy_order("AAPL", 10, price=185.50, order_type="market")
        assert result.status == "pending"
        assert result.order_id == "0001234567"

        # Verify the POST body sent OVRS_ORD_UNPR="0"
        call_args = adapter._session.post.call_args
        body = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else None
        if body is None:
            # aiohttp session.post(url, json=body, ...) or via _post helper
            body = call_args[1].get("json")
        assert body["OVRS_ORD_UNPR"] == "0"
        assert body["ORD_DVSN"] == "01"

    @pytest.mark.asyncio
    async def test_market_sell_sends_zero_price(self, adapter):
        """Market sell order must send OVRS_ORD_UNPR='0' even when price is provided."""
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234568"},
            },
        )

        result = await adapter.create_sell_order("AAPL", 10, price=185.50, order_type="market")
        assert result.status == "pending"

        call_args = adapter._session.post.call_args
        body = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else None
        if body is None:
            body = call_args[1].get("json")
        assert body["OVRS_ORD_UNPR"] == "0"
        assert body["ORD_DVSN"] == "01"

    @pytest.mark.asyncio
    async def test_market_order_no_price_sends_zero(self, adapter):
        """Market order without price also sends OVRS_ORD_UNPR='0'."""
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234569"},
            },
        )

        result = await adapter.create_buy_order("AAPL", 10, order_type="market")
        assert result.status == "pending"

        call_args = adapter._session.post.call_args
        body = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else None
        if body is None:
            body = call_args[1].get("json")
        assert body["OVRS_ORD_UNPR"] == "0"

    @pytest.mark.asyncio
    async def test_limit_order_sends_formatted_price(self, adapter):
        """Limit order must send the actual price formatted to 2 decimals."""
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234570"},
            },
        )

        result = await adapter.create_buy_order("AAPL", 10, price=185.50, order_type="limit")
        assert result.status == "pending"

        call_args = adapter._session.post.call_args
        body = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else None
        if body is None:
            body = call_args[1].get("json")
        assert body["OVRS_ORD_UNPR"] == "185.50"
        assert body["ORD_DVSN"] == "00"

    @pytest.mark.asyncio
    async def test_limit_order_no_price_sends_zero(self, adapter):
        """Limit order with no price falls back to '0'."""
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234571"},
            },
        )

        result = await adapter.create_buy_order("AAPL", 10, order_type="limit")
        assert result.status == "pending"

        call_args = adapter._session.post.call_args
        body = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else None
        if body is None:
            body = call_args[1].get("json")
        assert body["OVRS_ORD_UNPR"] == "0"


class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_buy_order(self, adapter):
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234567"},
            },
        )

        result = await adapter.create_buy_order("AAPL", 10, 185.50)
        assert result.order_id == "0001234567"
        assert result.side == "buy"
        assert result.status == "pending"
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_sell_order(self, adapter):
        _mock_post_response(
            adapter,
            {
                "rt_cd": "0",
                "output": {"ODNO": "0001234568"},
            },
        )

        result = await adapter.create_sell_order("AAPL", 10, 186.00)
        assert result.order_id == "0001234568"
        assert result.side == "sell"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_order(self, adapter):
        _mock_post_response(
            adapter,
            {
                "rt_cd": "-1",
                "msg_cd": "APBK1233",
                "msg1": "price validation error",
                "output": {},
            },
        )

        result = await adapter.create_buy_order("AAPL", 1000, 185.50)
        assert result.status == "failed"
