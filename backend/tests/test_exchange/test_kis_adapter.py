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
    """Helper to mock a POST response on the adapter.

    NOTE: This patches adapter._session directly, which bypasses aiohttp
    internals. The mock context manager is single-use — if _post() retried
    (e.g. on a 4xx), the mock would not return a fresh response. This is
    acceptable for happy-path tests (200 status) and tests where rt_cd="-1"
    does not trigger HTTP-level retry. Keep this in mind when adding tests
    that exercise retry paths.
    """
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=response_data)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._session = MagicMock()
    adapter._session.post = MagicMock(return_value=ctx)


def _extract_post_body(adapter) -> dict:
    """Extract the JSON body from the most recent POST call."""
    call_args = adapter._session.post.call_args
    # Support both keyword and positional passing of json body
    if "json" in call_args.kwargs:
        return call_args.kwargs["json"]
    return call_args[1]["json"]


_SUCCESS_RESPONSE = {
    "rt_cd": "0",
    "output": {"ODNO": "0001234567"},
}


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
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("AAPL", 10, price=185.50, order_type="market")
        assert result.status == "pending"
        assert result.order_id == "0001234567"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "0"
        assert body["ORD_DVSN"] == "01"

    @pytest.mark.asyncio
    async def test_market_sell_sends_zero_price(self, adapter):
        """Market sell order must send OVRS_ORD_UNPR='0' even when price is provided."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_sell_order("AAPL", 10, price=185.50, order_type="market")
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "0"
        assert body["ORD_DVSN"] == "01"

    @pytest.mark.asyncio
    async def test_market_order_no_price_sends_zero(self, adapter):
        """Market order without price also sends OVRS_ORD_UNPR='0'."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("AAPL", 10, order_type="market")
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "0"

    @pytest.mark.asyncio
    async def test_market_order_price_zero_sends_zero(self, adapter):
        """Market order with explicit price=0.0 still sends OVRS_ORD_UNPR='0'."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("AAPL", 5, price=0.0, order_type="market")
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "0"
        assert body["ORD_DVSN"] == "01"


class TestLimitOrderPrice:
    """Verify that limit orders send properly formatted prices."""

    @pytest.mark.asyncio
    async def test_limit_order_sends_formatted_price(self, adapter):
        """Limit order must send the actual price formatted to 2 decimals."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("AAPL", 10, price=185.50, order_type="limit")
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "185.50"
        assert body["ORD_DVSN"] == "00"

    @pytest.mark.asyncio
    async def test_limit_order_no_price_default_rejected(self, adapter):
        """Limit order with default (no price arg) is rejected — price is required."""
        result = await adapter.create_buy_order("AAPL", 10, order_type="limit")
        assert result.status == "failed"
        assert result.order_id == ""

    @pytest.mark.asyncio
    async def test_limit_order_price_zero_rejected(self, adapter):
        """Limit order with price=0.0 is rejected — $0 is never a valid limit price."""
        result = await adapter.create_buy_order("AAPL", 10, price=0.0, order_type="limit")
        assert result.status == "failed"
        assert result.order_id == ""

    @pytest.mark.asyncio
    async def test_limit_order_price_negative_rejected(self, adapter):
        """Limit order with negative price is rejected."""
        result = await adapter.create_buy_order("AAPL", 10, price=-5.0, order_type="limit")
        assert result.status == "failed"
        assert result.order_id == ""

    @pytest.mark.asyncio
    async def test_limit_order_no_price_rejected(self, adapter):
        """Limit order with no price (None) is rejected."""
        result = await adapter.create_buy_order("AAPL", 10, price=None, order_type="limit")
        assert result.status == "failed"
        assert result.order_id == ""

    @pytest.mark.asyncio
    async def test_limit_order_penny_price(self, adapter):
        """Limit order with sub-dollar price formats correctly."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("SIRI", 100, price=0.50, order_type="limit")
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["OVRS_ORD_UNPR"] == "0.50"


class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_buy_order(self, adapter):
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order("AAPL", 10, 185.50)
        assert result.order_id == "0001234567"
        assert result.side == "buy"
        assert result.status == "pending"
        assert result.symbol == "AAPL"

        # Verify request body sent to KIS API
        body = _extract_post_body(adapter)
        assert body["PDNO"] == "AAPL"
        assert body["ORD_QTY"] == "10"
        assert body["OVRS_ORD_UNPR"] == "185.50"
        assert body["ORD_DVSN"] == "00"  # limit order
        assert body["SLL_TYPE"] == ""  # buy has empty SLL_TYPE

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

        # Verify request body sent to KIS API
        body = _extract_post_body(adapter)
        assert body["PDNO"] == "AAPL"
        assert body["ORD_QTY"] == "10"
        assert body["OVRS_ORD_UNPR"] == "186.00"
        assert body["ORD_DVSN"] == "00"  # limit order
        assert body["SLL_TYPE"] == "00"  # sell has SLL_TYPE="00"

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


def _mock_get_responses(adapter, responses):
    """Helper to mock multiple sequential GET responses."""
    ctxs = []
    for resp_data in responses:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=resp_data)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctxs.append(ctx)
    adapter._session = MagicMock()
    adapter._session.get = MagicMock(side_effect=ctxs)


class TestFetchBalance:
    @pytest.mark.asyncio
    async def test_available_capped_when_exceeds_total(self, adapter):
        """STOCK-53: available must not exceed total (KRW auto-conversion inflates buying power)."""
        # present-balance response (tot_asst_amt in KRW)
        pb_resp = {
            "rt_cd": "0",
            "output3": [{"tot_asst_amt": "10444011", "tot_dncl_amt": "4137401", "frst_bltn_exrt": "1450.0", "frcr_evlu_tota": "0"}],
        }
        # inquire-balance (positions)
        bal_resp = {
            "rt_cd": "0",
            "output1": [
                {"ovrs_cblc_qty": "10", "now_pric2": "180.50"},  # position worth $1805
            ],
        }
        # buying power — exceeds total due to KRW auto-conversion
        bp_resp = {
            "rt_cd": "0",
            "output": {"frcr_ord_psbl_amt1": "9188.26"},
        }

        _mock_get_responses(adapter, [pb_resp, bal_resp, bp_resp])

        balance = await adapter.fetch_balance()
        assert balance.currency == "USD"
        total = 10444011 / 1450.0  # ~$7202.77
        assert balance.total == pytest.approx(total, rel=1e-3)
        # available must be capped: total - position_value
        assert balance.available <= balance.total
        assert balance.available == pytest.approx(total - 1805.0, rel=1e-3)
        # invested should be positive (there are real positions)
        invested = balance.total - balance.available
        assert invested > 0

    @pytest.mark.asyncio
    async def test_normal_balance_not_capped(self, adapter):
        """When available < total, no capping occurs."""
        pb_resp = {
            "rt_cd": "0",
            "output3": [{"tot_asst_amt": "15000000", "tot_dncl_amt": "5000000", "frst_bltn_exrt": "1500.0", "frcr_evlu_tota": "0"}],
        }
        bal_resp = {
            "rt_cd": "0",
            "output1": [
                {"ovrs_cblc_qty": "5", "now_pric2": "200.00"},
            ],
        }
        bp_resp = {
            "rt_cd": "0",
            "output": {"frcr_ord_psbl_amt1": "5000.00"},
        }

        _mock_get_responses(adapter, [pb_resp, bal_resp, bp_resp])

        balance = await adapter.fetch_balance()
        total = 15000000 / 1500.0  # $10,000
        assert balance.total == pytest.approx(total, rel=1e-3)
        assert balance.available == 5000.0  # not capped
        assert balance.available < balance.total
