"""Tests for KIS Korean domestic stock adapter."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exchange.kis_kr_adapter import TR_ID_KR_LIVE, TR_ID_KR_PAPER, KISKRAdapter


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
    return KISKRAdapter(config=mock_config, auth=mock_auth)


def _mock_post_response(adapter: KISKRAdapter, response_data: dict) -> None:
    """Helper to mock a POST response on the adapter."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=response_data)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._session = MagicMock()
    adapter._session.post = MagicMock(return_value=ctx)


def _extract_post_body(adapter: KISKRAdapter) -> dict:
    """Extract the JSON body from the most recent POST call."""
    call_args = adapter._session.post.call_args
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
        assert adapter._tr == TR_ID_KR_PAPER

    def test_live_mode(self, mock_auth):
        config = MagicMock()
        config.base_url = "https://openapi.koreainvestment.com:9443"
        a = KISKRAdapter(config=config, auth=mock_auth)
        assert a._is_paper is False
        assert a._tr == TR_ID_KR_LIVE


class TestFetchTicker:
    @pytest.mark.asyncio
    async def test_returns_ticker(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {
                "stck_prpr": "72300",
                "prdy_ctrt": "1.54",
                "acml_vol": "15234567",
            },
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        ticker = await adapter.fetch_ticker("005930", "KRX")
        assert ticker.symbol == "005930"
        assert ticker.price == 72300.0
        assert ticker.change_pct == 1.54
        assert ticker.volume == 15234567.0


class TestFetchBalance:
    @pytest.mark.asyncio
    async def test_returns_krw_balance(self, adapter):
        adapter._session = MagicMock()

        # Mock responses: first call = balance, second call = buying power
        balance_resp = AsyncMock()
        balance_resp.status = 200
        balance_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output1": [],
            "output2": [{
                "tot_evlu_amt": "50000000",
                "pchs_amt_smtl_amt": "30000000",
                "dnca_tot_amt": "20000000",
            }],
        })

        buying_power_resp = AsyncMock()
        buying_power_resp.status = 200
        buying_power_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {"ord_psbl_cash": "15000000"},
        })

        ctx1 = MagicMock()
        ctx1.__aenter__ = AsyncMock(return_value=balance_resp)
        ctx1.__aexit__ = AsyncMock(return_value=False)

        ctx2 = MagicMock()
        ctx2.__aenter__ = AsyncMock(return_value=buying_power_resp)
        ctx2.__aexit__ = AsyncMock(return_value=False)

        adapter._session.get = MagicMock(side_effect=[ctx1, ctx2])

        balance = await adapter.fetch_balance()
        assert balance.currency == "KRW"
        assert balance.total == 50000000.0
        assert balance.available == 15000000.0  # from 주문가능조회
        assert balance.locked == 30000000.0

    @pytest.mark.asyncio
    async def test_balance_fallback_on_buying_power_failure(self, adapter):
        """If 주문가능조회 fails, fall back to dnca_tot_amt."""
        adapter._session = MagicMock()

        balance_resp = AsyncMock()
        balance_resp.status = 200
        balance_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output2": [{"tot_evlu_amt": "50000000", "pchs_amt_smtl_amt": "30000000", "dnca_tot_amt": "20000000"}],
        })

        error_resp = AsyncMock()
        error_resp.status = 500
        error_resp.json = AsyncMock(return_value={"rt_cd": "-1", "msg1": "Error"})

        ctx1 = MagicMock()
        ctx1.__aenter__ = AsyncMock(return_value=balance_resp)
        ctx1.__aexit__ = AsyncMock(return_value=False)

        ctx2 = MagicMock()
        ctx2.__aenter__ = AsyncMock(return_value=error_resp)
        ctx2.__aexit__ = AsyncMock(return_value=False)

        adapter._session.get = MagicMock(side_effect=[ctx1, ctx2, ctx2, ctx2])

        balance = await adapter.fetch_balance()
        assert balance.available == 20000000.0  # fallback to dnca_tot_amt


class TestFetchPositions:
    @pytest.mark.asyncio
    async def test_returns_positions(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output1": [
                {
                    "pdno": "005930",
                    "hldg_qty": "100",
                    "pchs_avg_pric": "70000",
                    "prpr": "72300",
                    "evlu_pfls_amt": "230000",
                },
                {
                    "pdno": "035420",
                    "hldg_qty": "0",
                    "pchs_avg_pric": "300000",
                    "prpr": "310000",
                    "evlu_pfls_amt": "0",
                },
            ],
            "output2": [{}],
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        positions = await adapter.fetch_positions()
        assert len(positions) == 1  # qty=0 filtered out
        assert positions[0].symbol == "005930"
        assert positions[0].quantity == 100
        assert positions[0].avg_price == 70000.0
        assert positions[0].current_price == 72300.0
        assert positions[0].exchange == "KRX"


class TestCreateOrder:
    @pytest.mark.asyncio
    async def test_buy_order(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {"ODNO": "0001234567"},
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        result = await adapter.create_buy_order("005930", 10, 72000.0)
        assert result.order_id == "0001234567"
        assert result.side == "buy"
        assert result.status == "pending"
        assert result.symbol == "005930"

    @pytest.mark.asyncio
    async def test_sell_order(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {"ODNO": "0001234568"},
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        result = await adapter.create_sell_order("005930", 10, 73000.0)
        assert result.order_id == "0001234568"
        assert result.side == "sell"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_market_order(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {"ODNO": "0001234569"},
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        result = await adapter.create_buy_order(
            "005930", 10, order_type="market"
        )
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_order(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "-1",
            "msg_cd": "APBK0918",
            "msg1": "주문가능금액 부족",
            "output": {},
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        result = await adapter.create_buy_order("005930", 1000, 72000.0)
        assert result.status == "failed"


class TestMarketOrderBody:
    """STOCK-29: Verify market orders send ORD_UNPR='0' and ORD_DVSN='01'.

    This was the root cause of STOCK-25 (KR SL 전면 실패): market orders
    were sending the actual price instead of '0', causing KIS API rejection.
    """

    @pytest.mark.asyncio
    async def test_market_buy_sends_zero_price(self, adapter):
        """Market buy must send ORD_UNPR='0' even when price is provided."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order(
            "005930", 10, price=72000.0, order_type="market",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "0", "Market order must send ORD_UNPR=0"
        assert body["ORD_DVSN"] == "01", "Market order must send ORD_DVSN=01"

    @pytest.mark.asyncio
    async def test_market_sell_sends_zero_price(self, adapter):
        """Market sell (SL/TP) must send ORD_UNPR='0' even when price is provided."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_sell_order(
            "005930", 10, price=72000.0, order_type="market",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "0", "Market sell must send ORD_UNPR=0"
        assert body["ORD_DVSN"] == "01", "Market sell must send ORD_DVSN=01"

    @pytest.mark.asyncio
    async def test_market_buy_no_price_sends_zero(self, adapter):
        """Market buy without explicit price also sends ORD_UNPR='0'."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order(
            "005930", 10, order_type="market",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "0"

    @pytest.mark.asyncio
    async def test_market_sell_no_price_sends_zero(self, adapter):
        """Market sell without explicit price also sends ORD_UNPR='0'."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_sell_order(
            "005930", 10, order_type="market",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "0"

    @pytest.mark.asyncio
    async def test_market_order_body_has_correct_symbol(self, adapter):
        """Market order body must include PDNO=symbol."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        await adapter.create_buy_order("005930", 10, order_type="market")

        body = _extract_post_body(adapter)
        assert body["PDNO"] == "005930"
        assert body["ORD_QTY"] == "10"
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"


class TestLimitOrderBody:
    """STOCK-29: Verify limit orders send correct ORD_UNPR and ORD_DVSN='00'."""

    @pytest.mark.asyncio
    async def test_limit_buy_sends_integer_price(self, adapter):
        """KR limit buy must send price as integer string (no decimals)."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_buy_order(
            "005930", 10, price=72000.0, order_type="limit",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "72000", "KR limit price must be integer string"
        assert body["ORD_DVSN"] == "00", "Limit order must send ORD_DVSN=00"

    @pytest.mark.asyncio
    async def test_limit_sell_sends_integer_price(self, adapter):
        """KR limit sell must send price as integer string."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        result = await adapter.create_sell_order(
            "005930", 10, price=73000.0, order_type="limit",
        )
        assert result.status == "pending"

        body = _extract_post_body(adapter)
        assert body["ORD_UNPR"] == "73000", "KR limit price must be integer string"
        assert body["ORD_DVSN"] == "00", "Limit sell must send ORD_DVSN=00"

    @pytest.mark.asyncio
    async def test_buy_order_body_structure(self, adapter):
        """Full body structure verification for KR buy order."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        await adapter.create_buy_order("005930", 10, 72000.0)

        body = _extract_post_body(adapter)
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"
        assert body["PDNO"] == "005930"
        assert body["ORD_DVSN"] == "00"  # limit
        assert body["ORD_QTY"] == "10"
        assert body["ORD_UNPR"] == "72000"

    @pytest.mark.asyncio
    async def test_sell_order_body_structure(self, adapter):
        """Full body structure verification for KR sell order."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        await adapter.create_sell_order("035420", 5, 310000.0)

        body = _extract_post_body(adapter)
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"
        assert body["PDNO"] == "035420"
        assert body["ORD_DVSN"] == "00"  # limit
        assert body["ORD_QTY"] == "5"
        assert body["ORD_UNPR"] == "310000"

    @pytest.mark.asyncio
    async def test_limit_order_default_is_limit_type(self, adapter):
        """Default order_type is 'limit' — ORD_DVSN should be '00'."""
        _mock_post_response(adapter, _SUCCESS_RESPONSE)

        # No explicit order_type → defaults to "limit"
        await adapter.create_buy_order("005930", 10, 72000.0)

        body = _extract_post_body(adapter)
        assert body["ORD_DVSN"] == "00"


class TestFetchPendingOrders:
    @pytest.mark.asyncio
    async def test_returns_pending(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": [
                {
                    "odno": "0001234567",
                    "pdno": "005930",
                    "sll_buy_dvsn_cd": "02",  # buy
                    "ord_qty": "100",
                    "ord_unpr": "72000",
                    "psbl_qty": "50",  # 50 unfilled/cancellable
                },
                {
                    "odno": "0001234568",
                    "pdno": "035420",
                    "sll_buy_dvsn_cd": "01",  # sell
                    "ord_qty": "10",
                    "ord_unpr": "310000",
                    "psbl_qty": "0",  # fully filled, should be excluded
                },
            ],
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        orders = await adapter.fetch_pending_orders()
        assert len(orders) == 1
        assert orders[0].order_id == "0001234567"
        assert orders[0].side == "buy"
        assert orders[0].filled_quantity == 50.0


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_success(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"rt_cd": "0"})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        assert await adapter.cancel_order("0001234567", "005930") is True

    @pytest.mark.asyncio
    async def test_cancel_failure(self, adapter):
        adapter._session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"rt_cd": "-1"})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.post = MagicMock(return_value=ctx)

        assert await adapter.cancel_order("0001234567", "005930") is False


class TestDetectExchangeLogging:
    """STOCK-28: _detect_exchange should log warning on fallback."""

    def test_logs_warning_when_get_kr_exchange_raises(self, caplog):
        """When get_kr_exchange raises, should log warning and return KRX."""
        with patch(
            "scanner.kr_screener.get_kr_exchange",
            side_effect=ValueError("unknown symbol"),
        ):
            with caplog.at_level(logging.WARNING, logger="exchange.kis_kr_adapter"):
                result = KISKRAdapter._detect_exchange("999999")
        assert result == "KRX"
        assert any("Exchange detection failed" in r.message for r in caplog.records)
        assert any("999999" in r.message for r in caplog.records)

    def test_returns_valid_exchange(self):
        """Should always return a valid exchange value."""
        result = KISKRAdapter._detect_exchange("005930")
        assert result in ("KRX", "KOSDAQ")


class TestFetchOrderableAmountLogging:
    """STOCK-28: _fetch_orderable_amount should not raise NameError."""

    @pytest.mark.asyncio
    async def test_both_api_calls_fail_no_name_error(self, adapter):
        """When both OVRS=N and OVRS=Y return rt_cd != 0, should NOT raise NameError."""
        adapter._session = MagicMock()
        error_resp = AsyncMock()
        error_resp.status = 200
        error_resp.json = AsyncMock(return_value={
            "rt_cd": "-1",
            "msg_cd": "APBK0918",
            "msg1": "주문가능금액 조회 실패",
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=error_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        # Should NOT raise NameError — cash is now initialized to 0.0
        result = await adapter._fetch_orderable_amount()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_both_api_calls_fail_logs_warning(self, adapter, caplog):
        """When API returns error, should log the API error message."""
        adapter._session = MagicMock()
        error_resp = AsyncMock()
        error_resp.status = 200
        error_resp.json = AsyncMock(return_value={
            "rt_cd": "-1",
            "msg_cd": "APBK0918",
            "msg1": "주문가능금액 조회 실패",
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=error_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        with caplog.at_level(logging.WARNING, logger="exchange.kis_kr_adapter"):
            await adapter._fetch_orderable_amount()

        api_error_logs = [r for r in caplog.records if "주문가능조회 failed" in r.message]
        assert len(api_error_logs) == 2, f"Expected 2 warnings (OVRS=N and Y), got {len(api_error_logs)}"

    @pytest.mark.asyncio
    async def test_first_succeeds_returns_cash(self, adapter):
        """When first call (OVRS=N) succeeds, should return that cash value."""
        adapter._session = MagicMock()
        success_resp = AsyncMock()
        success_resp.status = 200
        success_resp.json = AsyncMock(return_value={
            "rt_cd": "0",
            "output": {"ord_psbl_cash": "15000000", "max_buy_amt": "20000000"},
        })
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=success_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=ctx)

        result = await adapter._fetch_orderable_amount()
        assert result == 15000000.0
