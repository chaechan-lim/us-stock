"""Tests for extended hours trading implementation.

Covers:
1. KIS KR adapter _place_nxt_order (NXT exchange orders)
2. KIS KR adapter _place_order with ord_dvsn_override
3. KIS KR adapter create_buy_order / create_sell_order with session values
4. Order manager session passthrough (place_buy / place_sell)
5. Position tracker check_all with session parameter
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from exchange.base import OrderResult, Position
from exchange.kis_kr_adapter import KISKRAdapter, TR_ID_KR_LIVE, TR_ID_KR_PAPER
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager, RiskParams
from engine.position_tracker import PositionTracker


# ── Fixtures ────────────────────────────────────────────────────────────


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
def live_config():
    """Live (non-paper) config for testing live TR_ID selection."""
    config = MagicMock()
    config.base_url = "https://openapi.koreainvestment.com:9443"
    config.account_no = "12345678"
    config.account_product = "01"
    return config


@pytest.fixture
def adapter(mock_config, mock_auth):
    """Paper-mode KIS KR adapter."""
    return KISKRAdapter(config=mock_config, auth=mock_auth)


@pytest.fixture
def live_adapter(live_config, mock_auth):
    """Live-mode KIS KR adapter."""
    return KISKRAdapter(config=live_config, auth=mock_auth)


def _mock_post_success(adapter, order_id="0001234567"):
    """Wire up adapter._session.post to return a successful order response."""
    adapter._session = MagicMock()
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "rt_cd": "0",
        "output": {"ODNO": order_id},
    })
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._session.post = MagicMock(return_value=ctx)
    return adapter._session


def _mock_post_failure(adapter, msg="주문 실패"):
    """Wire up adapter._session.post to return a failed order response."""
    adapter._session = MagicMock()
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "rt_cd": "-1",
        "msg_cd": "APBK0918",
        "msg1": msg,
        "output": {},
    })
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._session.post = MagicMock(return_value=ctx)
    return adapter._session


# ── 1. _place_nxt_order Tests ───────────────────────────────────────────


class TestPlaceNxtOrder:
    """Tests for KISKRAdapter._place_nxt_order (NXT exchange orders)."""

    async def test_nxt_buy_paper_tr_id(self, adapter):
        """Paper NXT buy uses VTTC0012U."""
        session = _mock_post_success(adapter, "NXT001")

        result = await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        assert result.status == "pending"
        assert result.order_id == "NXT001"
        assert result.side == "buy"
        assert result.order_type == "limit"
        assert result.quantity == 10
        assert result.price == 72000.0

        # Verify TR_ID in headers call (paper NXT buy = VTTC0012U)
        adapter._auth.get_auth_headers.assert_called_with("VTTC0012U", "test-hash")

    async def test_nxt_sell_paper_tr_id(self, adapter):
        """Paper NXT sell uses VTTC0011U."""
        _mock_post_success(adapter, "NXT002")

        result = await adapter._place_nxt_order("005930", "sell", 5, 73000.0)

        assert result.status == "pending"
        assert result.order_id == "NXT002"
        assert result.side == "sell"
        adapter._auth.get_auth_headers.assert_called_with("VTTC0011U", "test-hash")

    async def test_nxt_buy_live_tr_id(self, live_adapter):
        """Live NXT buy uses TTTC0012U."""
        _mock_post_success(live_adapter, "NXT003")

        result = await live_adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        assert result.status == "pending"
        live_adapter._auth.get_auth_headers.assert_called_with("TTTC0012U", "test-hash")

    async def test_nxt_sell_live_tr_id(self, live_adapter):
        """Live NXT sell uses TTTC0011U."""
        _mock_post_success(live_adapter, "NXT004")

        result = await live_adapter._place_nxt_order("005930", "sell", 5, 73000.0)

        assert result.status == "pending"
        live_adapter._auth.get_auth_headers.assert_called_with("TTTC0011U", "test-hash")

    async def test_nxt_order_body_structure(self, adapter):
        """NXT order includes EXCG_ID_DVSN_CD=NXT and ORD_DVSN=00 (limit)."""
        session = _mock_post_success(adapter, "NXT005")

        await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        # Inspect the body passed to POST
        post_call = session.post.call_args
        body = post_call[1]["json"] if "json" in post_call[1] else post_call[0][1] if len(post_call[0]) > 1 else None
        # The _post method is called via session.post(url, headers=..., json=body)
        # Since we mock _session.post, we need to check what _post sent
        assert body is not None
        assert body["EXCG_ID_DVSN_CD"] == "NXT"
        assert body["ORD_DVSN"] == "00"
        assert body["PDNO"] == "005930"
        assert body["ORD_QTY"] == "10"
        assert body["ORD_UNPR"] == "72000"
        assert body["CANO"] == "12345678"
        assert body["ACNT_PRDT_CD"] == "01"

    async def test_nxt_order_endpoint(self, adapter):
        """NXT order posts to /uapi/domestic-stock/v1/trading/order-cash."""
        session = _mock_post_success(adapter, "NXT006")

        await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        post_call = session.post.call_args
        url = post_call[0][0] if post_call[0] else post_call[1].get("url", "")
        assert "/uapi/domestic-stock/v1/trading/order-cash" in url

    async def test_nxt_order_failed(self, adapter):
        """Failed NXT order returns status='failed'."""
        _mock_post_failure(adapter, "NXT 거래 불가")

        result = await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        assert result.status == "failed"
        assert result.order_id == ""
        assert result.symbol == "005930"

    async def test_nxt_order_price_none(self, adapter):
        """NXT order with price=None sends ORD_UNPR='0'."""
        session = _mock_post_success(adapter, "NXT007")

        await adapter._place_nxt_order("005930", "buy", 10, None)

        post_call = session.post.call_args
        body = post_call[1]["json"]
        assert body["ORD_UNPR"] == "0"

    async def test_nxt_order_price_conversion(self, adapter):
        """NXT order converts float price to int string."""
        session = _mock_post_success(adapter, "NXT008")

        await adapter._place_nxt_order("005930", "buy", 10, 72500.7)

        post_call = session.post.call_args
        body = post_call[1]["json"]
        assert body["ORD_UNPR"] == "72500"

    async def test_nxt_order_ensures_valid_token(self, adapter):
        """NXT order calls ensure_valid_token before placing."""
        _mock_post_success(adapter, "NXT009")

        await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        adapter._auth.ensure_valid_token.assert_called_once()

    async def test_nxt_order_gets_hashkey(self, adapter):
        """NXT order requests hashkey for the order body."""
        _mock_post_success(adapter, "NXT010")

        await adapter._place_nxt_order("005930", "buy", 10, 72000.0)

        adapter._auth.get_hashkey.assert_called_once()
        body_arg = adapter._auth.get_hashkey.call_args[0][0]
        assert body_arg["PDNO"] == "005930"
        assert body_arg["EXCG_ID_DVSN_CD"] == "NXT"


# ── 2. _place_order with ord_dvsn_override Tests ───────────────────────


class TestPlaceOrderOrdDvsnOverride:
    """Tests for _place_order with ord_dvsn_override parameter."""

    async def test_override_sets_ord_dvsn_05(self, adapter):
        """ord_dvsn_override='05' sets ORD_DVSN to 05 (장전시간외)."""
        session = _mock_post_success(adapter, "OVR001")

        result = await adapter._place_order(
            symbol="005930", side="buy", quantity=10, price=72000.0,
            order_type="limit", tr_id=adapter._tr["BUY"],
            ord_dvsn_override="05",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "05"

    async def test_override_sets_ord_dvsn_06(self, adapter):
        """ord_dvsn_override='06' sets ORD_DVSN to 06 (장후시간외)."""
        session = _mock_post_success(adapter, "OVR002")

        result = await adapter._place_order(
            symbol="005930", side="sell", quantity=5, price=73000.0,
            order_type="limit", tr_id=adapter._tr["SELL"],
            ord_dvsn_override="06",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "06"

    async def test_no_override_limit_order(self, adapter):
        """Without override, limit order uses ORD_DVSN='00'."""
        session = _mock_post_success(adapter, "OVR003")

        await adapter._place_order(
            symbol="005930", side="buy", quantity=10, price=72000.0,
            order_type="limit", tr_id=adapter._tr["BUY"],
        )

        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "00"

    async def test_no_override_market_order(self, adapter):
        """Without override, market order uses ORD_DVSN='01'."""
        session = _mock_post_success(adapter, "OVR004")

        await adapter._place_order(
            symbol="005930", side="buy", quantity=10, price=None,
            order_type="market", tr_id=adapter._tr["BUY"],
        )

        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "01"

    async def test_override_takes_precedence_over_order_type(self, adapter):
        """ord_dvsn_override takes priority even when order_type='market'."""
        session = _mock_post_success(adapter, "OVR005")

        await adapter._place_order(
            symbol="005930", side="buy", quantity=10, price=72000.0,
            order_type="market", tr_id=adapter._tr["BUY"],
            ord_dvsn_override="05",
        )

        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "05"

    async def test_override_none_defaults_to_order_type(self, adapter):
        """ord_dvsn_override=None falls back to order_type-based logic."""
        session = _mock_post_success(adapter, "OVR006")

        await adapter._place_order(
            symbol="005930", side="buy", quantity=10, price=72000.0,
            order_type="limit", tr_id=adapter._tr["BUY"],
            ord_dvsn_override=None,
        )

        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "00"


# ── 3. create_buy_order / create_sell_order with session ────────────────


class TestCreateBuyOrderSessions:
    """Tests for create_buy_order with different session values."""

    async def test_regular_session_default(self, adapter):
        """Regular session routes to standard _place_order."""
        _mock_post_success(adapter, "BUY001")

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, order_type="limit", session="regular",
        )

        assert result.status == "pending"
        assert result.order_id == "BUY001"
        # Standard buy should use the default BUY TR_ID
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["BUY"], "test-hash",
        )

    async def test_pre_market_session(self, adapter):
        """pre_market session uses ORD_DVSN=05 (장전시간외)."""
        session = _mock_post_success(adapter, "BUY002")

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, session="pre_market",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "05"
        # Uses BUY TR_ID (not NXT)
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["BUY"], "test-hash",
        )

    async def test_after_hours_session(self, adapter):
        """after_hours session uses ORD_DVSN=06 (장후시간외)."""
        session = _mock_post_success(adapter, "BUY003")

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, session="after_hours",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "06"

    async def test_extended_nxt_session(self, adapter):
        """extended_nxt session routes to _place_nxt_order."""
        _mock_post_success(adapter, "BUY004")

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, session="extended_nxt",
        )

        assert result.status == "pending"
        assert result.order_id == "BUY004"
        # Paper NXT buy TR_ID
        adapter._auth.get_auth_headers.assert_called_with("VTTC0012U", "test-hash")

    async def test_regular_session_market_order(self, adapter):
        """Regular session with market order uses ORD_DVSN=01."""
        session = _mock_post_success(adapter, "BUY005")

        result = await adapter.create_buy_order(
            "005930", 10, order_type="market", session="regular",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "01"

    async def test_pre_market_ignores_order_type(self, adapter):
        """pre_market always uses limit (ORD_DVSN=05 override) regardless of order_type."""
        session = _mock_post_success(adapter, "BUY006")

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, order_type="market", session="pre_market",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "05"

    async def test_session_default_is_regular(self, adapter):
        """Default session parameter is 'regular'."""
        _mock_post_success(adapter, "BUY007")

        # Call without session parameter
        result = await adapter.create_buy_order("005930", 10, 72000.0)

        assert result.status == "pending"
        # Should use standard BUY TR_ID (not NXT)
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["BUY"], "test-hash",
        )


class TestCreateSellOrderSessions:
    """Tests for create_sell_order with different session values."""

    async def test_regular_session(self, adapter):
        """Regular session sell routes to standard _place_order."""
        _mock_post_success(adapter, "SELL001")

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="regular",
        )

        assert result.status == "pending"
        assert result.side == "sell"
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["SELL"], "test-hash",
        )

    async def test_pre_market_session(self, adapter):
        """pre_market sell uses ORD_DVSN=05."""
        session = _mock_post_success(adapter, "SELL002")

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="pre_market",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "05"
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["SELL"], "test-hash",
        )

    async def test_after_hours_session(self, adapter):
        """after_hours sell uses ORD_DVSN=06."""
        session = _mock_post_success(adapter, "SELL003")

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="after_hours",
        )

        assert result.status == "pending"
        body = session.post.call_args[1]["json"]
        assert body["ORD_DVSN"] == "06"

    async def test_extended_nxt_session(self, adapter):
        """extended_nxt sell routes to _place_nxt_order with sell TR_ID."""
        _mock_post_success(adapter, "SELL004")

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="extended_nxt",
        )

        assert result.status == "pending"
        # Paper NXT sell TR_ID
        adapter._auth.get_auth_headers.assert_called_with("VTTC0011U", "test-hash")

    async def test_extended_nxt_sell_live_tr_id(self, live_adapter):
        """Live NXT sell uses TTTC0011U."""
        _mock_post_success(live_adapter, "SELL005")

        result = await live_adapter.create_sell_order(
            "005930", 10, 73000.0, session="extended_nxt",
        )

        assert result.status == "pending"
        live_adapter._auth.get_auth_headers.assert_called_with("TTTC0011U", "test-hash")

    async def test_session_default_is_regular(self, adapter):
        """Default session parameter is 'regular' for sell orders too."""
        _mock_post_success(adapter, "SELL006")

        result = await adapter.create_sell_order("005930", 10, 73000.0)

        assert result.status == "pending"
        adapter._auth.get_auth_headers.assert_called_with(
            adapter._tr["SELL"], "test-hash",
        )


class TestExtendedHoursFailures:
    """Edge cases and failure paths for extended hours orders."""

    async def test_pre_market_buy_failed(self, adapter):
        """Failed pre_market buy returns status='failed'."""
        _mock_post_failure(adapter)

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, session="pre_market",
        )

        assert result.status == "failed"

    async def test_after_hours_sell_failed(self, adapter):
        """Failed after_hours sell returns status='failed'."""
        _mock_post_failure(adapter)

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="after_hours",
        )

        assert result.status == "failed"

    async def test_nxt_buy_failed(self, adapter):
        """Failed NXT buy returns status='failed'."""
        _mock_post_failure(adapter)

        result = await adapter.create_buy_order(
            "005930", 10, 72000.0, session="extended_nxt",
        )

        assert result.status == "failed"

    async def test_nxt_sell_failed(self, adapter):
        """Failed NXT sell returns status='failed'."""
        _mock_post_failure(adapter)

        result = await adapter.create_sell_order(
            "005930", 10, 73000.0, session="extended_nxt",
        )

        assert result.status == "failed"


# ── 4. Order Manager Session Passthrough ────────────────────────────────


class TestOrderManagerSessionPassthrough:
    """Tests that OrderManager passes session to adapter correctly."""

    @pytest.fixture
    def mock_adapter(self):
        adapter = AsyncMock()
        adapter.create_buy_order = AsyncMock(return_value=OrderResult(
            order_id="ORD001", symbol="005930", side="BUY",
            order_type="limit", quantity=10, price=72000.0,
            filled_quantity=10, filled_price=72000.0, status="filled",
        ))
        adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="ORD002", symbol="005930", side="SELL",
            order_type="limit", quantity=10, price=73000.0,
            filled_quantity=10, filled_price=73000.0, status="filled",
        ))
        return adapter

    @pytest.fixture
    def risk(self):
        return RiskManager(RiskParams(max_position_pct=0.10, max_positions=20))

    @pytest.fixture
    def om(self, mock_adapter, risk):
        return OrderManager(adapter=mock_adapter, risk_manager=risk)

    async def test_place_buy_passes_regular_session(self, om, mock_adapter):
        """place_buy with default session passes session='regular' to adapter."""
        await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
        )

        mock_adapter.create_buy_order.assert_called_once()
        kwargs = mock_adapter.create_buy_order.call_args[1]
        assert kwargs["session"] == "regular"

    async def test_place_buy_passes_pre_market_session(self, om, mock_adapter):
        """place_buy with session='pre_market' passes it to adapter."""
        await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
            session="pre_market",
        )

        kwargs = mock_adapter.create_buy_order.call_args[1]
        assert kwargs["session"] == "pre_market"

    async def test_place_buy_passes_after_hours_session(self, om, mock_adapter):
        """place_buy with session='after_hours' passes it to adapter."""
        await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
            session="after_hours",
        )

        kwargs = mock_adapter.create_buy_order.call_args[1]
        assert kwargs["session"] == "after_hours"

    async def test_place_buy_passes_extended_nxt_session(self, om, mock_adapter):
        """place_buy with session='extended_nxt' passes it to adapter."""
        await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
            session="extended_nxt",
        )

        kwargs = mock_adapter.create_buy_order.call_args[1]
        assert kwargs["session"] == "extended_nxt"

    async def test_place_sell_passes_regular_session(self, om, mock_adapter):
        """place_sell with default session passes session='regular' to adapter."""
        await om.place_sell(
            symbol="005930", quantity=10, price=73000.0,
            strategy_name="test",
        )

        kwargs = mock_adapter.create_sell_order.call_args[1]
        assert kwargs["session"] == "regular"

    async def test_place_sell_passes_after_hours_session(self, om, mock_adapter):
        """place_sell with session='after_hours' passes it to adapter."""
        await om.place_sell(
            symbol="005930", quantity=10, price=73000.0,
            strategy_name="test", session="after_hours",
        )

        kwargs = mock_adapter.create_sell_order.call_args[1]
        assert kwargs["session"] == "after_hours"

    async def test_place_sell_passes_extended_nxt_session(self, om, mock_adapter):
        """place_sell with session='extended_nxt' passes it to adapter."""
        await om.place_sell(
            symbol="005930", quantity=10, price=73000.0,
            strategy_name="test", session="extended_nxt",
        )

        kwargs = mock_adapter.create_sell_order.call_args[1]
        assert kwargs["session"] == "extended_nxt"

    async def test_extended_hours_forces_limit_order_buy(self, om, mock_adapter):
        """Extended hours buy forces order_type='limit' regardless of input."""
        await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
            order_type="market",
            session="pre_market",
        )

        kwargs = mock_adapter.create_buy_order.call_args[1]
        assert kwargs["order_type"] == "limit"

    async def test_extended_hours_forces_limit_order_sell(self, om, mock_adapter):
        """Extended hours sell forces order_type='limit' regardless of input."""
        await om.place_sell(
            symbol="005930", quantity=10, price=73000.0,
            strategy_name="test",
            order_type="market",
            session="after_hours",
        )

        kwargs = mock_adapter.create_sell_order.call_args[1]
        assert kwargs["order_type"] == "limit"

    async def test_regular_session_keeps_market_order(self, om, mock_adapter):
        """Regular session preserves market order_type."""
        await om.place_sell(
            symbol="005930", quantity=10, price=73000.0,
            strategy_name="test",
            order_type="market",
            session="regular",
        )

        kwargs = mock_adapter.create_sell_order.call_args[1]
        assert kwargs["order_type"] == "market"

    async def test_failed_extended_hours_buy_returns_none(self, mock_adapter, risk):
        """OrderManager returns None when extended hours buy fails at exchange."""
        mock_adapter.create_buy_order = AsyncMock(return_value=OrderResult(
            order_id="", symbol="005930", side="BUY",
            order_type="limit", quantity=10, price=72000.0,
            status="failed",
        ))
        om = OrderManager(adapter=mock_adapter, risk_manager=risk)

        result = await om.place_buy(
            symbol="005930", price=72000.0,
            portfolio_value=10_000_000, cash_available=5_000_000,
            current_positions=0, strategy_name="test",
            session="pre_market",
        )

        assert result is None


# ── 5. Position Tracker check_all with session ──────────────────────────


class TestPositionTrackerSession:
    """Tests for PositionTracker.check_all with session parameter."""

    @pytest.fixture
    def pt_adapter(self):
        a = AsyncMock()
        a.fetch_positions = AsyncMock(return_value=[])
        return a

    @pytest.fixture
    def pt_risk(self):
        return RiskManager(RiskParams(
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        ))

    @pytest.fixture
    def pt_order_mgr(self, pt_adapter, pt_risk):
        return OrderManager(adapter=pt_adapter, risk_manager=pt_risk)

    @pytest.fixture
    def pt_tracker(self, pt_adapter, pt_risk, pt_order_mgr):
        return PositionTracker(pt_adapter, pt_risk, pt_order_mgr)

    async def test_check_all_default_session(self, pt_tracker):
        """check_all with no session argument defaults to 'regular'."""
        result = await pt_tracker.check_all()
        assert result == []

    async def test_check_all_passes_session_to_execute_sell(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """When SL triggers, check_all passes session to _execute_sell."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=64000.0),  # -11% < -8% SL
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="SL001", symbol="005930", side="SELL",
            order_type="limit", quantity=10, status="filled",
            filled_price=64000.0,
        ))

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)

        triggered = await tracker.check_all(session="after_hours")

        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"

        # Verify sell order was placed with session and limit order type
        pt_adapter.create_sell_order.assert_called_once()
        sell_kwargs = pt_adapter.create_sell_order.call_args[1]
        assert sell_kwargs["session"] == "after_hours"
        assert sell_kwargs["order_type"] == "limit"

    async def test_check_all_regular_session_uses_limit_order(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """STOCK-77: Regular session SL sell uses limit order (KIS APBK1269 fix).

        KIS overseas API only supports limit orders (ORD_DVSN="00"). Market orders
        return APBK1269. order_type must always be 'limit', including regular session.
        """
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=64000.0),
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="SL002", symbol="005930", side="SELL",
            order_type="limit", quantity=10, status="filled",
            filled_price=64000.0,
        ))

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)

        await tracker.check_all(session="regular")

        sell_kwargs = pt_adapter.create_sell_order.call_args[1]
        assert sell_kwargs["session"] == "regular"
        assert sell_kwargs["order_type"] == "limit"

    async def test_check_all_pre_market_session(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """pre_market session SL sell uses limit order."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=64000.0),
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="SL003", symbol="005930", side="SELL",
            order_type="limit", quantity=10, status="filled",
            filled_price=64000.0,
        ))

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)

        await tracker.check_all(session="pre_market")

        sell_kwargs = pt_adapter.create_sell_order.call_args[1]
        assert sell_kwargs["session"] == "pre_market"
        assert sell_kwargs["order_type"] == "limit"

    async def test_take_profit_extended_session(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """Take profit in extended session uses limit order with session."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=87000.0),  # +20.8% > 20% TP
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="TP001", symbol="005930", side="SELL",
            order_type="limit", quantity=10, status="filled",
            filled_price=87000.0,
        ))

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)

        triggered = await tracker.check_all(session="extended_nxt")

        assert len(triggered) == 1
        assert triggered[0]["reason"] == "take_profit"

        sell_kwargs = pt_adapter.create_sell_order.call_args[1]
        assert sell_kwargs["session"] == "extended_nxt"
        assert sell_kwargs["order_type"] == "limit"

    async def test_multiple_positions_same_session(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """Multiple SL triggers in the same extended session all use correct session."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=64000.0),  # SL
            Position(symbol="035420", exchange="KRX", quantity=5,
                     avg_price=300000.0, current_price=270000.0),  # SL -10%
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="SL_MULTI", symbol="multi", side="SELL",
            order_type="limit", quantity=1, status="filled",
            filled_price=0.0,
        ))

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)
        tracker.track("035420", 300000.0, 5)

        triggered = await tracker.check_all(session="after_hours")

        assert len(triggered) == 2
        # All sell orders should have been placed with after_hours session
        assert pt_adapter.create_sell_order.call_count == 2
        for c in pt_adapter.create_sell_order.call_args_list:
            assert c[1]["session"] == "after_hours"
            assert c[1]["order_type"] == "limit"

    async def test_no_trigger_no_sell_placed(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """When no triggers fire, no sell orders are placed regardless of session."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=71000.0),  # -1.4%, within range
        ])

        tracker = PositionTracker(pt_adapter, pt_risk, pt_order_mgr)
        tracker.track("005930", 72000.0, 10)

        triggered = await tracker.check_all(session="after_hours")

        assert len(triggered) == 0
        pt_adapter.create_sell_order.assert_not_called()

    async def test_session_passed_through_notification(
        self, pt_adapter, pt_risk, pt_order_mgr,
    ):
        """Notification is sent even in extended hours session."""
        pt_adapter.fetch_positions = AsyncMock(return_value=[
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=72000.0, current_price=64000.0),
        ])
        pt_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="SL_NOTIF", symbol="005930", side="SELL",
            order_type="limit", quantity=10, status="filled",
            filled_price=64000.0,
        ))

        notif = AsyncMock()
        tracker = PositionTracker(
            pt_adapter, pt_risk, pt_order_mgr, notification=notif,
        )
        tracker.track("005930", 72000.0, 10)

        await tracker.check_all(session="after_hours")

        notif.notify_stop_loss.assert_called_once()
        args = notif.notify_stop_loss.call_args[0]
        assert args[0] == "005930"
