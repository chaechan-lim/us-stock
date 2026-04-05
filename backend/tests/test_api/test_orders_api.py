"""Tests for GET /orders endpoint (STOCK-86)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.orders import router as orders_router
from config.accounts import AccountConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_ACCOUNTS = [
    AccountConfig(
        account_id="ACC001",
        name="Default",
        app_key="k",
        app_secret="s",
        account_no="123",
        base_url="https://openapivts.koreainvestment.com:29443",
        markets=["US", "KR"],
    ),
]


def _make_order(
    order_id="ORD001",
    symbol="AAPL",
    side="BUY",
    quantity=10,
    price=150.0,
    status="filled",
    market="US",
    account_id="ACC001",
    pnl=None,
):
    """Create a MagicMock Order ORM object."""
    o = MagicMock()
    o.id = 1
    o.kis_order_id = order_id
    o.symbol = symbol
    o.side = side
    o.quantity = quantity
    o.price = price
    o.filled_price = price
    o.filled_quantity = quantity
    o.status = status
    o.strategy_name = "test_strategy"
    o.buy_strategy = ""
    o.pnl = pnl
    o.pnl_pct = None
    o.is_paper = False
    o.market = market
    o.session = "regular"
    o.account_id = account_id
    o.created_at = MagicMock()
    o.created_at.__str__ = lambda self: "2024-01-01 10:00:00"
    return o


@pytest.fixture(autouse=True)
def reset_accounts_cache():
    import api.accounts as mod

    mod._accounts_cache = None
    yield
    mod._accounts_cache = None


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(orders_router, prefix="/api/v1")
    return test_app


@pytest.fixture
def mock_orders():
    return [
        _make_order("ORD001", "AAPL", "BUY", market="US", account_id="ACC001"),
        _make_order("ORD002", "005930", "SELL", market="KR", account_id="ACC001", pnl=5000.0),
    ]


# ---------------------------------------------------------------------------
# Tests: basic retrieval
# ---------------------------------------------------------------------------


class TestGetOrdersBasic:
    def test_returns_order_list(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        session_ctx.execute = AsyncMock()
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_order_fields_present(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=[mock_orders[0]])

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/")

        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["symbol"] == "AAPL"
        assert entry["side"] == "BUY"
        assert entry["status"] == "filled"
        assert entry["market"] == "US"


class TestGetOrdersNoSessionFactory:
    def test_no_session_factory_returns_empty(self, app):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", None):
                c = TestClient(app)
                resp = c.get("/api/v1/orders/")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Tests: account_id filter
# ---------------------------------------------------------------------------


class TestGetOrdersAccountId:
    def test_valid_account_id_passes_to_repo(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/?account_id=ACC001")

        assert resp.status_code == 200
        # Verify account_id was passed to repo
        mock_repo.get_trade_history.assert_awaited_once()
        call_kwargs = mock_repo.get_trade_history.call_args.kwargs
        assert call_kwargs.get("account_id") == "ACC001"

    def test_unknown_account_id_returns_404(self, app):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            c = TestClient(app)
            resp = c.get("/api/v1/orders/?account_id=BADACCOUNT")
        assert resp.status_code == 404

    def test_omitted_account_id_returns_all(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/")

        assert resp.status_code == 200
        # account_id should be None when omitted
        call_kwargs = mock_repo.get_trade_history.call_args.kwargs
        assert call_kwargs.get("account_id") is None


# ---------------------------------------------------------------------------
# Tests: market filter
# ---------------------------------------------------------------------------


class TestGetOrdersMarketFilter:
    def _make_client(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        patches = [
            patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS),
            patch("api.trades._session_factory", mock_sf),
            patch("api.orders.TradeRepository", return_value=mock_repo),
        ]
        return patches, TestClient(app)

    def test_market_us_filter(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/?market=US")

        assert resp.status_code == 200
        data = resp.json()
        assert all(d["market"] == "US" for d in data)

    def test_market_kr_filter(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/?market=KR")

        assert resp.status_code == 200
        data = resp.json()
        assert all(d["market"] == "KR" for d in data)

    def test_market_all_returns_both(self, app, mock_orders):
        mock_sf = MagicMock()
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session_ctx)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_sf.return_value = session_ctx

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=mock_orders)

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("api.trades._session_factory", mock_sf):
                with patch("api.orders.TradeRepository", return_value=mock_repo):
                    c = TestClient(app)
                    resp = c.get("/api/v1/orders/?market=ALL")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
