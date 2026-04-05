"""Tests for GET /positions endpoint (STOCK-86)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.positions import router as positions_router
from config.accounts import AccountConfig
from exchange.base import Position

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

US_POSITIONS = [
    Position(
        symbol="AAPL",
        exchange="NASD",
        quantity=10,
        avg_price=150.0,
        current_price=160.0,
        unrealized_pnl=100.0,
        unrealized_pnl_pct=6.67,
    )
]

KR_POSITIONS = [
    Position(
        symbol="005930",
        exchange="KRX",
        quantity=5,
        avg_price=70000.0,
        current_price=75000.0,
        unrealized_pnl=25000.0,
        unrealized_pnl_pct=7.14,
    )
]


@pytest.fixture(autouse=True)
def reset_accounts_cache():
    import api.accounts as mod

    mod._accounts_cache = None
    yield
    mod._accounts_cache = None


@pytest.fixture
def app_with_positions():
    """FastAPI test app with mocked US + KR market data."""
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(positions_router, prefix="/api/v1")

    mock_us_md = AsyncMock()
    mock_us_md.get_positions = AsyncMock(return_value=US_POSITIONS)

    mock_kr_md = AsyncMock()
    mock_kr_md.get_positions = AsyncMock(return_value=KR_POSITIONS)

    test_app.state.market_data = mock_us_md
    test_app.state.kr_market_data = mock_kr_md
    test_app.state.position_tracker = None
    test_app.state.kr_position_tracker = None

    return test_app


@pytest.fixture
def client(app_with_positions):
    with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
        with patch("data.stock_name_service.get_name", return_value=""):
            with patch("data.stock_name_service.resolve_names", new=AsyncMock(return_value={})):
                yield TestClient(app_with_positions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetPositionsAll:
    def test_returns_combined_us_kr(self, client):
        resp = client.get("/api/v1/positions/")
        assert resp.status_code == 200
        data = resp.json()
        symbols = [p["symbol"] for p in data]
        assert "AAPL" in symbols
        assert "005930" in symbols

    def test_market_field_set_correctly(self, client):
        resp = client.get("/api/v1/positions/")
        data = resp.json()
        by_symbol = {p["symbol"]: p for p in data}
        assert by_symbol["AAPL"]["market"] == "US"
        assert by_symbol["005930"]["market"] == "KR"


class TestGetPositionsMarketFilter:
    def test_us_market_only(self, client):
        resp = client.get("/api/v1/positions/?market=US")
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["market"] == "US" for p in data)
        assert any(p["symbol"] == "AAPL" for p in data)

    def test_kr_market_only(self, client):
        resp = client.get("/api/v1/positions/?market=KR")
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["market"] == "KR" for p in data)
        assert any(p["symbol"] == "005930" for p in data)


class TestGetPositionsAccountId:
    def test_valid_account_id_returns_200(self, client):
        resp = client.get("/api/v1/positions/?account_id=ACC001")
        assert resp.status_code == 200

    def test_unknown_account_id_returns_404(self, client):
        resp = client.get("/api/v1/positions/?account_id=UNKNOWN")
        assert resp.status_code == 404

    def test_omitted_account_id_returns_all(self, client):
        resp = client.get("/api/v1/positions/")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1


class TestGetPositionsNoAdapter:
    def test_no_market_data_returns_empty(self):
        """When no market_data is set, positions endpoint returns []."""
        app = FastAPI()
        app.include_router(positions_router, prefix="/api/v1")
        # No market_data on app.state
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("data.stock_name_service.get_name", return_value=""):
                with patch(
                    "data.stock_name_service.resolve_names",
                    new=AsyncMock(return_value={}),
                ):
                    c = TestClient(app)
                    resp = c.get("/api/v1/positions/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_adapter_error_returns_partial(self, app_with_positions):
        """When one market adapter fails, other market still returns data."""
        app_with_positions.state.market_data.get_positions = AsyncMock(
            side_effect=RuntimeError("connection error")
        )
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with patch("data.stock_name_service.get_name", return_value=""):
                with patch(
                    "data.stock_name_service.resolve_names",
                    new=AsyncMock(return_value={}),
                ):
                    c = TestClient(app_with_positions)
                    resp = c.get("/api/v1/positions/")
        assert resp.status_code == 200
        data = resp.json()
        # Only KR positions should be returned (US adapter errored)
        assert any(p["symbol"] == "005930" for p in data)
