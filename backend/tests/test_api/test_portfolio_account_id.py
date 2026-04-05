"""Tests for account_id parameter on GET /portfolio/summary (STOCK-86)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.portfolio import router as portfolio_router
from config.accounts import AccountConfig
from exchange.base import Balance

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture(autouse=True)
def reset_accounts_cache():
    import api.accounts as mod

    mod._accounts_cache = None
    yield
    mod._accounts_cache = None


@pytest.fixture
def app():
    """FastAPI test app with mocked US + KR market data."""
    test_app = FastAPI()
    test_app.include_router(portfolio_router, prefix="/api/v1")

    mock_us_md = AsyncMock()
    mock_us_md.get_balance = AsyncMock(
        return_value=Balance(
            currency="USD",
            total=100_000.0,
            available=80_000.0,
            locked=20_000.0,
        )
    )
    mock_us_md.get_positions = AsyncMock(return_value=[])
    mock_us_md.get_exchange_rate = AsyncMock(return_value=1350.0)

    mock_kr_md = AsyncMock()
    mock_kr_md.get_balance = AsyncMock(
        return_value=Balance(
            currency="KRW",
            total=10_000_000.0,
            available=8_000_000.0,
            locked=2_000_000.0,
        )
    )
    mock_kr_md.get_positions = AsyncMock(return_value=[])

    test_app.state.market_data = mock_us_md
    test_app.state.kr_market_data = mock_kr_md
    test_app.state.adapter = MagicMock(
        _tot_asst_krw=None,
        _tot_dncl_krw=None,
        _full_account_usd=0,
    )
    test_app.state.kr_adapter = MagicMock(_tot_evlu_amt=None)

    return test_app


@pytest.fixture
def client(app):
    with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Tests: account_id validation on /portfolio/summary
# ---------------------------------------------------------------------------


class TestPortfolioSummaryAccountId:
    def test_no_account_id_returns_200(self, client):
        """Omitting account_id should return 200 (all-accounts view)."""
        resp = client.get("/api/v1/portfolio/summary")
        assert resp.status_code == 200

    def test_valid_account_id_returns_200(self, client):
        resp = client.get("/api/v1/portfolio/summary?account_id=ACC001")
        assert resp.status_code == 200

    def test_unknown_account_id_returns_404(self, client):
        resp = client.get("/api/v1/portfolio/summary?account_id=BADACCOUNT")
        assert resp.status_code == 404

    def test_unknown_account_id_error_message(self, client):
        resp = client.get("/api/v1/portfolio/summary?account_id=GHOST")
        assert resp.status_code == 404
        body = resp.json()
        assert "GHOST" in body.get("detail", "")

    def test_market_all_combined_summary(self, client):
        """market=ALL should return combined US+KR summary."""
        resp = client.get("/api/v1/portfolio/summary?market=ALL")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_equity" in data
        assert data["market"] == "ALL"

    def test_market_us_specific_summary(self, client):
        """market=US should return US-only balance."""
        resp = client.get("/api/v1/portfolio/summary?market=US")
        assert resp.status_code == 200
        data = resp.json()
        assert data["market"] == "US"

    def test_market_kr_specific_summary(self, client):
        """market=KR should return KR-only balance."""
        resp = client.get("/api/v1/portfolio/summary?market=KR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["market"] == "KR"

    def test_combined_with_valid_account_id(self, client):
        """account_id=ACC001 with market=ALL should work."""
        resp = client.get("/api/v1/portfolio/summary?account_id=ACC001&market=ALL")
        assert resp.status_code == 200

    def test_invalid_account_with_market_us(self, client):
        """Invalid account_id is rejected regardless of market param."""
        resp = client.get("/api/v1/portfolio/summary?account_id=INVALID&market=US")
        assert resp.status_code == 404
