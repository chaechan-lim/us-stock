"""Tests for GET /accounts endpoint and account helper utilities."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.accounts import (
    is_valid_account_id,
    validate_account_id_or_404,
)
from api.accounts import (
    router as accounts_router,
)
from config.accounts import AccountConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_ACCOUNTS = [
    AccountConfig(
        account_id="ACC001",
        name="Paper Account",
        app_key="key1",
        app_secret="secret1",
        account_no="11111111",
        base_url="https://openapivts.koreainvestment.com:29443",
        markets=["US", "KR"],
    ),
    AccountConfig(
        account_id="ACC002",
        name="Live US Account",
        app_key="key2",
        app_secret="secret2",
        account_no="22222222",
        base_url="https://openapi.koreainvestment.com:9443",
        markets=["US"],
    ),
]


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset module-level accounts cache before each test."""
    import api.accounts as mod

    mod._accounts_cache = None
    yield
    mod._accounts_cache = None


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(accounts_router, prefix="/api/v1")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: GET /accounts/
# ---------------------------------------------------------------------------


class TestListAccounts:
    def test_returns_account_list(self, client):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            resp = client.get("/api/v1/accounts/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_account_fields_present(self, client):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            resp = client.get("/api/v1/accounts/")
        data = resp.json()
        acc = data[0]
        assert acc["account_id"] == "ACC001"
        assert acc["name"] == "Paper Account"
        assert acc["markets"] == ["US", "KR"]
        assert acc["is_paper"] is True

    def test_no_sensitive_fields(self, client):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            resp = client.get("/api/v1/accounts/")
        for acc in resp.json():
            assert "app_key" not in acc
            assert "app_secret" not in acc
            assert "account_no" not in acc

    def test_cache_used_on_second_call(self, client):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS) as mock_load:
            client.get("/api/v1/accounts/")
            client.get("/api/v1/accounts/")
        # load_accounts should only be called once (second call uses cache)
        assert mock_load.call_count == 1


# ---------------------------------------------------------------------------
# Tests: is_valid_account_id
# ---------------------------------------------------------------------------


class TestIsValidAccountId:
    def test_known_account_returns_true(self):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            assert is_valid_account_id("ACC001") is True
            assert is_valid_account_id("ACC002") is True

    def test_unknown_account_returns_false(self):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            assert is_valid_account_id("ACC999") is False

    def test_empty_string_returns_false(self):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            assert is_valid_account_id("") is False


# ---------------------------------------------------------------------------
# Tests: validate_account_id_or_404 (FastAPI dependency)
# ---------------------------------------------------------------------------


class TestValidateAccountIdOrDefault:
    @pytest.mark.asyncio
    async def test_none_returns_none(self):
        result = await validate_account_id_or_404(account_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_account_id_returned(self):
        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            result = await validate_account_id_or_404(account_id="ACC001")
        assert result == "ACC001"

    @pytest.mark.asyncio
    async def test_unknown_account_raises_404(self):
        from fastapi import HTTPException

        with patch("api.accounts.load_accounts", return_value=MOCK_ACCOUNTS):
            with pytest.raises(HTTPException) as exc_info:
                await validate_account_id_or_404(account_id="UNKNOWN")
        assert exc_info.value.status_code == 404
        assert "UNKNOWN" in exc_info.value.detail
