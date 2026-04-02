"""Tests for AccountConfig model and AdapterRegistry.

Covers:
- AccountConfig field validation (happy path + edge cases)
- AccountConfig.is_paper / is_live properties
- load_accounts() from YAML
- load_accounts() backward-compat env-var fallback (ACC001)
- AdapterRegistry.get_adapter() — creates and caches adapter
- AdapterRegistry.get_default_adapter() — ACC001 preference + fallback
- AdapterRegistry.get_account() / list_accounts()
- AdapterRegistry.close_all() — graceful close
- AdapterRegistry with empty account list edge case
- KISAdapter backward compat: account_id defaults to "ACC001"
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.accounts import (
    DEFAULT_ACCOUNT_ID,
    MARKET_KR,
    MARKET_US,
    AccountConfig,
    load_accounts,
)
from exchange.adapter_registry import AdapterRegistry
from exchange.kis_adapter import KISAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def us_account() -> AccountConfig:
    return AccountConfig(
        account_id="ACC001",
        name="Test Paper Account",
        app_key="TESTKEY123",
        app_secret="TESTSECRET456",
        account_no="12345678",
        account_product="01",
        base_url="https://openapivts.koreainvestment.com:29443",
        markets=[MARKET_US, MARKET_KR],
    )


@pytest.fixture
def live_account() -> AccountConfig:
    return AccountConfig(
        account_id="ACC002",
        name="Live US Account",
        app_key="LIVEKEY789",
        app_secret="LIVESECRET012",
        account_no="87654321",
        account_product="01",
        base_url="https://openapi.koreainvestment.com:9443",
        markets=[MARKET_US],
    )


@pytest.fixture
def registry_single(us_account: AccountConfig) -> AdapterRegistry:
    """Registry with one paper account (ACC001)."""
    return AdapterRegistry(accounts=[us_account])


@pytest.fixture
def registry_multi(us_account: AccountConfig, live_account: AccountConfig) -> AdapterRegistry:
    """Registry with two accounts: ACC001 (paper) + ACC002 (live)."""
    return AdapterRegistry(accounts=[us_account, live_account])


# ---------------------------------------------------------------------------
# AccountConfig — validation
# ---------------------------------------------------------------------------


class TestAccountConfig:
    def test_required_account_id(self) -> None:
        """account_id is a required field."""
        with pytest.raises(Exception):
            AccountConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        """Fields that have defaults populate correctly."""
        acc = AccountConfig(account_id="X001")
        assert acc.name == ""
        assert acc.app_key == ""
        assert acc.app_secret == ""
        assert acc.account_no == ""
        assert acc.account_product == "01"
        assert "vts" in acc.base_url  # default is paper URL
        assert MARKET_US in acc.markets
        assert MARKET_KR in acc.markets

    def test_is_paper_vts_url(self, us_account: AccountConfig) -> None:
        """is_paper is True for VTS (virtual trading service) URL."""
        assert us_account.is_paper is True
        assert us_account.is_live is False

    def test_is_live_production_url(self, live_account: AccountConfig) -> None:
        """is_live is True for the production openapi URL."""
        assert live_account.is_live is True
        assert live_account.is_paper is False

    def test_markets_us_only(self) -> None:
        acc = AccountConfig(account_id="X001", markets=[MARKET_US])
        assert acc.markets == [MARKET_US]
        assert MARKET_KR not in acc.markets

    def test_markets_kr_only(self) -> None:
        acc = AccountConfig(account_id="X001", markets=[MARKET_KR])
        assert acc.markets == [MARKET_KR]

    def test_custom_fields(self, live_account: AccountConfig) -> None:
        assert live_account.account_id == "ACC002"
        assert live_account.app_key == "LIVEKEY789"
        assert live_account.account_no == "87654321"

    def test_default_account_id_constant(self) -> None:
        assert DEFAULT_ACCOUNT_ID == "ACC001"


# ---------------------------------------------------------------------------
# load_accounts — YAML loading
# ---------------------------------------------------------------------------


class TestLoadAccountsFromYaml:
    def test_loads_multiple_accounts(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent(
            """
            accounts:
              - account_id: ACC001
                name: Paper Account
                app_key: KEYABC
                app_secret: SECRETABC
                account_no: "11111111"
                base_url: "https://openapivts.koreainvestment.com:29443"
                markets: [US, KR]
              - account_id: ACC002
                name: Live Account
                app_key: KEYXYZ
                app_secret: SECRETXYZ
                account_no: "22222222"
                base_url: "https://openapi.koreainvestment.com:9443"
                markets: [US]
            """
        )
        config_file = tmp_path / "accounts.yaml"
        config_file.write_text(yaml_content)

        accounts = load_accounts(config_path=config_file)

        assert len(accounts) == 2
        assert accounts[0].account_id == "ACC001"
        assert accounts[0].app_key == "KEYABC"
        assert accounts[0].is_paper is True
        assert accounts[1].account_id == "ACC002"
        assert accounts[1].is_live is True
        assert accounts[1].markets == [MARKET_US]

    def test_empty_yaml_falls_back_to_env(self, tmp_path: Path) -> None:
        config_file = tmp_path / "accounts.yaml"
        config_file.write_text("accounts: []\n")

        with patch.dict(
            os.environ,
            {
                "KIS_APP_KEY": "ENVKEY",
                "KIS_APP_SECRET": "ENVSECRET",
                "KIS_ACCOUNT_NO": "99999999",
            },
            clear=False,
        ):
            accounts = load_accounts(config_path=config_file)

        assert len(accounts) == 1
        assert accounts[0].account_id == DEFAULT_ACCOUNT_ID
        assert accounts[0].app_key == "ENVKEY"

    def test_missing_yaml_falls_back_to_env(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.yaml"
        with patch.dict(
            os.environ,
            {
                "KIS_APP_KEY": "MYKEY",
                "KIS_APP_SECRET": "MYSECRET",
                "KIS_ACCOUNT_NO": "12345678",
            },
            clear=False,
        ):
            accounts = load_accounts(config_path=non_existent)

        assert len(accounts) == 1
        assert accounts[0].account_id == DEFAULT_ACCOUNT_ID

    def test_corrupt_yaml_falls_back_to_env(self, tmp_path: Path) -> None:
        config_file = tmp_path / "accounts.yaml"
        config_file.write_text(": : : invalid yaml :::\n")

        with patch.dict(os.environ, {"KIS_APP_KEY": "FALLBACK"}, clear=False):
            accounts = load_accounts(config_path=config_file)

        assert len(accounts) == 1
        assert accounts[0].account_id == DEFAULT_ACCOUNT_ID


# ---------------------------------------------------------------------------
# load_accounts — env-var fallback (backward compat)
# ---------------------------------------------------------------------------


class TestLoadAccountsEnvFallback:
    def test_creates_acc001_from_env_vars(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "no_file.yaml"
        with patch.dict(
            os.environ,
            {
                "KIS_APP_KEY": "TESTAPPKEY",
                "KIS_APP_SECRET": "TESTAPPSECRET",
                "KIS_ACCOUNT_NO": "55667788",
                "KIS_BASE_URL": "https://openapivts.koreainvestment.com:29443",
            },
            clear=False,
        ):
            accounts = load_accounts(config_path=non_existent)

        assert len(accounts) == 1
        acc = accounts[0]
        assert acc.account_id == "ACC001"
        assert acc.app_key == "TESTAPPKEY"
        assert acc.app_secret == "TESTAPPSECRET"
        assert acc.account_no == "55667788"
        assert acc.is_paper is True
        assert MARKET_US in acc.markets
        assert MARKET_KR in acc.markets

    def test_default_paper_url_when_base_url_absent(self, tmp_path: Path) -> None:
        """When KIS_BASE_URL is not set, the default VTS (paper) URL is used."""
        non_existent = tmp_path / "no_file.yaml"
        # Remove KIS_BASE_URL from environment to force the default value
        with patch.dict(os.environ, {}, clear=False):
            saved = os.environ.pop("KIS_BASE_URL", None)
            try:
                accounts = load_accounts(config_path=non_existent)
            finally:
                if saved is not None:
                    os.environ["KIS_BASE_URL"] = saved

        assert accounts[0].is_paper is True


# ---------------------------------------------------------------------------
# AdapterRegistry — core behaviour
# ---------------------------------------------------------------------------


class TestAdapterRegistrySingle:
    def test_list_accounts(self, registry_single: AdapterRegistry) -> None:
        accounts = registry_single.list_accounts()
        assert len(accounts) == 1
        assert accounts[0].account_id == "ACC001"

    def test_get_account_known(self, registry_single: AdapterRegistry) -> None:
        acc = registry_single.get_account("ACC001")
        assert acc is not None
        assert acc.account_id == "ACC001"

    def test_get_account_unknown(self, registry_single: AdapterRegistry) -> None:
        assert registry_single.get_account("UNKNOWN") is None

    def test_get_adapter_returns_kis_adapter(self, registry_single: AdapterRegistry) -> None:
        adapter = registry_single.get_adapter("ACC001")
        assert adapter is not None
        assert isinstance(adapter, KISAdapter)

    def test_get_adapter_cached(self, registry_single: AdapterRegistry) -> None:
        """Repeated calls return the same object (lazy-singleton per account)."""
        a1 = registry_single.get_adapter("ACC001")
        a2 = registry_single.get_adapter("ACC001")
        assert a1 is a2

    def test_get_adapter_unknown_returns_none(self, registry_single: AdapterRegistry) -> None:
        assert registry_single.get_adapter("ZZZZZZ") is None

    def test_get_default_adapter_acc001(self, registry_single: AdapterRegistry) -> None:
        default = registry_single.get_default_adapter()
        assert default is not None
        assert isinstance(default, KISAdapter)

    def test_adapter_account_id_stored(self, registry_single: AdapterRegistry) -> None:
        """KISAdapter must record the account_id it was created with."""
        adapter = registry_single.get_adapter("ACC001")
        assert adapter is not None
        assert adapter._account_id == "ACC001"


class TestAdapterRegistryMulti:
    def test_list_accounts_two(self, registry_multi: AdapterRegistry) -> None:
        accounts = registry_multi.list_accounts()
        assert len(accounts) == 2
        ids = {a.account_id for a in accounts}
        assert ids == {"ACC001", "ACC002"}

    def test_get_adapter_different_accounts(self, registry_multi: AdapterRegistry) -> None:
        a1 = registry_multi.get_adapter("ACC001")
        a2 = registry_multi.get_adapter("ACC002")
        assert a1 is not a2

    def test_default_prefers_acc001(self, registry_multi: AdapterRegistry) -> None:
        """get_default_adapter() returns ACC001 even when ACC002 is first."""
        default = registry_multi.get_default_adapter()
        assert default is not None
        # ACC001 is the default, ACC002 is live → verify by paper flag
        assert default._is_paper is True

    def test_get_default_fallback_first_account(self) -> None:
        """When ACC001 absent, first registered account becomes the default."""
        acc = AccountConfig(
            account_id="ACC999",
            name="Only Account",
            base_url="https://openapi.koreainvestment.com:9443",
        )
        registry = AdapterRegistry(accounts=[acc])
        default = registry.get_default_adapter()
        assert default is not None
        assert default._account_id == "ACC999"

    def test_acc002_adapter_is_live(self, registry_multi: AdapterRegistry) -> None:
        adapter = registry_multi.get_adapter("ACC002")
        assert adapter is not None
        assert adapter._is_paper is False


class TestAdapterRegistryEmpty:
    def test_empty_registry_default_returns_none(self) -> None:
        registry = AdapterRegistry(accounts=[])
        assert registry.get_default_adapter() is None

    def test_empty_registry_list_accounts_empty(self) -> None:
        registry = AdapterRegistry(accounts=[])
        assert registry.list_accounts() == []


# ---------------------------------------------------------------------------
# AdapterRegistry — close_all
# ---------------------------------------------------------------------------


class TestAdapterRegistryCloseAll:
    @pytest.mark.asyncio
    async def test_close_all_calls_close_on_each_adapter(
        self, registry_multi: AdapterRegistry
    ) -> None:
        # Force adapter creation for both accounts
        a1 = registry_multi.get_adapter("ACC001")
        a2 = registry_multi.get_adapter("ACC002")
        assert a1 is not None and a2 is not None

        # Replace with mocks to capture close() calls
        mock1 = AsyncMock()
        mock2 = AsyncMock()
        registry_multi._adapters["ACC001"] = mock1
        registry_multi._adapters["ACC002"] = mock2

        await registry_multi.close_all()

        mock1.close.assert_awaited_once()
        mock2.close.assert_awaited_once()
        # Cache must be cleared after close
        assert registry_multi._adapters == {}

    @pytest.mark.asyncio
    async def test_close_all_ignores_errors(self, registry_single: AdapterRegistry) -> None:
        """close_all() should not raise even if an adapter.close() fails."""
        bad_adapter = AsyncMock()
        bad_adapter.close.side_effect = RuntimeError("connection gone")
        registry_single._adapters["ACC001"] = bad_adapter

        # Must not raise
        await registry_single.close_all()
        assert registry_single._adapters == {}


# ---------------------------------------------------------------------------
# AdapterRegistry — auto-load from env (backward compat, no explicit accounts)
# ---------------------------------------------------------------------------


class TestAdapterRegistryAutoLoad:
    def test_auto_loads_when_no_accounts_passed(self, tmp_path: Path) -> None:
        """When accounts=None, registry calls load_accounts() automatically."""
        acc = AccountConfig(account_id="AUTO001")
        with patch("exchange.adapter_registry.load_accounts", return_value=[acc]) as mock_load:
            registry = AdapterRegistry()
        mock_load.assert_called_once()
        assert registry.get_account("AUTO001") is not None


# ---------------------------------------------------------------------------
# KISAdapter backward-compat: account_id defaults to "ACC001"
# ---------------------------------------------------------------------------


class TestKISAdapterAccountIdParam:
    def test_default_account_id(self) -> None:
        config = MagicMock()
        config.base_url = "https://openapivts.koreainvestment.com:29443"
        auth = MagicMock()
        adapter = KISAdapter(config=config, auth=auth)
        assert adapter._account_id == "ACC001"

    def test_custom_account_id(self) -> None:
        config = MagicMock()
        config.base_url = "https://openapivts.koreainvestment.com:29443"
        auth = MagicMock()
        adapter = KISAdapter(config=config, auth=auth, account_id="ACC002")
        assert adapter._account_id == "ACC002"

    def test_account_config_as_config(self) -> None:
        """KISAdapter should accept AccountConfig in place of KISConfig."""
        acc = AccountConfig(
            account_id="ACC001",
            app_key="K",
            app_secret="S",
            account_no="12345678",
            base_url="https://openapivts.koreainvestment.com:29443",
        )
        auth = MagicMock()
        adapter = KISAdapter(config=acc, auth=auth, account_id=acc.account_id)
        assert adapter._is_paper is True
        assert adapter._account_id == "ACC001"
