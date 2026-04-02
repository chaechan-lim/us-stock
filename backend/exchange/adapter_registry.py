"""AdapterRegistry — creates and caches KIS adapter instances per account.

Usage::

    from config.accounts import load_accounts
    from exchange.adapter_registry import AdapterRegistry

    registry = AdapterRegistry(accounts=load_accounts())
    adapter = registry.get_default_adapter()   # ACC001 or first account
    adapter = registry.get_adapter("ACC002")   # specific account
"""

import logging
from typing import Optional

from config.accounts import DEFAULT_ACCOUNT_ID, AccountConfig, load_accounts
from exchange.kis_adapter import KISAdapter
from exchange.kis_auth import KISAuth

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Manages KIS adapter instances, one per registered account.

    Adapters are created lazily on first access and cached for reuse.
    The registry holds strong references to all adapters — callers must not
    call :meth:`KISAdapter.close` independently; use :meth:`close_all` instead.

    Args:
        accounts: List of :class:`~config.accounts.AccountConfig` objects.
                  If *None*, :func:`~config.accounts.load_accounts` is called
                  automatically (accounts.yaml or env-var fallback).
        redis_client: Optional async Redis client forwarded to each
                      :class:`~exchange.kis_auth.KISAuth` instance for token caching.
    """

    def __init__(
        self,
        accounts: Optional[list[AccountConfig]] = None,
        redis_client: Optional[object] = None,
    ) -> None:
        if accounts is None:
            accounts = load_accounts()
        self._accounts: dict[str, AccountConfig] = {acc.account_id: acc for acc in accounts}
        self._adapters: dict[str, KISAdapter] = {}
        self._redis = redis_client
        logger.info(
            "AdapterRegistry initialised with %d account(s): %s",
            len(self._accounts),
            list(self._accounts.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_adapter(self, account_id: str) -> Optional[KISAdapter]:
        """Return the :class:`KISAdapter` for *account_id*.

        The adapter is created on first call and cached for subsequent calls.

        Args:
            account_id: The account identifier (e.g. ``"ACC001"``).

        Returns:
            A :class:`KISAdapter` instance, or *None* if *account_id* is not
            registered.
        """
        if account_id not in self._accounts:
            logger.warning(
                "AdapterRegistry.get_adapter: unknown account_id=%r (known: %s)",
                account_id,
                list(self._accounts.keys()),
            )
            return None
        if account_id not in self._adapters:
            self._adapters[account_id] = self._create_adapter(account_id)
        return self._adapters[account_id]

    def get_default_adapter(self) -> Optional[KISAdapter]:
        """Return the adapter for the default account.

        Resolution order:

        1. ``ACC001`` (conventional default id)
        2. First account in registration order

        Returns:
            A :class:`KISAdapter` instance, or *None* if no accounts are
            registered.
        """
        if DEFAULT_ACCOUNT_ID in self._accounts:
            return self.get_adapter(DEFAULT_ACCOUNT_ID)
        if self._accounts:
            first_id = next(iter(self._accounts))
            logger.debug(
                "Default account %r not found; falling back to %r",
                DEFAULT_ACCOUNT_ID,
                first_id,
            )
            return self.get_adapter(first_id)
        logger.warning("AdapterRegistry.get_default_adapter: no accounts registered")
        return None

    def get_account(self, account_id: str) -> Optional[AccountConfig]:
        """Return the :class:`AccountConfig` for *account_id*, or *None*."""
        return self._accounts.get(account_id)

    def list_accounts(self) -> list[AccountConfig]:
        """Return all registered account configurations (insertion order)."""
        return list(self._accounts.values())

    async def close_all(self) -> None:
        """Gracefully close all cached adapters (releases aiohttp sessions)."""
        for account_id, adapter in list(self._adapters.items()):
            try:
                await adapter.close()
                logger.debug("Closed adapter for account %s", account_id)
            except Exception as exc:
                logger.warning("Error closing adapter for %s: %s", account_id, exc)
        self._adapters.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_adapter(self, account_id: str) -> KISAdapter:
        """Instantiate a new :class:`KISAdapter` for *account_id*."""
        account = self._accounts[account_id]
        auth = KISAuth(
            app_key=account.app_key,
            app_secret=account.app_secret,
            base_url=account.base_url,
            redis_client=self._redis,
        )
        adapter = KISAdapter(
            config=account,
            auth=auth,
            account_id=account_id,
        )
        logger.info(
            "Created KIS adapter for account %s (mode=%s, markets=%s)",
            account_id,
            "paper" if account.is_paper else "live",
            account.markets,
        )
        return adapter
