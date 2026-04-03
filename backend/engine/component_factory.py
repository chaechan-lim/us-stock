"""EngineComponentFactory — per-account, per-market component instantiation.

Creates isolated sets of engine components (MarketDataService, RiskManager,
OrderManager, PositionTracker) for each account+market combination.  Each
bundle uses the account-specific KIS adapter from :class:`AdapterRegistry`
and tags all DB operations with the account's ``account_id``.

Usage::

    from config.accounts import load_accounts
    from exchange.adapter_registry import AdapterRegistry
    from engine.component_factory import EngineComponentFactory
    from engine.risk_manager import RiskParams

    registry = AdapterRegistry(accounts=load_accounts())
    factory = EngineComponentFactory(
        adapter_registry=registry,
        session_factory=async_session_factory,
    )

    # Create US components for ACC001
    components = factory.create("ACC001", market="US")

    # Access individual components
    components.order_manager.place_buy(...)
    components.position_tracker.track(...)

Decision notes:
- MarketDataService is **per-account** (not shared). Each account has its own
  adapter for balance/position queries — sharing would mix account data.
- RateLimiter is auto-derived from ``AccountConfig.is_paper``:
  5 req/sec for paper, 20 req/sec for live.
- EvaluationLoop is *not* created here — it requires additional dependencies
  (StrategyRegistry, SignalCombiner, etc.) that are beyond this factory's scope.
  Callers construct EvaluationLoop separately using the ``EngineComponents``
  bundle as input.
- ``create()`` is idempotent: the same (account_id, market) key returns the
  same cached instance on subsequent calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data.market_data_service import MarketDataService
from engine.order_manager import OrderManager
from engine.position_tracker import PositionTracker
from engine.risk_manager import RiskManager, RiskParams
from exchange.adapter_registry import AdapterRegistry
from exchange.base import ExchangeAdapter
from services.exchange_resolver import ExchangeResolver
from services.rate_limiter import RateLimiter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Rate limits per KIS API documentation
_RATE_LIMIT_LIVE: int = 20  # req/sec for live accounts
_RATE_LIMIT_PAPER: int = 5  # req/sec for paper (VTS) accounts


@dataclass
class EngineComponents:
    """Bundle of engine component instances for a single account+market.

    All components share the same ``account_id`` so DB records are tagged
    consistently and queries can be scoped per-account.

    Attributes:
        account_id: KIS account identifier (e.g. ``"ACC001"``).
        market: Market code — ``"US"`` or ``"KR"``.
        adapter: Exchange adapter for this account.
        market_data: Market data service wrapping the adapter.
        risk_manager: Per-account risk manager with independent PnL tracking.
        order_manager: Order lifecycle manager; tags all trades with ``account_id``.
        position_tracker: Position monitor; all DB ops are scoped to ``account_id``.
    """

    account_id: str
    market: str
    adapter: ExchangeAdapter
    market_data: MarketDataService
    risk_manager: RiskManager
    order_manager: OrderManager
    position_tracker: PositionTracker


class EngineComponentFactory:
    """Creates and caches per-account, per-market :class:`EngineComponents`.

    Args:
        adapter_registry: Source of per-account :class:`~exchange.kis_adapter.KISAdapter`
            instances.  Must already be populated with all relevant accounts.
        notification: Optional notification service passed to ``OrderManager``
            and ``PositionTracker`` for trade/alert messages.
        session_factory: Async SQLAlchemy session factory.  Forwarded to
            ``PositionTracker`` for position persistence.
        exchange_resolver: Optional :class:`~services.exchange_resolver.ExchangeResolver`
            shared across all component bundles (cached exchange lookups).
        default_risk_params: Fallback :class:`~engine.risk_manager.RiskParams` when
            no account-specific params are supplied to :meth:`create`.
            Defaults to the library defaults when *None*.
    """

    def __init__(
        self,
        adapter_registry: AdapterRegistry,
        notification: Any = None,
        session_factory: Any = None,
        exchange_resolver: ExchangeResolver | None = None,
        default_risk_params: RiskParams | None = None,
    ) -> None:
        self._registry = adapter_registry
        self._notification = notification
        self._session_factory = session_factory
        self._exchange_resolver = exchange_resolver or ExchangeResolver()
        self._default_risk_params = default_risk_params or RiskParams()
        # Cache: (account_id, market) -> EngineComponents
        self._cache: dict[str, EngineComponents] = {}
        logger.info(
            "EngineComponentFactory initialised (accounts=%s)",
            [acc.account_id for acc in adapter_registry.list_accounts()],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        account_id: str,
        market: str = "US",
        risk_params: RiskParams | None = None,
    ) -> EngineComponents:
        """Return :class:`EngineComponents` for *account_id* + *market*.

        The first call creates all components; subsequent calls return the
        cached bundle unchanged (idempotent).

        Args:
            account_id: Account identifier registered in :attr:`adapter_registry`.
            market: ``"US"`` or ``"KR"``.
            risk_params: Per-account risk configuration.  Falls back to
                :attr:`default_risk_params` when *None*.

        Returns:
            A fully wired :class:`EngineComponents` bundle.

        Raises:
            ValueError: If *account_id* is not registered or *market* is not
                supported for that account.
        """
        cache_key = f"{account_id}:{market}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        components = self._build(account_id, market, risk_params)
        self._cache[cache_key] = components
        logger.info(
            "EngineComponentFactory: created components for account=%s market=%s",
            account_id,
            market,
        )
        return components

    def get_cached(self, account_id: str, market: str = "US") -> EngineComponents | None:
        """Return cached :class:`EngineComponents`, or *None* if not yet created."""
        return self._cache.get(f"{account_id}:{market}")

    def list_components(self) -> list[EngineComponents]:
        """Return all currently cached component bundles."""
        return list(self._cache.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(
        self,
        account_id: str,
        market: str,
        risk_params: RiskParams | None,
    ) -> EngineComponents:
        """Instantiate a fresh :class:`EngineComponents` bundle."""
        # Resolve adapter — raises ValueError for unknown accounts
        adapter = self._registry.get_adapter(account_id)
        if adapter is None:
            raise ValueError(
                f"EngineComponentFactory: no adapter registered for account_id={account_id!r}. "
                f"Known accounts: {[acc.account_id for acc in self._registry.list_accounts()]}"
            )

        # Validate market against account configuration
        account = self._registry.get_account(account_id)
        if account is not None and market not in account.markets:
            raise ValueError(
                f"EngineComponentFactory: account {account_id!r} does not support "
                f"market={market!r}. Supported markets: {account.markets}"
            )

        # Rate limiter: paper = 5 req/sec, live = 20 req/sec
        is_paper = account.is_paper if account is not None else True
        rate_limit = _RATE_LIMIT_PAPER if is_paper else _RATE_LIMIT_LIVE
        rate_limiter = RateLimiter(max_per_second=rate_limit)

        market_data = MarketDataService(
            adapter=adapter,
            rate_limiter=rate_limiter,
        )

        params = risk_params or self._default_risk_params
        risk_manager = RiskManager(params=params, account_id=account_id)

        order_manager = OrderManager(
            adapter=adapter,
            risk_manager=risk_manager,
            notification=self._notification,
            market_data=market_data,
            market=market,
            is_paper=is_paper,
            account_id=account_id,
        )

        position_tracker = PositionTracker(
            adapter=adapter,
            risk_manager=risk_manager,
            order_manager=order_manager,
            notification=self._notification,
            market_data=market_data,
            market=market,
            exchange_resolver=self._exchange_resolver,
            session_factory=self._session_factory,
            account_id=account_id,
        )

        return EngineComponents(
            account_id=account_id,
            market=market,
            adapter=adapter,
            market_data=market_data,
            risk_manager=risk_manager,
            order_manager=order_manager,
            position_tracker=position_tracker,
        )
