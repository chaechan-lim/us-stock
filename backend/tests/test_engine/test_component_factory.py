"""Tests for EngineComponentFactory (STOCK-84: Phase 3).

Covers:
- Factory creation: correct component types and account_id wiring
- account_id propagation to all components
- Account isolation: separate instances per account+market
- Market isolation: separate instances per market within same account
- Risk independence: daily PnL tracked independently per account
- Order isolation: trade_data tagged with correct account_id
- Position isolation: DB queries scoped by account_id
- Error cases: unknown account, unsupported market
- Caching: same key returns same instance
- MarketDataService: per-account (not shared)
- RateLimiter: paper=5, live=20 req/sec
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from config.accounts import MARKET_KR, MARKET_US, AccountConfig
from core.models import Base, PositionRecord
from data.indicator_service import IndicatorService
from engine.component_factory import EngineComponentFactory, EngineComponents
from engine.evaluation_loop import EvaluationLoop
from engine.order_manager import OrderManager, set_db_recorder, set_trade_recorder
from engine.position_tracker import PositionTracker
from engine.risk_manager import RiskManager, RiskParams
from exchange.adapter_registry import AdapterRegistry
from exchange.base import OrderResult
from strategies.combiner import SignalCombiner
from strategies.registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAPER_URL = "https://openapivts.koreainvestment.com:29443"
LIVE_URL = "https://openapi.koreainvestment.com:9443"


@pytest.fixture
def paper_account() -> AccountConfig:
    return AccountConfig(
        account_id="ACC001",
        name="Paper Account",
        app_key="PAPERKEY",
        app_secret="PAPERSECRET",
        account_no="11111111",
        base_url=PAPER_URL,
        markets=[MARKET_US, MARKET_KR],
    )


@pytest.fixture
def live_account() -> AccountConfig:
    return AccountConfig(
        account_id="ACC002",
        name="Live Account",
        app_key="LIVEKEY",
        app_secret="LIVESECRET",
        account_no="22222222",
        base_url=LIVE_URL,
        markets=[MARKET_US],
    )


@pytest.fixture
def us_only_account() -> AccountConfig:
    return AccountConfig(
        account_id="ACC003",
        name="US-only Account",
        app_key="USKEY",
        app_secret="USSECRET",
        account_no="33333333",
        base_url=PAPER_URL,
        markets=[MARKET_US],
    )


def _make_mock_adapter():
    """Return a minimal AsyncMock adapter that satisfies component constructors."""
    adapter = MagicMock()
    adapter.fetch_positions = AsyncMock(return_value=[])
    adapter.fetch_balance = AsyncMock(return_value=MagicMock(total=100_000, cash=100_000))
    return adapter


@pytest.fixture
def registry_single(paper_account: AccountConfig) -> AdapterRegistry:
    """Registry with one paper account."""
    with (
        patch("exchange.adapter_registry.KISAdapter", return_value=_make_mock_adapter()),
        patch("exchange.adapter_registry.KISAuth"),
    ):
        reg = AdapterRegistry(accounts=[paper_account])
    return reg


@pytest.fixture
def registry_multi(
    paper_account: AccountConfig,
    live_account: AccountConfig,
) -> AdapterRegistry:
    """Registry with two accounts: ACC001 (paper) + ACC002 (live US)."""
    with (
        patch("exchange.adapter_registry.KISAdapter", return_value=_make_mock_adapter()),
        patch("exchange.adapter_registry.KISAuth"),
    ):
        reg = AdapterRegistry(accounts=[paper_account, live_account])
    return reg


@pytest.fixture
def factory_single(registry_single: AdapterRegistry) -> EngineComponentFactory:
    return EngineComponentFactory(adapter_registry=registry_single)


@pytest.fixture
def factory_multi(registry_multi: AdapterRegistry) -> EngineComponentFactory:
    return EngineComponentFactory(adapter_registry=registry_multi)


@pytest_asyncio.fixture
async def db_factory():
    """In-memory SQLite session factory for position isolation tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Factory creation tests
# ---------------------------------------------------------------------------


def test_create_returns_engine_components(factory_single: EngineComponentFactory) -> None:
    """create() returns an EngineComponents dataclass."""
    components = factory_single.create("ACC001", "US")
    assert isinstance(components, EngineComponents)


def test_components_have_correct_types(factory_single: EngineComponentFactory) -> None:
    """All component fields have the correct types."""
    from data.market_data_service import MarketDataService

    components = factory_single.create("ACC001", "US")
    assert isinstance(components.risk_manager, RiskManager)
    assert isinstance(components.order_manager, OrderManager)
    assert isinstance(components.position_tracker, PositionTracker)
    assert isinstance(components.market_data, MarketDataService)


def test_components_have_correct_account_id(factory_single: EngineComponentFactory) -> None:
    """Components report the expected account_id."""
    components = factory_single.create("ACC001", "US")
    assert components.account_id == "ACC001"
    assert components.risk_manager._account_id == "ACC001"
    assert components.order_manager._account_id == "ACC001"
    assert components.position_tracker._account_id == "ACC001"


def test_components_have_correct_market(factory_single: EngineComponentFactory) -> None:
    """Components report the expected market."""
    components = factory_single.create("ACC001", "US")
    assert components.market == "US"
    assert components.order_manager._market == "US"
    assert components.position_tracker._market == "US"


def test_create_kr_market(factory_single: EngineComponentFactory) -> None:
    """Factory can create KR components for a multi-market account."""
    components = factory_single.create("ACC001", "KR")
    assert components.market == "KR"
    assert components.account_id == "ACC001"
    assert components.order_manager._market == "KR"
    assert components.position_tracker._market == "KR"


def test_paper_account_is_paper_true(factory_single: EngineComponentFactory) -> None:
    """Paper account sets is_paper=True on OrderManager."""
    components = factory_single.create("ACC001", "US")
    assert components.order_manager._is_paper is True


def test_live_account_is_paper_false(factory_multi: EngineComponentFactory) -> None:
    """Live account sets is_paper=False on OrderManager."""
    components = factory_multi.create("ACC002", "US")
    assert components.order_manager._is_paper is False


def test_rate_limiter_paper(factory_single: EngineComponentFactory) -> None:
    """Paper accounts use 5 req/sec rate limit."""
    components = factory_single.create("ACC001", "US")
    assert components.market_data._rate_limiter._max_per_second == 5


def test_rate_limiter_live(factory_multi: EngineComponentFactory) -> None:
    """Live accounts use 20 req/sec rate limit."""
    components = factory_multi.create("ACC002", "US")
    assert components.market_data._rate_limiter._max_per_second == 20


def test_create_caches_result(factory_single: EngineComponentFactory) -> None:
    """Calling create() twice with the same key returns the same instance."""
    c1 = factory_single.create("ACC001", "US")
    c2 = factory_single.create("ACC001", "US")
    assert c1 is c2


def test_get_cached_returns_none_before_create(factory_single: EngineComponentFactory) -> None:
    """get_cached() returns None for an un-created key."""
    assert factory_single.get_cached("ACC001", "US") is None


def test_get_cached_returns_components_after_create(factory_single: EngineComponentFactory) -> None:
    """get_cached() returns the bundle after create()."""
    factory_single.create("ACC001", "US")
    result = factory_single.get_cached("ACC001", "US")
    assert result is not None
    assert isinstance(result, EngineComponents)


def test_list_components_empty_initially(factory_single: EngineComponentFactory) -> None:
    """list_components() returns empty list before any create()."""
    assert factory_single.list_components() == []


def test_list_components_after_create(factory_single: EngineComponentFactory) -> None:
    """list_components() includes created bundles."""
    factory_single.create("ACC001", "US")
    bundles = factory_single.list_components()
    assert len(bundles) == 1
    assert bundles[0].account_id == "ACC001"


# ---------------------------------------------------------------------------
# Account isolation tests
# ---------------------------------------------------------------------------


def test_different_accounts_get_different_instances(
    factory_multi: EngineComponentFactory,
) -> None:
    """Different accounts get independent component instances."""
    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")
    assert acc1 is not acc2
    assert acc1.order_manager is not acc2.order_manager
    assert acc1.risk_manager is not acc2.risk_manager
    assert acc1.position_tracker is not acc2.position_tracker
    assert acc1.market_data is not acc2.market_data


def test_different_markets_get_different_instances(
    factory_single: EngineComponentFactory,
) -> None:
    """Same account, different markets → independent components."""
    us = factory_single.create("ACC001", "US")
    kr = factory_single.create("ACC001", "KR")
    assert us is not kr
    assert us.order_manager is not kr.order_manager
    assert us.risk_manager is not kr.risk_manager
    assert us.position_tracker is not kr.position_tracker


def test_account_id_tagged_in_trade_data_buy(factory_multi: EngineComponentFactory) -> None:
    """OrderManager sets the correct account_id in trade_data for BUY orders.

    We inspect the _account_id attribute that is embedded in trade_data dicts
    passed to DB recorders.
    """
    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")
    assert acc1.order_manager._account_id == "ACC001"
    assert acc2.order_manager._account_id == "ACC002"


def test_account_id_tagged_in_position_tracker(
    factory_multi: EngineComponentFactory,
) -> None:
    """PositionTracker uses the correct account_id for DB queries."""
    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")
    assert acc1.position_tracker._account_id == "ACC001"
    assert acc2.position_tracker._account_id == "ACC002"


def test_list_components_tracks_all_created(
    factory_multi: EngineComponentFactory,
) -> None:
    """list_components() returns all account+market bundles."""
    factory_multi.create("ACC001", "US")
    factory_multi.create("ACC001", "KR")
    factory_multi.create("ACC002", "US")
    bundles = factory_multi.list_components()
    assert len(bundles) == 3
    keys = {(b.account_id, b.market) for b in bundles}
    assert ("ACC001", "US") in keys
    assert ("ACC001", "KR") in keys
    assert ("ACC002", "US") in keys


# ---------------------------------------------------------------------------
# Risk independence tests
# ---------------------------------------------------------------------------


def test_risk_managers_are_independent(factory_multi: EngineComponentFactory) -> None:
    """Risk managers for different accounts do not share daily PnL state."""
    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")

    # Update PnL on acc1's risk manager
    acc1.risk_manager.update_daily_pnl(-500.0)

    # acc2's risk manager must remain unaffected
    assert acc1.risk_manager.daily_pnl == -500.0
    assert acc2.risk_manager.daily_pnl == 0.0


def test_risk_managers_same_account_different_markets_are_independent(
    factory_single: EngineComponentFactory,
) -> None:
    """US and KR risk managers for the same account are independent instances."""
    us = factory_single.create("ACC001", "US")
    kr = factory_single.create("ACC001", "KR")

    us.risk_manager.update_daily_pnl(-1000.0)
    assert us.risk_manager.daily_pnl == -1000.0
    assert kr.risk_manager.daily_pnl == 0.0


def test_risk_manager_account_id(factory_multi: EngineComponentFactory) -> None:
    """RiskManager stores the account_id it was created with."""
    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")
    assert acc1.risk_manager._account_id == "ACC001"
    assert acc2.risk_manager._account_id == "ACC002"


def test_custom_risk_params_respected(factory_single: EngineComponentFactory) -> None:
    """Per-call risk_params override default params in the created RiskManager."""
    custom = RiskParams(max_positions=3, max_position_pct=0.15)
    components = factory_single.create("ACC001", "US", risk_params=custom)
    assert components.risk_manager.params.max_positions == 3
    assert components.risk_manager.params.max_position_pct == 0.15


def test_default_risk_params_used_when_none(factory_single: EngineComponentFactory) -> None:
    """Default risk params are used when create() is called without risk_params."""
    default = RiskParams(max_positions=7)
    factory_single._default_risk_params = default
    # Clear any cached state first
    factory_single._cache.clear()
    components = factory_single.create("ACC001", "US")
    assert components.risk_manager.params.max_positions == 7


# ---------------------------------------------------------------------------
# Position isolation via DB (account_id in queries)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_tracker_account_id_in_db_upsert(
    factory_multi: EngineComponentFactory,
    db_factory,
) -> None:
    """PositionTracker writes account_id when upserting positions to DB."""
    factory_multi._session_factory = db_factory
    factory_multi._cache.clear()

    acc1 = factory_multi.create("ACC001", "US")
    # ACC002 created so its PositionTracker is wired correctly (account_id = "ACC002")
    factory_multi.create("ACC002", "US")

    # Track a position in acc1
    acc1.position_tracker.track("AAPL", entry_price=150.0, quantity=10, strategy="trend")

    # Sync acc1 positions to DB
    await acc1.position_tracker.sync_to_db(db_factory)

    # Verify acc1's record has account_id == "ACC001"
    async with db_factory() as session:
        stmt = select(PositionRecord).where(PositionRecord.symbol == "AAPL")
        result = await session.execute(stmt)
        records = result.scalars().all()

    assert len(records) == 1
    assert records[0].account_id == "ACC001"
    assert records[0].symbol == "AAPL"

    # ACC002 has no tracked positions → query for acc2 returns nothing
    async with db_factory() as session:
        stmt = select(PositionRecord).where(
            PositionRecord.account_id == "ACC002",
            PositionRecord.symbol == "AAPL",
        )
        result = await session.execute(stmt)
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_two_accounts_same_symbol_isolated_in_db(
    factory_multi: EngineComponentFactory,
    db_factory,
) -> None:
    """Two accounts can track the same symbol independently in the DB."""
    factory_multi._session_factory = db_factory
    factory_multi._cache.clear()

    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")

    acc1.position_tracker.track("MSFT", entry_price=300.0, quantity=5, strategy="momentum")
    acc2.position_tracker.track("MSFT", entry_price=310.0, quantity=3, strategy="mean_rev")

    await acc1.position_tracker.sync_to_db(db_factory)
    await acc2.position_tracker.sync_to_db(db_factory)

    async with db_factory() as session:
        stmt = select(PositionRecord).where(PositionRecord.symbol == "MSFT")
        result = await session.execute(stmt)
        records = result.scalars().all()

    assert len(records) == 2
    by_account = {r.account_id: r for r in records}
    assert by_account["ACC001"].quantity == 5
    assert by_account["ACC002"].quantity == 3
    assert by_account["ACC001"].avg_price == 300.0
    assert by_account["ACC002"].avg_price == 310.0


@pytest.mark.asyncio
async def test_load_positions_scoped_by_account(
    factory_multi: EngineComponentFactory,
    db_factory,
) -> None:
    """_load_positions_from_db filters by account_id, not just market."""
    factory_multi._session_factory = db_factory
    factory_multi._cache.clear()

    acc1 = factory_multi.create("ACC001", "US")
    acc2 = factory_multi.create("ACC002", "US")

    acc1.position_tracker.track("NVDA", entry_price=500.0, quantity=2, strategy="trend")
    await acc1.position_tracker.sync_to_db(db_factory)

    # acc2 should load zero positions (even though same market+symbol exist for acc1)
    loaded_for_acc2 = await acc2.position_tracker._load_positions_from_db(db_factory)
    assert loaded_for_acc2 == []

    # acc1 should load its own position
    loaded_for_acc1 = await acc1.position_tracker._load_positions_from_db(db_factory)
    assert len(loaded_for_acc1) == 1
    assert loaded_for_acc1[0].symbol == "NVDA"


# ---------------------------------------------------------------------------
# Order isolation: account_id in trade_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buy_trade_data_includes_account_id() -> None:
    """OrderManager embeds account_id in trade_data for BUY orders."""
    rm = RiskManager()
    adapter = MagicMock()
    adapter.create_buy_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            order_type="limit",
            quantity=10,
            status="filled",
            filled_price=150.0,
            filled_quantity=10,
        )
    )

    om = OrderManager(
        adapter=adapter,
        risk_manager=rm,
        market="US",
        is_paper=True,
        account_id="ACC001",
    )

    # Capture the trade_data passed to the DB recorder
    captured: dict = {}

    async def fake_db_recorder(trade: dict) -> None:
        captured.update(trade)

    set_db_recorder(fake_db_recorder)
    set_trade_recorder(None)

    try:
        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="trend",
        )
    finally:
        set_db_recorder(None)

    # trade_data must carry account_id so DB recorder can tag the row
    assert captured.get("account_id") == "ACC001"


@pytest.mark.asyncio
async def test_sell_trade_data_includes_account_id() -> None:
    """OrderManager embeds account_id in trade_data for SELL orders."""
    rm = RiskManager()
    adapter = MagicMock()
    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD002",
            symbol="AAPL",
            side="SELL",
            order_type="limit",
            quantity=10,
            status="filled",
            filled_price=160.0,
            filled_quantity=10,
        )
    )

    om = OrderManager(
        adapter=adapter,
        risk_manager=rm,
        market="US",
        is_paper=False,
        account_id="ACC002",
    )

    captured: dict = {}

    async def fake_db_recorder(trade: dict) -> None:
        captured.update(trade)

    set_db_recorder(fake_db_recorder)
    set_trade_recorder(None)

    try:
        await om.place_sell(
            symbol="AAPL",
            quantity=10,
            price=160.0,
            strategy_name="trend",
        )
    finally:
        set_db_recorder(None)

    assert captured.get("account_id") == "ACC002"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_create_raises_for_unknown_account(factory_single: EngineComponentFactory) -> None:
    """create() raises ValueError for an account_id not in the registry."""
    with pytest.raises(ValueError, match="no adapter registered for account_id"):
        factory_single.create("UNKNOWN_ACCOUNT", "US")


def test_create_raises_for_unsupported_market(
    factory_multi: EngineComponentFactory,
) -> None:
    """create() raises ValueError when the market is not in account.markets."""
    # ACC002 only supports US
    with pytest.raises(ValueError, match="does not support market"):
        factory_multi.create("ACC002", "KR")


def test_create_with_notification_passed_to_components(
    factory_single: EngineComponentFactory,
) -> None:
    """Notification service is forwarded to OrderManager and PositionTracker."""
    notif = MagicMock()
    factory_single._notification = notif
    factory_single._cache.clear()

    components = factory_single.create("ACC001", "US")
    assert components.order_manager._notification is notif
    assert components.position_tracker._notification is notif


# ---------------------------------------------------------------------------
# EvaluationLoop account_id parameter
# ---------------------------------------------------------------------------


def test_evaluation_loop_accepts_account_id() -> None:
    """EvaluationLoop stores the account_id it is given."""
    adapter = MagicMock()
    market_data = MagicMock()
    rm = RiskManager()
    om = OrderManager(adapter=adapter, risk_manager=rm)

    loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=IndicatorService(),
        registry=StrategyRegistry(),
        combiner=SignalCombiner(),
        order_manager=om,
        risk_manager=rm,
        market="US",
        account_id="ACC002",
    )

    assert loop._account_id == "ACC002"


def test_evaluation_loop_default_account_id() -> None:
    """EvaluationLoop defaults to ACC001 when account_id is not supplied."""
    adapter = MagicMock()
    market_data = MagicMock()
    rm = RiskManager()
    om = OrderManager(adapter=adapter, risk_manager=rm)

    loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=IndicatorService(),
        registry=StrategyRegistry(),
        combiner=SignalCombiner(),
        order_manager=om,
        risk_manager=rm,
    )

    assert loop._account_id == "ACC001"


# ---------------------------------------------------------------------------
# account_id defaults — backward compatibility
# ---------------------------------------------------------------------------


def test_order_manager_default_account_id() -> None:
    """OrderManager defaults to ACC001 without account_id param."""
    adapter = MagicMock()
    rm = RiskManager()
    om = OrderManager(adapter=adapter, risk_manager=rm)
    assert om._account_id == "ACC001"


def test_position_tracker_default_account_id() -> None:
    """PositionTracker defaults to ACC001 without account_id param."""
    adapter = MagicMock()
    rm = RiskManager()
    om = OrderManager(adapter=adapter, risk_manager=rm)
    pt = PositionTracker(adapter=adapter, risk_manager=rm, order_manager=om)
    assert pt._account_id == "ACC001"


def test_risk_manager_default_account_id() -> None:
    """RiskManager defaults to ACC001 without account_id param."""
    rm = RiskManager()
    assert rm._account_id == "ACC001"
