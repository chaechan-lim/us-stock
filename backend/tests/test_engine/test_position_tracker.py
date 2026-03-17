"""Tests for PositionTracker."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base
from engine.order_manager import OrderManager
from engine.position_tracker import PositionTracker, TrackedPosition
from engine.risk_manager import RiskManager, RiskParams
from exchange.base import Position


@pytest.fixture
def adapter():
    a = AsyncMock()
    a.fetch_positions = AsyncMock(return_value=[])
    return a


@pytest.fixture
def risk():
    return RiskManager(
        RiskParams(
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
    )


@pytest.fixture
def order_mgr(adapter, risk):
    return OrderManager(adapter=adapter, risk_manager=risk)


@pytest.fixture
def tracker(adapter, risk, order_mgr):
    return PositionTracker(adapter, risk, order_mgr)


def test_track_and_untrack(tracker):
    tracker.track("AAPL", 150.0, 10, strategy="trend_following")
    assert "AAPL" in tracker.tracked_symbols
    assert len(tracker.tracked_symbols) == 1

    tracker.untrack("AAPL")
    assert "AAPL" not in tracker.tracked_symbols


def test_get_status(tracker):
    tracker.track("AAPL", 150.0, 10, strategy="trend_following")
    tracker.track("TSLA", 200.0, 5, strategy="macd_histogram")

    status = tracker.get_status()
    assert len(status) == 2
    symbols = {s["symbol"] for s in status}
    assert symbols == {"AAPL", "TSLA"}


@pytest.mark.asyncio
async def test_check_all_empty(tracker):
    result = await tracker.check_all()
    assert result == []


@pytest.mark.asyncio
async def test_stop_loss_triggered(adapter, risk, order_mgr):
    """Position drops below SL threshold -> sell triggered."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=135.0
            ),  # -10% < -8% SL
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell1",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=135.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "stop_loss"
    assert triggered[0]["symbol"] == "AAPL"
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_take_profit_triggered(adapter, risk, order_mgr):
    """Position rises above TP threshold -> sell triggered."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=365.0
            ),  # +21.7% > 20% TP
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell2",
            symbol="MSFT",
            side="SELL",
            order_type="market",
            quantity=5,
            status="filled",
            filled_price=365.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("MSFT", 300.0, 5)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "take_profit"
    assert triggered[0]["pnl"] == pytest.approx(325.0)


@pytest.mark.asyncio
async def test_trailing_stop_triggered(adapter, risk, order_mgr):
    """Price ran up then dropped -> trailing stop fires."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="GOOG", exchange="NASD", quantity=8, avg_price=100.0, current_price=108.0
            ),  # 8% above entry
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell3",
            symbol="GOOG",
            side="SELL",
            order_type="market",
            quantity=8,
            status="filled",
            filled_price=108.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("GOOG", 100.0, 8)
    # Enable trailing stop for this test
    tracker._tracked["GOOG"].trailing_activation_pct = 0.05
    tracker._tracked["GOOG"].trailing_stop_pct = 0.03
    # Simulate that price had gone to 115 before
    tracker._tracked["GOOG"].highest_price = 115.0
    # Drop from 115 to 108 = ~6.1% > 3% trail

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "trailing_stop"
    assert triggered[0]["highest"] == 115.0


@pytest.mark.asyncio
async def test_no_trigger_when_within_range(adapter, risk, order_mgr):
    """Price is within normal range -> no trigger."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=148.0
            ),  # -1.3%, well within SL
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 0
    assert "AAPL" in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_position_gone_removes_tracker(adapter, risk, order_mgr):
    """If position no longer exists on exchange, tracker is removed after grace period."""
    adapter.fetch_positions = AsyncMock(return_value=[])  # no positions

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert "AAPL" in tracker.tracked_symbols

    # Within grace period (5 min) — should NOT be removed
    await tracker.check_all()
    assert "AAPL" in tracker.tracked_symbols

    # Simulate grace period elapsed
    tracker._tracked["AAPL"].tracked_at -= 400  # 400s ago (> 300s grace)
    await tracker.check_all()
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_highest_price_update(adapter, risk, order_mgr):
    """Highest price tracks the peak for trailing stop."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert tracker._tracked["AAPL"].highest_price == 150.0

    await tracker.check_all()
    assert tracker._tracked["AAPL"].highest_price == 160.0


@pytest.mark.asyncio
async def test_notification_on_stop_loss(adapter, risk, order_mgr):
    """Notification is sent when stop-loss fires."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=135.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell4",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=135.0,
        )
    )

    notif = AsyncMock()
    tracker = PositionTracker(adapter, risk, order_mgr, notification=notif)
    tracker.track("AAPL", 150.0, 10)

    await tracker.check_all()
    notif.notify_stop_loss.assert_called_once()
    args = notif.notify_stop_loss.call_args
    assert args[0][0] == "AAPL"


@pytest.mark.asyncio
async def test_fetch_error_graceful(adapter, risk, order_mgr):
    """Adapter error doesn't crash the tracker."""
    adapter.fetch_positions = AsyncMock(side_effect=Exception("network"))

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    result = await tracker.check_all()
    assert result == []
    assert "AAPL" in tracker.tracked_symbols  # still tracked


# ── Startup Restoration Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_from_exchange_no_positions(adapter, risk, order_mgr):
    """Empty exchange -> nothing restored."""
    adapter.fetch_positions = AsyncMock(return_value=[])
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
    assert len(tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_from_exchange_with_positions(adapter, risk, order_mgr):
    """Exchange positions are restored into tracker."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=155.0
            ),
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
            ),
        ]
    )
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert len(restored) == 2
    assert set(tracker.tracked_symbols) == {"AAPL", "MSFT"}

    # Check correct entry prices from exchange avg_price
    assert tracker._tracked["AAPL"].entry_price == 150.0
    assert tracker._tracked["MSFT"].entry_price == 300.0

    # Check PnL calculation
    aapl = next(r for r in restored if r["symbol"] == "AAPL")
    assert aapl["pnl_pct"] == pytest.approx(3.33, abs=0.1)


@pytest.mark.asyncio
async def test_restore_skips_zero_quantity(adapter, risk, order_mgr):
    """Positions with 0 quantity are not restored."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=0, avg_price=150.0, current_price=155.0
            ),
        ]
    )
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
    assert len(tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_uses_default_risk_params(adapter, risk, order_mgr):
    """Restored positions get default SL/TP from RiskManager."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=155.0
            ),
        ]
    )
    tracker = PositionTracker(adapter, risk, order_mgr)

    await tracker.restore_from_exchange()
    tracked = tracker._tracked["AAPL"]
    assert tracked.stop_loss_pct == 0.08
    assert tracked.take_profit_pct == 0.20


@pytest.mark.asyncio
async def test_restore_fetch_error_graceful(adapter, risk, order_mgr):
    """Adapter error during restore returns empty list."""
    adapter.fetch_positions = AsyncMock(side_effect=Exception("timeout"))
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []


# ── DB Persistence Tests ────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session_factory():
    """In-memory SQLite session factory for position DB tests.

    Uses StaticPool so all sessions share the same in-memory database,
    which is required for fire-and-forget DB writes to be visible
    in subsequent queries.
    """
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def db_tracker(adapter, risk, order_mgr, db_session_factory):
    """PositionTracker with DB persistence enabled."""
    return PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )


@pytest.mark.asyncio
async def test_sync_to_db_persists_tracked_positions(db_tracker, db_session_factory):
    """sync_to_db writes all tracked positions to the positions table."""
    import asyncio

    db_tracker.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.08, take_profit_pct=0.20)
    db_tracker.track("MSFT", 300.0, 5, strategy="macd", stop_loss_pct=0.10, take_profit_pct=0.25)

    # Let fire-and-forget persist tasks complete
    await asyncio.sleep(0.05)

    synced = await db_tracker.sync_to_db()
    assert synced == 2

    # Verify in DB
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        records = await repo.get_all_positions(market="US")

    assert len(records) == 2
    symbols = {r.symbol for r in records}
    assert symbols == {"AAPL", "MSFT"}

    aapl = next(r for r in records if r.symbol == "AAPL")
    assert aapl.avg_price == 150.0
    assert aapl.quantity == 10
    assert aapl.stop_loss == 0.08
    assert aapl.take_profit == 0.20
    assert aapl.strategy_name == "trend"


@pytest.mark.asyncio
async def test_sync_to_db_removes_stale_entries(db_tracker, db_session_factory):
    """sync_to_db removes DB entries for positions no longer tracked."""
    import asyncio

    db_tracker.track("AAPL", 150.0, 10)
    db_tracker.track("MSFT", 300.0, 5)
    await asyncio.sleep(0.05)  # let fire-and-forget persist complete
    await db_tracker.sync_to_db()

    # Untrack MSFT (without DB — simulate memory-only change)
    db_tracker._tracked.pop("MSFT", None)

    synced = await db_tracker.sync_to_db()
    assert synced == 1  # only AAPL

    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        records = await repo.get_all_positions(market="US")

    assert len(records) == 1
    assert records[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_sync_to_db_empty_tracker(db_tracker):
    """sync_to_db with no tracked positions returns 0."""
    synced = await db_tracker.sync_to_db()
    assert synced == 0


@pytest.mark.asyncio
async def test_sync_to_db_no_session_factory(adapter, risk, order_mgr):
    """sync_to_db without session_factory returns 0 gracefully."""
    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    synced = await tracker.sync_to_db()
    assert synced == 0


@pytest.mark.asyncio
async def test_restore_from_db_restores_positions(db_tracker, db_session_factory):
    """restore_from_db loads positions from the DB into tracker."""
    # Pre-populate DB directly
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        await repo.upsert_position(
            symbol="AAPL",
            quantity=10,
            avg_price=150.0,
            stop_loss=0.08,
            take_profit=0.20,
            strategy_name="trend_following",
            market="US",
        )
        await repo.upsert_position(
            symbol="MSFT",
            quantity=5,
            avg_price=300.0,
            stop_loss=0.10,
            take_profit=0.25,
            strategy_name="macd_histogram",
            market="US",
        )

    restored = await db_tracker.restore_from_db()
    assert len(restored) == 2
    assert set(db_tracker.tracked_symbols) == {"AAPL", "MSFT"}

    # Check SL/TP restored correctly
    tracked_aapl = db_tracker._tracked["AAPL"]
    assert tracked_aapl.entry_price == 150.0
    assert tracked_aapl.quantity == 10
    assert tracked_aapl.stop_loss_pct == 0.08
    assert tracked_aapl.take_profit_pct == 0.20
    assert tracked_aapl.strategy == "trend_following"

    # Check restore summary
    aapl_info = next(r for r in restored if r["symbol"] == "AAPL")
    assert aapl_info["source"] == "db"


@pytest.mark.asyncio
async def test_restore_from_db_empty_table(db_tracker):
    """restore_from_db with empty positions table returns empty list."""
    restored = await db_tracker.restore_from_db()
    assert restored == []
    assert len(db_tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_from_db_skips_already_tracked(db_tracker, db_session_factory):
    """restore_from_db doesn't overwrite already-tracked positions."""

    # Pre-populate DB with AAPL and MSFT
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        await repo.upsert_position(
            symbol="AAPL",
            quantity=10,
            avg_price=150.0,
            stop_loss=0.08,
            strategy_name="old_strategy",
            market="US",
        )
        await repo.upsert_position(
            symbol="MSFT",
            quantity=5,
            avg_price=300.0,
            strategy_name="macd",
            market="US",
        )

    # Manually set AAPL as tracked (simulate it was already tracked before restore)
    db_tracker._tracked["AAPL"] = TrackedPosition(
        symbol="AAPL",
        entry_price=160.0,
        quantity=15,
        highest_price=160.0,
        strategy="override",
        stop_loss_pct=0.05,
    )

    restored = await db_tracker.restore_from_db()
    # Only MSFT should be restored (AAPL already tracked)
    assert len(restored) == 1
    assert restored[0]["symbol"] == "MSFT"

    # AAPL should keep its original tracking data
    assert db_tracker._tracked["AAPL"].entry_price == 160.0
    assert db_tracker._tracked["AAPL"].stop_loss_pct == 0.05


@pytest.mark.asyncio
async def test_restore_from_db_no_session_factory(adapter, risk, order_mgr):
    """restore_from_db without session_factory returns empty list."""
    tracker = PositionTracker(adapter, risk, order_mgr)
    restored = await tracker.restore_from_db()
    assert restored == []


@pytest.mark.asyncio
async def test_restore_from_db_skips_zero_quantity(db_tracker, db_session_factory):
    """restore_from_db skips positions with 0 quantity."""
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        await repo.upsert_position(
            symbol="AAPL",
            quantity=0,
            avg_price=150.0,
            market="US",
        )

    restored = await db_tracker.restore_from_db()
    assert restored == []
    assert len(db_tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_roundtrip_track_sync_restore(adapter, risk, order_mgr, db_session_factory):
    """Full roundtrip: track -> sync -> new tracker -> restore from DB."""
    import asyncio

    # Phase 1: Track and sync
    tracker1 = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )
    tracker1.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.08, take_profit_pct=0.20)
    tracker1.track("TSLA", 200.0, 5, strategy="macd", stop_loss_pct=0.10, take_profit_pct=0.30)
    await asyncio.sleep(0.05)  # let fire-and-forget persist complete
    await tracker1.sync_to_db()

    # Phase 2: New tracker (simulates server restart)
    tracker2 = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )
    assert len(tracker2.tracked_symbols) == 0

    # Restore from DB
    restored = await tracker2.restore_from_db()
    assert len(restored) == 2
    assert set(tracker2.tracked_symbols) == {"AAPL", "TSLA"}

    # SL/TP data survives roundtrip
    aapl = tracker2._tracked["AAPL"]
    assert aapl.entry_price == 150.0
    assert aapl.stop_loss_pct == 0.08
    assert aapl.take_profit_pct == 0.20
    assert aapl.strategy == "trend"


@pytest.mark.asyncio
async def test_restore_exchange_reconciles_with_db(adapter, risk, order_mgr, db_session_factory):
    """restore_from_exchange reconciles with positions already restored from DB."""
    # Pre-restore from DB
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        await repo.upsert_position(
            symbol="AAPL",
            quantity=10,
            avg_price=150.0,
            stop_loss=0.08,
            take_profit=0.20,
            strategy_name="trend",
            market="US",
        )

    tracker = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )
    await tracker.restore_from_db()
    assert "AAPL" in tracker.tracked_symbols

    # Exchange has AAPL (with updated price) and also MSFT (new)
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=152.0, current_price=160.0
            ),
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
            ),
        ]
    )

    restored = await tracker.restore_from_exchange()
    # Only MSFT should be newly restored (AAPL already tracked from DB)
    assert len(restored) == 1
    assert restored[0]["symbol"] == "MSFT"

    # AAPL entry_price updated from exchange avg_price
    assert tracker._tracked["AAPL"].entry_price == 152.0
    # Both tracked
    assert set(tracker.tracked_symbols) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_kr_market_db_persistence(adapter, risk, order_mgr, db_session_factory):
    """KR market positions are persisted separately from US."""
    import asyncio

    us_tracker = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )
    kr_tracker = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="KR",
    )

    us_tracker.track("AAPL", 150.0, 10, strategy="trend")
    kr_tracker.track("005930", 72000.0, 100, strategy="macd")
    await asyncio.sleep(0.05)  # let fire-and-forget persist complete

    await us_tracker.sync_to_db()
    await kr_tracker.sync_to_db()

    # Verify isolation
    from db.position_repository import PositionRepository

    async with db_session_factory() as session:
        repo = PositionRepository(session)
        us_positions = await repo.get_all_positions(market="US")
        kr_positions = await repo.get_all_positions(market="KR")

    assert len(us_positions) == 1
    assert us_positions[0].symbol == "AAPL"
    assert len(kr_positions) == 1
    assert kr_positions[0].symbol == "005930"

    # Restore each market independently
    us_tracker2 = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="US",
    )
    kr_tracker2 = PositionTracker(
        adapter,
        risk,
        order_mgr,
        session_factory=db_session_factory,
        market="KR",
    )

    us_restored = await us_tracker2.restore_from_db()
    kr_restored = await kr_tracker2.restore_from_db()

    assert len(us_restored) == 1
    assert us_restored[0]["symbol"] == "AAPL"
    assert len(kr_restored) == 1
    assert kr_restored[0]["symbol"] == "005930"
