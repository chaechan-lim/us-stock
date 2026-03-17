"""Tests for PositionTracker DB persistence (positions table sync).

Verifies that tracked positions are persisted to the `positions` DB table
via sync_to_db(), and removed on untrack, with periodic reconciliation.

Tests use trackers WITHOUT session_factory in __init__ (so track()/untrack()
don't fire background tasks), then pass session_factory explicitly to
sync_to_db() / _upsert_position_db / _remove_position_db.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.models import Base, Order, PositionRecord
from engine.order_manager import OrderManager
from engine.position_tracker import PositionTracker
from engine.risk_manager import RiskManager, RiskParams
from exchange.base import Position


@pytest_asyncio.fixture
async def db_factory():
    """Create in-memory SQLite engine and session factory.

    Uses StaticPool so all sessions share the same connection
    (required for in-memory SQLite to share data across sessions).
    """
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


def _make_tracker(adapter, risk, order_mgr, market: str = "US") -> PositionTracker:
    """Create a tracker WITHOUT session_factory to avoid background DB tasks."""
    return PositionTracker(
        adapter,
        risk,
        order_mgr,
        market=market,
    )


async def _count_positions(factory, market: str | None = None) -> int:
    """Helper: count positions in DB, optionally filtered by market."""
    async with factory() as session:
        stmt = select(PositionRecord)
        if market:
            stmt = stmt.where(PositionRecord.market == market)
        result = await session.execute(stmt)
        return len(result.scalars().all())


async def _get_position(factory, symbol: str, market: str = "US") -> PositionRecord | None:
    """Helper: get a position record from DB."""
    async with factory() as session:
        stmt = select(PositionRecord).where(
            PositionRecord.market == market,
            PositionRecord.symbol == symbol,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


# ── sync_to_db: Main Reconciliation ─────────────────────────────────


class TestSyncToDb:
    """Test periodic sync_to_db() reconciliation."""

    @pytest.mark.asyncio
    async def test_sync_creates_all_positions(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should create DB records for all tracked positions."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")
        tracker.track("MSFT", 300.0, 5, strategy="macd")

        synced = await tracker.sync_to_db(db_factory)
        assert synced == 2
        assert await _count_positions(db_factory, "US") == 2

    @pytest.mark.asyncio
    async def test_sync_stores_correct_fields(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should persist all tracked position fields."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track(
            "AAPL",
            150.0,
            10,
            strategy="trend_following",
            stop_loss_pct=0.10,
            take_profit_pct=0.25,
        )
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.symbol == "AAPL"
        assert record.market == "US"
        assert record.quantity == 10
        assert record.avg_price == 150.0
        assert record.strategy_name == "trend_following"
        assert record.stop_loss == 0.10
        assert record.take_profit == 0.25
        assert record.exchange == "NASD"
        assert record.opened_at is not None
        assert record.updated_at is not None

    @pytest.mark.asyncio
    async def test_sync_removes_stale_db_positions(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should remove DB positions no longer in memory."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        tracker.track("MSFT", 300.0, 5)
        tracker.track("GOOG", 140.0, 8)
        await tracker.sync_to_db(db_factory)
        assert await _count_positions(db_factory, "US") == 3

        # Remove GOOG from memory (not from DB)
        tracker._tracked.pop("GOOG")

        # Sync should remove GOOG from DB
        synced = await tracker.sync_to_db(db_factory)
        assert synced == 2
        assert await _count_positions(db_factory, "US") == 2
        assert await _get_position(db_factory, "GOOG") is None
        assert await _get_position(db_factory, "AAPL") is not None

    @pytest.mark.asyncio
    async def test_sync_updates_changed_positions(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should update positions that changed in memory."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="old_strategy", stop_loss_pct=0.08)
        await tracker.sync_to_db(db_factory)

        # Modify the tracked position in memory
        tracker._tracked["AAPL"].strategy = "new_strategy"
        tracker._tracked["AAPL"].stop_loss_pct = 0.12
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record.strategy_name == "new_strategy"
        assert record.stop_loss == 0.12

    @pytest.mark.asyncio
    async def test_sync_empty_clears_all(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db with empty tracked dict should clear all DB positions."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        await tracker.sync_to_db(db_factory)
        assert await _count_positions(db_factory, "US") == 1

        tracker._tracked.clear()
        await tracker.sync_to_db(db_factory)
        assert await _count_positions(db_factory, "US") == 0

    @pytest.mark.asyncio
    async def test_sync_returns_zero_without_session_factory(self, adapter, risk, order_mgr):
        """sync_to_db returns 0 if no session_factory is provided."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        result = await tracker.sync_to_db()
        assert result == 0

    @pytest.mark.asyncio
    async def test_sync_idempotent(self, adapter, risk, order_mgr, db_factory):
        """Calling sync_to_db multiple times produces the same result."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        tracker.track("MSFT", 300.0, 5)

        await tracker.sync_to_db(db_factory)
        await tracker.sync_to_db(db_factory)
        await tracker.sync_to_db(db_factory)

        assert await _count_positions(db_factory, "US") == 2


# ── Direct _upsert / _remove Methods ────────────────────────────────


class TestDirectDbMethods:
    """Test low-level _upsert_position_db and _remove_position_db methods."""

    @pytest.mark.asyncio
    async def test_upsert_creates_record(self, adapter, risk, order_mgr, db_factory):
        """_upsert_position_db creates a new record."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.10)
        # Set session_factory AFTER track() to avoid background task race
        tracker._session_factory = db_factory

        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.quantity == 10

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, adapter, risk, order_mgr, db_factory):
        """Upserting same symbol updates, not duplicates."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="old")
        tracker._session_factory = db_factory
        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])

        # Update in memory and upsert again
        tracker._tracked["AAPL"].quantity = 20
        tracker._tracked["AAPL"].strategy = "new"
        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])

        assert await _count_positions(db_factory, "US") == 1
        record = await _get_position(db_factory, "AAPL")
        assert record.quantity == 20
        assert record.strategy_name == "new"

    @pytest.mark.asyncio
    async def test_remove_deletes_from_db(self, adapter, risk, order_mgr, db_factory):
        """_remove_position_db deletes the record."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        tracker._session_factory = db_factory
        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])
        assert await _count_positions(db_factory) == 1

        await tracker._remove_position_db("AAPL")
        assert await _count_positions(db_factory) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_no_error(self, adapter, risk, order_mgr, db_factory):
        """Removing non-existent position should not raise."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker._session_factory = db_factory
        await tracker._remove_position_db("NONEXIST")
        assert await _count_positions(db_factory) == 0

    @pytest.mark.asyncio
    async def test_remove_only_target_symbol(self, adapter, risk, order_mgr, db_factory):
        """Removing one symbol preserves others."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        tracker.track("MSFT", 300.0, 5)
        tracker._session_factory = db_factory
        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])
        await tracker._upsert_position_db("MSFT", tracker._tracked["MSFT"])
        assert await _count_positions(db_factory) == 2

        await tracker._remove_position_db("AAPL")
        assert await _count_positions(db_factory) == 1
        assert await _get_position(db_factory, "MSFT") is not None

    @pytest.mark.asyncio
    async def test_upsert_without_session_factory(self, adapter, risk, order_mgr):
        """Upsert without session_factory silently skips."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)
        # Should not raise
        await tracker._upsert_position_db("AAPL", tracker._tracked["AAPL"])

    @pytest.mark.asyncio
    async def test_remove_without_session_factory(self, adapter, risk, order_mgr):
        """Remove without session_factory silently skips."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        await tracker._remove_position_db("AAPL")

    @pytest.mark.asyncio
    async def test_upsert_multiple_symbols(self, adapter, risk, order_mgr, db_factory):
        """Track multiple symbols -> multiple DB records."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        for sym, price, qty in [("AAPL", 150.0, 10), ("MSFT", 300.0, 5), ("GOOG", 140.0, 8)]:
            tracker.track(sym, price, qty, strategy="test")
        tracker._session_factory = db_factory
        for sym in ["AAPL", "MSFT", "GOOG"]:
            await tracker._upsert_position_db(sym, tracker._tracked[sym])

        assert await _count_positions(db_factory) == 3


# ── Dual Market Isolation ────────────────────────────────────────────


class TestDualMarketIsolation:
    """Test that US and KR positions are isolated in the DB."""

    @pytest.mark.asyncio
    async def test_us_and_kr_positions_separate(self, adapter, risk, order_mgr, db_factory):
        """US and KR trackers should not interfere with each other's DB rows."""
        us_tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        kr_tracker = _make_tracker(adapter, risk, order_mgr, market="KR")

        us_tracker.track("AAPL", 150.0, 10, strategy="us_trend")
        kr_tracker.track("005930", 72000.0, 5, strategy="kr_trend")

        await us_tracker.sync_to_db(db_factory)
        await kr_tracker.sync_to_db(db_factory)

        assert await _count_positions(db_factory, "US") == 1
        assert await _count_positions(db_factory, "KR") == 1

    @pytest.mark.asyncio
    async def test_sync_only_removes_own_market(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db for US should not remove KR positions."""
        us_tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        kr_tracker = _make_tracker(adapter, risk, order_mgr, market="KR")

        us_tracker.track("AAPL", 150.0, 10)
        kr_tracker.track("005930", 72000.0, 5)
        await us_tracker.sync_to_db(db_factory)
        await kr_tracker.sync_to_db(db_factory)

        # Clear US tracker and sync
        us_tracker._tracked.clear()
        await us_tracker.sync_to_db(db_factory)

        # KR should still be there
        assert await _count_positions(db_factory, "US") == 0
        assert await _count_positions(db_factory, "KR") == 1

    @pytest.mark.asyncio
    async def test_kr_exchange_code(self, adapter, risk, order_mgr, db_factory):
        """KR positions should get KRX exchange code."""
        kr_tracker = _make_tracker(adapter, risk, order_mgr, market="KR")
        kr_tracker.track("005930", 72000.0, 5)
        await kr_tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "005930", "KR")
        assert record.exchange == "KRX"

    @pytest.mark.asyncio
    async def test_us_exchange_code(self, adapter, risk, order_mgr, db_factory):
        """US positions should get NASD exchange code."""
        tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        tracker.track("AAPL", 150.0, 10)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL", "US")
        assert record.exchange == "NASD"


# ── Restore from Exchange with DB Sync ───────────────────────────────


class TestRestoreWithDbSync:
    """Test that restore_from_exchange persists restored positions to DB."""

    @pytest.mark.asyncio
    async def test_restore_persists_to_db(self, adapter, risk, order_mgr, db_factory):
        """Restored positions should be saved to DB after restore."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=155.0,
                ),
                Position(
                    symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)

        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 2
        assert await _count_positions(db_factory, "US") == 2

        aapl = await _get_position(db_factory, "AAPL")
        assert aapl is not None
        assert aapl.quantity == 10
        assert aapl.avg_price == 150.0
        assert aapl.stop_loss == 0.08  # default SL
        assert aapl.take_profit == 0.20  # default TP

    @pytest.mark.asyncio
    async def test_restore_empty_clears_stale_db(self, adapter, risk, order_mgr, db_factory):
        """When exchange returns no positions, stale DB entries should be cleaned."""
        # Pre-populate DB with a position
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("STALE", 100.0, 5)
        await tracker.sync_to_db(db_factory)
        assert await _count_positions(db_factory, "US") == 1

        # Fresh tracker restores from empty exchange
        tracker2 = _make_tracker(adapter, risk, order_mgr)
        adapter.fetch_positions = AsyncMock(return_value=[])
        restored = await tracker2.restore_from_exchange(db_factory)

        assert restored == []
        # Stale DB entry should be cleared
        assert await _count_positions(db_factory, "US") == 0

    @pytest.mark.asyncio
    async def test_restore_without_session_factory(self, adapter, risk, order_mgr):
        """Restore without session_factory still works (in-memory only)."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=155.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)

        restored = await tracker.restore_from_exchange()
        assert len(restored) == 1
        assert "AAPL" in tracker.tracked_symbols


# ── Edge Cases ───────────────────────────────────────────────────────


class TestDbEdgeCases:
    """Edge cases for DB persistence."""

    @pytest.mark.asyncio
    async def test_all_position_fields_preserved(self, adapter, risk, order_mgr, db_factory):
        """Verify all TrackedPosition fields are mapped to PositionRecord."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track(
            "TSLA",
            250.0,
            20,
            strategy="bollinger_squeeze",
            stop_loss_pct=0.15,
            take_profit_pct=0.35,
        )
        tracker._tracked["TSLA"].trailing_stop_pct = 0.05
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "TSLA")
        assert record.symbol == "TSLA"
        assert record.market == "US"
        assert record.quantity == 20
        assert record.avg_price == 250.0
        assert record.strategy_name == "bollinger_squeeze"
        assert record.stop_loss == 0.15
        assert record.take_profit == 0.35
        assert record.trailing_stop == 0.05
        assert record.opened_at is not None
        assert record.updated_at is not None

    @pytest.mark.asyncio
    async def test_none_sl_tp_stored_as_null(self, adapter, risk, order_mgr, db_factory):
        """Positions with None SL/TP should be stored correctly."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, stop_loss_pct=None, take_profit_pct=None)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.stop_loss is None
        assert record.take_profit is None

    @pytest.mark.asyncio
    async def test_sync_after_quantity_update(self, adapter, risk, order_mgr, db_factory):
        """Position quantity update is reflected in DB after sync."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="test")
        await tracker.sync_to_db(db_factory)

        # Simulate quantity change (e.g., partial sell)
        tracker._tracked["AAPL"].quantity = 5
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record.quantity == 5

    @pytest.mark.asyncio
    async def test_market_parameter_default(self, adapter, risk, order_mgr):
        """Default market is 'US'."""
        tracker = PositionTracker(adapter, risk, order_mgr)
        assert tracker._market == "US"

    @pytest.mark.asyncio
    async def test_market_parameter_kr(self, adapter, risk, order_mgr):
        """Market can be set to 'KR'."""
        tracker = PositionTracker(adapter, risk, order_mgr, market="KR")
        assert tracker._market == "KR"


# ── STOCK-9: opened_at from earliest BUY order ──────────────────────


class TestOpenedAtFromOrders:
    """Verify opened_at is set to earliest BUY order timestamp, not utcnow().

    STOCK-9: All positions had opened_at reset to sync time because
    _upsert_position_record always used datetime.utcnow() for new records.
    """

    @pytest.mark.asyncio
    async def test_restore_sets_opened_at_from_earliest_buy(self, adapter, risk, order_mgr):
        """restore_from_exchange should set opened_at from the earliest BUY order."""
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        # Create two BUY orders at different times
        first_buy_time = datetime(2026, 1, 15, 10, 30, 0)
        second_buy_time = datetime(2026, 2, 20, 14, 0, 0)
        async with factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    exchange="NASD",
                    side="BUY",
                    order_type="limit",
                    quantity=10,
                    price=150.0,
                    filled_quantity=10,
                    filled_price=150.0,
                    status="filled",
                    strategy_name="trend_following",
                    is_paper=False,
                    market="US",
                    created_at=first_buy_time,
                )
            )
            session.add(
                Order(
                    symbol="AAPL",
                    exchange="NASD",
                    side="BUY",
                    order_type="limit",
                    quantity=5,
                    price=155.0,
                    filled_quantity=5,
                    filled_price=155.0,
                    status="filled",
                    strategy_name="macd_histogram",
                    is_paper=False,
                    market="US",
                    created_at=second_buy_time,
                )
            )
            await session.commit()

        # Exchange shows position
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=15,
                    avg_price=152.0,
                    current_price=160.0,
                ),
            ]
        )

        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_exchange(factory)

        assert len(restored) == 1

        # Verify in-memory tracked position has correct opened_at
        assert tracker._tracked["AAPL"].opened_at == first_buy_time

        # Verify DB record has correct opened_at (from earliest buy)
        record = await _get_position(factory, "AAPL")
        assert record is not None
        assert record.opened_at == first_buy_time

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_restore_opened_at_per_symbol(self, adapter, risk, order_mgr):
        """Each position should get its own earliest BUY timestamp."""
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        aapl_first = datetime(2026, 1, 10, 9, 0, 0)
        msft_first = datetime(2026, 2, 5, 11, 0, 0)
        async with factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    exchange="NASD",
                    side="BUY",
                    order_type="market",
                    quantity=10,
                    price=150.0,
                    filled_quantity=10,
                    filled_price=150.0,
                    status="filled",
                    strategy_name="trend",
                    is_paper=False,
                    market="US",
                    created_at=aapl_first,
                )
            )
            session.add(
                Order(
                    symbol="MSFT",
                    exchange="NASD",
                    side="BUY",
                    order_type="market",
                    quantity=5,
                    price=300.0,
                    filled_quantity=5,
                    filled_price=300.0,
                    status="filled",
                    strategy_name="macd",
                    is_paper=False,
                    market="US",
                    created_at=msft_first,
                )
            )
            await session.commit()

        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=160.0,
                ),
                Position(
                    symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
                ),
            ]
        )

        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_exchange(factory)

        assert len(restored) == 2
        assert tracker._tracked["AAPL"].opened_at == aapl_first
        assert tracker._tracked["MSFT"].opened_at == msft_first

        # DB records should reflect per-symbol opened_at
        aapl_rec = await _get_position(factory, "AAPL")
        msft_rec = await _get_position(factory, "MSFT")
        assert aapl_rec.opened_at == aapl_first
        assert msft_rec.opened_at == msft_first

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_restore_no_orders_uses_utcnow(self, adapter, risk, order_mgr):
        """When no BUY orders exist, opened_at should fall back to utcnow()."""
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        # No orders in DB at all
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=155.0,
                ),
            ]
        )

        before = datetime.utcnow()
        tracker = _make_tracker(adapter, risk, order_mgr)
        await tracker.restore_from_exchange(factory)
        after = datetime.utcnow()

        # opened_at should be None in memory (no order found)
        assert tracker._tracked["AAPL"].opened_at is None

        # DB record should use utcnow() fallback
        record = await _get_position(factory, "AAPL")
        assert record.opened_at is not None
        assert before <= record.opened_at <= after

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_restore_ignores_paper_orders_for_opened_at(self, adapter, risk, order_mgr):
        """Paper BUY orders should NOT be used for opened_at calculation."""
        engine = create_async_engine(
            "sqlite+aiosqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        paper_time = datetime(2025, 12, 1, 8, 0, 0)  # earlier but paper
        live_time = datetime(2026, 1, 15, 10, 0, 0)  # later but live
        async with factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    exchange="NASD",
                    side="BUY",
                    order_type="market",
                    quantity=38,
                    price=145.0,
                    filled_quantity=38,
                    filled_price=145.0,
                    status="filled",
                    strategy_name="paper_strategy",
                    is_paper=True,
                    market="US",
                    created_at=paper_time,
                )
            )
            session.add(
                Order(
                    symbol="AAPL",
                    exchange="NASD",
                    side="BUY",
                    order_type="limit",
                    quantity=42,
                    price=150.0,
                    filled_quantity=42,
                    filled_price=150.0,
                    status="filled",
                    strategy_name="trend_following",
                    is_paper=False,
                    market="US",
                    created_at=live_time,
                )
            )
            await session.commit()

        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=42,
                    avg_price=150.0,
                    current_price=155.0,
                ),
            ]
        )

        tracker = _make_tracker(adapter, risk, order_mgr)
        await tracker.restore_from_exchange(factory)

        # Should use live order time, not paper order time
        assert tracker._tracked["AAPL"].opened_at == live_time

        record = await _get_position(factory, "AAPL")
        assert record.opened_at == live_time

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_upsert_preserves_existing_opened_at(self, adapter, risk, order_mgr, db_factory):
        """Upserting an existing position should NOT change opened_at."""
        original_time = datetime(2026, 1, 10, 9, 0, 0)

        # Create tracker and manually set opened_at
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend", opened_at=original_time)
        await tracker.sync_to_db(db_factory)

        # Verify original opened_at was stored
        record = await _get_position(db_factory, "AAPL")
        assert record.opened_at == original_time

        # Update the position (simulate re-sync)
        tracker._tracked["AAPL"].quantity = 20
        tracker._tracked["AAPL"].strategy = "new_strategy"
        await tracker.sync_to_db(db_factory)

        # opened_at should be unchanged
        record = await _get_position(db_factory, "AAPL")
        assert record.quantity == 20
        assert record.strategy_name == "new_strategy"
        assert record.opened_at == original_time

    @pytest.mark.asyncio
    async def test_track_with_opened_at_persists_to_db(self, adapter, risk, order_mgr, db_factory):
        """track() with opened_at should persist the timestamp to DB."""
        buy_time = datetime(2026, 2, 1, 14, 30, 0)

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("TSLA", 250.0, 20, strategy="momentum", opened_at=buy_time)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "TSLA")
        assert record is not None
        assert record.opened_at == buy_time

    @pytest.mark.asyncio
    async def test_restore_existing_db_record_preserves_opened_at(
        self, adapter, risk, order_mgr, db_factory
    ):
        """When DB record already exists, restore should preserve its opened_at."""
        original_time = datetime(2026, 1, 5, 12, 0, 0)

        # Pre-populate DB with a position having a correct opened_at
        tracker1 = _make_tracker(adapter, risk, order_mgr)
        tracker1.track("AAPL", 150.0, 10, strategy="trend", opened_at=original_time)
        await tracker1.sync_to_db(db_factory)

        record_before = await _get_position(db_factory, "AAPL")
        assert record_before.opened_at == original_time

        # Now restore from exchange (simulating restart)
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=155.0,
                ),
            ]
        )

        tracker2 = _make_tracker(adapter, risk, order_mgr)
        await tracker2.restore_from_exchange(db_factory)

        # DB record's opened_at should still be the original time
        # (upsert update path does not touch opened_at)
        record_after = await _get_position(db_factory, "AAPL")
        assert record_after.opened_at == original_time
