"""Tests for PositionTracker DB persistence (positions table sync).

Verifies that tracked positions are persisted to the `positions` DB table
via sync_to_db(), and removed on untrack, with periodic reconciliation.

Tests use trackers WITHOUT session_factory in __init__ (so track()/untrack()
don't fire background tasks), then pass session_factory explicitly to
sync_to_db() / _upsert_position_db / _remove_position_db.
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.models import Base, PositionRecord
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


# ── restore_from_db (STOCK-7) ────────────────────────────────────────


class TestRestoreFromDb:
    """Test restore_from_db() — DB-backed position recovery fallback."""

    @pytest.mark.asyncio
    async def test_restore_from_db_populates_tracker(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should populate tracker from DB records."""
        # First, create DB records via a different tracker
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.10, take_profit_pct=0.25)
        writer.track("MSFT", 300.0, 5, strategy="macd", stop_loss_pct=0.08, take_profit_pct=0.20)
        await writer.sync_to_db(db_factory)

        # Fresh tracker restores from DB
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)

        assert restored == 2
        assert set(tracker.tracked_symbols) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_restore_from_db_preserves_sl_tp(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should preserve SL/TP values from DB."""
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.12, take_profit_pct=0.50)
        await writer.sync_to_db(db_factory)

        tracker = _make_tracker(adapter, risk, order_mgr)
        await tracker.restore_from_db(db_factory)

        tracked = tracker._tracked["AAPL"]
        assert tracked.entry_price == 150.0
        assert tracked.quantity == 10
        assert tracked.strategy == "trend"
        assert tracked.stop_loss_pct == 0.12
        assert tracked.take_profit_pct == 0.50

    @pytest.mark.asyncio
    async def test_restore_from_db_empty_table(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db returns 0 when DB has no positions."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)

        assert restored == 0
        assert len(tracker.tracked_symbols) == 0

    @pytest.mark.asyncio
    async def test_restore_from_db_skips_already_tracked(
        self,
        adapter,
        risk,
        order_mgr,
        db_factory,
    ):
        """restore_from_db should not overwrite positions already in tracker."""
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("AAPL", 150.0, 10, strategy="db_strat")
        await writer.sync_to_db(db_factory)

        tracker = _make_tracker(adapter, risk, order_mgr)
        # Pre-track AAPL with different info
        tracker.track("AAPL", 160.0, 20, strategy="live_strat")
        restored = await tracker.restore_from_db(db_factory)

        assert restored == 0
        # Original in-memory data preserved
        assert tracker._tracked["AAPL"].entry_price == 160.0
        assert tracker._tracked["AAPL"].strategy == "live_strat"

    @pytest.mark.asyncio
    async def test_restore_from_db_without_session_factory(self, adapter, risk, order_mgr):
        """restore_from_db returns 0 if no session_factory."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db()
        assert restored == 0

    @pytest.mark.asyncio
    async def test_restore_from_db_market_isolation(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should only restore positions for its own market."""
        us_writer = _make_tracker(adapter, risk, order_mgr, market="US")
        kr_writer = _make_tracker(adapter, risk, order_mgr, market="KR")
        us_writer.track("AAPL", 150.0, 10, strategy="us_trend")
        kr_writer.track("005930", 72000.0, 5, strategy="kr_trend")
        await us_writer.sync_to_db(db_factory)
        await kr_writer.sync_to_db(db_factory)

        # US tracker should only get AAPL
        us_tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        restored = await us_tracker.restore_from_db(db_factory)
        assert restored == 1
        assert "AAPL" in us_tracker.tracked_symbols
        assert "005930" not in us_tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_restore_from_db_skips_zero_quantity(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should skip records with quantity <= 0."""
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("AAPL", 150.0, 10, strategy="trend")
        await writer.sync_to_db(db_factory)

        # Manually set quantity to 0 in DB
        async with db_factory() as session:
            from sqlalchemy import select

            stmt = select(PositionRecord).where(PositionRecord.symbol == "AAPL")
            result = await session.execute(stmt)
            record = result.scalar_one()
            record.quantity = 0
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)
        assert restored == 0


# ── Auto-recover Untracked (STOCK-7) ─────────────────────────────────


class TestAutoRecoverUntracked:
    """Test _auto_recover_untracked() — defense-in-depth for check_all()."""

    @pytest.mark.asyncio
    async def test_auto_recover_populates_empty_tracker(self, adapter, risk, order_mgr):
        """When tracker is empty but exchange has positions, auto-recover them."""
        exchange_positions = [
            Position(symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.5),
            Position(symbol="ALLO", exchange="NASD", quantity=38, avg_price=3.0, current_price=2.8),
        ]
        tracker = _make_tracker(adapter, risk, order_mgr)

        await tracker._auto_recover_untracked(exchange_positions)

        assert set(tracker.tracked_symbols) == {"LION", "ALLO"}
        assert tracker._tracked["LION"].quantity == 44
        assert tracker._tracked["LION"].entry_price == 5.0
        assert tracker._tracked["LION"].strategy == "auto_recovered"
        assert tracker._tracked["LION"].stop_loss_pct == 0.08  # default
        assert tracker._tracked["LION"].take_profit_pct == 0.20  # default

    @pytest.mark.asyncio
    async def test_auto_recover_uses_db_info(self, adapter, risk, order_mgr, db_factory):
        """Auto-recovery should use DB info when available (preserves SL/TP)."""
        # Pre-populate DB with position info
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("LION", 5.0, 44, strategy="trend", stop_loss_pct=0.12, take_profit_pct=0.50)
        await writer.sync_to_db(db_factory)

        exchange_positions = [
            Position(symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.5),
        ]

        # Fresh tracker with session_factory (so _load_positions_from_db works)
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker._session_factory = db_factory

        await tracker._auto_recover_untracked(exchange_positions)

        assert "LION" in tracker.tracked_symbols
        assert tracker._tracked["LION"].strategy == "trend"
        assert tracker._tracked["LION"].stop_loss_pct == 0.12
        assert tracker._tracked["LION"].take_profit_pct == 0.50

    @pytest.mark.asyncio
    async def test_auto_recover_skips_already_tracked(self, adapter, risk, order_mgr):
        """Already tracked positions should not be re-tracked."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="existing")

        exchange_positions = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=155.0
            ),
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
            ),
        ]

        await tracker._auto_recover_untracked(exchange_positions)

        # AAPL should not be overwritten
        assert tracker._tracked["AAPL"].strategy == "existing"
        # MSFT should be auto-recovered
        assert "MSFT" in tracker.tracked_symbols
        assert tracker._tracked["MSFT"].strategy == "auto_recovered"

    @pytest.mark.asyncio
    async def test_auto_recover_skips_zero_quantity(self, adapter, risk, order_mgr):
        """Positions with 0 quantity should not be recovered."""
        exchange_positions = [
            Position(
                symbol="SOLD", exchange="NASD", quantity=0, avg_price=100.0, current_price=100.0
            ),
        ]
        tracker = _make_tracker(adapter, risk, order_mgr)

        await tracker._auto_recover_untracked(exchange_positions)
        assert len(tracker.tracked_symbols) == 0

    @pytest.mark.asyncio
    async def test_auto_recover_rate_limited(self, adapter, risk, order_mgr):
        """Auto-recovery should be rate-limited (cooldown between attempts)."""
        exchange_positions = [
            Position(symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.5),
        ]
        tracker = _make_tracker(adapter, risk, order_mgr)

        # First call: should recover
        await tracker._auto_recover_untracked(exchange_positions)
        assert "LION" in tracker.tracked_symbols

        # Remove LION from tracker (simulate issue)
        tracker._tracked.clear()

        # Second call within cooldown: should NOT recover (rate-limited)
        new_positions = [
            Position(
                symbol="NEWPOS", exchange="NASD", quantity=10, avg_price=50.0, current_price=55.0
            ),
        ]
        await tracker._auto_recover_untracked(new_positions)
        assert "NEWPOS" not in tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_auto_recover_after_cooldown(self, adapter, risk, order_mgr):
        """After cooldown expires, auto-recovery should work again."""
        exchange_positions = [
            Position(symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.5),
        ]
        tracker = _make_tracker(adapter, risk, order_mgr)

        # First recovery
        await tracker._auto_recover_untracked(exchange_positions)
        assert "LION" in tracker.tracked_symbols

        tracker._tracked.clear()

        # Simulate cooldown elapsed
        tracker._last_auto_recover -= tracker._AUTO_RECOVER_COOLDOWN + 1

        new_positions = [
            Position(
                symbol="NEWPOS", exchange="NASD", quantity=10, avg_price=50.0, current_price=55.0
            ),
        ]
        await tracker._auto_recover_untracked(new_positions)
        assert "NEWPOS" in tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_auto_recover_empty_positions_noop(self, adapter, risk, order_mgr):
        """No exchange positions -> no recovery attempt."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        await tracker._auto_recover_untracked([])
        assert len(tracker.tracked_symbols) == 0


# ── check_all with Auto-Recovery (STOCK-7) ───────────────────────────


class TestCheckAllAutoRecovery:
    """Test that check_all() auto-recovers positions for SL/TP monitoring."""

    @pytest.mark.asyncio
    async def test_check_all_recovers_empty_tracker(self, adapter, risk, order_mgr):
        """check_all should auto-recover when tracker is empty but exchange has positions."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.8
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        assert len(tracker.tracked_symbols) == 0

        await tracker.check_all()

        # LION should now be tracked (auto-recovered)
        assert "LION" in tracker.tracked_symbols
        assert tracker._tracked["LION"].quantity == 44
        assert tracker._tracked["LION"].stop_loss_pct == 0.08

    @pytest.mark.asyncio
    async def test_check_all_evaluates_recovered_positions(self, adapter, risk, order_mgr):
        """After auto-recovery, check_all should evaluate SL/TP on recovered positions."""
        # Price dropped to trigger SL: entry=5.0, current=4.0 -> -20% > 8% SL
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="LION", exchange="NASD", quantity=44, avg_price=5.0, current_price=4.0
                ),
            ]
        )
        from exchange.base import OrderResult

        adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="sl1",
                symbol="LION",
                side="SELL",
                order_type="market",
                quantity=44,
                status="filled",
                filled_price=4.0,
            )
        )

        tracker = _make_tracker(adapter, risk, order_mgr)
        triggered = await tracker.check_all()

        # Should have auto-recovered AND triggered SL in same cycle
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"
        assert triggered[0]["symbol"] == "LION"

    @pytest.mark.asyncio
    async def test_check_all_no_positions_returns_empty(self, adapter, risk, order_mgr):
        """check_all with no exchange positions returns empty list."""
        adapter.fetch_positions = AsyncMock(return_value=[])
        tracker = _make_tracker(adapter, risk, order_mgr)

        triggered = await tracker.check_all()
        assert triggered == []

    @pytest.mark.asyncio
    async def test_check_all_recovery_with_db_info(self, adapter, risk, order_mgr, db_factory):
        """check_all auto-recovery should use DB info for SL/TP when available."""
        # Pre-populate DB
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("VG", 22.0, 22, strategy="macd", stop_loss_pct=0.15, take_profit_pct=0.40)
        await writer.sync_to_db(db_factory)

        # Fresh tracker with session_factory but empty in-memory
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="VG", exchange="NASD", quantity=22, avg_price=22.0, current_price=21.5
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker._session_factory = db_factory

        await tracker.check_all()

        # VG auto-recovered with DB info
        assert "VG" in tracker.tracked_symbols
        assert tracker._tracked["VG"].strategy == "macd"
        assert tracker._tracked["VG"].stop_loss_pct == 0.15
        assert tracker._tracked["VG"].take_profit_pct == 0.40


# ── _load_positions_from_db (STOCK-7) ─────────────────────────────────


class TestLoadPositionsFromDb:
    """Test _load_positions_from_db() helper."""

    @pytest.mark.asyncio
    async def test_load_returns_correct_data(self, adapter, risk, order_mgr, db_factory):
        """Should return dict with correct fields per symbol."""
        writer = _make_tracker(adapter, risk, order_mgr)
        writer.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.12, take_profit_pct=0.50)
        writer.track("MSFT", 300.0, 5, strategy="macd", stop_loss_pct=0.08, take_profit_pct=0.20)
        await writer.sync_to_db(db_factory)

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker._session_factory = db_factory
        result = await tracker._load_positions_from_db()

        assert len(result) == 2
        assert result["AAPL"]["avg_price"] == 150.0
        assert result["AAPL"]["stop_loss"] == 0.12
        assert result["AAPL"]["take_profit"] == 0.50
        assert result["AAPL"]["strategy"] == "trend"
        assert result["MSFT"]["avg_price"] == 300.0

    @pytest.mark.asyncio
    async def test_load_empty_db(self, adapter, risk, order_mgr, db_factory):
        """Empty DB should return empty dict."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker._session_factory = db_factory
        result = await tracker._load_positions_from_db()
        assert result == {}

    @pytest.mark.asyncio
    async def test_load_without_session_factory(self, adapter, risk, order_mgr):
        """No session_factory should return empty dict."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        result = await tracker._load_positions_from_db()
        assert result == {}

    @pytest.mark.asyncio
    async def test_load_market_isolation(self, adapter, risk, order_mgr, db_factory):
        """Should only return positions for the tracker's market."""
        us_writer = _make_tracker(adapter, risk, order_mgr, market="US")
        kr_writer = _make_tracker(adapter, risk, order_mgr, market="KR")
        us_writer.track("AAPL", 150.0, 10, strategy="us")
        kr_writer.track("005930", 72000.0, 5, strategy="kr")
        await us_writer.sync_to_db(db_factory)
        await kr_writer.sync_to_db(db_factory)

        us_tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        us_tracker._session_factory = db_factory
        result = await us_tracker._load_positions_from_db()

        assert len(result) == 1
        assert "AAPL" in result
        assert "005930" not in result
