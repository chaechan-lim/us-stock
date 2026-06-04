"""Tests for PositionTracker DB persistence (positions table sync).

Verifies that tracked positions are persisted to the `positions` DB table
via sync_to_db(), and removed on untrack, with periodic reconciliation.

Tests use trackers WITHOUT session_factory in __init__ (so track()/untrack()
don't fire background tasks), then pass session_factory explicitly to
sync_to_db() / _upsert_position_db / _remove_position_db.
"""

from unittest.mock import AsyncMock, MagicMock

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


def _make_tracker(
    adapter,
    risk,
    order_mgr,
    market: str = "US",
    exchange_resolver=None,
) -> PositionTracker:
    """Create a tracker WITHOUT session_factory to avoid background DB tasks."""
    return PositionTracker(
        adapter,
        risk,
        order_mgr,
        market=market,
        exchange_resolver=exchange_resolver,
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
        """US positions should get exchange code from resolver."""
        resolver = MagicMock()
        resolver.resolve = MagicMock(return_value="NASD")
        tracker = _make_tracker(adapter, risk, order_mgr, market="US", exchange_resolver=resolver)
        tracker.track("AAPL", 150.0, 10)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL", "US")
        assert record.exchange == "NASD"

    @pytest.mark.asyncio
    async def test_us_nyse_exchange_code(self, adapter, risk, order_mgr, db_factory):
        """NYSE positions should get NYSE exchange code from resolver."""
        resolver = MagicMock()
        resolver.resolve = MagicMock(return_value="NYSE")
        tracker = _make_tracker(adapter, risk, order_mgr, market="US", exchange_resolver=resolver)
        tracker.track("JPM", 180.0, 20)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "JPM", "US")
        assert record.exchange == "NYSE"

    @pytest.mark.asyncio
    async def test_us_amex_exchange_code(self, adapter, risk, order_mgr, db_factory):
        """AMEX positions should get AMEX exchange code from resolver."""
        resolver = MagicMock()
        resolver.resolve = MagicMock(return_value="AMEX")
        tracker = _make_tracker(adapter, risk, order_mgr, market="US", exchange_resolver=resolver)
        tracker.track("XLE", 90.0, 15)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "XLE", "US")
        assert record.exchange == "AMEX"


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


# ── current_price / unrealized_pnl in DB Sync ──────────────────────


class TestSyncCurrentPrice:
    """Test that sync_to_db populates current_price and unrealized_pnl."""

    @pytest.mark.asyncio
    async def test_sync_stores_current_price(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should fetch prices and store current_price in DB."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=165.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.current_price == 165.0

    @pytest.mark.asyncio
    async def test_sync_calculates_unrealized_pnl(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should calculate unrealized_pnl = (current - entry) * qty."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=165.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        # (165.0 - 150.0) * 10 = 150.0
        assert record.unrealized_pnl == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_sync_negative_unrealized_pnl(self, adapter, risk, order_mgr, db_factory):
        """unrealized_pnl should be negative when current < entry."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=285.0
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("MSFT", 300.0, 5, strategy="macd")

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "MSFT")
        assert record is not None
        assert record.current_price == 285.0
        # (285.0 - 300.0) * 5 = -75.0
        assert record.unrealized_pnl == pytest.approx(-75.0)

    @pytest.mark.asyncio
    async def test_sync_multiple_positions_prices(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should populate prices for all tracked positions."""
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
        tracker.track("AAPL", 150.0, 10)
        tracker.track("MSFT", 300.0, 5)

        await tracker.sync_to_db(db_factory)

        aapl = await _get_position(db_factory, "AAPL")
        msft = await _get_position(db_factory, "MSFT")
        assert aapl.current_price == 160.0
        assert aapl.unrealized_pnl == pytest.approx(100.0)  # (160-150)*10
        assert msft.current_price == 310.0
        assert msft.unrealized_pnl == pytest.approx(50.0)  # (310-300)*5

    @pytest.mark.asyncio
    async def test_sync_price_fetch_failure_still_syncs(self, adapter, risk, order_mgr, db_factory):
        """If price fetch fails, sync should still persist positions (without price)."""
        adapter.fetch_positions = AsyncMock(side_effect=Exception("API down"))
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")

        synced = await tracker.sync_to_db(db_factory)

        assert synced == 1
        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.strategy_name == "trend"
        # Price fields remain None when fetch fails
        assert record.current_price is None
        assert record.unrealized_pnl is None

    @pytest.mark.asyncio
    async def test_sync_zero_price_excluded_from_map(self, adapter, risk, order_mgr, db_factory):
        """Positions with current_price=0 should not populate DB current_price."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=0.0
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.current_price is None

    @pytest.mark.asyncio
    async def test_sync_price_update_on_repeated_sync(self, adapter, risk, order_mgr, db_factory):
        """Prices should update on subsequent syncs."""
        # First sync with price 155
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
        tracker.track("AAPL", 150.0, 10)
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price == 155.0
        assert record.unrealized_pnl == pytest.approx(50.0)

        # Second sync with updated price 170
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=170.0,
                ),
            ]
        )
        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price == 170.0
        assert record.unrealized_pnl == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_sync_market_data_used_when_available(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should prefer market_data over adapter for price fetch."""
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=175.0,
                ),
            ]
        )
        # Adapter returns different price (should NOT be used)
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=160.0,
                ),
            ]
        )
        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )
        tracker.track("AAPL", 150.0, 10)

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "AAPL")
        # Should use market_data price (175), not adapter price (160)
        assert record.current_price == 175.0

    @pytest.mark.asyncio
    async def test_sync_kr_positions_with_prices(self, adapter, risk, order_mgr, db_factory):
        """KR positions should also get current_price and unrealized_pnl."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="005930",
                    exchange="KRX",
                    quantity=5,
                    avg_price=72000.0,
                    current_price=75000.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr, market="KR")
        tracker.track("005930", 72000.0, 5, strategy="kr_trend")

        await tracker.sync_to_db(db_factory)

        record = await _get_position(db_factory, "005930", market="KR")
        assert record is not None
        assert record.current_price == 75000.0
        # (75000 - 72000) * 5 = 15000
        assert record.unrealized_pnl == pytest.approx(15000.0)


# ── Upsert with current_price ────────────────────────────────────────


class TestUpsertWithCurrentPrice:
    """Test _upsert_position_record with current_price parameter."""

    @pytest.mark.asyncio
    async def test_upsert_with_price_creates_record(self, adapter, risk, order_mgr, db_factory):
        """Upsert with current_price should create record with price fields."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")

        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
                current_price=165.0,
            )
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price == 165.0
        assert record.unrealized_pnl == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_upsert_without_price_leaves_null(self, adapter, risk, order_mgr, db_factory):
        """Upsert without current_price should leave fields as None."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
            )
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price is None
        assert record.unrealized_pnl is None

    @pytest.mark.asyncio
    async def test_upsert_update_adds_price(self, adapter, risk, order_mgr, db_factory):
        """Updating an existing record should set current_price."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        # First upsert without price
        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
            )
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price is None

        # Second upsert with price
        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
                current_price=160.0,
            )
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.current_price == 160.0
        assert record.unrealized_pnl == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_upsert_preserves_price_when_not_provided(
        self,
        adapter,
        risk,
        order_mgr,
        db_factory,
    ):
        """Updating without current_price should preserve existing price in DB."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        # First upsert with price
        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
                current_price=160.0,
            )
            await session.commit()

        # Second upsert without price (e.g., from track() background task)
        async with db_factory() as session:
            await tracker._upsert_position_record(
                session,
                "AAPL",
                tracker._tracked["AAPL"],
            )
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        # Price should be preserved from previous upsert
        assert record.current_price == 160.0


# ── Strategy Resolution ──────────────────────────────────────────────


class TestStrategyResolution:
    """Test improved strategy name resolution during restore and sync."""

    @pytest.mark.asyncio
    async def test_restore_uses_buy_order_strategy(self, adapter, risk, order_mgr, db_factory):
        """restore_from_exchange should use BUY order's strategy_name."""
        from datetime import datetime

        from core.models import Order

        # Create a BUY order in DB
        async with db_factory() as session:
            order = Order(
                symbol="AAPL",
                side="BUY",
                order_type="market",
                quantity=10,
                price=150.0,
                status="filled",
                strategy_name="trend_following",
                is_paper=False,
                created_at=datetime.utcnow(),
            )
            session.add(order)
            await session.commit()

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
        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 1
        assert restored[0]["strategy"] == "trend_following"

    @pytest.mark.asyncio
    async def test_restore_fallback_to_any_buy(self, adapter, risk, order_mgr, db_factory):
        """Should fall back to paper BUY order if no live BUY found."""
        from datetime import datetime

        from core.models import Order

        # Only paper BUY order
        async with db_factory() as session:
            order = Order(
                symbol="AAPL",
                side="BUY",
                order_type="market",
                quantity=10,
                price=150.0,
                status="filled",
                strategy_name="paper_macd",
                is_paper=True,
                created_at=datetime.utcnow(),
            )
            session.add(order)
            await session.commit()

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
        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 1
        assert restored[0]["strategy"] == "paper_macd"

    @pytest.mark.asyncio
    async def test_restore_fallback_to_sell_buy_strategy(
        self,
        adapter,
        risk,
        order_mgr,
        db_factory,
    ):
        """Should fall back to SELL order's buy_strategy if no BUY orders found."""
        from datetime import datetime

        from core.models import Order

        # Only SELL order with buy_strategy
        async with db_factory() as session:
            order = Order(
                symbol="AAPL",
                side="SELL",
                order_type="market",
                quantity=10,
                price=165.0,
                status="filled",
                strategy_name="trend_following:take_profit",
                buy_strategy="trend_following",
                is_paper=False,
                created_at=datetime.utcnow(),
            )
            session.add(order)
            await session.commit()

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
        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 1
        assert restored[0]["strategy"] == "trend_following"

    @pytest.mark.asyncio
    async def test_restore_unknown_when_no_orders(self, adapter, risk, order_mgr, db_factory):
        """Should default to 'unknown' when no orders found at all."""
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
        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 1
        assert restored[0]["strategy"] == "unknown"

    @pytest.mark.asyncio
    async def test_sync_resolves_unknown_strategy(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should re-resolve 'unknown' strategies from order history."""
        from datetime import datetime

        from core.models import Order

        # Add a BUY order to DB after initial tracking
        async with db_factory() as session:
            order = Order(
                symbol="AAPL",
                side="BUY",
                order_type="market",
                quantity=10,
                price=150.0,
                status="filled",
                strategy_name="rsi_divergence",
                is_paper=False,
                created_at=datetime.utcnow(),
            )
            session.add(order)
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
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        # Track with unknown strategy (simulates initial restore failure)
        tracker.track("AAPL", 150.0, 10, strategy="unknown")

        await tracker.sync_to_db(db_factory)

        # Strategy should be resolved in memory
        assert tracker._tracked["AAPL"].strategy == "rsi_divergence"
        # And persisted to DB
        record = await _get_position(db_factory, "AAPL")
        assert record.strategy_name == "rsi_divergence"

    @pytest.mark.asyncio
    async def test_sync_resolves_empty_strategy(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should also re-resolve empty string strategies."""
        from datetime import datetime

        from core.models import Order

        async with db_factory() as session:
            order = Order(
                symbol="AAPL",
                side="BUY",
                order_type="market",
                quantity=10,
                price=150.0,
                status="filled",
                strategy_name="bollinger_squeeze",
                is_paper=False,
                created_at=datetime.utcnow(),
            )
            session.add(order)
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
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="")

        await tracker.sync_to_db(db_factory)

        assert tracker._tracked["AAPL"].strategy == "bollinger_squeeze"

    @pytest.mark.asyncio
    async def test_sync_keeps_known_strategy(self, adapter, risk, order_mgr, db_factory):
        """sync_to_db should NOT re-resolve strategies that are already known."""
        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=160.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="my_strategy")

        await tracker.sync_to_db(db_factory)

        # Strategy should remain unchanged
        assert tracker._tracked["AAPL"].strategy == "my_strategy"
        record = await _get_position(db_factory, "AAPL")
        assert record.strategy_name == "my_strategy"


# ── opened_at from Orders ──────────────────────────────────────────


class TestOpenedAtFromOrders:
    """Test that opened_at is set from the earliest BUY order, not utcnow().

    STOCK-9: When positions are recreated (e.g., during restore_from_exchange
    or sync_to_db), opened_at should reflect the actual first buy time from
    the orders table, not the moment the PositionRecord is created.
    """

    @pytest.mark.asyncio
    async def test_new_position_uses_first_buy_order_time(
        self, adapter, risk, order_mgr, db_factory
    ):
        """New PositionRecord should get opened_at from earliest BUY order."""
        from datetime import datetime

        from core.models import Order

        buy_time = datetime(2025, 6, 15, 14, 30, 0)
        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="market",
                    quantity=10,
                    price=150.0,
                    status="filled",
                    strategy_name="trend_following",
                    is_paper=False,
                    created_at=buy_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend_following")

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record is not None
        assert record.opened_at == buy_time

    @pytest.mark.asyncio
    async def test_opened_at_picks_earliest_buy(self, adapter, risk, order_mgr, db_factory):
        """When multiple BUY orders exist, opened_at uses the earliest one."""
        from datetime import datetime

        from core.models import Order

        early_time = datetime(2025, 3, 1, 10, 0, 0)
        late_time = datetime(2025, 6, 15, 14, 30, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="market",
                    quantity=5,
                    price=140.0,
                    status="filled",
                    strategy_name="trend_following",
                    is_paper=False,
                    created_at=early_time,
                )
            )
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="market",
                    quantity=5,
                    price=150.0,
                    status="filled",
                    strategy_name="macd_cross",
                    is_paper=False,
                    created_at=late_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 145.0, 10)

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.opened_at == early_time

    @pytest.mark.asyncio
    async def test_opened_at_falls_back_without_orders(self, adapter, risk, order_mgr, db_factory):
        """When no BUY orders exist, opened_at falls back to ~utcnow()."""
        from datetime import datetime, timedelta

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("NEWSTOCK", 100.0, 5)

        before = datetime.utcnow()
        async with db_factory() as session:
            await tracker._upsert_position_record(session, "NEWSTOCK", tracker._tracked["NEWSTOCK"])
            await session.commit()
        after = datetime.utcnow()

        record = await _get_position(db_factory, "NEWSTOCK")
        assert record is not None
        # opened_at should be approximately now (within a few seconds)
        assert before - timedelta(seconds=2) <= record.opened_at <= after + timedelta(seconds=2)

    @pytest.mark.asyncio
    async def test_opened_at_ignores_paper_orders(self, adapter, risk, order_mgr, db_factory):
        """Paper orders (is_paper=True) should NOT be used for opened_at."""
        from datetime import datetime, timedelta

        from core.models import Order

        paper_time = datetime(2024, 1, 1, 10, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="market",
                    quantity=10,
                    price=150.0,
                    status="filled",
                    strategy_name="trend",
                    is_paper=True,
                    created_at=paper_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        # Should NOT use paper_time (2024-01-01), should be ~now
        assert record.opened_at > paper_time + timedelta(days=1)

    @pytest.mark.asyncio
    async def test_opened_at_ignores_sell_orders(self, adapter, risk, order_mgr, db_factory):
        """SELL orders should NOT be used for opened_at."""
        from datetime import datetime, timedelta

        from core.models import Order

        sell_time = datetime(2024, 6, 1, 10, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="SELL",
                    order_type="market",
                    quantity=10,
                    price=170.0,
                    status="filled",
                    strategy_name="trend:take_profit",
                    is_paper=False,
                    created_at=sell_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        # Should NOT use sell_time, should be ~now
        assert record.opened_at > sell_time + timedelta(days=1)

    @pytest.mark.asyncio
    async def test_opened_at_preserved_on_update(self, adapter, risk, order_mgr, db_factory):
        """Updating an existing PositionRecord should NOT reset opened_at."""
        from datetime import datetime

        from core.models import Order

        buy_time = datetime(2025, 1, 10, 9, 30, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="market",
                    quantity=10,
                    price=150.0,
                    status="filled",
                    strategy_name="trend",
                    is_paper=False,
                    created_at=buy_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")

        # First upsert creates the record
        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.opened_at == buy_time

        # Second upsert updates the record (should NOT reset opened_at)
        tracker._tracked["AAPL"].quantity = 20
        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        assert record.opened_at == buy_time
        assert record.quantity == 20

    @pytest.mark.asyncio
    async def test_opened_at_survives_resync(self, adapter, risk, order_mgr, db_factory):
        """Delete and recreate position should recover correct opened_at."""
        from datetime import datetime

        from sqlalchemy import delete

        from core.models import Order, PositionRecord

        buy_time = datetime(2025, 2, 20, 11, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="TSLA",
                    side="BUY",
                    order_type="market",
                    quantity=5,
                    price=200.0,
                    status="filled",
                    strategy_name="momentum",
                    is_paper=False,
                    created_at=buy_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("TSLA", 200.0, 5, strategy="momentum")

        # Create initial record
        async with db_factory() as session:
            await tracker._upsert_position_record(session, "TSLA", tracker._tracked["TSLA"])
            await session.commit()

        record = await _get_position(db_factory, "TSLA")
        assert record.opened_at == buy_time

        # Delete the position record (simulates DB clear / STOCK-2 scenario)
        async with db_factory() as session:
            await session.execute(delete(PositionRecord).where(PositionRecord.symbol == "TSLA"))
            await session.commit()

        # Recreate — should recover opened_at from orders
        async with db_factory() as session:
            await tracker._upsert_position_record(session, "TSLA", tracker._tracked["TSLA"])
            await session.commit()

        record = await _get_position(db_factory, "TSLA")
        assert record.opened_at == buy_time

    @pytest.mark.asyncio
    async def test_restore_from_exchange_preserves_opened_at(
        self, adapter, risk, order_mgr, db_factory
    ):
        """Full restore_from_exchange path should set correct opened_at."""
        from datetime import datetime

        from core.models import Order

        buy_time = datetime(2025, 5, 10, 15, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="MSFT",
                    side="BUY",
                    order_type="market",
                    quantity=8,
                    price=400.0,
                    status="filled",
                    strategy_name="rsi_divergence",
                    is_paper=False,
                    created_at=buy_time,
                )
            )
            await session.commit()

        adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="MSFT",
                    exchange="NASD",
                    quantity=8,
                    avg_price=400.0,
                    current_price=420.0,
                ),
            ]
        )
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_exchange(db_factory)

        assert len(restored) == 1
        record = await _get_position(db_factory, "MSFT")
        assert record is not None
        assert record.opened_at == buy_time

    @pytest.mark.asyncio
    async def test_kr_position_opened_at_from_orders(self, adapter, risk, order_mgr, db_factory):
        """KR market positions should also get opened_at from orders."""
        from datetime import datetime

        from core.models import Order

        buy_time = datetime(2025, 4, 1, 9, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="005930",
                    side="BUY",
                    order_type="market",
                    quantity=10,
                    price=72000.0,
                    status="filled",
                    strategy_name="kr_trend",
                    is_paper=False,
                    created_at=buy_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr, market="KR")
        tracker.track("005930", 72000.0, 10, strategy="kr_trend")

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "005930", tracker._tracked["005930"])
            await session.commit()

        record = await _get_position(db_factory, "005930", market="KR")
        assert record is not None
        assert record.opened_at == buy_time

    @pytest.mark.asyncio
    async def test_opened_at_ignores_cancelled_orders(self, adapter, risk, order_mgr, db_factory):
        """Cancelled orders should NOT be used for opened_at."""
        from datetime import datetime, timedelta

        from core.models import Order

        cancelled_time = datetime(2024, 12, 1, 10, 0, 0)

        async with db_factory() as session:
            session.add(
                Order(
                    symbol="AAPL",
                    side="BUY",
                    order_type="limit",
                    quantity=10,
                    price=150.0,
                    status="cancelled",
                    strategy_name="trend",
                    is_paper=False,
                    created_at=cancelled_time,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10)

        async with db_factory() as session:
            await tracker._upsert_position_record(session, "AAPL", tracker._tracked["AAPL"])
            await session.commit()

        record = await _get_position(db_factory, "AAPL")
        # Should NOT use cancelled order time, should be ~now
        assert record.opened_at > cancelled_time + timedelta(days=1)


# ── restore_from_db: Fallback Recovery (STOCK-7) ───────────────────


class TestRestoreFromDb:
    """Test restore_from_db() fallback when exchange API is unavailable."""

    @pytest.mark.asyncio
    async def test_restore_from_db_basic(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should re-populate tracker from positions DB table."""
        # First, populate DB via normal flow
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")
        tracker.track("MSFT", 300.0, 5, strategy="macd")
        await tracker.sync_to_db(db_factory)

        # Create a fresh tracker (simulating restart)
        tracker2 = _make_tracker(adapter, risk, order_mgr)
        assert len(tracker2.tracked_symbols) == 0

        # Restore from DB
        restored = await tracker2.restore_from_db(db_factory)
        assert len(restored) == 2
        assert set(tracker2.tracked_symbols) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_restore_from_db_preserves_sl_tp(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should preserve SL/TP from DB records."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend", stop_loss_pct=0.12, take_profit_pct=0.30)
        await tracker.sync_to_db(db_factory)

        tracker2 = _make_tracker(adapter, risk, order_mgr)
        await tracker2.restore_from_db(db_factory)

        status = {s["symbol"]: s for s in tracker2.get_status()}
        assert status["AAPL"]["stop_loss_pct"] == 0.12
        assert status["AAPL"]["take_profit_pct"] == 0.30
        assert status["AAPL"]["strategy"] == "trend"

    @pytest.mark.asyncio
    async def test_restore_from_db_empty(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db returns empty list when no DB positions exist."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)
        assert restored == []
        assert len(tracker.tracked_symbols) == 0

    @pytest.mark.asyncio
    async def test_restore_from_db_no_session_factory(self, adapter, risk, order_mgr):
        """restore_from_db returns empty when no session_factory is available."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db()
        assert restored == []

    @pytest.mark.asyncio
    async def test_restore_from_db_skips_zero_quantity(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db skips records with quantity <= 0."""
        # Manually insert a zero-qty record
        async with db_factory() as session:
            session.add(
                PositionRecord(
                    market="US",
                    symbol="DEAD",
                    exchange="NASD",
                    quantity=0,
                    avg_price=50.0,
                    strategy_name="test",
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)
        assert len(restored) == 0
        assert "DEAD" not in tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_restore_from_db_skips_already_tracked(
        self,
        adapter,
        risk,
        order_mgr,
        db_factory,
    ):
        """restore_from_db should not overwrite existing tracked positions."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="existing")
        await tracker.sync_to_db(db_factory)

        # Tracker already has AAPL, restore should skip it
        restored = await tracker.restore_from_db(db_factory)
        assert len(restored) == 0
        status = {s["symbol"]: s for s in tracker.get_status()}
        assert status["AAPL"]["strategy"] == "existing"

    @pytest.mark.asyncio
    async def test_restore_from_db_market_isolation(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should only restore positions for its own market."""
        # Create US positions via tracker
        us_tracker = _make_tracker(adapter, risk, order_mgr, market="US")
        us_tracker.track("AAPL", 150.0, 10, strategy="trend")
        await us_tracker.sync_to_db(db_factory)

        # Create KR positions via tracker
        kr_tracker = _make_tracker(adapter, risk, order_mgr, market="KR")
        kr_tracker.track("005930", 70000.0, 5, strategy="macd")
        await kr_tracker.sync_to_db(db_factory)

        # Fresh US tracker should only get AAPL
        us_tracker2 = _make_tracker(adapter, risk, order_mgr, market="US")
        us_restored = await us_tracker2.restore_from_db(db_factory)
        assert len(us_restored) == 1
        assert us_restored[0]["symbol"] == "AAPL"

        # Fresh KR tracker should only get 005930
        kr_tracker2 = _make_tracker(adapter, risk, order_mgr, market="KR")
        kr_restored = await kr_tracker2.restore_from_db(db_factory)
        assert len(kr_restored) == 1
        assert kr_restored[0]["symbol"] == "005930"

    @pytest.mark.asyncio
    async def test_restore_from_db_returns_source_db(self, adapter, risk, order_mgr, db_factory):
        """restore_from_db should include source='db' in results."""
        tracker = _make_tracker(adapter, risk, order_mgr)
        tracker.track("AAPL", 150.0, 10, strategy="trend")
        await tracker.sync_to_db(db_factory)

        tracker2 = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker2.restore_from_db(db_factory)
        assert len(restored) == 1
        assert restored[0]["source"] == "db"

    @pytest.mark.asyncio
    async def test_restore_from_db_populates_entry_unix(
        self, adapter, risk, order_mgr, db_factory
    ):
        """entry_unix must be restored from PositionRecord.opened_at.

        2026-06-04 regression: monotonic tracked_at resets on every process
        restart, which made the min_hold gate think every position was
        freshly opened. The wall-clock entry_unix derived from opened_at
        is the source of truth that survives restart.
        """
        import time as _t
        from datetime import datetime, timedelta, timezone

        from core.models import PositionRecord

        opened = datetime.utcnow() - timedelta(days=5)
        async with db_factory() as session:
            session.add(
                PositionRecord(
                    market="US",
                    symbol="OLD",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    strategy_name="trend",
                    opened_at=opened,
                )
            )
            await session.commit()

        tracker = _make_tracker(adapter, risk, order_mgr)
        restored = await tracker.restore_from_db(db_factory)
        assert len(restored) == 1

        tracked = tracker._tracked["OLD"]
        expected = opened.replace(tzinfo=timezone.utc).timestamp()
        # Allow small float jitter from timestamp() round-trip.
        assert abs(tracked.entry_unix - expected) < 1.0
        # And the wall-clock elapsed since entry_unix should be ~5 days.
        elapsed = _t.time() - tracked.entry_unix
        assert 4.9 * 86400 < elapsed < 5.1 * 86400


# ── Auto-Recovery in check_all (STOCK-7) ────────────────────────────


class TestAutoRecoverUntracked:
    """Test _auto_recover_untracked() called from check_all()."""

    @pytest.mark.asyncio
    async def test_check_all_triggers_auto_recover_when_empty(self, risk, order_mgr):
        """check_all should auto-recover positions when tracker is empty."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=155.0,
                ),
                Position(
                    symbol="MSFT",
                    exchange="NASD",
                    quantity=5,
                    avg_price=300.0,
                    current_price=310.0,
                ),
            ]
        )

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )
        assert len(tracker.tracked_symbols) == 0

        # check_all should auto-recover
        await tracker.check_all()

        assert set(tracker.tracked_symbols) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_auto_recover_cooldown(self, risk, order_mgr):
        """Auto-recovery should respect cooldown interval."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        # First call: empty, second call: has positions
        market_data.get_positions = AsyncMock(return_value=[])

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )

        # First call - should attempt recovery (cooldown starts)
        await tracker.check_all()
        assert market_data.get_positions.call_count == 1

        # Second call - should skip recovery (within cooldown)
        await tracker.check_all()
        # Still 1 call because cooldown prevents re-check
        assert market_data.get_positions.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_recover_exchange_failure_falls_back_to_db(
        self,
        risk,
        order_mgr,
        db_factory,
    ):
        """When exchange fetch fails, auto-recovery falls back to DB."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(side_effect=Exception("API down"))

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            session_factory=db_factory,
            market="US",
        )

        # Populate DB with a position (simulating previous session)
        async with db_factory() as session:
            session.add(
                PositionRecord(
                    market="US",
                    symbol="GOOG",
                    exchange="NASD",
                    quantity=3,
                    avg_price=140.0,
                    current_price=145.0,
                    strategy_name="macd",
                    stop_loss=0.08,
                    take_profit=0.20,
                )
            )
            await session.commit()

        await tracker.check_all()
        assert "GOOG" in tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_auto_recover_skips_already_tracked(self, risk, order_mgr):
        """Auto-recovery should not overwrite already-tracked positions."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(
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

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )
        # Pre-track with specific strategy
        tracker.track("AAPL", 148.0, 10, strategy="my_strategy")

        # Untrack triggers empty, but auto-recover should not happen
        # because tracker is not empty
        status_before = tracker.get_status()
        assert status_before[0]["strategy"] == "my_strategy"

    @pytest.mark.asyncio
    async def test_auto_recover_skips_zero_quantity(self, risk, order_mgr):
        """Auto-recovery should skip positions with zero quantity."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="DEAD",
                    exchange="NASD",
                    quantity=0,
                    avg_price=50.0,
                    current_price=45.0,
                ),
                Position(
                    symbol="LIVE",
                    exchange="NASD",
                    quantity=5,
                    avg_price=100.0,
                    current_price=105.0,
                ),
            ]
        )

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )
        await tracker.check_all()
        assert "DEAD" not in tracker.tracked_symbols
        assert "LIVE" in tracker.tracked_symbols

    @pytest.mark.asyncio
    async def test_check_all_no_auto_recover_when_tracked_exists(
        self,
        risk,
        order_mgr,
    ):
        """check_all should NOT auto-recover when tracker has positions."""
        adapter = AsyncMock()
        market_data = AsyncMock()
        market_data.get_positions = AsyncMock(
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

        tracker = PositionTracker(
            adapter,
            risk,
            order_mgr,
            market_data=market_data,
            market="US",
        )
        tracker.track("AAPL", 150.0, 10, strategy="existing")

        # check_all should evaluate normally without auto-recover
        await tracker.check_all()
        # get_positions called for SL/TP evaluation, not for recovery
        assert "AAPL" in tracker.tracked_symbols
