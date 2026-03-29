"""Tests for auto-migration of missing columns and indexes (STOCK-11/STOCK-13/STOCK-63).

Validates that ensure_columns() correctly detects and adds columns that
exist in the ORM model but are missing from the physical DB schema, and
that ensure_indexes() creates missing indexes.

STOCK-11: Auto-migration for is_paper column addition.
STOCK-13: Also creates idx_orders_is_paper index and enforces NOT NULL.
STOCK-63: run_alembic_upgrade() runs alembic upgrade head on startup.
"""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import Column, DateTime, Float, Integer, String, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from db.migrations import (
    _INITIAL_REVISION,
    _MIGRATION_SENTINELS,
    ensure_columns,
    ensure_indexes,
    run_alembic_upgrade,
)


class _OldBase(DeclarativeBase):
    """Simulates the old schema without is_paper / buy_strategy."""

    pass


class _OldOrder(_OldBase):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String(20), default="pending")
    strategy_name = Column(String(50))
    kis_order_id = Column(String(50))
    pnl = Column(Float)
    created_at = Column(DateTime)
    exchange = Column(String(10), default="NASD")
    market = Column(String(2), default="US")
    session = Column(String(20), default="regular")
    filled_quantity = Column(Float, default=0)
    filled_price = Column(Float)
    filled_at = Column(DateTime)


@pytest_asyncio.fixture
async def old_engine():
    """Create an engine with the OLD schema (missing is_paper, buy_strategy)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(_OldBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def full_engine():
    """Create an engine with the FULL schema (all columns present)."""
    from core.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def empty_engine():
    """Create an engine with NO tables at all."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


# ── Column migration tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_adds_missing_is_paper_column(old_engine):
    """ensure_columns adds is_paper to orders table when missing."""
    # Verify is_paper is NOT in the old schema
    async with old_engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "is_paper" not in cols

    # Run migration
    added = await ensure_columns(old_engine)

    assert "orders.is_paper" in added

    # Verify column now exists
    async with old_engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "is_paper" in cols


@pytest.mark.asyncio
async def test_adds_missing_buy_strategy_column(old_engine):
    """ensure_columns adds buy_strategy to orders table when missing."""
    async with old_engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "buy_strategy" not in cols

    added = await ensure_columns(old_engine)

    assert "orders.buy_strategy" in added

    async with old_engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "buy_strategy" in cols


@pytest.mark.asyncio
async def test_adds_all_missing_columns(old_engine):
    """ensure_columns adds all missing columns in one call."""
    added = await ensure_columns(old_engine)

    assert len(added) == 3
    assert "orders.is_paper" in added
    assert "orders.buy_strategy" in added
    assert "orders.pnl_pct" in added


@pytest.mark.asyncio
async def test_idempotent_no_change_when_columns_present(full_engine):
    """ensure_columns does nothing when all columns already exist."""
    added = await ensure_columns(full_engine)

    assert added == []


@pytest.mark.asyncio
async def test_idempotent_second_run(old_engine):
    """Running ensure_columns twice does not fail or re-add columns."""
    first = await ensure_columns(old_engine)
    assert len(first) == 3

    second = await ensure_columns(old_engine)
    assert second == []


@pytest.mark.asyncio
async def test_no_tables_present(empty_engine):
    """ensure_columns is a no-op when the target table does not exist."""
    added = await ensure_columns(empty_engine)

    assert added == []


@pytest.mark.asyncio
async def test_is_paper_default_value(old_engine):
    """Newly added is_paper column defaults to FALSE for existing rows."""
    # Insert a row BEFORE migration (simulate existing prod data)
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, status) "
                "VALUES ('AAPL', 'BUY', 'market', 10, 150.0, 'filled')"
            )
        )

    # Run migration
    await ensure_columns(old_engine)

    # Verify the default value is applied
    async with old_engine.connect() as conn:
        result = await conn.execute(text("SELECT is_paper FROM orders WHERE symbol='AAPL'"))
        row = result.fetchone()
    # SQLite stores FALSE as 0
    assert row[0] in (False, 0)


@pytest.mark.asyncio
async def test_queries_work_after_migration(old_engine):
    """After migration, ORM queries referencing is_paper work correctly."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from core.models import Order

    # Insert data before migration
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, status) "
                "VALUES ('AAPL', 'BUY', 'market', 10, 150.0, 'filled')"
            )
        )

    # Run migration
    await ensure_columns(old_engine)

    # Now ORM queries referencing is_paper should work
    from sqlalchemy import select

    factory = async_sessionmaker(old_engine, expire_on_commit=False)
    async with factory() as session:
        stmt = select(Order).where(Order.is_paper == False)  # noqa: E712
        result = await session.execute(stmt)
        orders = list(result.scalars().all())

    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].is_paper in (False, 0)


@pytest.mark.asyncio
async def test_trade_history_works_after_migration(old_engine):
    """After migration, get_trade_history(exclude_paper=True) works.

    This is the exact query that failed in production, causing all PnL = 0.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from db.trade_repository import TradeRepository

    # Insert data before migration
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                "status, exchange, market) "
                "VALUES ('AAPL', 'SELL', 'market', 10, 180.0, 'filled', 'NASD', 'US')"
            )
        )

    # Run migration
    await ensure_columns(old_engine)

    # The exact query that was failing in production
    factory = async_sessionmaker(old_engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)
        trades = await repo.get_trade_history(limit=50, exclude_paper=True)

    assert len(trades) == 1
    assert trades[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_is_paper_not_null_enforced(old_engine):
    """After migration, is_paper column rejects NULL values (STOCK-13)."""
    await ensure_columns(old_engine)

    # Attempt to insert a row with NULL is_paper should fail
    with pytest.raises(Exception):
        async with old_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                    "status, is_paper) "
                    "VALUES ('TSLA', 'BUY', 'market', 5, 200.0, 'filled', NULL)"
                )
            )


# ── Index migration tests (STOCK-13) ──────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_indexes_creates_is_paper_index(old_engine):
    """ensure_indexes creates idx_orders_is_paper after column is added."""
    # First add the column
    await ensure_columns(old_engine)

    # Then create indexes
    created = await ensure_indexes(old_engine)

    assert "idx_orders_is_paper" in created


@pytest.mark.asyncio
async def test_ensure_indexes_idempotent(old_engine):
    """Running ensure_indexes twice does not fail or re-create indexes."""
    await ensure_columns(old_engine)

    first = await ensure_indexes(old_engine)
    assert "idx_orders_is_paper" in first

    second = await ensure_indexes(old_engine)
    assert second == []


@pytest.mark.asyncio
async def test_ensure_indexes_skips_when_column_missing(old_engine):
    """ensure_indexes skips index creation if the column doesn't exist yet."""
    # Don't run ensure_columns — is_paper column doesn't exist
    created = await ensure_indexes(old_engine)

    assert created == []


@pytest.mark.asyncio
async def test_ensure_indexes_no_tables(empty_engine):
    """ensure_indexes is a no-op when the target table does not exist."""
    created = await ensure_indexes(empty_engine)

    assert created == []


@pytest.mark.asyncio
async def test_ensure_indexes_noop_when_all_present(full_engine):
    """ensure_indexes does nothing when all indexes already exist."""
    # full_engine uses Base.metadata.create_all which creates all indexes
    created = await ensure_indexes(full_engine)

    assert created == []


@pytest.mark.asyncio
async def test_index_actually_used_in_query(old_engine):
    """After migration, the index is usable for is_paper filter queries."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from db.trade_repository import TradeRepository

    await ensure_columns(old_engine)
    await ensure_indexes(old_engine)

    # Insert paper + live orders
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                "status, exchange, market, is_paper) VALUES "
                "('AAPL', 'BUY', 'market', 10, 150.0, 'filled', 'NASD', 'US', 0), "
                "('TSLA', 'BUY', 'market', 5, 200.0, 'filled', 'NASD', 'US', 1)"
            )
        )

    factory = async_sessionmaker(old_engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)

        # exclude_paper should filter out TSLA
        live_trades = await repo.get_trade_history(limit=50, exclude_paper=True)
        assert len(live_trades) == 1
        assert live_trades[0].symbol == "AAPL"

        # include all should return both
        all_trades = await repo.get_trade_history(limit=50, exclude_paper=False)
        assert len(all_trades) == 2


@pytest.mark.asyncio
async def test_full_migration_sequence(old_engine):
    """Full migration sequence: columns then indexes, matching production startup."""
    # Simulate production startup sequence
    added_cols = await ensure_columns(old_engine)
    created_idxs = await ensure_indexes(old_engine)

    assert "orders.is_paper" in added_cols
    assert "orders.buy_strategy" in added_cols
    assert "orders.pnl_pct" in added_cols
    assert "idx_orders_is_paper" in created_idxs

    # Verify everything is in place
    async with old_engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
        indexes = await conn.run_sync(
            lambda sc: {idx["name"] for idx in inspect(sc).get_indexes("orders")}
        )

    assert "is_paper" in cols
    assert "buy_strategy" in cols
    assert "pnl_pct" in cols
    assert "idx_orders_is_paper" in indexes

    # Second run is a complete no-op
    assert await ensure_columns(old_engine) == []
    assert await ensure_indexes(old_engine) == []


@pytest.mark.asyncio
async def test_position_tracker_query_works_after_migration(old_engine):
    """After migration, position_tracker's is_paper filter query works.

    This reproduces the production error:
    'position_tracker: Failed to look up entry info from DB: column orders.is_paper does not exist'
    """
    from sqlalchemy import desc, select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from core.models import Order

    # Insert data before migration
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                "status, strategy_name, exchange, market) "
                "VALUES ('AAPL', 'BUY', 'market', 10, 150.0, 'filled', "
                "'momentum', 'NASD', 'US')"
            )
        )

    # Run full migration
    await ensure_columns(old_engine)
    await ensure_indexes(old_engine)

    # Execute the exact query from position_tracker.restore_from_exchange()
    factory = async_sessionmaker(old_engine, expire_on_commit=False)
    async with factory() as session:
        stmt = (
            select(Order)
            .where(
                Order.symbol == "AAPL",
                Order.side == "BUY",
                Order.status.in_(["filled", "submitted"]),
                Order.is_paper == False,  # noqa: E712
            )
            .order_by(desc(Order.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        order = result.scalar_one_or_none()

    assert order is not None
    assert order.symbol == "AAPL"
    assert order.strategy_name == "momentum"


@pytest.mark.asyncio
async def test_reconcile_query_works_after_migration(old_engine):
    """After migration, reconcile_pending_orders' is_paper filter works.

    This reproduces:
    'api.trades: Failed to reconcile pending DB orders: column orders.is_paper does not exist'
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from db.trade_repository import TradeRepository

    # Insert a pending order before migration
    async with old_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                "status, exchange, market) "
                "VALUES ('MSFT', 'BUY', 'market', 5, 400.0, 'pending', 'NASD', 'US')"
            )
        )

    await ensure_columns(old_engine)
    await ensure_indexes(old_engine)

    # Execute the exact query from reconcile_pending_orders
    factory = async_sessionmaker(old_engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)
        pending = await repo.get_open_orders(exclude_paper=True)

    assert len(pending) == 1
    assert pending[0].symbol == "MSFT"
    assert pending[0].status == "pending"


# ── run_alembic_upgrade tests (STOCK-63) ──────────────────────────────


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a mock CompletedProcess for subprocess.run."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


@pytest.fixture
def alembic_dir(tmp_path: Path) -> Path:
    """Dummy backend dir for subprocess cwd (doesn't need real alembic.ini)."""
    return tmp_path


@pytest_asyncio.fixture
async def fresh_engine():
    """Engine with NO tables — simulates a brand-new database.

    Note: in the normal ``main.py`` startup sequence ``Base.metadata.create_all``
    runs *before* ``run_alembic_upgrade``, so ``run_alembic_upgrade`` never
    receives a truly empty DB through the production path.  This fixture
    exercises the function in isolation (e.g. when called directly without the
    preceding ``create_all``).  The production new-install scenario is captured
    by ``legacy_engine_with_stock58`` (tables created by ``create_all`` with the
    current ORM models already applied).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def legacy_engine_without_stock58():
    """Engine with orders/portfolio_snapshots tables but WITHOUT stock-58 columns.

    Simulates a production DB that was bootstrapped via create_all at the
    initial schema but never had STOCK-58 migration applied.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side TEXT, "
                "order_type TEXT, quantity REAL, price REAL, status TEXT, "
                "exchange TEXT DEFAULT 'NASD', market TEXT DEFAULT 'US', "
                "strategy_name TEXT, kis_order_id TEXT, pnl REAL, "
                "created_at DATETIME, filled_at DATETIME, "
                "filled_quantity REAL DEFAULT 0, filled_price REAL)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE portfolio_snapshots ("
                "id INTEGER PRIMARY KEY, total_value_usd REAL NOT NULL, "
                "cash_usd REAL NOT NULL, invested_usd REAL NOT NULL, "
                "realized_pnl REAL, unrealized_pnl REAL, daily_pnl REAL, "
                "drawdown_pct REAL, recorded_at DATETIME)"
            )
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def legacy_engine_with_stock58():
    """Engine with tables AND stock-58 columns — simulates fresh DB via create_all.

    When create_all builds tables from current ORM models, all columns are
    present.  We should stamp at the latest sentinel revision and then run
    upgrade (which will be a no-op since schema is already current).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side TEXT, "
                "order_type TEXT, quantity REAL, price REAL, status TEXT, "
                "exchange TEXT DEFAULT 'NASD', market TEXT DEFAULT 'US', "
                "strategy_name TEXT, kis_order_id TEXT, pnl REAL, "
                "created_at DATETIME, filled_at DATETIME, "
                "filled_quantity REAL DEFAULT 0, filled_price REAL)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE portfolio_snapshots ("
                "id INTEGER PRIMARY KEY, total_value_usd REAL NOT NULL, "
                "cash_usd REAL NOT NULL, invested_usd REAL NOT NULL, "
                "realized_pnl REAL, unrealized_pnl REAL, daily_pnl REAL, "
                "drawdown_pct REAL, recorded_at DATETIME, "
                "usd_krw_rate REAL)"  # STOCK-58 column present
            )
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def tracked_engine():
    """Engine with tables and alembic_version — simulates fully tracked DB."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE TABLE orders (id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side TEXT)")
        )
        await conn.execute(
            text("CREATE TABLE alembic_version (version_num TEXT NOT NULL PRIMARY KEY)")
        )
        await conn.execute(text("INSERT INTO alembic_version VALUES ('607feca4f8b7')"))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def legacy_engine_only_orders():
    """Engine with only the ``orders`` table — ``portfolio_snapshots`` does NOT exist.

    Exercises the edge case where ``has_base_tables=True`` but
    ``has_stock58_columns=False`` because the sentinel table itself is absent.
    The code should stamp at ``_INITIAL_REVISION`` (not head) and then run
    upgrade, which will create/alter ``portfolio_snapshots`` via the STOCK-58
    migration.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, side TEXT, "
                "order_type TEXT, quantity REAL, price REAL, status TEXT, "
                "exchange TEXT DEFAULT 'NASD', market TEXT DEFAULT 'US', "
                "strategy_name TEXT, kis_order_id TEXT, pnl REAL, "
                "created_at DATETIME, filled_at DATETIME, "
                "filled_quantity REAL DEFAULT 0, filled_price REAL)"
            )
        )
        # portfolio_snapshots intentionally NOT created here
    yield engine
    await engine.dispose()


# ── Fresh DB (no tables) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_fresh_db_no_stamp(fresh_engine, alembic_dir):
    """Fresh DB: no tables → no stamp call, just upgrade head.

    Note: in production ``create_all`` always runs before ``run_alembic_upgrade``
    (see ``main.py``), so this zero-table state is only reached when calling the
    function directly.  The production new-install path is covered by
    ``test_run_alembic_upgrade_legacy_with_all_sentinels_stamps_at_latest_revision``.
    """
    success_proc = _make_proc(0, "No migrations to apply.\n")

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(fresh_engine, alembic_dir=alembic_dir)

    # Only upgrade head should have been called — no stamp needed
    calls = [c.args[0] for c in mock_cmd.call_args_list]
    assert calls == [["upgrade", "head"]]


# ── Legacy DB without STOCK-58 columns ───────────────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_legacy_without_stock58_stamps_initial(
    legacy_engine_without_stock58, alembic_dir
):
    """Legacy DB without STOCK-58 columns → stamp at _INITIAL_REVISION then upgrade."""
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(legacy_engine_without_stock58, alembic_dir=alembic_dir)

    calls = [c.args[0] for c in mock_cmd.call_args_list]
    assert calls[0] == ["stamp", _INITIAL_REVISION]
    assert calls[1] == ["upgrade", "head"]


# ── Legacy DB with all sentinel columns present ───────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_legacy_with_all_sentinels_stamps_at_latest_revision(
    legacy_engine_with_stock58, alembic_dir
):
    """Legacy DB with all sentinel columns → stamp at newest sentinel, then upgrade."""
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(legacy_engine_with_stock58, alembic_dir=alembic_dir)

    calls = [c.args[0] for c in mock_cmd.call_args_list]
    # Should stamp at the newest sentinel's revision (not the dynamic "head" string),
    # so that the stamp target is deterministic regardless of when alembic resolves it.
    assert calls[0] == ["stamp", _MIGRATION_SENTINELS[0][0]]
    assert calls[1] == ["upgrade", "head"]


# ── Tracked DB (alembic_version table present) ───────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_tracked_db_no_stamp(tracked_engine, alembic_dir):
    """Tracked DB: alembic_version exists → no stamp, just upgrade head."""
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(tracked_engine, alembic_dir=alembic_dir)

    calls = [c.args[0] for c in mock_cmd.call_args_list]
    assert calls == [["upgrade", "head"]]


# ── Failure / error handling ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_raises_on_upgrade_failure(tracked_engine, alembic_dir):
    """RuntimeError raised when alembic upgrade head exits non-zero."""
    failure_proc = _make_proc(1, stderr="FATAL: column does not exist\n")

    with patch("db.migrations._run_alembic_cmd", return_value=failure_proc):
        with pytest.raises(RuntimeError, match="alembic upgrade head failed"):
            await run_alembic_upgrade(tracked_engine, alembic_dir=alembic_dir)


@pytest.mark.asyncio
async def test_run_alembic_upgrade_raises_on_stamp_failure(
    legacy_engine_without_stock58, alembic_dir
):
    """RuntimeError raised when alembic stamp exits non-zero."""
    stamp_failure = _make_proc(1, stderr="stamp error\n")
    success_proc = _make_proc(0)

    # stamp fails, upgrade should never be reached
    with patch(
        "db.migrations._run_alembic_cmd",
        side_effect=[stamp_failure, success_proc],
    ):
        with pytest.raises(RuntimeError, match=f"alembic stamp {_INITIAL_REVISION} failed"):
            await run_alembic_upgrade(legacy_engine_without_stock58, alembic_dir=alembic_dir)


@pytest.mark.asyncio
async def test_run_alembic_upgrade_default_ini_path(tracked_engine):
    """Default alembic_dir resolves to backend/ directory."""
    success_proc = _make_proc(0)
    captured_cwd: list[Path] = []

    def _capturing_cmd(args: list, cwd: Path) -> MagicMock:  # type: ignore[no-untyped-def]
        captured_cwd.append(cwd)
        return success_proc

    with patch("db.migrations._run_alembic_cmd", side_effect=_capturing_cmd):
        await run_alembic_upgrade(tracked_engine)

    assert len(captured_cwd) == 1
    # Should point to the backend/ directory (parent of db/)
    assert captured_cwd[0].name == "backend"


@pytest.mark.asyncio
async def test_run_alembic_upgrade_upgrade_head_called_last(
    legacy_engine_without_stock58, alembic_dir
):
    """upgrade head is always the final alembic command."""
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(legacy_engine_without_stock58, alembic_dir=alembic_dir)

    last_call_args = mock_cmd.call_args_list[-1].args[0]
    assert last_call_args == ["upgrade", "head"]


# ── Edge case: orders table only, portfolio_snapshots absent ──────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_only_orders_no_snapshots_stamps_initial(
    legacy_engine_only_orders, alembic_dir
):
    """orders exists but portfolio_snapshots does NOT → stamp at _INITIAL_REVISION.

    When only the orders table is present (portfolio_snapshots was never created),
    has_base_tables=True but the sentinel column cannot be found because the
    sentinel table itself is absent.  The code should stamp at _INITIAL_REVISION
    so that the STOCK-58 migration (which creates/alters portfolio_snapshots) is
    applied by the subsequent upgrade head.
    """
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc) as mock_cmd:
        await run_alembic_upgrade(legacy_engine_only_orders, alembic_dir=alembic_dir)

    calls = [c.args[0] for c in mock_cmd.call_args_list]
    # Must stamp at _INITIAL_REVISION (not head) so upgrade applies STOCK-58
    assert calls[0] == ["stamp", _INITIAL_REVISION]
    assert calls[1] == ["upgrade", "head"]


# ── Timeout error handling ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_alembic_upgrade_raises_on_upgrade_timeout(tracked_engine, alembic_dir):
    """RuntimeError raised when alembic upgrade head times out (TimeoutExpired)."""
    with patch(
        "db.migrations._run_alembic_cmd",
        side_effect=subprocess.TimeoutExpired(cmd=[], timeout=180),
    ):
        with pytest.raises(RuntimeError, match="alembic upgrade head timed out after 180 s"):
            await run_alembic_upgrade(tracked_engine, alembic_dir=alembic_dir)


@pytest.mark.asyncio
async def test_run_alembic_upgrade_raises_on_stamp_timeout(
    legacy_engine_without_stock58, alembic_dir
):
    """RuntimeError raised when alembic stamp times out (TimeoutExpired)."""
    with patch(
        "db.migrations._run_alembic_cmd",
        side_effect=subprocess.TimeoutExpired(cmd=[], timeout=180),
    ):
        with pytest.raises(RuntimeError, match=f"alembic stamp {_INITIAL_REVISION} timed out"):
            await run_alembic_upgrade(legacy_engine_without_stock58, alembic_dir=alembic_dir)


# ── PostgreSQL advisory lock ──────────────────────────────────────────


def _make_pg_engine(mock_conn: AsyncMock) -> MagicMock:
    """Build a minimal mock engine that reports its dialect as PostgreSQL."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.connect.return_value = cm
    return mock_engine


@pytest.mark.asyncio
async def test_pg_advisory_lock_acquired_and_released(alembic_dir):
    """pg_advisory_lock is acquired before migration and released after success."""
    executed_sqls: list[str] = []

    mock_conn = AsyncMock()

    def _record(stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        executed_sqls.append(str(stmt))

    mock_conn.execute.side_effect = _record
    # Simulate a tracked DB so no stamp is attempted
    mock_conn.run_sync = AsyncMock(return_value=(True, True, _INITIAL_REVISION))

    mock_engine = _make_pg_engine(mock_conn)
    success_proc = _make_proc(0)

    with patch("db.migrations._run_alembic_cmd", return_value=success_proc):
        await run_alembic_upgrade(mock_engine, alembic_dir=alembic_dir)

    lock_sqls = [s for s in executed_sqls if "pg_advisory_lock" in s and "unlock" not in s]
    unlock_sqls = [s for s in executed_sqls if "pg_advisory_unlock" in s]
    assert len(lock_sqls) == 1, f"expected 1 lock SQL, got: {lock_sqls}"
    assert len(unlock_sqls) == 1, f"expected 1 unlock SQL, got: {unlock_sqls}"


@pytest.mark.asyncio
async def test_pg_advisory_lock_released_on_failure(alembic_dir):
    """pg_advisory_unlock is called in the finally block even when upgrade raises."""
    executed_sqls: list[str] = []

    mock_conn = AsyncMock()

    def _record(stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        executed_sqls.append(str(stmt))

    mock_conn.execute.side_effect = _record
    mock_conn.run_sync = AsyncMock(return_value=(True, True, _INITIAL_REVISION))

    mock_engine = _make_pg_engine(mock_conn)
    failure_proc = _make_proc(1, stderr="upgrade failed\n")

    with patch("db.migrations._run_alembic_cmd", return_value=failure_proc):
        with pytest.raises(RuntimeError, match="alembic upgrade head failed"):
            await run_alembic_upgrade(mock_engine, alembic_dir=alembic_dir)

    # Lock must be released even though upgrade raised
    unlock_sqls = [s for s in executed_sqls if "pg_advisory_unlock" in s]
    assert len(unlock_sqls) == 1, f"expected 1 unlock SQL, got: {unlock_sqls}"
