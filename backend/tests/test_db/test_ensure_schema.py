"""Tests for ensure_schema_columns — adds missing ORM columns to existing DB tables.

Reproduces the STOCK-11 bug: is_paper column exists in Order model but not in
the database (initial alembic migration predates the column addition).
"""

import pytest
import pytest_asyncio
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from db.session import _build_default_clause, _find_missing_columns, ensure_schema_columns


class _Base(DeclarativeBase):
    pass


class _TestTable(_Base):
    """ORM model with columns that may not exist in DB yet."""

    __tablename__ = "test_orders"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    # Columns added later (simulate is_paper scenario)
    is_paper = Column(Boolean, nullable=False, default=False)
    market = Column(String(2), nullable=False, default="US")
    session = Column(String(20), nullable=True, default="regular")
    pnl = Column(Float)


@pytest_asyncio.fixture
async def engine_with_partial_schema():
    """Create engine where table exists but is missing some columns.

    Simulates production DB created by old alembic migration that lacks
    is_paper, market, session columns.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # Create table manually with only id + symbol (simulating old migration)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                'CREATE TABLE "test_orders" ('
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  symbol VARCHAR(20) NOT NULL"
                ")"
            )
        )
        # Insert a row to verify data preservation
        await conn.execute(text("INSERT INTO test_orders (symbol) VALUES ('AAPL')"))

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def engine_full_schema():
    """Create engine where table has all columns (fresh install)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield engine
    await engine.dispose()


# --- ensure_schema_columns: adds missing columns ---


@pytest.mark.asyncio
async def test_adds_missing_columns(engine_with_partial_schema):
    """Missing columns are added via ALTER TABLE."""
    # Patch Base to use our test model's metadata
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        added = await ensure_schema_columns(engine_with_partial_schema)
    finally:
        session_mod.Base = original_base

    # Should have added is_paper, market, session, pnl
    assert len(added) == 4
    assert "test_orders.is_paper" in added
    assert "test_orders.market" in added
    assert "test_orders.session" in added
    assert "test_orders.pnl" in added


@pytest.mark.asyncio
async def test_preserves_existing_data(engine_with_partial_schema):
    """Existing rows are preserved after columns are added."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        await ensure_schema_columns(engine_with_partial_schema)
    finally:
        session_mod.Base = original_base

    # Verify existing data is intact
    async with engine_with_partial_schema.connect() as conn:
        result = await conn.execute(text("SELECT symbol FROM test_orders"))
        rows = result.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "AAPL"


@pytest.mark.asyncio
async def test_new_columns_have_defaults(engine_with_partial_schema):
    """Added columns get default values for existing rows."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        await ensure_schema_columns(engine_with_partial_schema)
    finally:
        session_mod.Base = original_base

    # Verify defaults are applied
    async with engine_with_partial_schema.connect() as conn:
        result = await conn.execute(text("SELECT is_paper, market, session FROM test_orders"))
        row = result.fetchone()
        # is_paper defaults to FALSE, market to 'US', session to 'regular'
        assert row[0] in (0, False)  # SQLite stores booleans as 0/1
        assert row[1] == "US"
        assert row[2] == "regular"


@pytest.mark.asyncio
async def test_noop_when_all_columns_exist(engine_full_schema):
    """No columns added when schema is already up-to-date."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        added = await ensure_schema_columns(engine_full_schema)
    finally:
        session_mod.Base = original_base

    assert added == []


@pytest.mark.asyncio
async def test_noop_when_table_does_not_exist():
    """No error when ORM model table doesn't exist yet (create_all handles it)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    # Don't create any tables — ensure_schema_columns should skip gracefully

    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        added = await ensure_schema_columns(engine)
    finally:
        session_mod.Base = original_base

    assert added == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_idempotent(engine_with_partial_schema):
    """Running ensure_schema_columns twice is safe (idempotent)."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        added1 = await ensure_schema_columns(engine_with_partial_schema)
        added2 = await ensure_schema_columns(engine_with_partial_schema)
    finally:
        session_mod.Base = original_base

    assert len(added1) == 4
    assert added2 == []  # No-op on second run


# --- _find_missing_columns ---


@pytest.mark.asyncio
async def test_find_missing_columns_partial(engine_with_partial_schema):
    """_find_missing_columns detects missing columns correctly."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        async with engine_with_partial_schema.connect() as conn:
            missing = await conn.run_sync(_find_missing_columns)
    finally:
        session_mod.Base = original_base

    col_names = [m[1] for m in missing]
    assert "is_paper" in col_names
    assert "market" in col_names
    assert "session" in col_names
    assert "pnl" in col_names
    # id and symbol should NOT be missing
    assert "id" not in col_names
    assert "symbol" not in col_names


@pytest.mark.asyncio
async def test_find_missing_columns_complete(engine_full_schema):
    """_find_missing_columns returns empty for complete schema."""
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = _Base
    try:
        async with engine_full_schema.connect() as conn:
            missing = await conn.run_sync(_find_missing_columns)
    finally:
        session_mod.Base = original_base

    assert missing == []


# --- _build_default_clause ---


class TestBuildDefaultClause:
    def test_bool_false(self):
        col = Column("test", Boolean, default=False)
        col.name = "test"
        assert _build_default_clause(col) == " DEFAULT FALSE"

    def test_bool_true(self):
        col = Column("test", Boolean, default=True)
        col.name = "test"
        assert _build_default_clause(col) == " DEFAULT TRUE"

    def test_int_default(self):
        col = Column("test", Integer, default=0)
        col.name = "test"
        assert _build_default_clause(col) == " DEFAULT 0"

    def test_float_default(self):
        col = Column("test", Float, default=1.5)
        col.name = "test"
        assert _build_default_clause(col) == " DEFAULT 1.5"

    def test_string_default(self):
        col = Column("test", String, default="hello")
        col.name = "test"
        assert _build_default_clause(col) == " DEFAULT 'hello'"

    def test_callable_default_skipped(self):
        """Callable defaults (e.g. datetime.utcnow) return empty string."""
        from datetime import datetime

        col = Column("test", DateTime, default=datetime.utcnow)
        col.name = "test"
        assert _build_default_clause(col) == ""

    def test_no_default(self):
        col = Column("test", Float)
        col.name = "test"
        assert _build_default_clause(col) == ""


# --- Integration: reproduce STOCK-11 scenario with actual Order model ---


@pytest.mark.asyncio
async def test_stock11_is_paper_column_fix():
    """Reproduce STOCK-11: orders table exists without is_paper column.

    Simulate the exact production scenario:
    1. Create orders table with the initial migration schema (no is_paper)
    2. Run ensure_schema_columns()
    3. Verify is_paper column is added and queries work
    """
    from core.models import Base as RealBase
    from core.models import Order

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # Step 1: Create orders table WITHOUT is_paper (mimics initial alembic migration)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                'CREATE TABLE "orders" ('
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  symbol VARCHAR(20) NOT NULL,"
                "  exchange VARCHAR(10) NOT NULL DEFAULT 'NASD',"
                "  side VARCHAR(4) NOT NULL,"
                "  order_type VARCHAR(10) NOT NULL,"
                "  quantity FLOAT NOT NULL,"
                "  price FLOAT,"
                "  filled_quantity FLOAT DEFAULT 0,"
                "  filled_price FLOAT,"
                "  status VARCHAR(20) NOT NULL DEFAULT 'pending',"
                "  strategy_name VARCHAR(50),"
                "  kis_order_id VARCHAR(50),"
                "  pnl FLOAT,"
                "  created_at DATETIME,"
                "  filled_at DATETIME"
                ")"
            )
        )
        # Insert an existing order (pre-is_paper era)
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, status, pnl) "
                "VALUES ('AAPL', 'SELL', 'market', 10, 180.0, 'filled', 50.0)"
            )
        )

    # Step 2: Run ensure_schema_columns (should add is_paper, market, etc.)
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = RealBase
    try:
        added = await ensure_schema_columns(engine)
    finally:
        session_mod.Base = original_base

    # Verify is_paper was added
    assert any("orders.is_paper" in col for col in added)

    # Step 3: Verify the column works — queries that previously crashed now succeed
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        # This query would have crashed before the fix:
        # "column orders.is_paper does not exist"
        stmt = select(Order).where(Order.is_paper == False)  # noqa: E712
        result = await session.execute(stmt)
        orders = list(result.scalars().all())

        # Existing order should be returned (is_paper defaults to FALSE)
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].pnl == 50.0

    await engine.dispose()


@pytest.mark.asyncio
async def test_stock11_trade_history_with_ensure_schema():
    """After ensure_schema_columns, TradeRepository.get_trade_history works."""
    from core.models import Base as RealBase
    from db.trade_repository import TradeRepository

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # Create orders table WITHOUT is_paper (old migration)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                'CREATE TABLE "orders" ('
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  symbol VARCHAR(20) NOT NULL,"
                "  exchange VARCHAR(10) NOT NULL DEFAULT 'NASD',"
                "  side VARCHAR(4) NOT NULL,"
                "  order_type VARCHAR(10) NOT NULL,"
                "  quantity FLOAT NOT NULL,"
                "  price FLOAT,"
                "  filled_quantity FLOAT DEFAULT 0,"
                "  filled_price FLOAT,"
                "  status VARCHAR(20) NOT NULL DEFAULT 'pending',"
                "  strategy_name VARCHAR(50),"
                "  kis_order_id VARCHAR(50),"
                "  pnl FLOAT,"
                "  created_at DATETIME,"
                "  filled_at DATETIME"
                ")"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO orders"
                " (symbol, side, order_type, quantity, price,"
                "  status, pnl, filled_at)"
                " VALUES ('MSFT', 'SELL', 'market', 5, 400.0,"
                "  'filled', 100.0, '2026-03-17 10:00:00')"
            )
        )

    # Apply schema fix
    import db.session as session_mod

    original_base = session_mod.Base
    session_mod.Base = RealBase
    try:
        await ensure_schema_columns(engine)
    finally:
        session_mod.Base = original_base

    # TradeRepository should now work (previously crashed with "column does not exist")
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)

        # get_trade_history with exclude_paper should work
        history = await repo.get_trade_history(limit=50, exclude_paper=True)
        assert len(history) == 1
        assert history[0].symbol == "MSFT"
        assert history[0].pnl == 100.0

    await engine.dispose()
