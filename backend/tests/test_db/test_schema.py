"""Tests for db.schema — startup schema migration helper."""

import pytest
import pytest_asyncio
from sqlalchemy import Column, DateTime, Float, Integer, String, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from db.schema import _column_exists, _index_exists, ensure_schema


class _Base(DeclarativeBase):
    pass


class _OrdersLegacy(_Base):
    """Simulates an old orders table WITHOUT the is_paper column."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String(20), nullable=False, default="pending")
    strategy_name = Column(String(50))
    created_at = Column(DateTime)


@pytest_asyncio.fixture
async def legacy_engine():
    """Create an in-memory SQLite DB with old orders schema (no is_paper)."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def full_engine():
    """Create an in-memory SQLite DB with the full current schema (has is_paper)."""
    from core.models import Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


class TestEnsureSchema:
    """Test ensure_schema() startup migration."""

    @pytest.mark.asyncio
    async def test_adds_missing_is_paper_column(self, legacy_engine):
        """ensure_schema adds is_paper column when missing."""
        # Verify column is missing before migration
        async with legacy_engine.begin() as conn:
            has_col = await _column_exists(conn, "orders", "is_paper", "sqlite")
            assert has_col is False

        # Run migration
        applied = await ensure_schema(legacy_engine)

        # Verify column was added
        assert len(applied) >= 1
        assert any("is_paper" in msg for msg in applied)

        # Verify column exists and works
        async with legacy_engine.begin() as conn:
            has_col = await _column_exists(conn, "orders", "is_paper", "sqlite")
            assert has_col is True

    @pytest.mark.asyncio
    async def test_creates_index_on_is_paper(self, legacy_engine):
        """ensure_schema creates idx_orders_is_paper index."""
        applied = await ensure_schema(legacy_engine)

        # Verify index was created
        assert any("idx_orders_is_paper" in msg for msg in applied)

        async with legacy_engine.begin() as conn:
            has_idx = await _index_exists(conn, "idx_orders_is_paper", "orders", "sqlite")
            assert has_idx is True

    @pytest.mark.asyncio
    async def test_idempotent_no_changes_when_up_to_date(self, full_engine):
        """ensure_schema returns empty list when schema is already correct."""
        applied = await ensure_schema(full_engine)
        assert applied == []

    @pytest.mark.asyncio
    async def test_idempotent_second_run(self, legacy_engine):
        """Running ensure_schema twice is safe — second run does nothing."""
        first = await ensure_schema(legacy_engine)
        assert len(first) >= 1

        second = await ensure_schema(legacy_engine)
        assert second == []

    @pytest.mark.asyncio
    async def test_default_value_is_false(self, legacy_engine):
        """New is_paper column defaults to FALSE for existing rows."""
        # Insert a row before migration
        async with legacy_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO orders (symbol, side, order_type, quantity, status) "
                    "VALUES ('AAPL', 'BUY', 'market', 10, 'filled')"
                )
            )

        # Run migration
        await ensure_schema(legacy_engine)

        # Existing row should have is_paper = FALSE
        async with legacy_engine.begin() as conn:
            result = await conn.execute(text("SELECT is_paper FROM orders WHERE symbol = 'AAPL'"))
            row = result.fetchone()
            assert row is not None
            assert row[0] == 0  # SQLite represents FALSE as 0

    @pytest.mark.asyncio
    async def test_new_rows_work_after_migration(self, legacy_engine):
        """After migration, inserting rows with is_paper=TRUE works."""
        await ensure_schema(legacy_engine)

        async with legacy_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO orders (symbol, side, order_type, quantity, status, is_paper) "
                    "VALUES ('TSLA', 'BUY', 'market', 5, 'filled', 1)"
                )
            )
            result = await conn.execute(text("SELECT is_paper FROM orders WHERE symbol = 'TSLA'"))
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1  # SQLite represents TRUE as 1


class TestColumnExists:
    """Test the _column_exists helper."""

    @pytest.mark.asyncio
    async def test_existing_column(self, legacy_engine):
        """Detects columns that exist."""
        async with legacy_engine.begin() as conn:
            assert await _column_exists(conn, "orders", "symbol", "sqlite") is True
            assert await _column_exists(conn, "orders", "side", "sqlite") is True

    @pytest.mark.asyncio
    async def test_missing_column(self, legacy_engine):
        """Detects columns that don't exist."""
        async with legacy_engine.begin() as conn:
            assert await _column_exists(conn, "orders", "is_paper", "sqlite") is False
            assert await _column_exists(conn, "orders", "nonexistent", "sqlite") is False


class TestIndexExists:
    """Test the _index_exists helper."""

    @pytest.mark.asyncio
    async def test_missing_index(self, legacy_engine):
        """Detects indexes that don't exist."""
        async with legacy_engine.begin() as conn:
            assert await _index_exists(conn, "idx_orders_is_paper", "orders", "sqlite") is False

    @pytest.mark.asyncio
    async def test_existing_index(self, legacy_engine):
        """Detects indexes that exist after creation."""
        async with legacy_engine.begin() as conn:
            await conn.execute(text("CREATE INDEX test_idx ON orders(symbol)"))

        async with legacy_engine.begin() as conn:
            assert await _index_exists(conn, "test_idx", "orders", "sqlite") is True
