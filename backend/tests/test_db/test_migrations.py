"""Tests for auto-migration of missing columns (STOCK-11).

Validates that ensure_columns() correctly detects and adds columns that
exist in the ORM model but are missing from the physical DB schema.
This is the fix for the 'orders.is_paper does not exist' error that
caused all PnL statistics to show as 0.
"""

import pytest
import pytest_asyncio
from sqlalchemy import Column, DateTime, Float, Integer, String, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from db.migrations import ensure_columns


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
async def test_adds_both_missing_columns(old_engine):
    """ensure_columns adds all missing columns in one call."""
    added = await ensure_columns(old_engine)

    assert len(added) == 2
    assert "orders.is_paper" in added
    assert "orders.buy_strategy" in added


@pytest.mark.asyncio
async def test_idempotent_no_change_when_columns_present(full_engine):
    """ensure_columns does nothing when all columns already exist."""
    added = await ensure_columns(full_engine)

    assert added == []


@pytest.mark.asyncio
async def test_idempotent_second_run(old_engine):
    """Running ensure_columns twice does not fail or re-add columns."""
    first = await ensure_columns(old_engine)
    assert len(first) == 2

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
