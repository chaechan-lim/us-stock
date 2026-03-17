"""Tests for PositionRepository using in-memory SQLite."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base
from db.position_repository import PositionRepository


@pytest_asyncio.fixture
async def session():
    """Create in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess

    await engine.dispose()


@pytest_asyncio.fixture
async def repo(session):
    return PositionRepository(session)


# ── Basic CRUD ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_creates_new_position(repo):
    record = await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        stop_loss=0.08,
        take_profit=0.20,
        strategy_name="trend_following",
        market="US",
    )
    assert record.id is not None
    assert record.symbol == "AAPL"
    assert record.quantity == 10
    assert record.avg_price == 150.0
    assert record.stop_loss == 0.08
    assert record.take_profit == 0.20
    assert record.strategy_name == "trend_following"
    assert record.market == "US"


@pytest.mark.asyncio
async def test_upsert_updates_existing_position(repo):
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        strategy_name="trend_following",
        market="US",
    )
    updated = await repo.upsert_position(
        symbol="AAPL",
        quantity=20,
        avg_price=155.0,
        stop_loss=0.10,
        take_profit=0.25,
        strategy_name="macd_histogram",
        market="US",
    )
    assert updated.quantity == 20
    assert updated.avg_price == 155.0
    assert updated.stop_loss == 0.10
    assert updated.take_profit == 0.25
    assert updated.strategy_name == "macd_histogram"

    # Should be only one record
    all_positions = await repo.get_all_positions(market="US")
    assert len(all_positions) == 1


@pytest.mark.asyncio
async def test_upsert_preserves_strategy_name_if_empty(repo):
    """If new strategy_name is empty, keep the old one."""
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        strategy_name="trend_following",
        market="US",
    )
    updated = await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        strategy_name="",
        market="US",
    )
    assert updated.strategy_name == "trend_following"


@pytest.mark.asyncio
async def test_remove_position(repo):
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        market="US",
    )
    removed = await repo.remove_position("AAPL", market="US")
    assert removed is True

    positions = await repo.get_all_positions(market="US")
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_remove_nonexistent_position(repo):
    removed = await repo.remove_position("FAKE", market="US")
    assert removed is False


@pytest.mark.asyncio
async def test_get_all_positions(repo):
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        market="US",
    )
    await repo.upsert_position(
        symbol="MSFT",
        quantity=5,
        avg_price=300.0,
        market="US",
    )
    await repo.upsert_position(
        symbol="005930",
        quantity=100,
        avg_price=72000.0,
        market="KR",
    )

    us_positions = await repo.get_all_positions(market="US")
    assert len(us_positions) == 2
    symbols = {p.symbol for p in us_positions}
    assert symbols == {"AAPL", "MSFT"}

    kr_positions = await repo.get_all_positions(market="KR")
    assert len(kr_positions) == 1
    assert kr_positions[0].symbol == "005930"

    all_positions = await repo.get_all_positions()
    assert len(all_positions) == 3


@pytest.mark.asyncio
async def test_remove_all(repo):
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        market="US",
    )
    await repo.upsert_position(
        symbol="MSFT",
        quantity=5,
        avg_price=300.0,
        market="US",
    )
    await repo.upsert_position(
        symbol="005930",
        quantity=100,
        avg_price=72000.0,
        market="KR",
    )

    deleted = await repo.remove_all(market="US")
    assert deleted == 2

    remaining = await repo.get_all_positions()
    assert len(remaining) == 1
    assert remaining[0].symbol == "005930"


@pytest.mark.asyncio
async def test_remove_all_no_market_filter(repo):
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        market="US",
    )
    await repo.upsert_position(
        symbol="005930",
        quantity=100,
        avg_price=72000.0,
        market="KR",
    )

    deleted = await repo.remove_all()
    assert deleted == 2

    remaining = await repo.get_all_positions()
    assert len(remaining) == 0


# ── Edge Cases ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_with_current_price_and_pnl(repo):
    record = await repo.upsert_position(
        symbol="TSLA",
        quantity=3,
        avg_price=200.0,
        current_price=250.0,
        unrealized_pnl=150.0,
        market="US",
    )
    assert record.current_price == 250.0
    assert record.unrealized_pnl == 150.0


@pytest.mark.asyncio
async def test_different_markets_same_symbol_pattern(repo):
    """Ensure US and KR positions are isolated by market."""
    await repo.upsert_position(
        symbol="AAPL",
        quantity=10,
        avg_price=150.0,
        market="US",
    )
    # Hypothetical: same symbol name in different market
    await repo.upsert_position(
        symbol="AAPL",
        quantity=5,
        avg_price=100.0,
        market="KR",
    )

    us = await repo.get_all_positions(market="US")
    kr = await repo.get_all_positions(market="KR")
    assert len(us) == 1
    assert us[0].quantity == 10
    assert len(kr) == 1
    assert kr[0].quantity == 5
