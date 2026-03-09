"""Tests for TradeRepository using in-memory SQLite."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.models import Base
from db.trade_repository import TradeRepository


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
    return TradeRepository(session)


@pytest.mark.asyncio
async def test_save_and_get_order(repo):
    order = await repo.save_order(
        symbol="AAPL", side="buy", order_type="limit",
        quantity=10, price=180.0, status="filled",
        strategy_name="trend_following", filled_price=180.0,
        filled_quantity=10,
    )
    assert order.id is not None
    assert order.symbol == "AAPL"

    history = await repo.get_trade_history(limit=10)
    assert len(history) == 1
    assert history[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_update_order_status(repo):
    order = await repo.save_order(
        symbol="MSFT", side="buy", order_type="limit",
        quantity=5, price=400.0, status="pending",
    )
    updated = await repo.update_order_status(
        order.id, status="filled", filled_price=399.50, pnl=None,
    )
    assert updated.status == "filled"
    assert updated.filled_price == 399.50


@pytest.mark.asyncio
async def test_get_trade_history_filter_by_symbol(repo):
    await repo.save_order(symbol="AAPL", side="buy", order_type="market", quantity=10, price=180.0)
    await repo.save_order(symbol="MSFT", side="buy", order_type="market", quantity=5, price=400.0)
    await repo.save_order(symbol="AAPL", side="sell", order_type="market", quantity=10, price=190.0)

    aapl_trades = await repo.get_trade_history(symbol="AAPL")
    assert len(aapl_trades) == 2
    assert all(t.symbol == "AAPL" for t in aapl_trades)

    all_trades = await repo.get_trade_history()
    assert len(all_trades) == 3


@pytest.mark.asyncio
async def test_get_open_orders(repo):
    await repo.save_order(symbol="AAPL", side="buy", order_type="limit", quantity=10, price=180.0, status="pending")
    await repo.save_order(symbol="MSFT", side="buy", order_type="limit", quantity=5, price=400.0, status="filled")

    open_orders = await repo.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_watchlist_add_and_get(repo):
    await repo.add_to_watchlist("AAPL", name="Apple Inc.")
    await repo.add_to_watchlist("MSFT", name="Microsoft")

    wl = await repo.get_watchlist()
    assert len(wl) == 2
    symbols = [w.symbol for w in wl]
    assert "AAPL" in symbols
    assert "MSFT" in symbols


@pytest.mark.asyncio
async def test_watchlist_remove(repo):
    await repo.add_to_watchlist("AAPL")
    await repo.add_to_watchlist("MSFT")

    removed = await repo.remove_from_watchlist("AAPL")
    assert removed is True

    wl = await repo.get_watchlist(active_only=True)
    assert len(wl) == 1
    assert wl[0].symbol == "MSFT"


@pytest.mark.asyncio
async def test_watchlist_add_duplicate_reactivates(repo):
    await repo.add_to_watchlist("AAPL")
    await repo.remove_from_watchlist("AAPL")

    wl = await repo.get_watchlist(active_only=True)
    assert len(wl) == 0

    # Re-add same symbol
    await repo.add_to_watchlist("AAPL")
    wl = await repo.get_watchlist(active_only=True)
    assert len(wl) == 1


@pytest.mark.asyncio
async def test_watchlist_remove_nonexistent(repo):
    result = await repo.remove_from_watchlist("FAKE")
    assert result is False


@pytest.mark.asyncio
async def test_get_recent_trades(repo):
    """get_recent_trades returns only filled orders within time window."""
    from datetime import datetime

    # Filled order — should appear
    order = await repo.save_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=10, price=180.0, status="pending",
    )
    await repo.update_order_status(order.id, status="filled", filled_price=180.0)

    # Pending order — should NOT appear
    await repo.save_order(
        symbol="MSFT", side="buy", order_type="limit",
        quantity=5, price=400.0, status="pending",
    )

    recent = await repo.get_recent_trades(hours=24)
    assert len(recent) == 1
    assert recent[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_get_recent_trades_empty(repo):
    recent = await repo.get_recent_trades(hours=24)
    assert recent == []


@pytest.mark.asyncio
async def test_watchlist_market_filter(repo):
    """Watchlist can be filtered by market."""
    await repo.add_to_watchlist("AAPL", market="US")
    await repo.add_to_watchlist("005930", exchange="KRX", market="KR")
    await repo.add_to_watchlist("MSFT", market="US")

    all_wl = await repo.get_watchlist()
    assert len(all_wl) == 3

    us_wl = await repo.get_watchlist(market="US")
    assert len(us_wl) == 2
    assert {w.symbol for w in us_wl} == {"AAPL", "MSFT"}

    kr_wl = await repo.get_watchlist(market="KR")
    assert len(kr_wl) == 1
    assert kr_wl[0].symbol == "005930"


@pytest.mark.asyncio
async def test_watchlist_same_symbol_different_markets(repo):
    """Same symbol code can exist in US and KR markets."""
    await repo.add_to_watchlist("TEST01", market="US")
    await repo.add_to_watchlist("TEST01", exchange="KRX", market="KR")

    all_wl = await repo.get_watchlist()
    assert len(all_wl) == 2

    us_wl = await repo.get_watchlist(market="US")
    assert len(us_wl) == 1
    assert us_wl[0].market == "US"

    kr_wl = await repo.get_watchlist(market="KR")
    assert len(kr_wl) == 1
    assert kr_wl[0].market == "KR"


@pytest.mark.asyncio
async def test_watchlist_remove_with_market(repo):
    """Remove operates within correct market scope."""
    await repo.add_to_watchlist("AAPL", market="US")
    await repo.add_to_watchlist("005930", exchange="KRX", market="KR")

    # Remove from KR market only
    await repo.remove_from_watchlist("005930", market="KR")

    us_wl = await repo.get_watchlist(market="US")
    assert len(us_wl) == 1  # US untouched

    kr_wl = await repo.get_watchlist(market="KR")
    assert len(kr_wl) == 0


@pytest.mark.asyncio
async def test_watchlist_add_kr_with_market(repo):
    """KR stocks added with correct market and exchange."""
    item = await repo.add_to_watchlist(
        "005930", exchange="KRX", market="KR", source="scanner",
    )
    assert item.market == "KR"
    assert item.exchange == "KRX"
    assert item.source == "scanner"
