"""Tests for TradeRepository using in-memory SQLite."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        price=180.0,
        status="filled",
        strategy_name="trend_following",
        filled_price=180.0,
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
        symbol="MSFT",
        side="buy",
        order_type="limit",
        quantity=5,
        price=400.0,
        status="pending",
    )
    updated = await repo.update_order_status(
        order.id,
        status="filled",
        filled_price=399.50,
        pnl=None,
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
    await repo.save_order(
        symbol="AAPL", side="buy", order_type="limit", quantity=10, price=180.0, status="pending"
    )
    await repo.save_order(
        symbol="MSFT", side="buy", order_type="limit", quantity=5, price=400.0, status="filled"
    )

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
    # Filled order — should appear
    order = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=180.0,
        status="pending",
    )
    await repo.update_order_status(order.id, status="filled", filled_price=180.0)

    # Pending order — should NOT appear
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="limit",
        quantity=5,
        price=400.0,
        status="pending",
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
        "005930",
        exchange="KRX",
        market="KR",
        source="scanner",
    )
    assert item.market == "KR"
    assert item.exchange == "KRX"
    assert item.source == "scanner"


# --- Paper/Live order separation (STOCK-6) ---


@pytest.mark.asyncio
async def test_save_order_with_is_paper(repo):
    """Paper orders are saved with is_paper=True."""
    paper = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        status="filled",
        is_paper=True,
    )
    assert paper.is_paper is True

    live = await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="limit",
        quantity=5,
        price=400.0,
        status="filled",
        is_paper=False,
    )
    assert live.is_paper is False


@pytest.mark.asyncio
async def test_save_order_is_paper_default_false(repo):
    """Orders default to is_paper=False (live)."""
    order = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
    )
    assert order.is_paper is False


@pytest.mark.asyncio
async def test_get_trade_history_exclude_paper(repo):
    """get_trade_history with exclude_paper=True filters out paper orders."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        is_paper=True,
    )
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="market",
        quantity=5,
        price=400.0,
        is_paper=False,
    )
    await repo.save_order(
        symbol="GOOGL",
        side="buy",
        order_type="market",
        quantity=3,
        price=170.0,
        is_paper=False,
    )

    # Without filter: all 3
    all_trades = await repo.get_trade_history()
    assert len(all_trades) == 3

    # With filter: only live (2)
    live_trades = await repo.get_trade_history(exclude_paper=True)
    assert len(live_trades) == 2
    assert all(not t.is_paper for t in live_trades)


@pytest.mark.asyncio
async def test_get_open_orders_exclude_paper(repo):
    """get_open_orders with exclude_paper=True filters out paper orders."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        price=150.0,
        status="pending",
        is_paper=True,
    )
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="limit",
        quantity=5,
        price=400.0,
        status="pending",
        is_paper=False,
    )

    # Without filter: both
    all_open = await repo.get_open_orders()
    assert len(all_open) == 2

    # With filter: only live
    live_open = await repo.get_open_orders(exclude_paper=True)
    assert len(live_open) == 1
    assert live_open[0].symbol == "MSFT"


@pytest.mark.asyncio
async def test_get_recent_trades_exclude_paper(repo):
    """get_recent_trades with exclude_paper=True filters out paper orders."""
    # Paper filled order
    paper = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        status="pending",
        is_paper=True,
    )
    await repo.update_order_status(paper.id, "filled", filled_price=150.0)

    # Live filled order
    live = await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="market",
        quantity=5,
        price=400.0,
        status="pending",
        is_paper=False,
    )
    await repo.update_order_status(live.id, "filled", filled_price=400.0)

    # Without filter: both
    all_recent = await repo.get_recent_trades(hours=24)
    assert len(all_recent) == 2

    # With filter: only live
    live_recent = await repo.get_recent_trades(hours=24, exclude_paper=True)
    assert len(live_recent) == 1
    assert live_recent[0].symbol == "MSFT"


# --- UPSERT dedup for kis_order_id (STOCK-3) ---


@pytest.mark.asyncio
async def test_save_order_upsert_prevents_duplicate_kis_order_id(repo):
    """save_order with same kis_order_id updates existing row instead of inserting."""
    # First insert: order placed with pending status
    order1 = await repo.save_order(
        symbol="263750",
        side="buy",
        order_type="limit",
        quantity=60,
        price=35000.0,
        status="pending",
        strategy_name="trend_following",
        kis_order_id="0013045000",
        market="KR",
        exchange="KRX",
    )
    assert order1.id is not None
    assert order1.status == "pending"
    original_id = order1.id

    # Second insert: same kis_order_id, now filled (from reconciliation)
    order2 = await repo.save_order(
        symbol="263750",
        side="buy",
        order_type="limit",
        quantity=60,
        price=35000.0,
        filled_quantity=60,
        filled_price=35000.0,
        status="filled",
        strategy_name="trend_following",
        kis_order_id="0013045000",
        market="KR",
        exchange="KRX",
    )

    # Should return the same row, not a new one
    assert order2.id == original_id
    assert order2.status == "filled"
    assert order2.filled_quantity == 60
    assert order2.filled_price == 35000.0
    assert order2.filled_at is not None

    # Verify only 1 row exists in DB
    history = await repo.get_trade_history(limit=100)
    assert len(history) == 1
    assert history[0].kis_order_id == "0013045000"


@pytest.mark.asyncio
async def test_save_order_upsert_does_not_downgrade_filled_status(repo):
    """UPSERT should not downgrade a filled order back to pending."""
    # Insert as filled
    order1 = await repo.save_order(
        symbol="005930",
        side="buy",
        order_type="limit",
        quantity=10,
        price=70000.0,
        filled_quantity=10,
        filled_price=70000.0,
        status="filled",
        kis_order_id="0009999999",
        market="KR",
    )
    assert order1.status == "filled"

    # Try to save again with pending status
    order2 = await repo.save_order(
        symbol="005930",
        side="buy",
        order_type="limit",
        quantity=10,
        price=70000.0,
        status="pending",
        kis_order_id="0009999999",
        market="KR",
    )

    # Should keep filled status
    assert order2.id == order1.id
    assert order2.status == "filled"


@pytest.mark.asyncio
async def test_save_order_upsert_updates_pnl(repo):
    """UPSERT should update PnL when provided."""
    order1 = await repo.save_order(
        symbol="AAPL",
        side="sell",
        order_type="market",
        quantity=10,
        price=180.0,
        status="filled",
        kis_order_id="US12345",
    )
    assert order1.pnl is None

    order2 = await repo.save_order(
        symbol="AAPL",
        side="sell",
        order_type="market",
        quantity=10,
        price=180.0,
        status="filled",
        kis_order_id="US12345",
        pnl=500.0,
    )

    assert order2.id == order1.id
    assert order2.pnl == 500.0


@pytest.mark.asyncio
async def test_save_order_upsert_updates_buy_strategy(repo):
    """UPSERT should update buy_strategy when provided."""
    order1 = await repo.save_order(
        symbol="AAPL",
        side="sell",
        order_type="market",
        quantity=10,
        price=180.0,
        status="filled",
        kis_order_id="US12345",
    )
    assert order1.buy_strategy is None

    order2 = await repo.save_order(
        symbol="AAPL",
        side="sell",
        order_type="market",
        quantity=10,
        price=180.0,
        status="filled",
        kis_order_id="US12345",
        buy_strategy="momentum",
    )

    assert order2.id == order1.id
    assert order2.buy_strategy == "momentum"


@pytest.mark.asyncio
async def test_save_order_no_upsert_without_kis_order_id(repo):
    """Orders without kis_order_id should always insert (no dedup)."""
    order1 = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=180.0,
    )
    order2 = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=180.0,
    )

    # Different rows
    assert order1.id != order2.id
    history = await repo.get_trade_history(limit=100)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_save_order_different_kis_order_ids_insert_separately(repo):
    """Different kis_order_ids should create separate rows."""
    order1 = await repo.save_order(
        symbol="263750",
        side="buy",
        order_type="limit",
        quantity=30,
        price=35000.0,
        kis_order_id="0013045000",
        market="KR",
    )
    order2 = await repo.save_order(
        symbol="263750",
        side="buy",
        order_type="limit",
        quantity=30,
        price=35000.0,
        kis_order_id="0013045001",
        market="KR",
    )

    assert order1.id != order2.id
    history = await repo.get_trade_history(limit=100)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_save_order_upsert_higher_filled_quantity_wins(repo):
    """UPSERT should only increase filled_quantity, not decrease."""
    order1 = await repo.save_order(
        symbol="005930",
        side="buy",
        order_type="limit",
        quantity=100,
        price=70000.0,
        filled_quantity=50,
        status="partial",
        kis_order_id="KR_PARTIAL",
        market="KR",
    )
    assert order1.filled_quantity == 50

    # Update with more fills
    order2 = await repo.save_order(
        symbol="005930",
        side="buy",
        order_type="limit",
        quantity=100,
        price=70000.0,
        filled_quantity=100,
        filled_price=70000.0,
        status="filled",
        kis_order_id="KR_PARTIAL",
        market="KR",
    )

    assert order2.id == order1.id
    assert order2.filled_quantity == 100
    assert order2.status == "filled"


# --- Duplicate cleanup (STOCK-3) ---


@pytest.mark.asyncio
async def test_cleanup_duplicate_orders(session):
    """cleanup_duplicate_orders removes duplicates, keeping the best row."""
    repo = TradeRepository(session)

    # Manually insert duplicates (bypassing UPSERT by using direct ORM)
    from datetime import datetime

    from core.models import Order

    # Duplicate pair: same kis_order_id "0013045000"
    order_a = Order(
        symbol="263750",
        exchange="KRX",
        side="buy",
        order_type="limit",
        quantity=60,
        price=35000.0,
        filled_quantity=0,
        status="pending",
        kis_order_id="0013045000",
        market="KR",
    )
    session.add(order_a)
    await session.flush()

    order_b = Order(
        symbol="263750",
        exchange="KRX",
        side="buy",
        order_type="limit",
        quantity=60,
        price=35000.0,
        filled_quantity=60,
        filled_price=35000.0,
        status="filled",
        kis_order_id="0013045000",
        market="KR",
        filled_at=datetime.utcnow(),
    )
    session.add(order_b)
    await session.flush()

    # Unique order (should not be affected)
    unique = Order(
        symbol="005930",
        exchange="KRX",
        side="buy",
        order_type="limit",
        quantity=10,
        price=70000.0,
        status="filled",
        kis_order_id="UNIQUE001",
        market="KR",
    )
    session.add(unique)
    await session.commit()

    # Before cleanup: 3 rows
    history = await repo.get_trade_history(limit=100)
    assert len(history) == 3

    # Run cleanup
    deleted = await repo.cleanup_duplicate_orders()
    assert deleted == 1

    # After cleanup: 2 rows (1 duplicate removed)
    history = await repo.get_trade_history(limit=100)
    assert len(history) == 2

    # The kept row should be the filled one (order_b, with filled_at)
    remaining = await repo.find_by_kis_order_id("0013045000")
    assert remaining is not None
    assert remaining.status == "filled"
    assert remaining.filled_quantity == 60

    # Unique order untouched
    unique_check = await repo.find_by_kis_order_id("UNIQUE001")
    assert unique_check is not None


@pytest.mark.asyncio
async def test_cleanup_no_duplicates(repo):
    """cleanup_duplicate_orders returns 0 when no duplicates exist."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=180.0,
        kis_order_id="US001",
    )
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="market",
        quantity=5,
        price=400.0,
        kis_order_id="US002",
    )

    deleted = await repo.cleanup_duplicate_orders()
    assert deleted == 0


# --- Exchange field propagation (STOCK-5) ---


@pytest.mark.asyncio
async def test_save_order_kr_exchange_krx(repo):
    """KR orders are saved with exchange='KRX', not default 'NASD'."""
    order = await repo.save_order(
        symbol="005930",
        side="buy",
        order_type="limit",
        quantity=10,
        price=70000.0,
        status="filled",
        strategy_name="supertrend",
        exchange="KRX",
        market="KR",
    )
    assert order.exchange == "KRX"
    assert order.market == "KR"


@pytest.mark.asyncio
async def test_save_order_us_exchange_nyse(repo):
    """US NYSE orders are saved with exchange='NYSE'."""
    order = await repo.save_order(
        symbol="BAC",
        side="buy",
        order_type="limit",
        quantity=50,
        price=40.0,
        status="filled",
        exchange="NYSE",
        market="US",
    )
    assert order.exchange == "NYSE"


@pytest.mark.asyncio
async def test_save_order_us_exchange_amex(repo):
    """US AMEX orders are saved with exchange='AMEX'."""
    order = await repo.save_order(
        symbol="SOXL",
        side="buy",
        order_type="limit",
        quantity=20,
        price=30.0,
        status="filled",
        exchange="AMEX",
        market="US",
    )
    assert order.exchange == "AMEX"


@pytest.mark.asyncio
async def test_save_order_default_exchange_nasd(repo):
    """Orders without explicit exchange default to 'NASD'."""
    order = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
    )
    assert order.exchange == "NASD"


@pytest.mark.asyncio
async def test_cleanup_ignores_empty_kis_order_id(session):
    """cleanup_duplicate_orders ignores orders with empty kis_order_id."""
    repo = TradeRepository(session)

    from core.models import Order

    # Multiple orders with empty kis_order_id — not duplicates
    for _ in range(3):
        order = Order(
            symbol="AAPL",
            exchange="NASD",
            side="buy",
            order_type="market",
            quantity=10,
            price=180.0,
            status="filled",
            kis_order_id="",
            market="US",
        )
        session.add(order)
    await session.commit()

    deleted = await repo.cleanup_duplicate_orders()
    assert deleted == 0

    history = await repo.get_trade_history(limit=100)
    assert len(history) == 3  # All preserved


# --- STOCK-37: Protect orders with PnL from not_found ---


@pytest.mark.asyncio
async def test_save_order_upsert_prevents_not_found_when_pnl_exists(repo):
    """STOCK-37: UPSERT should not downgrade to not_found when PnL exists."""
    # Insert order with PnL (simulates place_sell recording)
    order1 = await repo.save_order(
        symbol="AMPX",
        side="sell",
        order_type="market",
        quantity=10,
        price=25.0,
        status="submitted",
        kis_order_id="US_SELL_001",
        pnl=19.26,
        market="US",
    )
    assert order1.status == "submitted"
    assert order1.pnl == 19.26

    # Reconciliation tries to set not_found
    order2 = await repo.save_order(
        symbol="AMPX",
        side="sell",
        order_type="market",
        quantity=10,
        price=25.0,
        status="not_found",
        kis_order_id="US_SELL_001",
        market="US",
    )

    # Should be overridden to "filled" because PnL exists
    assert order2.id == order1.id
    assert order2.status == "filled"
    assert order2.pnl == 19.26
    assert order2.filled_at is not None


@pytest.mark.asyncio
async def test_save_order_upsert_allows_not_found_when_no_pnl(repo):
    """STOCK-37: not_found is allowed when there's no PnL (order never filled)."""
    order1 = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        price=150.0,
        status="submitted",
        kis_order_id="US_BUY_001",
        market="US",
    )
    assert order1.pnl is None

    order2 = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        price=150.0,
        status="not_found",
        kis_order_id="US_BUY_001",
        market="US",
    )

    # No PnL → not_found is allowed
    assert order2.id == order1.id
    assert order2.status == "not_found"


@pytest.mark.asyncio
async def test_get_recent_trades_includes_not_found_with_pnl(repo):
    """STOCK-37: get_recent_trades returns not_found orders with PnL."""
    # Create a filled order
    filled = await repo.save_order(
        symbol="AAPL",
        side="sell",
        order_type="market",
        quantity=10,
        price=180.0,
        status="pending",
        kis_order_id="FILLED_001",
        pnl=50.0,
        market="US",
    )
    await repo.update_order_status(filled.id, "filled", filled_price=180.0)

    # Create a not_found order with PnL (STOCK-37 scenario)
    await repo.save_order(
        symbol="AMPX",
        side="sell",
        order_type="market",
        quantity=10,
        price=25.0,
        status="not_found",
        kis_order_id="NOTFOUND_001",
        pnl=19.26,
        market="US",
    )

    recent = await repo.get_recent_trades(hours=24)
    symbols = {o.symbol for o in recent}
    assert "AAPL" in symbols
    assert "AMPX" in symbols  # not_found with PnL should be included


@pytest.mark.asyncio
async def test_get_recent_trades_excludes_not_found_without_pnl(repo):
    """STOCK-37: not_found orders without PnL are still excluded."""
    await repo.save_order(
        symbol="TSLA",
        side="buy",
        order_type="limit",
        quantity=5,
        price=200.0,
        status="not_found",
        kis_order_id="NOTFOUND_002",
        market="US",
    )

    recent = await repo.get_recent_trades(hours=24)
    assert len(recent) == 0  # No PnL → not included


@pytest.mark.asyncio
async def test_save_order_upsert_not_found_with_negative_pnl(repo):
    """STOCK-37: Negative PnL also protects from not_found (loss is still a fill)."""
    order1 = await repo.save_order(
        symbol="LION",
        side="sell",
        order_type="market",
        quantity=10,
        price=8.0,
        status="submitted",
        kis_order_id="US_SELL_LOSS",
        pnl=-51.62,
        market="US",
    )

    order2 = await repo.save_order(
        symbol="LION",
        side="sell",
        order_type="market",
        quantity=10,
        price=8.0,
        status="not_found",
        kis_order_id="US_SELL_LOSS",
        market="US",
    )

    assert order2.id == order1.id
    assert order2.status == "filled"
    assert order2.pnl == -51.62


# --- STOCK-38: recover_not_found_orders ---


@pytest.mark.asyncio
async def test_recover_not_found_orders_with_pnl(repo, session):
    """Orders with not_found status and PnL are recovered to filled."""
    from core.models import Order

    # not_found order WITH pnl — should be recovered
    order1 = Order(
        symbol="AAPL",
        exchange="NASD",
        side="SELL",
        order_type="market",
        quantity=10,
        price=155.0,
        status="not_found",
        pnl=50.0,
        kis_order_id="KIS001",
        market="US",
    )
    session.add(order1)

    # not_found order WITHOUT pnl — should NOT be recovered
    order2 = Order(
        symbol="MSFT",
        exchange="NASD",
        side="BUY",
        order_type="market",
        quantity=5,
        price=400.0,
        status="not_found",
        pnl=None,
        kis_order_id="KIS002",
        market="US",
    )
    session.add(order2)

    # filled order — should NOT be touched
    order3 = Order(
        symbol="GOOGL",
        exchange="NASD",
        side="SELL",
        order_type="market",
        quantity=3,
        price=140.0,
        filled_price=141.0,
        status="filled",
        pnl=3.0,
        kis_order_id="KIS003",
        market="US",
    )
    session.add(order3)

    await session.commit()

    recovered_ids = await repo.recover_not_found_orders()
    assert len(recovered_ids) == 1
    assert "KIS001" in recovered_ids

    await session.refresh(order1)
    assert order1.status == "filled"
    assert order1.filled_price == 155.0  # Set from price since filled_price was None
    assert order1.filled_quantity == 10  # Set from quantity since filled_quantity was None

    await session.refresh(order2)
    assert order2.status == "not_found"  # No PnL → not recovered

    await session.refresh(order3)
    assert order3.status == "filled"
    assert order3.filled_price == 141.0  # Unchanged


@pytest.mark.asyncio
async def test_recover_not_found_preserves_existing_filled_price(repo, session):
    """Recovery does not overwrite existing filled_price."""
    from core.models import Order

    order = Order(
        symbol="NVDA",
        exchange="NASD",
        side="SELL",
        order_type="market",
        quantity=5,
        price=300.0,
        filled_price=305.0,  # Already has filled_price
        status="not_found",
        pnl=25.0,
        kis_order_id="KIS004",
        market="US",
    )
    session.add(order)
    await session.commit()

    recovered_ids = await repo.recover_not_found_orders()
    assert len(recovered_ids) == 1

    await session.refresh(order)
    assert order.status == "filled"
    assert order.filled_price == 305.0  # Preserved, not overwritten with 300.0


@pytest.mark.asyncio
async def test_recover_not_found_zero_when_none_match(repo, session):
    """Returns 0 when no not_found orders with PnL exist."""
    from core.models import Order

    # Only has pending order
    order = Order(
        symbol="AAPL",
        exchange="NASD",
        side="BUY",
        order_type="market",
        quantity=10,
        price=150.0,
        status="pending",
        market="US",
    )
    session.add(order)
    await session.commit()

    recovered_ids = await repo.recover_not_found_orders()
    assert len(recovered_ids) == 0


@pytest.mark.asyncio
async def test_recover_not_found_kr_market(repo, session):
    """Recovery works for KR market orders too."""
    from core.models import Order

    order = Order(
        symbol="005930",
        exchange="KRX",
        side="SELL",
        order_type="market",
        quantity=10,
        price=70000.0,
        status="not_found",
        pnl=5000.0,
        kis_order_id="KR001",
        market="KR",
    )
    session.add(order)
    await session.commit()

    recovered_ids = await repo.recover_not_found_orders()
    assert len(recovered_ids) == 1
    assert "KR001" in recovered_ids

    await session.refresh(order)
    assert order.status == "filled"
    assert order.filled_price == 70000.0


@pytest.mark.asyncio
async def test_recover_not_found_commits_orders_without_kis_order_id(repo, session):
    """Orders without kis_order_id are still committed to DB.

    Regression test: previously the commit was gated on `if recovered_ids`,
    but recovered_ids only includes orders with non-empty kis_order_id.
    Orders without kis_order_id had their ORM mutations silently discarded.
    """
    from core.models import Order

    order = Order(
        symbol="AAPL",
        exchange="NASD",
        side="SELL",
        order_type="market",
        quantity=10,
        price=155.0,
        status="not_found",
        pnl=50.0,
        kis_order_id="",  # Empty kis_order_id
        market="US",
    )
    session.add(order)
    await session.commit()

    recovered_ids = await repo.recover_not_found_orders()
    # No kis_order_id → not in recovered_ids, but still committed
    assert len(recovered_ids) == 0

    await session.refresh(order)
    assert order.status == "filled"
    assert order.filled_at == order.created_at
    assert order.filled_price == 155.0
    assert order.filled_quantity == 10


# --- STOCK-83: account_id multi-account support ---


@pytest.mark.asyncio
async def test_save_order_with_explicit_account_id(repo):
    """save_order persists the given account_id on the order row."""
    order = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        account_id="ACC002",
    )
    assert order.account_id == "ACC002"


@pytest.mark.asyncio
async def test_save_order_default_account_id(repo):
    """save_order defaults account_id to 'ACC001' (backward compat)."""
    order = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
    )
    assert order.account_id == "ACC001"


@pytest.mark.asyncio
async def test_get_trade_history_filter_by_account_id(repo):
    """get_trade_history(account_id=...) returns only that account's orders."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        account_id="ACC001",
    )
    await repo.save_order(
        symbol="TSLA",
        side="buy",
        order_type="market",
        quantity=5,
        price=200.0,
        account_id="ACC002",
    )
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="market",
        quantity=3,
        price=400.0,
        account_id="ACC001",
    )

    acc1 = await repo.get_trade_history(account_id="ACC001")
    assert len(acc1) == 2
    assert all(o.account_id == "ACC001" for o in acc1)

    acc2 = await repo.get_trade_history(account_id="ACC002")
    assert len(acc2) == 1
    assert acc2[0].symbol == "TSLA"

    all_orders = await repo.get_trade_history()
    assert len(all_orders) == 3


@pytest.mark.asyncio
async def test_get_open_orders_filter_by_account_id(repo):
    """get_open_orders(account_id=...) filters to a specific account."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="limit",
        quantity=10,
        price=150.0,
        status="pending",
        account_id="ACC001",
    )
    await repo.save_order(
        symbol="TSLA",
        side="buy",
        order_type="limit",
        quantity=5,
        price=200.0,
        status="pending",
        account_id="ACC002",
    )

    acc1_open = await repo.get_open_orders(account_id="ACC001")
    assert len(acc1_open) == 1
    assert acc1_open[0].symbol == "AAPL"

    acc2_open = await repo.get_open_orders(account_id="ACC002")
    assert len(acc2_open) == 1
    assert acc2_open[0].symbol == "TSLA"

    all_open = await repo.get_open_orders()
    assert len(all_open) == 2


@pytest.mark.asyncio
async def test_get_recent_trades_filter_by_account_id(repo):
    """get_recent_trades(account_id=...) filters to a specific account."""
    # Create filled orders for two accounts
    order1 = await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        status="pending",
        account_id="ACC001",
    )
    await repo.update_order_status(order1.id, "filled", filled_price=150.0)

    order2 = await repo.save_order(
        symbol="TSLA",
        side="buy",
        order_type="market",
        quantity=5,
        price=200.0,
        status="pending",
        account_id="ACC002",
    )
    await repo.update_order_status(order2.id, "filled", filled_price=200.0)

    acc1_recent = await repo.get_recent_trades(hours=24, account_id="ACC001")
    assert len(acc1_recent) == 1
    assert acc1_recent[0].symbol == "AAPL"

    acc2_recent = await repo.get_recent_trades(hours=24, account_id="ACC002")
    assert len(acc2_recent) == 1
    assert acc2_recent[0].symbol == "TSLA"

    all_recent = await repo.get_recent_trades(hours=24)
    assert len(all_recent) == 2


@pytest.mark.asyncio
async def test_account_id_combined_with_symbol_filter(repo):
    """account_id filter composes correctly with symbol filter."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        account_id="ACC001",
    )
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=5,
        price=155.0,
        account_id="ACC002",
    )

    result = await repo.get_trade_history(symbol="AAPL", account_id="ACC001")
    assert len(result) == 1
    assert result[0].account_id == "ACC001"


@pytest.mark.asyncio
async def test_account_id_combined_with_exclude_paper(repo):
    """account_id filter composes correctly with exclude_paper filter."""
    await repo.save_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        price=150.0,
        account_id="ACC001",
        is_paper=True,
    )
    await repo.save_order(
        symbol="MSFT",
        side="buy",
        order_type="market",
        quantity=5,
        price=400.0,
        account_id="ACC001",
        is_paper=False,
    )
    await repo.save_order(
        symbol="TSLA",
        side="buy",
        order_type="market",
        quantity=3,
        price=200.0,
        account_id="ACC002",
        is_paper=False,
    )

    # Only live orders for ACC001
    result = await repo.get_trade_history(account_id="ACC001", exclude_paper=True)
    assert len(result) == 1
    assert result[0].symbol == "MSFT"
