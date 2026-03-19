"""Tests for STOCK-37: trade-summary includes not_found orders with PnL.

Validates:
- trade_summary_periods includes not_found orders with PnL
- Period filtering uses created_at as fallback when filled_at is NULL
- total_buys includes not_found orders
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base, Order


@pytest.fixture
async def db_session():
    """Create in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session, factory
    await engine.dispose()


def _make_order(
    symbol: str,
    side: str,
    status: str,
    pnl: float | None = None,
    pnl_pct: float | None = None,
    filled_at: datetime | None = None,
    created_at: datetime | None = None,
    filled_price: float | None = None,
    quantity: float = 10,
    price: float = 100.0,
    market: str = "US",
) -> Order:
    """Helper to create an Order ORM object."""
    return Order(
        symbol=symbol,
        exchange="NASD",
        side=side,
        order_type="market",
        quantity=quantity,
        price=price,
        filled_price=filled_price,
        filled_quantity=quantity if status == "filled" else 0,
        status=status,
        strategy_name="test",
        pnl=pnl,
        pnl_pct=pnl_pct,
        market=market,
        filled_at=filled_at,
        created_at=created_at or datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_trade_summary_includes_not_found_with_pnl(db_session):
    """STOCK-37: not_found orders with PnL should appear in trade summary."""
    session, factory = db_session
    now = datetime.utcnow()

    # Filled order (normal)
    filled = _make_order(
        "AAPL",
        "SELL",
        "filled",
        pnl=50.0,
        pnl_pct=5.0,
        filled_at=now - timedelta(hours=1),
        filled_price=105.0,
    )
    # Not-found order with PnL (STOCK-37 scenario)
    not_found = _make_order(
        "AMPX",
        "SELL",
        "not_found",
        pnl=19.26,
        pnl_pct=2.0,
        filled_at=None,  # NULL — this is the problem case
        created_at=now - timedelta(hours=2),
        filled_price=None,
    )
    session.add_all([filled, not_found])
    await session.commit()

    # Patch _session_factory for trade_summary_periods
    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request, market="US")

    # Both orders should be counted
    assert result["all_time"]["trades"] == 2
    assert result["all_time"]["pnl"] == pytest.approx(69.26)  # 50.0 + 19.26
    assert result["all_time"]["wins"] == 2
    assert result["total_sells"] == 2


@pytest.mark.asyncio
async def test_trade_summary_not_found_in_today_period(db_session):
    """STOCK-37: not_found orders with PnL appear in today period using created_at."""
    session, factory = db_session
    now = datetime.utcnow()

    not_found = _make_order(
        "DOCN",
        "SELL",
        "not_found",
        pnl=41.60,
        pnl_pct=3.0,
        filled_at=None,
        created_at=now - timedelta(hours=1),
    )
    session.add(not_found)
    await session.commit()

    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request)

    # Should appear in today's trades
    assert result["today"]["trades"] == 1
    assert result["today"]["pnl"] == pytest.approx(41.60)


@pytest.mark.asyncio
async def test_trade_summary_excludes_not_found_without_pnl(db_session):
    """STOCK-37: not_found orders WITHOUT PnL should still be excluded."""
    session, factory = db_session

    not_found_no_pnl = _make_order(
        "TSLA",
        "SELL",
        "not_found",
        pnl=None,
        filled_at=None,
    )
    session.add(not_found_no_pnl)
    await session.commit()

    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request)

    # Should not be counted
    assert result["all_time"]["trades"] == 0
    assert result["total_sells"] == 0


@pytest.mark.asyncio
async def test_trade_summary_total_buys_includes_not_found(db_session):
    """STOCK-37: total_buys should count not_found BUY orders."""
    session, factory = db_session

    buy_filled = _make_order("AAPL", "BUY", "filled")
    buy_not_found = _make_order("MSFT", "BUY", "not_found")
    session.add_all([buy_filled, buy_not_found])
    await session.commit()

    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request)

    assert result["total_buys"] == 2


@pytest.mark.asyncio
async def test_trade_summary_not_found_losses_counted(db_session):
    """STOCK-37: not_found orders with negative PnL (losses) are counted."""
    session, factory = db_session
    now = datetime.utcnow()

    loss = _make_order(
        "LION",
        "SELL",
        "not_found",
        pnl=-51.62,
        pnl_pct=-8.0,
        filled_at=None,
        created_at=now - timedelta(hours=1),
    )
    session.add(loss)
    await session.commit()

    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request)

    assert result["all_time"]["trades"] == 1
    assert result["all_time"]["losses"] == 1
    assert result["all_time"]["pnl"] == pytest.approx(-51.62)


@pytest.mark.asyncio
async def test_trade_summary_mixed_filled_and_not_found(db_session):
    """STOCK-37: Full scenario matching the issue — 11 not_found trades with PnL."""
    session, factory = db_session
    now = datetime.utcnow()

    # Replicate the 11 orders from the issue
    orders_data = [
        ("AMPX", 19.26),
        ("XLU", -0.09),
        ("ADTN", -7.6),
        ("CVE", 1.91),
        ("CNQ", 62.02),
        ("AEP", -3.27),
        ("DOCN", 41.6),
        ("CF", 24.56),
        ("LION", -51.62),
        ("VG", 67.41),
        ("NFLX", 0.34),
    ]
    for symbol, pnl in orders_data:
        order = _make_order(
            symbol,
            "SELL",
            "not_found",
            pnl=pnl,
            filled_at=None,
            created_at=now - timedelta(hours=1),
        )
        session.add(order)
    await session.commit()

    with patch("api.portfolio._session_factory", factory):
        with patch("api.trades._session_factory", factory):
            from api.portfolio import trade_summary_periods

            mock_request = MagicMock()
            result = await trade_summary_periods(mock_request)

    assert result["all_time"]["trades"] == 11
    assert result["all_time"]["pnl"] == pytest.approx(154.52)
    assert result["all_time"]["wins"] == 7  # positive PnL (AMPX, CVE, CNQ, DOCN, CF, VG, NFLX)
    assert result["all_time"]["losses"] == 4  # negative PnL (XLU, ADTN, AEP, LION)
    assert result["today"]["trades"] == 11
    assert result["total_sells"] == 11
