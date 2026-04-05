"""Tests for PnL percentage (pnl_pct) feature — STOCK-16.

Covers:
- Order model pnl_pct column
- OrderManager pnl_pct calculation on sell
- Trade API pnl_pct in responses (_order_to_dict, DB fallback, _persist_trade)
- Portfolio summary total_unrealized_pnl_pct
- Trade period summary avg pnl_pct
- Notification service pnl_pct in SL/TP/trailing stop messages
- Position tracker pnl_pct calculation and propagation to notifications
- DB migration for pnl_pct column
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# ── Order model tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_order_model_has_pnl_pct_column():
    """Order ORM model includes pnl_pct column."""
    from core.models import Order

    # Check the column exists on the model
    assert hasattr(Order, "pnl_pct")


@pytest.mark.asyncio
async def test_order_pnl_pct_persisted_to_db():
    """pnl_pct value is stored and retrieved from DB correctly."""
    from core.models import Base, Order

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        order = Order(
            symbol="AAPL",
            exchange="NASD",
            side="SELL",
            order_type="market",
            quantity=10,
            price=165.0,
            filled_price=165.0,
            filled_quantity=10,
            status="filled",
            pnl=150.0,
            pnl_pct=10.0,
            market="US",
        )
        session.add(order)
        await session.commit()
        await session.refresh(order)

    async with factory() as session:
        from sqlalchemy import select

        result = await session.execute(select(Order).where(Order.symbol == "AAPL"))
        loaded = result.scalar_one()
        assert loaded.pnl_pct == 10.0
        assert loaded.pnl == 150.0

    await engine.dispose()


@pytest.mark.asyncio
async def test_order_pnl_pct_nullable():
    """pnl_pct can be NULL (BUY orders or old data)."""
    from core.models import Base, Order

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        order = Order(
            symbol="MSFT",
            exchange="NASD",
            side="BUY",
            order_type="market",
            quantity=5,
            price=400.0,
            status="filled",
            market="US",
        )
        session.add(order)
        await session.commit()
        await session.refresh(order)
        assert order.pnl_pct is None

    await engine.dispose()


# ── OrderManager pnl_pct calculation tests ────────────────────────────


@pytest.mark.asyncio
async def test_order_manager_calculates_pnl_pct_on_sell():
    """place_sell calculates pnl_pct = ((sell - entry) / entry) * 100."""
    from engine.order_manager import OrderManager, set_trade_recorder
    from exchange.base import OrderResult

    recorded = []
    set_trade_recorder(lambda t, **kw: recorded.append(t))

    adapter = AsyncMock()
    adapter.create_sell_order.return_value = OrderResult(
        order_id="ORD1", symbol="AAPL", side="SELL", order_type="market",
        quantity=10, status="filled", filled_price=165.0, filled_quantity=10,
    )

    risk = MagicMock()
    om = OrderManager(adapter, risk, market="US")

    await om.place_sell(
        symbol="AAPL",
        quantity=10,
        price=165.0,
        strategy_name="test",
        entry_price=150.0,
    )

    assert len(recorded) == 1
    trade = recorded[0]
    assert trade["pnl"] == 150.0  # (165 - 150) * 10
    assert trade["pnl_pct"] == 10.0  # (165 - 150) / 150 * 100

    # Clean up
    set_trade_recorder(None)


@pytest.mark.asyncio
async def test_order_manager_pnl_pct_negative():
    """pnl_pct is negative when sell price < entry price."""
    from engine.order_manager import OrderManager, set_trade_recorder
    from exchange.base import OrderResult

    recorded = []
    set_trade_recorder(lambda t, **kw: recorded.append(t))

    adapter = AsyncMock()
    adapter.create_sell_order.return_value = OrderResult(
        order_id="ORD2", symbol="AAPL", side="SELL", order_type="market",
        quantity=10, status="filled", filled_price=135.0, filled_quantity=10,
    )

    risk = MagicMock()
    om = OrderManager(adapter, risk, market="US")

    await om.place_sell(
        symbol="AAPL",
        quantity=10,
        price=135.0,
        strategy_name="test",
        entry_price=150.0,
    )

    assert len(recorded) == 1
    trade = recorded[0]
    assert trade["pnl"] == -150.0
    assert trade["pnl_pct"] == -10.0

    set_trade_recorder(None)


@pytest.mark.asyncio
async def test_order_manager_pnl_pct_none_without_entry_price():
    """pnl_pct is None when entry_price is not provided."""
    from engine.order_manager import OrderManager, set_trade_recorder
    from exchange.base import OrderResult

    recorded = []
    set_trade_recorder(lambda t, **kw: recorded.append(t))

    adapter = AsyncMock()
    adapter.create_sell_order.return_value = OrderResult(
        order_id="ORD3", symbol="AAPL", side="SELL", order_type="market",
        quantity=10, status="filled", filled_price=165.0, filled_quantity=10,
    )

    risk = MagicMock()
    om = OrderManager(adapter, risk, market="US")

    await om.place_sell(
        symbol="AAPL",
        quantity=10,
        price=165.0,
        strategy_name="test",
        entry_price=None,
    )

    assert len(recorded) == 1
    trade = recorded[0]
    assert trade["pnl"] is None
    assert trade["pnl_pct"] is None

    set_trade_recorder(None)


@pytest.mark.asyncio
async def test_order_manager_pnl_pct_zero_entry_price():
    """pnl_pct is None when entry_price is 0 (avoid division by zero)."""
    from engine.order_manager import OrderManager, set_trade_recorder
    from exchange.base import OrderResult

    recorded = []
    set_trade_recorder(lambda t, **kw: recorded.append(t))

    adapter = AsyncMock()
    adapter.create_sell_order.return_value = OrderResult(
        order_id="ORD4", symbol="AAPL", side="SELL", order_type="market",
        quantity=10, status="filled", filled_price=165.0, filled_quantity=10,
    )

    risk = MagicMock()
    om = OrderManager(adapter, risk, market="US")

    # entry_price=0 is falsy, so pnl block won't be entered
    await om.place_sell(
        symbol="AAPL",
        quantity=10,
        price=165.0,
        strategy_name="test",
        entry_price=0,
    )

    assert len(recorded) == 1
    trade = recorded[0]
    assert trade["pnl"] is None
    assert trade["pnl_pct"] is None

    set_trade_recorder(None)


# ── Trade API tests ───────────────────────────────────────────────────


def test_order_to_dict_includes_pnl_pct():
    """_order_to_dict includes pnl_pct from Order object."""
    from api.trades import order_to_dict

    order = MagicMock()
    order.kis_order_id = "ORD1"
    order.symbol = "AAPL"
    order.side = "SELL"
    order.quantity = 10
    order.price = 165.0
    order.filled_price = 165.0
    order.filled_quantity = 10
    order.status = "filled"
    order.strategy_name = "test"
    order.buy_strategy = ""
    order.pnl = 150.0
    order.pnl_pct = 10.0
    order.is_paper = False
    order.market = "US"
    order.session = "regular"
    order.created_at = datetime(2024, 1, 1)
    order.id = 1

    d = order_to_dict(order)
    assert d["pnl_pct"] == 10.0


def test_order_to_dict_pnl_pct_none():
    """_order_to_dict handles missing pnl_pct (old orders)."""
    from api.trades import order_to_dict

    order = MagicMock(spec=[])
    order.kis_order_id = "ORD2"
    order.symbol = "MSFT"
    order.side = "BUY"
    order.quantity = 5
    order.price = 400.0
    order.filled_price = 400.0
    order.filled_quantity = 5
    order.status = "filled"
    order.strategy_name = "test"
    order.pnl = None
    order.id = 2
    order.created_at = datetime(2024, 1, 1)
    # pnl_pct not set (simulates old data)

    d = order_to_dict(order)
    assert d["pnl_pct"] is None


@pytest.mark.asyncio
async def test_persist_trade_passes_pnl_pct():
    """_persist_trade passes pnl_pct to TradeRepository.save_order."""
    from api import trades

    old_factory = trades._session_factory
    try:
        mock_session = AsyncMock()
        mock_repo = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        trades._session_factory = MagicMock(return_value=ctx)

        with patch("api.trades.TradeRepository", return_value=mock_repo):
            await trades._persist_trade({
                "symbol": "AAPL",
                "side": "SELL",
                "quantity": 10,
                "price": 165.0,
                "filled_price": 165.0,
                "filled_quantity": 10,
                "status": "filled",
                "strategy": "test",
                "buy_strategy": "",
                "order_id": "ORD1",
                "pnl": 150.0,
                "pnl_pct": 10.0,
                "exchange": "NASD",
                "market": "US",
                "session": "regular",
                "is_paper": False,
            })

            mock_repo.save_order.assert_called_once()
            call_kwargs = mock_repo.save_order.call_args
            assert call_kwargs.kwargs.get("pnl_pct") == 10.0
    finally:
        trades._session_factory = old_factory


# ── Trade Repository tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trade_repository_save_order_with_pnl_pct():
    """TradeRepository.save_order stores pnl_pct correctly."""
    from core.models import Base
    from db.trade_repository import TradeRepository

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)
        order = await repo.save_order(
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            price=165.0,
            filled_price=165.0,
            filled_quantity=10,
            status="filled",
            pnl=150.0,
            pnl_pct=10.0,
        )
        assert order.pnl_pct == 10.0

    await engine.dispose()


@pytest.mark.asyncio
async def test_trade_repository_upsert_updates_pnl_pct():
    """TradeRepository.save_order upsert path updates pnl_pct."""
    from core.models import Base
    from db.trade_repository import TradeRepository

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        repo = TradeRepository(session)
        # Initial save without pnl_pct
        await repo.save_order(
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            price=165.0,
            status="filled",
            kis_order_id="KIS123",
        )
        # Upsert with pnl_pct
        order = await repo.save_order(
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            price=165.0,
            filled_price=165.0,
            filled_quantity=10,
            status="filled",
            kis_order_id="KIS123",
            pnl=150.0,
            pnl_pct=10.0,
        )
        assert order.pnl_pct == 10.0

    await engine.dispose()


# ── Portfolio summary tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_portfolio_summary_includes_total_unrealized_pnl_pct():
    """Single-market portfolio summary includes total_unrealized_pnl_pct."""
    from api.portfolio import portfolio_summary

    # Create mock position
    pos = MagicMock()
    pos.unrealized_pnl = 30.0  # $30 gain
    pos.avg_price = 150.0
    pos.quantity = 10  # cost = $1500
    pos.unrealized_pnl_pct = 2.0

    balance = MagicMock()
    balance.currency = "USD"
    balance.total = 10000.0
    balance.available = 8500.0
    balance.locked = 1500.0

    md = AsyncMock()
    md.get_balance.return_value = balance
    md.get_positions.return_value = [pos]

    request = MagicMock()
    request.app.state.market_data = md
    request.app.state.kr_market_data = None

    result = await portfolio_summary(request, market="US")
    assert "total_unrealized_pnl_pct" in result
    assert result["total_unrealized_pnl_pct"] == 2.0  # 30 / 1500 * 100


@pytest.mark.asyncio
async def test_portfolio_summary_pnl_pct_zero_when_no_positions():
    """total_unrealized_pnl_pct is 0 when there are no positions."""
    from api.portfolio import portfolio_summary

    balance = MagicMock()
    balance.currency = "USD"
    balance.total = 10000.0
    balance.available = 10000.0
    balance.locked = 0.0

    md = AsyncMock()
    md.get_balance.return_value = balance
    md.get_positions.return_value = []

    request = MagicMock()
    request.app.state.market_data = md
    request.app.state.kr_market_data = None

    result = await portfolio_summary(request, market="US")
    assert result["total_unrealized_pnl_pct"] == 0.0


# ── Notification tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notify_stop_loss_includes_pnl_pct():
    """notify_stop_loss message includes percentage when pnl_pct provided."""
    from services.notification import NotificationService

    svc = NotificationService(enabled=True)
    adapter = AsyncMock()
    adapter.is_configured = True
    adapter.name = "test"
    adapter.send_rich = AsyncMock(return_value=True)
    svc._adapters = [adapter]

    with patch("data.stock_name_service.get_name", return_value=None):
        await svc.notify_stop_loss("AAPL", 10, 150.0, 135.0, -150.0, pnl_pct=-10.0)

    adapter.send_rich.assert_called_once()
    call_args = adapter.send_rich.call_args
    plain_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("body", "")
    assert "-10.00%" in plain_text

    # Check fields contain P&L %
    fields = call_args.kwargs.get("fields") or call_args.args[3] if len(call_args.args) > 3 else {}
    assert "P&L %" in fields


@pytest.mark.asyncio
async def test_notify_take_profit_includes_pnl_pct():
    """notify_take_profit message includes percentage when pnl_pct provided."""
    from services.notification import NotificationService

    svc = NotificationService(enabled=True)
    adapter = AsyncMock()
    adapter.is_configured = True
    adapter.name = "test"
    adapter.send_rich = AsyncMock(return_value=True)
    svc._adapters = [adapter]

    with patch("data.stock_name_service.get_name", return_value=None):
        await svc.notify_take_profit("AAPL", 10, 150.0, 180.0, 300.0, pnl_pct=20.0)

    adapter.send_rich.assert_called_once()
    call_args = adapter.send_rich.call_args
    plain_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("body", "")
    assert "+20.00%" in plain_text


@pytest.mark.asyncio
async def test_notify_trailing_stop_includes_pnl_pct():
    """notify_trailing_stop message includes percentage when pnl_pct provided."""
    from services.notification import NotificationService

    svc = NotificationService(enabled=True)
    adapter = AsyncMock()
    adapter.is_configured = True
    adapter.name = "test"
    adapter.send_rich = AsyncMock(return_value=True)
    svc._adapters = [adapter]

    with patch("data.stock_name_service.get_name", return_value=None):
        await svc.notify_trailing_stop(
            "AAPL", 10, 150.0, 170.0, 180.0, 200.0, pnl_pct=13.33
        )

    adapter.send_rich.assert_called_once()
    call_args = adapter.send_rich.call_args
    plain_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("body", "")
    assert "+13.33%" in plain_text


@pytest.mark.asyncio
async def test_notify_stop_loss_without_pnl_pct():
    """notify_stop_loss works fine without pnl_pct (backward compatible)."""
    from services.notification import NotificationService

    svc = NotificationService(enabled=True)
    adapter = AsyncMock()
    adapter.is_configured = True
    adapter.name = "test"
    adapter.send_rich = AsyncMock(return_value=True)
    svc._adapters = [adapter]

    with patch("data.stock_name_service.get_name", return_value=None):
        await svc.notify_stop_loss("AAPL", 10, 150.0, 135.0, -150.0)

    adapter.send_rich.assert_called_once()
    call_args = adapter.send_rich.call_args
    plain_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("body", "")
    # No percentage should be in the message when pnl_pct is None
    assert "%" not in plain_text or "P&L %" not in str(call_args.kwargs.get("fields", {}))


# ── Trade period summary tests ────────────────────────────────────────


def test_empty_summary_includes_pnl_pct():
    """_empty_summary includes pnl_pct field set to None."""
    from api.portfolio import _empty_summary

    result = _empty_summary()
    assert result["today"]["pnl_pct"] is None
    assert result["week"]["pnl_pct"] is None
    assert result["month"]["pnl_pct"] is None
    assert result["all_time"]["pnl_pct"] is None


# ── DB migration tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_adds_pnl_pct_column():
    """ensure_columns adds pnl_pct to orders table when missing."""
    from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, inspect
    from sqlalchemy.orm import DeclarativeBase

    from db.migrations import ensure_columns

    class _Base(DeclarativeBase):
        pass

    class _OldOrder(_Base):
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
        is_paper = Column(Boolean, default=False)
        buy_strategy = Column(String(50))
        created_at = Column(DateTime)
        exchange = Column(String(10), default="NASD")
        market = Column(String(2), default="US")
        session = Column(String(20), default="regular")
        filled_quantity = Column(Float, default=0)
        filled_price = Column(Float)
        filled_at = Column(DateTime)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)

    # Verify pnl_pct is NOT in the schema yet
    async with engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "pnl_pct" not in cols

    # Run migration
    added = await ensure_columns(engine)
    assert "orders.pnl_pct" in added

    # Verify column now exists
    async with engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in inspect(sc).get_columns("orders")}
        )
    assert "pnl_pct" in cols

    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_idempotent_with_pnl_pct():
    """ensure_columns is idempotent when pnl_pct already exists."""
    from core.models import Base
    from db.migrations import ensure_columns

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # All columns already present
    added = await ensure_columns(engine)
    assert "orders.pnl_pct" not in added

    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_preserves_existing_data():
    """Migration doesn't corrupt existing rows when adding pnl_pct."""
    from sqlalchemy import Boolean, Column, Float, Integer, String
    from sqlalchemy.orm import DeclarativeBase

    from db.migrations import ensure_columns

    class _Base(DeclarativeBase):
        pass

    class _OldOrder(_Base):
        __tablename__ = "orders"
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20), nullable=False)
        side = Column(String(4), nullable=False)
        order_type = Column(String(10), nullable=False)
        quantity = Column(Float, nullable=False)
        price = Column(Float)
        status = Column(String(20), default="pending")
        pnl = Column(Float)
        is_paper = Column(Boolean, default=False)
        buy_strategy = Column(String(50))
        exchange = Column(String(10), default="NASD")
        market = Column(String(2), default="US")
        session = Column(String(20), default="regular")

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)

    # Insert row before migration
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (symbol, side, order_type, quantity, price, "
                "status, pnl, is_paper) "
                "VALUES ('AAPL', 'SELL', 'market', 10, 165.0, 'filled', 150.0, 0)"
            )
        )

    await ensure_columns(engine)

    # Verify existing data + new column
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT symbol, pnl, pnl_pct FROM orders WHERE symbol='AAPL'")
        )
        row = result.fetchone()
    assert row[0] == "AAPL"
    assert row[1] == 150.0
    assert row[2] is None  # pnl_pct is NULL for existing data

    await engine.dispose()
