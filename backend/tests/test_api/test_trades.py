"""Tests for trade history module: restore, reconciliation, persistence."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.trades import (
    _trade_log,
    record_trade,
    restore_trade_log,
    reconcile_pending_orders,
    update_order_in_db,
    init_trades,
    _order_to_dict,
)


@pytest.fixture(autouse=True)
def clear_trade_log():
    """Clear in-memory trade log before/after each test."""
    _trade_log.clear()
    yield
    _trade_log.clear()


class TestRecordTrade:
    def test_appends_to_in_memory_log(self):
        record_trade({"symbol": "AAPL", "side": "BUY", "status": "pending"})
        assert len(_trade_log) == 1
        assert _trade_log[0]["symbol"] == "AAPL"

    def test_includes_order_id(self):
        record_trade({
            "order_id": "KIS123",
            "symbol": "MSFT",
            "side": "BUY",
            "status": "pending",
        })
        assert _trade_log[0]["order_id"] == "KIS123"


class TestRestoreTradeLog:
    @pytest.mark.asyncio
    async def test_restore_from_db(self):
        """Restore trade log populates _trade_log from DB orders."""
        mock_order = MagicMock()
        mock_order.id = 1
        mock_order.kis_order_id = "KIS001"
        mock_order.symbol = "AAPL"
        mock_order.side = "BUY"
        mock_order.quantity = 10
        mock_order.price = 150.0
        mock_order.filled_price = 150.5
        mock_order.filled_quantity = 10
        mock_order.status = "filled"
        mock_order.strategy_name = "trend_following"
        mock_order.pnl = None
        mock_order.market = "US"
        mock_order.created_at = "2026-03-10 10:00:00"

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=[mock_order])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("db.trade_repository.TradeRepository", return_value=mock_repo):
                count = await restore_trade_log()

        assert count == 1
        assert _trade_log[0]["symbol"] == "AAPL"
        assert _trade_log[0]["order_id"] == "KIS001"
        assert _trade_log[0]["status"] == "filled"

    @pytest.mark.asyncio
    async def test_restore_no_session_factory(self):
        with patch("api.trades._session_factory", None):
            count = await restore_trade_log()
        assert count == 0


class TestReconcilePendingOrders:
    @pytest.mark.asyncio
    async def test_buy_held_marked_filled(self):
        """Pending BUY order for held symbol → filled."""
        mock_order = MagicMock()
        mock_order.id = 1
        mock_order.symbol = "AAPL"
        mock_order.side = "BUY"
        mock_order.price = 150.0
        mock_order.quantity = 10

        mock_repo = AsyncMock()
        mock_repo.get_open_orders = AsyncMock(return_value=[mock_order])
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("db.trade_repository.TradeRepository", return_value=mock_repo):
                updated = await reconcile_pending_orders({"AAPL", "MSFT"})

        assert updated == 1
        mock_repo.update_order_status.assert_called_once_with(
            1, "filled", filled_price=150.0, filled_quantity=10,
        )

    @pytest.mark.asyncio
    async def test_buy_not_held_marked_cancelled(self):
        """Pending BUY order for non-held symbol → cancelled."""
        mock_order = MagicMock()
        mock_order.id = 2
        mock_order.symbol = "TSLA"
        mock_order.side = "BUY"
        mock_order.price = 200.0
        mock_order.quantity = 5

        mock_repo = AsyncMock()
        mock_repo.get_open_orders = AsyncMock(return_value=[mock_order])
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("db.trade_repository.TradeRepository", return_value=mock_repo):
                updated = await reconcile_pending_orders({"AAPL"})

        assert updated == 1
        mock_repo.update_order_status.assert_called_once_with(2, "cancelled")

    @pytest.mark.asyncio
    async def test_sell_not_held_marked_filled(self):
        """Pending SELL order for non-held symbol → filled."""
        mock_order = MagicMock()
        mock_order.id = 3
        mock_order.symbol = "AAPL"
        mock_order.side = "SELL"
        mock_order.price = 160.0
        mock_order.quantity = 10

        mock_repo = AsyncMock()
        mock_repo.get_open_orders = AsyncMock(return_value=[mock_order])
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("db.trade_repository.TradeRepository", return_value=mock_repo):
                updated = await reconcile_pending_orders(set())

        assert updated == 1
        mock_repo.update_order_status.assert_called_once_with(
            3, "filled", filled_price=160.0, filled_quantity=10,
        )


class TestUpdateOrderInDb:
    @pytest.mark.asyncio
    async def test_updates_db_and_trade_log(self):
        """Update DB and in-memory trade log by kis_order_id."""
        _trade_log.append({
            "order_id": "KIS123",
            "symbol": "AAPL",
            "status": "pending",
            "filled_price": None,
        })

        mock_order = MagicMock()
        mock_order.id = 1

        mock_repo = AsyncMock()
        mock_repo.find_by_kis_order_id = AsyncMock(return_value=mock_order)
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("db.trade_repository.TradeRepository", return_value=mock_repo):
                ok = await update_order_in_db("KIS123", "filled", 155.0, 10)

        assert ok is True
        assert _trade_log[0]["status"] == "filled"
        assert _trade_log[0]["filled_price"] == 155.0

    @pytest.mark.asyncio
    async def test_no_order_id_returns_false(self):
        ok = await update_order_in_db("", "filled")
        assert ok is False


class TestOrderToDict:
    def test_converts_order_to_dict(self):
        mock_order = MagicMock()
        mock_order.id = 5
        mock_order.kis_order_id = "KIS555"
        mock_order.symbol = "NVDA"
        mock_order.side = "BUY"
        mock_order.quantity = 20
        mock_order.price = 300.0
        mock_order.filled_price = 301.0
        mock_order.filled_quantity = 20
        mock_order.status = "filled"
        mock_order.strategy_name = "bnf_deviation"
        mock_order.pnl = 50.0
        mock_order.market = "US"
        mock_order.created_at = "2026-03-10 15:30:00"

        d = _order_to_dict(mock_order)
        assert d["order_id"] == "KIS555"
        assert d["symbol"] == "NVDA"
        assert d["status"] == "filled"
        assert d["db_id"] == 5
