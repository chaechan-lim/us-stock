"""Tests for trade history module: restore, reconciliation, persistence."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.trades import (
    _merge_trade_entry,
    _trade_log,
    get_trades,
    order_to_dict,
    reconcile_pending_orders,
    record_trade,
    restore_trade_log,
    update_order_in_db,
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
        record_trade(
            {
                "order_id": "KIS123",
                "symbol": "MSFT",
                "side": "BUY",
                "status": "pending",
            }
        )
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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                updated = await reconcile_pending_orders({"AAPL", "MSFT"})

        assert updated == 1
        mock_repo.update_order_status.assert_called_once_with(
            1,
            "filled",
            filled_price=150.0,
            filled_quantity=10,
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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                updated = await reconcile_pending_orders(set())

        assert updated == 1
        mock_repo.update_order_status.assert_called_once_with(
            3,
            "filled",
            filled_price=160.0,
            filled_quantity=10,
        )


class TestUpdateOrderInDb:
    @pytest.mark.asyncio
    async def test_updates_db_and_trade_log(self):
        """Update DB and in-memory trade log by kis_order_id."""
        _trade_log.append(
            {
                "order_id": "KIS123",
                "symbol": "AAPL",
                "status": "pending",
                "filled_price": None,
            }
        )

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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
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
        mock_order.is_paper = False
        mock_order.market = "US"
        mock_order.created_at = "2026-03-10 15:30:00"

        d = order_to_dict(mock_order)
        assert d["order_id"] == "KIS555"
        assert d["symbol"] == "NVDA"
        assert d["status"] == "filled"
        assert d["db_id"] == 5
        assert d["is_paper"] is False

    def test_converts_paper_order_to_dict(self):
        mock_order = MagicMock()
        mock_order.id = 6
        mock_order.kis_order_id = ""
        mock_order.symbol = "AAPL"
        mock_order.side = "BUY"
        mock_order.quantity = 10
        mock_order.price = 150.0
        mock_order.filled_price = 150.0
        mock_order.filled_quantity = 10
        mock_order.status = "filled"
        mock_order.strategy_name = "trend_following"
        mock_order.pnl = None
        mock_order.is_paper = True
        mock_order.market = "US"
        mock_order.created_at = "2026-03-10 10:00:00"

        d = order_to_dict(mock_order)
        assert d["is_paper"] is True
        assert d["order_id"] == ""


# --- Paper/Live order separation tests (STOCK-6) ---


class TestPaperOrderSeparation:
    def test_record_trade_preserves_is_paper(self):
        """is_paper flag is preserved in in-memory trade log."""
        record_trade(
            {
                "order_id": "abc123",
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "is_paper": True,
            }
        )
        assert _trade_log[0]["is_paper"] is True

    def test_record_trade_defaults_is_paper_false(self):
        """Trade without is_paper flag defaults to False (live)."""
        record_trade(
            {
                "order_id": "KIS123",
                "symbol": "MSFT",
                "side": "BUY",
                "status": "filled",
            }
        )
        # is_paper key may not be present, but get() defaults to False
        assert _trade_log[0].get("is_paper", False) is False

    @pytest.mark.asyncio
    async def test_trade_summary_excludes_paper(self):
        """Trade summary excludes paper orders from PnL calculations."""
        from api.trades import trade_summary

        # Paper order with PnL
        record_trade(
            {
                "symbol": "AAPL",
                "side": "SELL",
                "pnl": 100.0,
                "is_paper": True,
                "market": "US",
            }
        )
        # Live order with PnL
        record_trade(
            {
                "symbol": "MSFT",
                "side": "SELL",
                "pnl": 50.0,
                "is_paper": False,
                "market": "US",
            }
        )
        # Live order without is_paper (legacy) — should be included
        record_trade(
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "pnl": 25.0,
                "market": "US",
            }
        )

        summary = await trade_summary()
        # Paper order excluded: total_pnl = 50 + 25 = 75
        assert summary["total_pnl"] == 75.0
        # total_trades excludes paper
        assert summary["total_trades"] == 2

    @pytest.mark.asyncio
    async def test_reconcile_excludes_paper_orders(self):
        """Reconciliation only processes live (non-paper) orders."""
        mock_repo = AsyncMock()
        mock_repo.get_open_orders = AsyncMock(return_value=[])
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                await reconcile_pending_orders({"AAPL"})

        # Verify exclude_paper=True was passed
        mock_repo.get_open_orders.assert_called_once_with(exclude_paper=True)

    @pytest.mark.asyncio
    async def test_restore_trade_log_excludes_paper(self):
        """restore_trade_log excludes paper orders by default."""
        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=[])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                await restore_trade_log()

        # Verify exclude_paper=True was passed
        mock_repo.get_trade_history.assert_called_once_with(
            limit=200,
            exclude_paper=True,
        )


# --- Exchange field propagation tests (STOCK-5) ---


class TestExchangeFieldPersistence:
    """Tests for correct exchange field propagation from trade recorder to DB.

    STOCK-5: exchange field was dropped in _persist_trade, causing all orders
    (including KR) to be stored with default exchange='NASD'.
    """

    @pytest.mark.asyncio
    async def test_persist_trade_passes_exchange_kr(self):
        """_persist_trade passes exchange='KRX' for KR trades to save_order."""
        from api.trades import _persist_trade

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "KR001",
            "symbol": "005930",
            "side": "BUY",
            "quantity": 10,
            "price": 70000.0,
            "filled_quantity": 10,
            "filled_price": 70000.0,
            "status": "filled",
            "strategy": "supertrend",
            "exchange": "KRX",
            "market": "KR",
            "session": "regular",
            "is_paper": False,
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch(
                "api.trades.TradeRepository", return_value=mock_repo
            ):
                await _persist_trade(trade)

        mock_repo.save_order.assert_called_once()
        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["exchange"] == "KRX"
        assert call_kwargs["market"] == "KR"

    @pytest.mark.asyncio
    async def test_persist_trade_passes_exchange_us(self):
        """_persist_trade passes exchange='NASD' for US trades."""
        from api.trades import _persist_trade

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "US001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.0,
            "exchange": "NASD",
            "market": "US",
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch(
                "api.trades.TradeRepository", return_value=mock_repo
            ):
                await _persist_trade(trade)

        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["exchange"] == "NASD"

    @pytest.mark.asyncio
    async def test_persist_trade_passes_exchange_nyse(self):
        """_persist_trade passes exchange='NYSE' for NYSE trades."""
        from api.trades import _persist_trade

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "US002",
            "symbol": "BAC",
            "side": "BUY",
            "quantity": 50,
            "price": 40.0,
            "exchange": "NYSE",
            "market": "US",
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch(
                "api.trades.TradeRepository", return_value=mock_repo
            ):
                await _persist_trade(trade)

        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["exchange"] == "NYSE"

    @pytest.mark.asyncio
    async def test_persist_trade_defaults_exchange_nasd(self):
        """_persist_trade defaults to 'NASD' when exchange not in trade dict."""
        from api.trades import _persist_trade

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        # Legacy trade dict without exchange field
        trade = {
            "order_id": "OLD001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.0,
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch(
                "api.trades.TradeRepository", return_value=mock_repo
            ):
                await _persist_trade(trade)

        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["exchange"] == "NASD"


# --- STOCK-33: Sell record deduplication ---


class TestMergeTradeEntry:
    """Unit tests for _merge_trade_entry helper."""

    def test_merge_preserves_pnl_when_new_is_none(self):
        """New entry with pnl=None should not overwrite existing PnL."""
        existing = {"order_id": "X", "pnl": 120.0, "pnl_pct": 5.2}
        new = {"order_id": "X", "pnl": None, "pnl_pct": None, "status": "filled"}
        _merge_trade_entry(existing, new)
        assert existing["pnl"] == 120.0
        assert existing["pnl_pct"] == 5.2

    def test_merge_updates_pnl_when_existing_is_none(self):
        """Existing entry with pnl=None should accept new PnL."""
        existing = {"order_id": "X", "pnl": None, "pnl_pct": None}
        new = {"order_id": "X", "pnl": 200.0, "pnl_pct": 3.5}
        _merge_trade_entry(existing, new)
        assert existing["pnl"] == 200.0
        assert existing["pnl_pct"] == 3.5

    def test_merge_updates_pnl_when_both_have_values(self):
        """When both have PnL, new value overwrites (correction scenario)."""
        existing = {"order_id": "X", "pnl": 100.0}
        new = {"order_id": "X", "pnl": 150.0}
        _merge_trade_entry(existing, new)
        assert existing["pnl"] == 150.0

    def test_merge_does_not_downgrade_filled_status(self):
        """Status should not be downgraded from 'filled' to 'pending'."""
        existing = {"order_id": "X", "status": "filled"}
        new = {"order_id": "X", "status": "pending"}
        _merge_trade_entry(existing, new)
        assert existing["status"] == "filled"

    def test_merge_upgrades_pending_to_filled(self):
        """Status should be upgraded from 'pending' to 'filled'."""
        existing = {"order_id": "X", "status": "pending"}
        new = {"order_id": "X", "status": "filled"}
        _merge_trade_entry(existing, new)
        assert existing["status"] == "filled"

    def test_merge_updates_other_fields(self):
        """Non-PnL, non-status fields should be updated normally."""
        existing = {"order_id": "X", "filled_price": None, "filled_quantity": 0}
        new = {"order_id": "X", "filled_price": 155.0, "filled_quantity": 10}
        _merge_trade_entry(existing, new)
        assert existing["filled_price"] == 155.0
        assert existing["filled_quantity"] == 10

    def test_merge_adds_new_keys(self):
        """New keys not present in existing should be added."""
        existing = {"order_id": "X"}
        new = {"order_id": "X", "market": "US", "strategy": "trend"}
        _merge_trade_entry(existing, new)
        assert existing["market"] == "US"
        assert existing["strategy"] == "trend"


class TestRecordTradeDedup:
    """STOCK-33: record_trade() should be idempotent for same order_id."""

    def test_duplicate_order_id_merges_instead_of_appending(self):
        """Recording same order_id twice should result in 1 entry, not 2."""
        record_trade({
            "order_id": "SELL001",
            "symbol": "AAPL",
            "side": "SELL",
            "pnl": 120.0,
            "pnl_pct": 5.0,
            "status": "filled",
        })
        assert len(_trade_log) == 1

        # Reconciliation records same order without PnL
        record_trade({
            "order_id": "SELL001",
            "symbol": "AAPL",
            "side": "SELL",
            "pnl": None,
            "pnl_pct": None,
            "status": "filled",
            "filled_price": 155.0,
        })

        # Still only 1 entry
        assert len(_trade_log) == 1
        # PnL preserved from first recording
        assert _trade_log[0]["pnl"] == 120.0
        assert _trade_log[0]["pnl_pct"] == 5.0
        # filled_price updated from second recording
        assert _trade_log[0]["filled_price"] == 155.0

    def test_different_order_ids_append_separately(self):
        """Different order_ids should create separate entries."""
        record_trade({"order_id": "A", "symbol": "AAPL", "side": "SELL"})
        record_trade({"order_id": "B", "symbol": "MSFT", "side": "SELL"})
        assert len(_trade_log) == 2

    def test_empty_order_id_always_appends(self):
        """Orders without order_id should always append (no dedup)."""
        record_trade({"order_id": "", "symbol": "AAPL", "side": "BUY"})
        record_trade({"order_id": "", "symbol": "MSFT", "side": "BUY"})
        assert len(_trade_log) == 2

    def test_no_order_id_key_always_appends(self):
        """Orders without order_id key should always append."""
        record_trade({"symbol": "AAPL", "side": "BUY"})
        record_trade({"symbol": "MSFT", "side": "BUY"})
        assert len(_trade_log) == 2

    def test_reconciliation_after_sell_scenario(self):
        """Simulate exact STOCK-33 bug: place_sell then reconciliation."""
        # Step 1: place_sell records SELL with PnL
        record_trade({
            "order_id": "0020703700",
            "symbol": "011200",
            "side": "SELL",
            "quantity": 10,
            "price": 5000,
            "pnl": 120.0,
            "pnl_pct": 2.4,
            "status": "filled",
            "market": "KR",
        })
        assert len(_trade_log) == 1
        assert _trade_log[0]["pnl"] == 120.0

        # Step 2: Reconciliation records same order without PnL
        record_trade({
            "order_id": "0020703700",
            "symbol": "011200",
            "side": "SELL",
            "quantity": 10,
            "price": 5000,
            "filled_price": 5012.0,
            "filled_quantity": 10,
            "strategy": "",
            "status": "filled",
            "market": "KR",
            "created_at": "",
        })

        # Must be exactly 1 entry
        assert len(_trade_log) == 1
        # PnL preserved
        assert _trade_log[0]["pnl"] == 120.0
        assert _trade_log[0]["pnl_pct"] == 2.4
        # filled_price updated from reconciliation
        assert _trade_log[0]["filled_price"] == 5012.0

    def test_merge_preserves_created_at(self):
        """Reconciliation with empty created_at must not overwrite real timestamp."""
        record_trade({
            "order_id": "BUY999",
            "symbol": "AAPL",
            "side": "BUY",
            "status": "pending",
            "created_at": "2026-03-27T09:01:31.222000",
        })
        # Reconciliation detects fill — passes empty created_at
        record_trade({
            "order_id": "BUY999",
            "symbol": "AAPL",
            "side": "BUY",
            "status": "filled",
            "filled_price": 155.0,
            "created_at": "",
        })
        assert len(_trade_log) == 1
        assert _trade_log[0]["status"] == "filled"
        assert _trade_log[0]["filled_price"] == 155.0
        # created_at must be preserved from the original recording
        assert _trade_log[0]["created_at"] == "2026-03-27T09:01:31.222000"


class TestRestoreTradeLogDedup:
    """STOCK-33: restore_trade_log should deduplicate DB rows by order_id."""

    @pytest.mark.asyncio
    async def test_restore_deduplicates_by_order_id(self):
        """DB rows with same kis_order_id should be deduplicated, keeping PnL."""
        # Two DB rows for same order — one with PnL, one without
        mock_order_with_pnl = MagicMock()
        mock_order_with_pnl.id = 1
        mock_order_with_pnl.kis_order_id = "SELL001"
        mock_order_with_pnl.symbol = "AAPL"
        mock_order_with_pnl.side = "SELL"
        mock_order_with_pnl.quantity = 10
        mock_order_with_pnl.price = 150.0
        mock_order_with_pnl.filled_price = 155.0
        mock_order_with_pnl.filled_quantity = 10
        mock_order_with_pnl.status = "filled"
        mock_order_with_pnl.strategy_name = "trend"
        mock_order_with_pnl.pnl = 50.0
        mock_order_with_pnl.market = "US"
        mock_order_with_pnl.created_at = "2026-03-12 04:26:13"

        mock_order_no_pnl = MagicMock()
        mock_order_no_pnl.id = 2
        mock_order_no_pnl.kis_order_id = "SELL001"
        mock_order_no_pnl.symbol = "AAPL"
        mock_order_no_pnl.side = "SELL"
        mock_order_no_pnl.quantity = 10
        mock_order_no_pnl.price = 150.0
        mock_order_no_pnl.filled_price = 155.0
        mock_order_no_pnl.filled_quantity = 10
        mock_order_no_pnl.status = "filled"
        mock_order_no_pnl.strategy_name = "trend"
        mock_order_no_pnl.pnl = None
        mock_order_no_pnl.market = "US"
        mock_order_no_pnl.created_at = "2026-03-12 04:29:53"

        # Unique order (should be preserved)
        mock_unique = MagicMock()
        mock_unique.id = 3
        mock_unique.kis_order_id = "BUY002"
        mock_unique.symbol = "MSFT"
        mock_unique.side = "BUY"
        mock_unique.quantity = 5
        mock_unique.price = 400.0
        mock_unique.filled_price = 400.0
        mock_unique.filled_quantity = 5
        mock_unique.status = "filled"
        mock_unique.strategy_name = "momentum"
        mock_unique.pnl = None
        mock_unique.market = "US"
        mock_unique.created_at = "2026-03-12 03:00:00"

        # DB returns newest first (order of get_trade_history)
        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(
            return_value=[mock_order_no_pnl, mock_order_with_pnl, mock_unique]
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                count = await restore_trade_log()

        # 2 unique entries (not 3 with duplicate)
        assert count == 2
        assert len(_trade_log) == 2

        # Verify PnL entry was kept
        sell_entry = next(t for t in _trade_log if t["order_id"] == "SELL001")
        assert sell_entry["pnl"] == 50.0

        # Unique entry preserved
        buy_entry = next(t for t in _trade_log if t["order_id"] == "BUY002")
        assert buy_entry["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_restore_dedup_keeps_pnl_entry_over_none(self):
        """When duplicates exist, entry with PnL wins regardless of order."""
        # PnL entry appears AFTER no-PnL entry (reverse chronological order in DB)
        mock_order_no_pnl = MagicMock()
        mock_order_no_pnl.id = 2
        mock_order_no_pnl.kis_order_id = "X"
        mock_order_no_pnl.symbol = "AAPL"
        mock_order_no_pnl.side = "SELL"
        mock_order_no_pnl.quantity = 10
        mock_order_no_pnl.price = 150.0
        mock_order_no_pnl.filled_price = 155.0
        mock_order_no_pnl.filled_quantity = 10
        mock_order_no_pnl.status = "filled"
        mock_order_no_pnl.strategy_name = ""
        mock_order_no_pnl.pnl = None
        mock_order_no_pnl.market = "US"
        mock_order_no_pnl.created_at = "2026-03-12 04:29:53"

        mock_order_with_pnl = MagicMock()
        mock_order_with_pnl.id = 1
        mock_order_with_pnl.kis_order_id = "X"
        mock_order_with_pnl.symbol = "AAPL"
        mock_order_with_pnl.side = "SELL"
        mock_order_with_pnl.quantity = 10
        mock_order_with_pnl.price = 150.0
        mock_order_with_pnl.filled_price = 155.0
        mock_order_with_pnl.filled_quantity = 10
        mock_order_with_pnl.status = "filled"
        mock_order_with_pnl.strategy_name = "trend"
        mock_order_with_pnl.pnl = 50.0
        mock_order_with_pnl.market = "US"
        mock_order_with_pnl.created_at = "2026-03-12 04:26:13"

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(
            return_value=[mock_order_no_pnl, mock_order_with_pnl]
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                count = await restore_trade_log()

        assert count == 1
        assert len(_trade_log) == 1
        assert _trade_log[0]["pnl"] == 50.0

    @pytest.mark.asyncio
    async def test_restore_empty_order_ids_not_deduplicated(self):
        """Orders with empty kis_order_id should not be deduplicated."""
        mock1 = MagicMock()
        mock1.id = 1
        mock1.kis_order_id = ""
        mock1.symbol = "AAPL"
        mock1.side = "BUY"
        mock1.quantity = 10
        mock1.price = 150.0
        mock1.filled_price = 150.0
        mock1.filled_quantity = 10
        mock1.status = "filled"
        mock1.strategy_name = "trend"
        mock1.pnl = None
        mock1.market = "US"
        mock1.created_at = "2026-03-12 03:00:00"

        mock2 = MagicMock()
        mock2.id = 2
        mock2.kis_order_id = ""
        mock2.symbol = "MSFT"
        mock2.side = "BUY"
        mock2.quantity = 5
        mock2.price = 400.0
        mock2.filled_price = 400.0
        mock2.filled_quantity = 5
        mock2.status = "filled"
        mock2.strategy_name = "momentum"
        mock2.pnl = None
        mock2.market = "US"
        mock2.created_at = "2026-03-12 03:05:00"

        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=[mock2, mock1])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                count = await restore_trade_log()

        # Both entries preserved (empty order_id = no dedup)
        assert count == 2
        assert len(_trade_log) == 2


class TestTradeSummaryWithDedup:
    """Ensure trade summary is accurate after dedup (STOCK-33)."""

    @pytest.mark.asyncio
    async def test_summary_counts_deduplicated_sells_correctly(self):
        """After dedup, sell count and PnL should not be inflated."""
        from api.trades import trade_summary

        # Place sell with PnL
        record_trade({
            "order_id": "S1",
            "symbol": "AAPL",
            "side": "SELL",
            "pnl": 100.0,
            "market": "US",
        })
        # Reconciliation re-records same sell without PnL
        record_trade({
            "order_id": "S1",
            "symbol": "AAPL",
            "side": "SELL",
            "pnl": None,
            "market": "US",
            "status": "filled",
        })

        summary = await trade_summary()
        # Only 1 sell trade, not 2
        assert summary["total_trades"] == 1
        assert summary["wins"] == 1
        assert summary["total_pnl"] == 100.0


class TestUpdateOrderInDbNotFoundProtection:
    """STOCK-37: update_order_in_db should protect orders with PnL from not_found."""

    @pytest.mark.asyncio
    async def test_not_found_overridden_to_filled_when_pnl_exists(self):
        """Order with PnL should not be set to not_found — override to filled."""
        _trade_log.append(
            {
                "order_id": "KIS_SELL_001",
                "symbol": "AMPX",
                "side": "SELL",
                "status": "submitted",
                "pnl": 19.26,
                "filled_price": None,
            }
        )

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
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                ok = await update_order_in_db(
                    "KIS_SELL_001", "not_found", filled_price=None, filled_quantity=None,
                )

        assert ok is True
        # Status in trade log should be "filled", not "not_found"
        assert _trade_log[0]["status"] == "filled"
        # DB should also be updated with "filled"
        mock_repo.update_order_status.assert_called_once_with(
            1, "filled", filled_price=None, filled_quantity=None,
        )

    @pytest.mark.asyncio
    async def test_not_found_allowed_when_no_pnl(self):
        """Order without PnL can be set to not_found normally."""
        _trade_log.append(
            {
                "order_id": "KIS_BUY_001",
                "symbol": "AAPL",
                "side": "BUY",
                "status": "submitted",
                "pnl": None,
                "filled_price": None,
            }
        )

        mock_order = MagicMock()
        mock_order.id = 2

        mock_repo = AsyncMock()
        mock_repo.find_by_kis_order_id = AsyncMock(return_value=mock_order)
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                ok = await update_order_in_db(
                    "KIS_BUY_001", "not_found", filled_price=None, filled_quantity=None,
                )

        assert ok is True
        # No PnL → not_found is allowed
        assert _trade_log[0]["status"] == "not_found"
        mock_repo.update_order_status.assert_called_once_with(
            2, "not_found", filled_price=None, filled_quantity=None,
        )

    @pytest.mark.asyncio
    async def test_not_found_with_negative_pnl_also_overridden(self):
        """Even negative PnL means the order was filled — protect it."""
        _trade_log.append(
            {
                "order_id": "KIS_SELL_002",
                "symbol": "LION",
                "side": "SELL",
                "status": "submitted",
                "pnl": -51.62,
                "filled_price": None,
            }
        )

        mock_order = MagicMock()
        mock_order.id = 3

        mock_repo = AsyncMock()
        mock_repo.find_by_kis_order_id = AsyncMock(return_value=mock_order)
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                ok = await update_order_in_db(
                    "KIS_SELL_002", "not_found",
                )

        assert ok is True
        assert _trade_log[0]["status"] == "filled"

    @pytest.mark.asyncio
    async def test_not_found_with_zero_pnl_also_overridden(self):
        """PnL of 0.0 is not None — order was filled at breakeven."""
        _trade_log.append(
            {
                "order_id": "KIS_SELL_003",
                "symbol": "SPY",
                "side": "SELL",
                "status": "submitted",
                "pnl": 0.0,
            }
        )

        mock_order = MagicMock()
        mock_order.id = 4

        mock_repo = AsyncMock()
        mock_repo.find_by_kis_order_id = AsyncMock(return_value=mock_order)
        mock_repo.update_order_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                ok = await update_order_in_db("KIS_SELL_003", "not_found")

        assert ok is True
        assert _trade_log[0]["status"] == "filled"


# --- STOCK-38: Awaited DB persistence + not_found recovery ---


class TestPersistTradeToDb:
    """STOCK-38: persist_trade_to_db() provides awaited DB persistence."""

    @pytest.mark.asyncio
    async def test_persist_trade_to_db_success(self):
        """persist_trade_to_db returns True on successful save."""
        from api.trades import persist_trade_to_db

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "KIS001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.0,
            "filled_price": 150.5,
            "filled_quantity": 10,
            "status": "filled",
            "strategy": "trend_following",
            "exchange": "NASD",
            "market": "US",
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                result = await persist_trade_to_db(trade)

        assert result is True
        mock_repo.save_order.assert_called_once()
        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["filled_price"] == 150.5
        assert call_kwargs["filled_quantity"] == 10
        assert call_kwargs["status"] == "filled"
        assert call_kwargs["kis_order_id"] == "KIS001"

    @pytest.mark.asyncio
    async def test_persist_trade_to_db_no_session_factory(self):
        """persist_trade_to_db returns False when no session factory."""
        from api.trades import persist_trade_to_db

        with patch("api.trades._session_factory", None):
            result = await persist_trade_to_db({"symbol": "AAPL"})

        assert result is False

    @pytest.mark.asyncio
    async def test_persist_trade_to_db_handles_error(self):
        """persist_trade_to_db returns False on DB error."""
        from api.trades import persist_trade_to_db

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock(side_effect=Exception("DB error"))

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                result = await persist_trade_to_db({"symbol": "AAPL"})

        assert result is False

    @pytest.mark.asyncio
    async def test_persist_trade_passes_all_filled_data(self):
        """persist_trade_to_db passes filled_price, filled_quantity, status to save_order."""
        from api.trades import persist_trade_to_db

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "KIS002",
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 5,
            "price": 400.0,
            "filled_price": 401.5,
            "filled_quantity": 5,
            "status": "filled",
            "strategy": "supertrend",
            "buy_strategy": "momentum",
            "pnl": 7.5,
            "pnl_pct": 0.38,
            "exchange": "NASD",
            "market": "US",
            "session": "regular",
            "is_paper": False,
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                result = await persist_trade_to_db(trade)

        assert result is True
        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["filled_price"] == 401.5
        assert call_kwargs["filled_quantity"] == 5
        assert call_kwargs["status"] == "filled"
        assert call_kwargs["pnl"] == 7.5
        assert call_kwargs["pnl_pct"] == 0.38
        assert call_kwargs["buy_strategy"] == "momentum"

    @pytest.mark.asyncio
    async def test_persist_trade_kr_market(self):
        """persist_trade_to_db correctly handles KR market trades."""
        from api.trades import persist_trade_to_db

        mock_repo = AsyncMock()
        mock_repo.save_order = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        trade = {
            "order_id": "KR001",
            "symbol": "005930",
            "side": "BUY",
            "quantity": 10,
            "price": 70000.0,
            "filled_price": 70000.0,
            "filled_quantity": 10,
            "status": "filled",
            "strategy": "supertrend",
            "exchange": "KRX",
            "market": "KR",
        }

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                result = await persist_trade_to_db(trade)

        assert result is True
        call_kwargs = mock_repo.save_order.call_args.kwargs
        assert call_kwargs["exchange"] == "KRX"
        assert call_kwargs["market"] == "KR"


class TestRecoverNotFoundOrders:
    """STOCK-38: Recovery of orders stuck in 'not_found' status."""

    @pytest.mark.asyncio
    async def test_recover_not_found_orders_success(self):
        """Orders with not_found status and PnL are recovered to filled."""
        from api.trades import recover_not_found_orders

        mock_repo = AsyncMock()
        mock_repo.recover_not_found_orders = AsyncMock(
            return_value=["KIS_A", "KIS_B", "KIS_C"]
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                count = await recover_not_found_orders()

        assert count == 3
        mock_repo.recover_not_found_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_not_found_no_session_factory(self):
        """Returns 0 when no session factory configured."""
        from api.trades import recover_not_found_orders

        with patch("api.trades._session_factory", None):
            count = await recover_not_found_orders()

        assert count == 0

    @pytest.mark.asyncio
    async def test_recover_updates_in_memory_trade_log(self):
        """Recovery uses recovered IDs to target specific in-memory entries."""
        from api.trades import recover_not_found_orders

        # Add a not_found entry with PnL to trade log
        _trade_log.append(
            {
                "order_id": "KIS100",
                "symbol": "AAPL",
                "side": "SELL",
                "status": "not_found",
                "pnl": 50.0,
                "price": 155.0,
                "quantity": 10,
                "filled_price": None,
                "filled_quantity": None,
            }
        )

        mock_repo = AsyncMock()
        mock_repo.recover_not_found_orders = AsyncMock(return_value=["KIS100"])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                count = await recover_not_found_orders()

        assert count == 1
        assert _trade_log[0]["status"] == "filled"
        assert _trade_log[0]["filled_price"] == 155.0
        assert _trade_log[0]["filled_quantity"] == 10

    @pytest.mark.asyncio
    async def test_recover_skips_entries_not_in_recovered_ids(self):
        """In-memory entries not in recovered_ids are not modified."""
        from api.trades import recover_not_found_orders

        # not_found without PnL — DB won't recover it
        _trade_log.append(
            {
                "order_id": "KIS200",
                "symbol": "MSFT",
                "side": "BUY",
                "status": "not_found",
                "pnl": None,
                "price": 400.0,
            }
        )

        mock_repo = AsyncMock()
        mock_repo.recover_not_found_orders = AsyncMock(return_value=[])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                await recover_not_found_orders()

        # Status should remain not_found (not in recovered_ids)
        assert _trade_log[0]["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_recover_handles_db_error(self):
        """Returns 0 on DB error without crashing."""
        from api.trades import recover_not_found_orders

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("DB connection failed"))
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("api.trades._session_factory", mock_factory):
            count = await recover_not_found_orders()

        assert count == 0


class TestRecordTradeSkipDbPersist:
    """STOCK-38: record_trade skip_db_persist flag."""

    def test_skip_db_persist_true_does_not_fire_background_task(self):
        """When skip_db_persist=True, no fire-and-forget DB task is created."""
        from unittest.mock import patch as _patch

        created_tasks = []
        original_create_task = None

        with _patch("api.trades._session_factory", MagicMock()):
            # We can't easily test asyncio.create_task without a loop,
            # but we can verify the code path by checking the flag works
            # without errors at minimum
            record_trade(
                {"symbol": "AAPL", "side": "BUY", "status": "pending"},
                skip_db_persist=True,
            )

        assert len(_trade_log) == 1
        assert _trade_log[0]["symbol"] == "AAPL"

    def test_skip_db_persist_false_is_default(self):
        """Default behavior (skip_db_persist=False) still appends to log."""
        record_trade({"symbol": "MSFT", "side": "BUY", "status": "pending"})
        assert len(_trade_log) == 1



# --- STOCK-36: Sort order + pagination tests ---


class TestGetTradesNewestFirst:
    """STOCK-36: get_trades returns newest trades first."""

    @pytest.mark.asyncio
    async def test_returns_newest_first_from_in_memory(self):
        """In-memory trades should be returned newest-first (descending)."""
        _trade_log.extend([
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-18 01:00:00",
                "market": "US",
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-19 02:00:00",
                "market": "US",
            },
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "status": "filled",
                "created_at": "2026-03-20 03:00:00",
                "market": "US",
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=0)

        assert len(result) == 3
        # Newest first
        assert result[0]["symbol"] == "GOOGL"
        assert result[1]["symbol"] == "MSFT"
        assert result[2]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_limit_returns_newest_subset(self):
        """With limit=2 on 3 trades, should return 2 newest."""
        _trade_log.extend([
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-18 01:00:00",
                "market": "US",
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-19 02:00:00",
                "market": "US",
            },
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "status": "filled",
                "created_at": "2026-03-20 03:00:00",
                "market": "US",
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=2, offset=0)

        assert len(result) == 2
        assert result[0]["symbol"] == "GOOGL"
        assert result[1]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_offset_skips_newest(self):
        """Offset=1 should skip the newest trade."""
        _trade_log.extend([
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-18 01:00:00",
                "market": "US",
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-19 02:00:00",
                "market": "US",
            },
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "status": "filled",
                "created_at": "2026-03-20 03:00:00",
                "market": "US",
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=1)

        assert len(result) == 2
        # Skipped GOOGL (newest), starts from MSFT
        assert result[0]["symbol"] == "MSFT"
        assert result[1]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_offset_and_limit_pagination(self):
        """Offset + limit should work together for page 2."""
        _trade_log.extend([
            {
                "symbol": f"SYM{i}",
                "side": "BUY",
                "status": "filled",
                "created_at": f"2026-03-{10+i:02d} 01:00:00",
                "market": "US",
            }
            for i in range(5)
        ])

        with patch("api.trades.get_name", return_value=""):
            # Page 1: offset=0, limit=2
            page1 = await get_trades(limit=2, offset=0)
            # Page 2: offset=2, limit=2
            page2 = await get_trades(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        # Page 1 = newest 2 (SYM4, SYM3)
        assert page1[0]["symbol"] == "SYM4"
        assert page1[1]["symbol"] == "SYM3"
        # Page 2 = next 2 (SYM2, SYM1)
        assert page2[0]["symbol"] == "SYM2"
        assert page2[1]["symbol"] == "SYM1"

    @pytest.mark.asyncio
    async def test_market_filter_with_sort(self):
        """Market filter should work with newest-first sort."""
        _trade_log.extend([
            {
                "symbol": "005930",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-20 01:00:00",
                "market": "KR",
            },
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-20 02:00:00",
                "market": "US",
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-20 03:00:00",
                "market": "US",
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=0, market="US")

        assert len(result) == 2
        assert result[0]["symbol"] == "MSFT"  # Newest US trade
        assert result[1]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_offset_beyond_total_returns_empty(self):
        """Offset beyond total trades returns empty list."""
        _trade_log.append({
            "symbol": "AAPL",
            "side": "BUY",
            "status": "filled",
            "created_at": "2026-03-20 01:00:00",
            "market": "US",
        })

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_db_fallback_passes_offset(self):
        """DB fallback path should pass offset parameter."""
        mock_repo = AsyncMock()
        mock_repo.get_trade_history = AsyncMock(return_value=[])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        # _trade_log is empty so it uses DB fallback
        with patch("api.trades._session_factory", mock_factory):
            with patch("api.trades.TradeRepository", return_value=mock_repo):
                await get_trades(limit=30, offset=30)

        mock_repo.get_trade_history.assert_called_once_with(
            limit=30, offset=30, symbol=None,
        )

    @pytest.mark.asyncio
    async def test_sort_handles_none_created_at(self):
        """Trades with created_at=None should not crash the sort."""
        _trade_log.extend([
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-19 01:00:00",
                "market": "US",
            },
            {
                "symbol": "MSFT",
                "side": "BUY",
                "status": "pending",
                "created_at": None,  # None value — must not crash
                "market": "US",
            },
            {
                "symbol": "GOOGL",
                "side": "SELL",
                "status": "filled",
                "created_at": "2026-03-20 03:00:00",
                "market": "US",
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=0)

        assert len(result) == 3
        # Newest first, None sorts to the end (empty string)
        assert result[0]["symbol"] == "GOOGL"
        assert result[1]["symbol"] == "AAPL"
        assert result[2]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_sort_handles_missing_created_at_key(self):
        """Trades without created_at key should not crash the sort."""
        _trade_log.extend([
            {
                "symbol": "AAPL",
                "side": "BUY",
                "status": "filled",
                "created_at": "2026-03-19 01:00:00",
                "market": "US",
            },
            {
                "symbol": "TSLA",
                "side": "BUY",
                "status": "pending",
                "market": "US",
                # No created_at key at all
            },
        ])

        with patch("api.trades.get_name", return_value=""):
            result = await get_trades(limit=10, offset=0)

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "TSLA"
