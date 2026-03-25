"""Tests for Order Manager."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from engine.order_manager import OrderManager, ManagedOrder
from engine.risk_manager import RiskManager, RiskParams
from exchange.base import OrderResult, Position


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.create_buy_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            order_type="limit",
            quantity=10,
            price=150.0,
            filled_quantity=10,
            filled_price=150.0,
            status="filled",
        )
    )
    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD002",
            symbol="AAPL",
            side="SELL",
            order_type="limit",
            quantity=10,
            price=160.0,
            filled_quantity=10,
            filled_price=160.0,
            status="filled",
        )
    )
    adapter.cancel_order = AsyncMock(return_value=True)
    adapter.fetch_order = AsyncMock(
        return_value=OrderResult(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            order_type="limit",
            quantity=10,
            price=150.0,
            status="filled",
            filled_price=150.0,
            filled_quantity=10,
        )
    )
    return adapter


@pytest.fixture
def risk_manager():
    return RiskManager(RiskParams(max_position_pct=0.10, max_positions=20))


@pytest.fixture
def order_manager(mock_adapter, risk_manager):
    return OrderManager(adapter=mock_adapter, risk_manager=risk_manager)


class TestOrderManager:
    async def test_place_buy_success(self, order_manager, mock_adapter):
        order = await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="trend_following",
        )
        assert order is not None
        assert order.order_id == "ORD001"
        assert order.side == "BUY"
        assert order.strategy_name == "trend_following"
        mock_adapter.create_buy_order.assert_called_once()

    async def test_place_buy_rejected_by_risk(self, mock_adapter):
        rm = RiskManager(RiskParams(max_positions=0))
        om = OrderManager(adapter=mock_adapter, risk_manager=rm)
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_place_sell_success(self, order_manager, mock_adapter):
        order = await order_manager.place_sell(
            symbol="AAPL",
            quantity=10,
            price=160.0,
            strategy_name="trend_following",
        )
        assert order is not None
        assert order.order_id == "ORD002"
        assert order.side == "SELL"

    async def test_cancel_order(self, order_manager, mock_adapter):
        # Place then cancel
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        success = await order_manager.cancel("ORD001", "AAPL")
        assert success is True

    async def test_sync_order_status(self, order_manager):
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        managed = await order_manager.sync_order_status("ORD001", "AAPL")
        assert managed is not None
        assert managed.status == "filled"

    async def test_active_orders_tracked(self, order_manager):
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert "ORD001" in order_manager.active_orders

    async def test_clear_completed(self, order_manager):
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        order_manager.clear_completed()
        assert len(order_manager.active_orders) == 0  # Status was "filled"

    async def test_place_buy_adapter_error(self, order_manager, mock_adapter):
        mock_adapter.create_buy_order.side_effect = Exception("Network error")
        order = await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is None

    async def test_place_sell_adapter_error(self, order_manager, mock_adapter):
        mock_adapter.create_sell_order.side_effect = Exception("Network error")
        order = await order_manager.place_sell(
            symbol="AAPL",
            quantity=10,
            price=160.0,
        )
        assert order is None


class TestDuplicateOrderPrevention:
    """Tests for signal deduplication."""

    async def test_has_pending_order_false_when_empty(self, order_manager):
        assert order_manager.has_pending_order("AAPL") is False

    async def test_has_pending_order_true_for_pending(self, mock_adapter, risk_manager):
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert om.has_pending_order("AAPL") is True
        assert om.has_pending_order("AAPL", "BUY") is True
        assert om.has_pending_order("AAPL", "SELL") is False

    async def test_duplicate_buy_blocked(self, mock_adapter, risk_manager):
        """Second buy for same symbol is blocked when first is pending."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)

        # First buy succeeds
        first = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert first is not None

        # Second buy for same symbol is blocked
        second = await om.place_buy(
            symbol="AAPL",
            price=155.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            strategy_name="test2",
        )
        assert second is None
        assert mock_adapter.create_buy_order.call_count == 1

    async def test_different_symbol_not_blocked(self, mock_adapter, risk_manager):
        """Buy for different symbol is allowed."""
        call_count = 0

        async def create_buy(**kwargs):
            nonlocal call_count
            call_count += 1
            return OrderResult(
                order_id=f"ORD{call_count:03d}",
                symbol=kwargs["symbol"],
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                status="pending",
            )

        mock_adapter.create_buy_order = create_buy
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)

        first = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        second = await om.place_buy(
            symbol="NVDA",
            price=800.0,
            portfolio_value=100_000,
            cash_available=40_000,
            current_positions=1,
            strategy_name="test",
        )
        assert first is not None
        assert second is not None

    async def test_filled_order_allows_new_buy(self, order_manager):
        """After order fills, new buy for same symbol is allowed."""
        # First buy fills immediately (fixture default)
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        # Status is "filled", so has_pending_order should return False
        assert order_manager.has_pending_order("AAPL") is False


class TestSlippageTracking:
    """Tests for slippage measurement."""

    async def test_buy_slippage_positive(self, mock_adapter, risk_manager):
        """Track positive slippage (filled higher than intended)."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=10,
                filled_price=150.05,
                status="filled",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        assert abs(order.slippage - 0.05) < 0.001

    async def test_sell_slippage_negative(self, mock_adapter, risk_manager):
        """Track negative slippage (filled lower than intended)."""
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD002",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                filled_quantity=10,
                filled_price=159.95,
                status="filled",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_sell(
            symbol="AAPL",
            quantity=10,
            price=160.0,
            strategy_name="test",
        )
        assert order is not None
        assert abs(order.slippage - (-0.05)) < 0.001

    async def test_zero_slippage_on_exact_fill(self, order_manager):
        """No slippage when filled at intended price."""
        order = await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        assert order.slippage == 0.0


class TestPartialFill:
    """Tests for partial fill tracking."""

    async def test_partial_fill_tracked(self, mock_adapter, risk_manager):
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=100,
                price=150.0,
                filled_quantity=60,
                filled_price=150.0,
                status="partial",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        assert order.filled_quantity == 60
        assert order.quantity == 46  # risk-sized quantity (7% regime cap at uptrend)

    async def test_zero_fill_tracked(self, mock_adapter, risk_manager):
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        assert order.filled_quantity == 0


class TestReconciliation:
    """Tests for order reconciliation with exchange."""

    async def test_reconcile_detects_status_change(self, mock_adapter, risk_manager):
        """Reconciliation detects when order status changes."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        mock_adapter.fetch_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=10,
                filled_price=150.0,
                status="filled",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )

        changes = await om.reconcile_all()
        assert len(changes) == 1
        assert changes[0]["old_status"] == "pending"
        assert changes[0]["new_status"] == "filled"

    async def test_reconcile_no_changes(self, order_manager):
        """No changes when order status hasn't changed."""
        # Default fixture creates orders with "filled" status
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        changes = await order_manager.reconcile_all()
        # Order was already "filled", so no status change detected
        assert len(changes) == 0

    async def test_reconcile_empty(self, order_manager):
        """No-op when no active orders."""
        changes = await order_manager.reconcile_all()
        assert changes == []

    async def test_reconcile_clears_completed(self, mock_adapter, risk_manager):
        """Reconcile clears filled orders from tracking."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        mock_adapter.fetch_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=10,
                filled_price=150.0,
                status="filled",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert len(om.active_orders) == 1
        await om.reconcile_all()
        assert len(om.active_orders) == 0  # Cleared after fill


class TestStaleOrderCancel:
    """Tests for stale (unfilled) order auto-cancellation."""

    async def test_cancel_stale_order(self, mock_adapter, risk_manager):
        """Orders older than TTL are cancelled."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="005930",
                side="BUY",
                order_type="limit",
                quantity=5,
                price=70000.0,
                filled_quantity=0,
                filled_price=None,
                status="open",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="test",
        )
        # Backdate created_at to 20 minutes ago
        om._active_orders["ORD001"].created_at = (
            datetime.now() - timedelta(minutes=20)
        ).isoformat()

        cancelled = await om.cancel_stale_orders(ttl_minutes=15)
        assert len(cancelled) == 1
        assert cancelled[0]["symbol"] == "005930"
        assert cancelled[0]["side"] == "BUY"
        mock_adapter.cancel_order.assert_called_once_with("ORD001", "005930")

    async def test_fresh_order_not_cancelled(self, mock_adapter, risk_manager):
        """Orders within TTL are NOT cancelled."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="005930",
                side="BUY",
                order_type="limit",
                quantity=5,
                price=70000.0,
                filled_quantity=0,
                filled_price=None,
                status="open",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="test",
        )
        # created_at is "now" — well within TTL
        cancelled = await om.cancel_stale_orders(ttl_minutes=15)
        assert len(cancelled) == 0
        mock_adapter.cancel_order.assert_not_called()

    async def test_filled_order_not_cancelled(self, order_manager, mock_adapter):
        """Filled orders are never cancelled even if old."""
        await order_manager.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        # Backdate but status is "filled"
        order_manager._active_orders["ORD001"].created_at = (
            datetime.now() - timedelta(minutes=60)
        ).isoformat()

        cancelled = await order_manager.cancel_stale_orders(ttl_minutes=15)
        assert len(cancelled) == 0

    async def test_cancel_stale_no_active_orders(self, order_manager):
        """No-op when no active orders."""
        cancelled = await order_manager.cancel_stale_orders(ttl_minutes=15)
        assert cancelled == []

    async def test_cancel_stale_ttl_zero_disabled(self, mock_adapter, risk_manager):
        """TTL=0 disables stale order cancellation."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="open",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        om._active_orders["ORD001"].created_at = (datetime.now() - timedelta(hours=2)).isoformat()

        cancelled = await om.cancel_stale_orders(ttl_minutes=0)
        assert len(cancelled) == 0

    async def test_cancel_stale_frees_duplicate_lock(self, mock_adapter, risk_manager):
        """After stale cancel, same symbol can be bought again."""
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="005930",
                side="BUY",
                order_type="limit",
                quantity=5,
                price=70000.0,
                filled_quantity=0,
                filled_price=None,
                status="open",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_buy(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="test",
        )
        # Duplicate blocked while open
        assert om.has_pending_order("005930", "BUY") is True

        # Backdate and cancel
        om._active_orders["ORD001"].created_at = (
            datetime.now() - timedelta(minutes=20)
        ).isoformat()
        await om.cancel_stale_orders(ttl_minutes=15)

        # Now the duplicate lock is released
        assert om.has_pending_order("005930", "BUY") is False


# --- Paper/Live flag propagation (STOCK-6) ---


class TestPaperFlag:
    """Tests for is_paper flag in OrderManager."""

    async def test_default_is_paper_false(self, mock_adapter, risk_manager):
        """OrderManager defaults to is_paper=False (live mode)."""
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        assert om._is_paper is False

    async def test_is_paper_true(self, mock_adapter, risk_manager):
        """OrderManager accepts is_paper=True (paper mode)."""
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            is_paper=True,
        )
        assert om._is_paper is True

    async def test_buy_trade_record_includes_is_paper(self, mock_adapter, risk_manager):
        """Trade recorder dict includes is_paper flag on BUY."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                is_paper=True,
            )
            await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="test",
            )
            assert len(recorded) == 1
            assert recorded[0]["is_paper"] is True
        finally:
            set_trade_recorder(old_recorder)

    async def test_sell_trade_record_includes_is_paper(self, mock_adapter, risk_manager):
        """Trade recorder dict includes is_paper flag on SELL."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                is_paper=True,
            )
            await om.place_sell(
                symbol="AAPL",
                quantity=10,
                price=160.0,
                strategy_name="test",
            )
            assert len(recorded) == 1
            assert recorded[0]["is_paper"] is True
        finally:
            set_trade_recorder(old_recorder)

    async def test_live_mode_is_paper_false_in_record(self, mock_adapter, risk_manager):
        """Live mode OrderManager records is_paper=False."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                is_paper=False,
            )
            await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="test",
            )
            assert len(recorded) == 1
            assert recorded[0]["is_paper"] is False
        finally:
            set_trade_recorder(old_recorder)


class TestExchangePositionDuplicateBlock:
    """Test defense-in-depth: order_manager blocks buys for already-held symbols.

    STOCK-4: exchange position check in place_buy prevents duplicate buys
    even when in-memory dedup state is lost after restart.
    """

    @pytest.fixture
    def mock_market_data(self):
        md = AsyncMock()
        md.get_positions = AsyncMock(return_value=[])
        md.invalidate_cache = lambda: None
        return md

    async def test_buy_blocked_when_already_held(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """place_buy should refuse if exchange shows existing position."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_allowed_when_not_held(self, mock_adapter, risk_manager, mock_market_data):
        """place_buy should proceed when exchange has no matching position."""
        mock_market_data.get_positions.return_value = []
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_allowed_for_different_symbol(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """Holding TSLA should not block buying AAPL."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="TSLA", exchange="NASD", quantity=5, avg_price=200.0),
        ]
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            strategy_name="test",
        )
        assert order is not None

    async def test_buy_proceeds_without_market_data(self, mock_adapter, risk_manager):
        """Without market_data, position check is skipped (backward compatible)."""
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=None,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None

    async def test_buy_rejected_when_position_check_errors(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """STOCK-26: If get_positions throws, buy is REJECTED as safety precaution.

        Previously this silently swallowed errors (pass), allowing duplicate buys
        when the API was down. Now fail-safe: if we can't confirm we don't hold
        the symbol, refuse to buy.
        """
        mock_market_data.get_positions.side_effect = RuntimeError("API error")
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_blocked_kr_market(self, mock_adapter, risk_manager, mock_market_data):
        """KR market duplicate buy also blocked by exchange position check."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="263750", exchange="KRX", quantity=10, avg_price=61400.0),
        ]
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
            market="KR",
        )
        order = await om.place_buy(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="supertrend",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_zero_quantity_position_not_blocked(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """Position with quantity=0 should not block new buy."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=0, avg_price=140.0),
        ]
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None


# --- Exchange field propagation (STOCK-5) ---


class TestExchangeFieldPropagation:
    """Tests for correct exchange field in ManagedOrder and trade recorder.

    STOCK-5: KR orders were stored with exchange='NASD' because:
    1. trade_recorder dict didn't include exchange
    2. _persist_trade didn't pass exchange to save_order
    3. position_tracker._execute_sell didn't pass exchange to place_sell
    """

    async def test_buy_managed_order_has_exchange(self, mock_adapter, risk_manager):
        """ManagedOrder.exchange reflects the passed exchange value."""
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_buy(
            symbol="005930",
            price=70000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="test",
            exchange="KRX",
        )
        assert order is not None
        assert order.exchange == "KRX"

    async def test_sell_managed_order_has_exchange(self, mock_adapter, risk_manager):
        """ManagedOrder.exchange reflects the passed exchange value on sell."""
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_sell(
            symbol="005930",
            quantity=10,
            price=72000.0,
            strategy_name="test",
            exchange="KRX",
        )
        assert order is not None
        assert order.exchange == "KRX"

    async def test_buy_trade_record_includes_exchange_kr(self, mock_adapter, risk_manager):
        """Trade recorder dict includes exchange='KRX' for KR buy orders."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                market="KR",
            )
            await om.place_buy(
                symbol="005930",
                price=70000.0,
                portfolio_value=10_000_000,
                cash_available=5_000_000,
                current_positions=0,
                strategy_name="supertrend",
                exchange="KRX",
            )
            assert len(recorded) == 1
            assert recorded[0]["exchange"] == "KRX"
            assert recorded[0]["market"] == "KR"
        finally:
            set_trade_recorder(old_recorder)

    async def test_sell_trade_record_includes_exchange_kr(self, mock_adapter, risk_manager):
        """Trade recorder dict includes exchange='KRX' for KR sell orders."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                market="KR",
            )
            await om.place_sell(
                symbol="005930",
                quantity=10,
                price=72000.0,
                strategy_name="supertrend",
                exchange="KRX",
            )
            assert len(recorded) == 1
            assert recorded[0]["exchange"] == "KRX"
            assert recorded[0]["market"] == "KR"
        finally:
            set_trade_recorder(old_recorder)

    async def test_buy_trade_record_includes_exchange_us(self, mock_adapter, risk_manager):
        """Trade recorder dict includes exchange for US buy orders (NASD/NYSE/AMEX)."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                market="US",
            )
            await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="trend_following",
                exchange="NASD",
            )
            assert len(recorded) == 1
            assert recorded[0]["exchange"] == "NASD"

            # NYSE stock
            await om.place_buy(
                symbol="BAC",
                price=40.0,
                portfolio_value=100_000,
                cash_available=45_000,
                current_positions=1,
                strategy_name="trend_following",
                exchange="NYSE",
            )
            assert len(recorded) == 2
            assert recorded[1]["exchange"] == "NYSE"
        finally:
            set_trade_recorder(old_recorder)

    async def test_sell_trade_record_includes_exchange_us(self, mock_adapter, risk_manager):
        """Trade recorder dict includes exchange for US sell orders."""
        recorded = []
        from engine.order_manager import set_trade_recorder, _trade_recorder

        old_recorder = _trade_recorder
        set_trade_recorder(lambda t, **kw: recorded.append(t))

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                market="US",
            )
            await om.place_sell(
                symbol="AAPL",
                quantity=10,
                price=160.0,
                strategy_name="trend_following",
                exchange="NASD",
            )
            assert len(recorded) == 1
            assert recorded[0]["exchange"] == "NASD"
        finally:
            set_trade_recorder(old_recorder)

    async def test_default_exchange_is_nasd(self, mock_adapter, risk_manager):
        """Default exchange parameter is 'NASD' for backward compatibility."""
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None
        assert order.exchange == "NASD"


# --- STOCK-26: Duplicate buy prevention (fail-safe on position check error) ---


class TestStock26FailSafePositionCheck:
    """STOCK-26: Verify position check failure rejects buy (fail-safe).

    Previously `except Exception: pass` allowed buys when get_positions()
    failed, enabling 17 duplicate buys of 263750.
    """

    @pytest.fixture
    def mock_market_data(self):
        md = AsyncMock()
        md.get_positions = AsyncMock(return_value=[])
        md.invalidate_cache = lambda: None
        return md

    async def test_timeout_error_rejects_buy(self, mock_adapter, risk_manager, mock_market_data):
        """Timeout during position check rejects buy (fail-safe)."""
        import asyncio

        mock_market_data.get_positions.side_effect = asyncio.TimeoutError()
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="supertrend",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_connection_error_rejects_buy(self, mock_adapter, risk_manager, mock_market_data):
        """Connection error during position check rejects buy."""
        mock_market_data.get_positions.side_effect = ConnectionError("disconnected")
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is None
        mock_adapter.create_buy_order.assert_not_called()

    async def test_position_check_failure_counter(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """Position check failures increment the counter for monitoring."""
        mock_market_data.get_positions.side_effect = ConnectionError("disconnected")
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )
        assert om.position_check_failures == 0

        await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert om.position_check_failures == 1

        await om.place_buy(
            symbol="TSLA",
            price=200.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert om.position_check_failures == 2

    async def test_no_market_data_still_allows_buy(self, mock_adapter, risk_manager):
        """Without market_data service, position check skipped (backward compat)."""
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=None,
        )
        order = await om.place_buy(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            strategy_name="test",
        )
        assert order is not None

    async def test_consecutive_buys_same_symbol_blocked(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """STOCK-26 scenario: After first buy fills, second buy for same symbol
        is blocked by exchange position check (position now shows in exchange).
        """
        call_count = 0

        async def create_buy(**kwargs):
            nonlocal call_count
            call_count += 1
            return OrderResult(
                order_id=f"ORD{call_count:03d}",
                symbol=kwargs["symbol"],
                side="BUY",
                order_type="limit",
                quantity=kwargs["quantity"],
                price=kwargs.get("price", 64600.0),
                filled_quantity=kwargs["quantity"],
                filled_price=kwargs.get("price", 64600.0),
                status="filled",
            )

        mock_adapter.create_buy_order = create_buy

        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
        )

        # First buy: no position yet
        mock_market_data.get_positions.return_value = []
        first = await om.place_buy(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=0,
            strategy_name="supertrend",
        )
        assert first is not None

        # Second buy: exchange now shows position (first buy filled)
        mock_market_data.get_positions.return_value = [
            Position(symbol="263750", exchange="KRX", quantity=10, avg_price=64600.0),
        ]
        second = await om.place_buy(
            symbol="263750",
            price=64800.0,
            portfolio_value=10_000_000,
            cash_available=4_500_000,
            current_positions=1,
            strategy_name="supertrend",
        )
        assert second is None  # Blocked by exchange position check
        assert call_count == 1  # Only one order placed

    async def test_263750_scenario_17_buys_prevented(
        self, mock_adapter, risk_manager, mock_market_data
    ):
        """STOCK-26 regression test: Simulates the 263750 scenario where
        17 BUY orders were placed. After the fix, only 1 should succeed.
        """
        buy_count = 0

        async def create_buy(**kwargs):
            nonlocal buy_count
            buy_count += 1
            return OrderResult(
                order_id=f"ORD{buy_count:03d}",
                symbol="263750",
                side="BUY",
                order_type="limit",
                quantity=7,
                price=64600.0,
                filled_quantity=7,
                filled_price=64600.0,
                status="filled",
            )

        mock_adapter.create_buy_order = create_buy
        om = OrderManager(
            adapter=mock_adapter,
            risk_manager=risk_manager,
            market_data=mock_market_data,
            market="KR",
        )

        # Simulate 17 buy attempts — only first should succeed
        for i in range(17):
            # After first buy, exchange reports existing position
            if i >= 1:
                mock_market_data.get_positions.return_value = [
                    Position(
                        symbol="263750",
                        exchange="KRX",
                        quantity=7 * i,
                        avg_price=64600.0,
                    ),
                ]
            else:
                mock_market_data.get_positions.return_value = []

            await om.place_buy(
                symbol="263750",
                price=64600.0,
                portfolio_value=10_000_000,
                cash_available=5_000_000,
                current_positions=i,
                strategy_name="supertrend",
            )

        # Only 1 buy order should have been placed, not 17
        assert buy_count == 1


class TestReconcileNotFoundPreservesData:
    """STOCK-37: reconcile_all should preserve filled data when fetch_order returns not_found."""

    @pytest.mark.asyncio
    async def test_not_found_preserves_existing_filled_price(self, risk_manager):
        """When fetch_order returns not_found, existing filled_price should be preserved."""
        mock_adapter = AsyncMock()
        # Place order with filled data
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="SELL_001",
                symbol="AMPX",
                side="SELL",
                order_type="market",
                quantity=10,
                price=25.0,
                filled_quantity=10,
                filled_price=24.50,
                status="submitted",
            )
        )
        # Reconciliation returns not_found
        mock_adapter.fetch_order = AsyncMock(
            return_value=OrderResult(
                order_id="SELL_001",
                symbol="AMPX",
                side="unknown",
                order_type="unknown",
                quantity=0,
                status="not_found",
            )
        )

        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_sell(
            symbol="AMPX",
            quantity=10,
            price=25.0,
            strategy_name="test",
        )

        # Verify initial state
        managed = om.active_orders["SELL_001"]
        assert managed.filled_price == 24.50
        assert managed.filled_quantity == 10

        changes = await om.reconcile_all()

        # Status changes to not_found (we can't prevent that at this level)
        assert len(changes) == 1
        assert changes[0]["new_status"] == "not_found"
        # But filled_price/filled_quantity should be preserved in the change
        assert changes[0]["filled_price"] == 24.50
        assert changes[0]["filled_quantity"] == 10

    @pytest.mark.asyncio
    async def test_not_found_without_prior_filled_data(self, risk_manager):
        """When no prior filled data exists, not_found behaves normally."""
        mock_adapter = AsyncMock()
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="SELL_002",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=150.0,
                filled_quantity=0,
                filled_price=None,
                status="submitted",
            )
        )
        mock_adapter.fetch_order = AsyncMock(
            return_value=OrderResult(
                order_id="SELL_002",
                symbol="AAPL",
                side="unknown",
                order_type="unknown",
                quantity=0,
                status="not_found",
            )
        )

        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
        await om.place_sell(
            symbol="AAPL",
            quantity=10,
            price=150.0,
            strategy_name="test",
        )

        changes = await om.reconcile_all()
        assert len(changes) == 1
        assert changes[0]["new_status"] == "not_found"
        # No prior filled data → filled_price stays None
        assert changes[0]["filled_price"] is None
        assert changes[0]["filled_quantity"] == 0


# --- STOCK-38: DB recorder callback for immediate DB persistence ---


class TestDbRecorderCallback:
    """STOCK-38: Verify _db_recorder is called with filled data on order placement."""

    @pytest.mark.asyncio
    async def test_buy_calls_db_recorder_with_filled_data(self, mock_adapter, risk_manager):
        """place_buy awaits _db_recorder with filled_price/status."""
        db_records = []
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        try:
            om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
            order = await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="trend_following",
            )

            assert order is not None
            assert len(db_records) == 1
            assert db_records[0]["symbol"] == "AAPL"
            assert db_records[0]["side"] == "BUY"
            assert db_records[0]["filled_price"] == 150.0
            assert db_records[0]["filled_quantity"] == 10
            assert db_records[0]["status"] == "filled"
            assert db_records[0]["order_id"] == "ORD001"
            assert db_records[0]["strategy"] == "trend_following"
        finally:
            set_db_recorder(old_recorder)

    @pytest.mark.asyncio
    async def test_sell_calls_db_recorder_with_filled_data(self, mock_adapter, risk_manager):
        """place_sell awaits _db_recorder with filled_price/status/pnl."""
        db_records = []
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        try:
            om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
            order = await om.place_sell(
                symbol="AAPL",
                quantity=10,
                price=160.0,
                strategy_name="stop_loss",
                entry_price=150.0,
            )

            assert order is not None
            assert len(db_records) == 1
            assert db_records[0]["symbol"] == "AAPL"
            assert db_records[0]["side"] == "SELL"
            assert db_records[0]["filled_price"] == 160.0
            assert db_records[0]["filled_quantity"] == 10
            assert db_records[0]["status"] == "filled"
            assert db_records[0]["order_id"] == "ORD002"
            assert db_records[0]["pnl"] is not None
            assert db_records[0]["pnl"] == 100.0  # (160 - 150) * 10
        finally:
            set_db_recorder(old_recorder)

    @pytest.mark.asyncio
    async def test_db_recorder_not_called_when_none(self, mock_adapter, risk_manager):
        """When _db_recorder is None, place_buy/sell still work without error."""
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder
        set_db_recorder(None)

        try:
            om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
            order = await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="test",
            )
            assert order is not None
        finally:
            set_db_recorder(old_recorder)

    @pytest.mark.asyncio
    async def test_db_recorder_called_for_kr_market(self, mock_adapter, risk_manager):
        """place_buy for KR market passes correct exchange/market to _db_recorder."""
        db_records = []
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        try:
            om = OrderManager(
                adapter=mock_adapter,
                risk_manager=risk_manager,
                market="KR",
            )
            order = await om.place_buy(
                symbol="005930",
                price=70000.0,
                portfolio_value=10_000_000,
                cash_available=5_000_000,
                current_positions=0,
                strategy_name="supertrend",
                exchange="KRX",
            )

            assert order is not None
            assert len(db_records) == 1
            assert db_records[0]["market"] == "KR"
            assert db_records[0]["exchange"] == "KRX"
        finally:
            set_db_recorder(old_recorder)

    @pytest.mark.asyncio
    async def test_db_recorder_called_alongside_trade_recorder(self, mock_adapter, risk_manager):
        """Both _trade_recorder and _db_recorder are called on order placement."""
        trade_records = []
        db_records = []
        from engine.order_manager import (
            set_trade_recorder,
            set_db_recorder,
            _trade_recorder,
            _db_recorder,
        )

        old_trade_recorder = _trade_recorder
        old_db_recorder = _db_recorder

        set_trade_recorder(lambda t, **kw: trade_records.append(t))

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        try:
            om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
            await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="test",
            )

            # Both recorders called
            assert len(trade_records) == 1
            assert len(db_records) == 1
            # Same data
            assert trade_records[0]["order_id"] == db_records[0]["order_id"]
            assert trade_records[0]["filled_price"] == db_records[0]["filled_price"]
        finally:
            set_trade_recorder(old_trade_recorder)
            set_db_recorder(old_db_recorder)

    @pytest.mark.asyncio
    async def test_db_recorder_not_called_on_failed_order(self, risk_manager):
        """_db_recorder is not called when order fails."""
        db_records = []
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        # Adapter returns failed status
        failed_adapter = AsyncMock()
        failed_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="FAIL001",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=150.0,
                status="failed",
            )
        )

        try:
            om = OrderManager(adapter=failed_adapter, risk_manager=risk_manager)
            order = await om.place_buy(
                symbol="AAPL",
                price=150.0,
                portfolio_value=100_000,
                cash_available=50_000,
                current_positions=0,
                strategy_name="test",
            )
            assert order is None
            assert len(db_records) == 0  # No DB write for failed orders
        finally:
            set_db_recorder(old_recorder)

    @pytest.mark.asyncio
    async def test_sell_db_recorder_includes_pnl_data(self, mock_adapter, risk_manager):
        """place_sell passes pnl and pnl_pct to _db_recorder."""
        db_records = []
        from engine.order_manager import set_db_recorder, _db_recorder

        old_recorder = _db_recorder

        async def mock_db_recorder(trade):
            db_records.append(trade)

        set_db_recorder(mock_db_recorder)

        try:
            om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)
            await om.place_sell(
                symbol="AAPL",
                quantity=10,
                price=160.0,
                strategy_name="take_profit",
                entry_price=150.0,
                buy_strategy="momentum",
            )

            assert len(db_records) == 1
            record = db_records[0]
            # PnL: (160 - 150) * 10 = 100
            assert record["pnl"] == 100.0
            # PnL %: ((160 - 150) / 150) * 100 = 6.67%
            assert record["pnl_pct"] == pytest.approx(6.67, abs=0.01)
            assert record["buy_strategy"] == "momentum"
        finally:
            set_db_recorder(old_recorder)


class TestDuplicateSellPrevention:
    """STOCK-52: Duplicate sell prevention in place_sell."""

    async def test_duplicate_sell_blocked(self, mock_adapter, risk_manager):
        """Second sell for same symbol is blocked when first is pending."""
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="ORD001",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                filled_quantity=0,
                filled_price=None,
                status="pending",
            )
        )
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)

        first = await om.place_sell(
            symbol="AAPL", quantity=10, price=160.0, strategy_name="stop_loss"
        )
        assert first is not None

        # Second sell for same symbol is blocked
        second = await om.place_sell(
            symbol="AAPL", quantity=10, price=155.0, strategy_name="regime_protect"
        )
        assert second is None
        assert mock_adapter.create_sell_order.call_count == 1

    async def test_sell_different_symbol_not_blocked(self, mock_adapter, risk_manager):
        """Sell for different symbol is allowed."""
        call_count = 0

        async def create_sell(**kwargs):
            nonlocal call_count
            call_count += 1
            return OrderResult(
                order_id=f"ORD{call_count:03d}",
                symbol=kwargs["symbol"],
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                filled_quantity=0,
                status="pending",
            )

        mock_adapter.create_sell_order = create_sell
        om = OrderManager(adapter=mock_adapter, risk_manager=risk_manager)

        first = await om.place_sell(
            symbol="AAPL", quantity=10, price=160.0, strategy_name="stop_loss"
        )
        second = await om.place_sell(
            symbol="NVDA", quantity=5, price=800.0, strategy_name="stop_loss"
        )
        assert first is not None
        assert second is not None

    async def test_filled_sell_allows_new_sell(self, order_manager):
        """After a sell is filled and cleared, another sell is allowed."""
        order = await order_manager.place_sell(
            symbol="AAPL", quantity=10, price=160.0, strategy_name="stop_loss"
        )
        assert order is not None
        # Filled order → clear completed → new sell should work
        order_manager.clear_completed()
        second = await order_manager.place_sell(
            symbol="AAPL", quantity=10, price=155.0, strategy_name="take_profit"
        )
        assert second is not None
