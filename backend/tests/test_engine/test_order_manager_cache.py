"""Tests for OrderManager cache invalidation on reconcile — STOCK-1.

Validates:
- reconcile_all() invalidates balance cache when fills are detected
- reconcile_all() does NOT invalidate cache for non-fill transitions
- sync_order_status preserves existing behavior
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.order_manager import ManagedOrder, OrderManager
from engine.risk_manager import RiskManager
from exchange.base import OrderResult


def _order_result(
    order_id: str = "ORD001",
    symbol: str = "AAPL",
    status: str = "filled",
    filled_quantity: float = 0,
    filled_price: float | None = None,
) -> OrderResult:
    """Helper to create OrderResult with required fields filled in."""
    return OrderResult(
        order_id=order_id,
        symbol=symbol,
        side="BUY",
        order_type="limit",
        quantity=10,
        status=status,
        filled_quantity=filled_quantity,
        filled_price=filled_price,
    )


@pytest.fixture
def adapter():
    return AsyncMock()


@pytest.fixture
def risk_manager():
    rm = RiskManager()
    return rm


@pytest.fixture
def market_data():
    md = MagicMock()
    md.invalidate_cache = MagicMock()
    md.invalidate_balance_cache = MagicMock()
    return md


@pytest.fixture
def order_manager(adapter, risk_manager, market_data):
    return OrderManager(
        adapter=adapter,
        risk_manager=risk_manager,
        market_data=market_data,
    )


class TestReconcileAllCacheInvalidation:
    """Cache invalidation when reconcile_all detects fills."""

    async def test_invalidates_on_fill(self, order_manager, adapter, market_data):
        """When an order transitions to 'filled', balance cache should be invalidated."""
        order_manager._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )

        adapter.fetch_order = AsyncMock(
            return_value=_order_result(
                order_id="ORD001",
                status="filled",
                filled_quantity=10,
                filled_price=150.5,
            ),
        )

        changes = await order_manager.reconcile_all()

        assert len(changes) == 1
        assert changes[0]["new_status"] == "filled"
        market_data.invalidate_balance_cache.assert_called_once()

    async def test_no_invalidation_on_non_fill_transition(
        self, order_manager, adapter, market_data,
    ):
        """Non-fill transitions (e.g., pending->cancelled) should not invalidate."""
        order_manager._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )

        adapter.fetch_order = AsyncMock(
            return_value=_order_result(
                order_id="ORD001",
                status="cancelled",
                filled_quantity=0,
                filled_price=None,
            ),
        )

        changes = await order_manager.reconcile_all()

        assert len(changes) == 1
        assert changes[0]["new_status"] == "cancelled"
        market_data.invalidate_balance_cache.assert_not_called()

    async def test_no_invalidation_when_no_status_change(self, order_manager, adapter, market_data):
        """No state change -> no cache invalidation."""
        order_manager._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )

        adapter.fetch_order = AsyncMock(
            return_value=_order_result(
                order_id="ORD001",
                status="submitted",
                filled_quantity=0,
                filled_price=None,
            ),
        )

        changes = await order_manager.reconcile_all()
        assert len(changes) == 0
        market_data.invalidate_balance_cache.assert_not_called()

    async def test_invalidates_once_for_multiple_fills(self, order_manager, adapter, market_data):
        """Multiple fills in same reconcile -> invalidates once."""
        order_manager._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )
        order_manager._active_orders["ORD002"] = ManagedOrder(
            order_id="ORD002",
            symbol="TSLA",
            side="SELL",
            quantity=5,
            price=250.0,
            strategy_name="test",
            status="submitted",
        )

        async def _fetch(order_id, symbol):
            return _order_result(
                order_id=order_id,
                symbol=symbol,
                status="filled",
                filled_quantity=10 if order_id == "ORD001" else 5,
                filled_price=150.5 if order_id == "ORD001" else 249.5,
            )

        adapter.fetch_order = _fetch

        changes = await order_manager.reconcile_all()
        assert len(changes) == 2
        # Should be called exactly once even with multiple fills
        market_data.invalidate_balance_cache.assert_called_once()

    async def test_no_market_data_no_crash(self, adapter, risk_manager):
        """OrderManager without market_data doesn't crash on reconcile fill."""
        om = OrderManager(adapter=adapter, risk_manager=risk_manager)
        om._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )

        adapter.fetch_order = AsyncMock(
            return_value=_order_result(
                order_id="ORD001",
                status="filled",
                filled_quantity=10,
                filled_price=150.5,
            ),
        )

        # Should not crash when market_data is None
        changes = await om.reconcile_all()
        assert len(changes) == 1

    async def test_empty_orders_no_action(self, order_manager, market_data):
        """Empty active orders -> no reconcile, no cache invalidation."""
        changes = await order_manager.reconcile_all()
        assert changes == []
        market_data.invalidate_balance_cache.assert_not_called()

    async def test_mixed_fill_and_cancel(self, order_manager, adapter, market_data):
        """Mix of filled and cancelled orders -> invalidates because of the fill."""
        order_manager._active_orders["ORD001"] = ManagedOrder(
            order_id="ORD001",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            strategy_name="test",
            status="submitted",
        )
        order_manager._active_orders["ORD002"] = ManagedOrder(
            order_id="ORD002",
            symbol="TSLA",
            side="BUY",
            quantity=5,
            price=250.0,
            strategy_name="test",
            status="submitted",
        )

        async def _fetch(order_id, symbol):
            if order_id == "ORD001":
                return _order_result(
                    order_id=order_id, symbol=symbol,
                    status="filled", filled_quantity=10, filled_price=150.5,
                )
            return _order_result(
                order_id=order_id, symbol=symbol,
                status="cancelled", filled_quantity=0, filled_price=None,
            )

        adapter.fetch_order = _fetch

        changes = await order_manager.reconcile_all()
        assert len(changes) == 2
        market_data.invalidate_balance_cache.assert_called_once()
