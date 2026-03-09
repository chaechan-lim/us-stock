"""Order manager - handles order creation, tracking, and lifecycle.

Bridges strategy signals with exchange adapter for order execution.
Includes duplicate order prevention, partial fill tracking, and slippage monitoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from exchange.base import ExchangeAdapter, OrderResult
from engine.risk_manager import RiskManager, PositionSizeResult

logger = logging.getLogger(__name__)

# Optional trade recorder (set by main.py at startup)
_trade_recorder = None


def set_trade_recorder(recorder):
    global _trade_recorder
    _trade_recorder = recorder


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float | None
    strategy_name: str
    status: str = "pending"
    filled_quantity: int = 0
    filled_price: float | None = None
    slippage: float = 0.0
    created_at: str = ""
    exchange: str = "NASD"


class OrderManager:
    """Manage order lifecycle: create, track, cancel."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        risk_manager: RiskManager,
        notification=None,
        market_data=None,
    ):
        self._adapter = adapter
        self._risk = risk_manager
        self._notification = notification
        self._market_data = market_data
        self._active_orders: dict[str, ManagedOrder] = {}

    def has_pending_order(self, symbol: str, side: str | None = None) -> bool:
        """Check if there is already a pending/submitted order for this symbol."""
        for o in self._active_orders.values():
            if o.symbol == symbol and o.status in ("pending", "submitted"):
                if side is None or o.side == side:
                    return True
        return False

    async def place_buy(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        cash_available: float,
        current_positions: int,
        strategy_name: str,
        order_type: str = "limit",
        exchange: str = "NASD",
        atr: float | None = None,
        sizing_override: PositionSizeResult | None = None,
    ) -> ManagedOrder | None:
        """Place a buy order after risk checks and deduplication.

        Args:
            sizing_override: Pre-computed sizing (e.g. from Kelly). Skips
                internal sizing calculation when provided.
        """
        # Duplicate check: prevent double-buying same symbol
        if self.has_pending_order(symbol, "BUY"):
            logger.info("Buy skipped for %s: pending order already exists", symbol)
            return None

        if sizing_override is not None:
            sizing = sizing_override
        else:
            sizing = self._risk.calculate_position_size(
                symbol=symbol,
                price=price,
                portfolio_value=portfolio_value,
                cash_available=cash_available,
                current_positions=current_positions,
                atr=atr,
            )

        if not sizing.allowed:
            logger.info("Buy rejected for %s: %s", symbol, sizing.reason)
            if self._notification:
                await self._notification.notify_order_rejected(symbol, sizing.reason)
            return None

        try:
            result = await self._adapter.create_buy_order(
                symbol=symbol,
                quantity=sizing.quantity,
                price=price if order_type == "limit" else None,
                order_type=order_type,
                exchange=exchange,
            )

            # Track slippage (filled_price vs intended price)
            slippage = 0.0
            if result.filled_price and price:
                slippage = result.filled_price - price

            filled_qty = int(result.filled_quantity) if result.filled_quantity else 0

            # Check if order actually succeeded
            if result.status == "failed":
                logger.warning(
                    "Buy order FAILED for %s %d shares @ $%.2f (%s)",
                    symbol, sizing.quantity, price, strategy_name,
                )
                if self._notification:
                    await self._notification.notify_order_rejected(
                        symbol, "Order failed at exchange",
                    )
                return None

            order = ManagedOrder(
                order_id=result.order_id,
                symbol=symbol,
                side="BUY",
                quantity=sizing.quantity,
                price=price,
                strategy_name=strategy_name,
                status=result.status,
                filled_quantity=filled_qty,
                filled_price=result.filled_price,
                slippage=slippage,
                created_at=datetime.now().isoformat(),
                exchange=exchange,
            )
            self._active_orders[result.order_id] = order

            # Invalidate balance/positions cache so next fetch gets fresh data
            if self._market_data:
                self._market_data.invalidate_cache()

            if slippage != 0:
                logger.info(
                    "Buy order placed: %s %d shares @ $%.2f (filled=%d @ $%.2f, "
                    "slippage=$%.4f) (%s)",
                    symbol, sizing.quantity, price,
                    filled_qty, result.filled_price or 0, slippage,
                    strategy_name,
                )
            else:
                logger.info(
                    "Buy order placed: %s %d shares @ $%.2f (%s)",
                    symbol, sizing.quantity, price, strategy_name,
                )

            if self._notification:
                await self._notification.notify_trade_executed(
                    symbol, "BUY", sizing.quantity, price, strategy_name,
                )
            if _trade_recorder:
                _trade_recorder({
                    "symbol": symbol, "side": "BUY", "quantity": sizing.quantity,
                    "price": price, "filled_price": result.filled_price,
                    "filled_quantity": filled_qty,
                    "slippage": slippage,
                    "strategy": strategy_name, "status": result.status,
                    "created_at": order.created_at,
                })
            return order

        except Exception as e:
            logger.error("Failed to place buy order for %s: %s", symbol, e)
            return None

    async def place_sell(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        strategy_name: str = "",
        order_type: str = "limit",
        exchange: str = "NASD",
    ) -> ManagedOrder | None:
        """Place a sell order."""
        try:
            result = await self._adapter.create_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                exchange=exchange,
            )

            # Track slippage
            slippage = 0.0
            if result.filled_price and price:
                slippage = result.filled_price - price

            filled_qty = int(result.filled_quantity) if result.filled_quantity else 0

            # Check if order actually succeeded
            if result.status == "failed":
                logger.warning(
                    "Sell order FAILED for %s %d shares @ %s (%s)",
                    symbol, quantity, f"${price:.2f}" if price else "market",
                    strategy_name,
                )
                return None

            order = ManagedOrder(
                order_id=result.order_id,
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                strategy_name=strategy_name,
                status=result.status,
                filled_quantity=filled_qty,
                filled_price=result.filled_price,
                slippage=slippage,
                created_at=datetime.now().isoformat(),
                exchange=exchange,
            )
            self._active_orders[result.order_id] = order

            # Invalidate balance/positions cache so next fetch gets fresh data
            if self._market_data:
                self._market_data.invalidate_cache()

            logger.info(
                "Sell order placed: %s %d shares @ %s (%s)",
                symbol, quantity, f"${price:.2f}" if price else "market", strategy_name,
            )
            if self._notification:
                await self._notification.notify_trade_executed(
                    symbol, "SELL", quantity, price or 0, strategy_name,
                )
            if _trade_recorder:
                _trade_recorder({
                    "symbol": symbol, "side": "SELL", "quantity": quantity,
                    "price": price, "filled_price": result.filled_price,
                    "filled_quantity": filled_qty,
                    "slippage": slippage,
                    "strategy": strategy_name, "status": result.status,
                    "pnl": None,  # caller can update
                    "created_at": order.created_at,
                })
            return order

        except Exception as e:
            logger.error("Failed to place sell order for %s: %s", symbol, e)
            return None

    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Cancel an active order."""
        try:
            success = await self._adapter.cancel_order(order_id, symbol)
            if success and order_id in self._active_orders:
                self._active_orders[order_id].status = "cancelled"
            return success
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    async def sync_order_status(self, order_id: str, symbol: str) -> ManagedOrder | None:
        """Sync order status from exchange."""
        managed = self._active_orders.get(order_id)
        if not managed:
            return None
        try:
            result = await self._adapter.fetch_order(order_id, symbol)
            managed.status = result.status
            managed.filled_price = result.filled_price
            managed.filled_quantity = int(result.filled_quantity) if result.filled_quantity else 0
            if result.filled_price and managed.price:
                managed.slippage = result.filled_price - managed.price
            return managed
        except Exception as e:
            logger.error("Failed to sync order %s: %s", order_id, e)
            return managed

    async def reconcile_all(self) -> list[dict]:
        """Sync all active orders with exchange. Returns list of state changes."""
        if not self._active_orders:
            return []

        import asyncio

        # Fetch all pending orders in parallel
        pending = [
            (oid, order) for oid, order in self._active_orders.items()
            if order.status not in ("filled", "cancelled")
        ]
        if not pending:
            return []

        async def _fetch(oid: str, order: ManagedOrder):
            try:
                return oid, order, await self._adapter.fetch_order(oid, order.symbol)
            except Exception as e:
                logger.error("Reconcile failed for order %s: %s", oid, e)
                return oid, order, None

        results = await asyncio.gather(*[_fetch(oid, o) for oid, o in pending])

        changes = []
        for order_id, order, result in results:
            if result is None:
                continue
            old_status = order.status
            order.status = result.status
            order.filled_price = result.filled_price
            order.filled_quantity = int(result.filled_quantity) if result.filled_quantity else 0
            if result.filled_price and order.price:
                order.slippage = result.filled_price - order.price

            if old_status != result.status:
                changes.append({
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "old_status": old_status,
                    "new_status": result.status,
                    "filled_quantity": order.filled_quantity,
                    "quantity": order.quantity,
                })
                logger.info(
                    "Order %s (%s %s): %s -> %s (filled=%d/%d)",
                    order_id, order.side, order.symbol,
                    old_status, result.status,
                    order.filled_quantity, order.quantity,
                )

        # Clean up completed orders
        self.clear_completed()
        return changes

    @property
    def active_orders(self) -> dict[str, ManagedOrder]:
        return dict(self._active_orders)

    def clear_completed(self) -> None:
        """Remove completed/cancelled orders from tracking."""
        to_remove = [
            oid for oid, o in self._active_orders.items()
            if o.status in ("filled", "cancelled")
        ]
        for oid in to_remove:
            del self._active_orders[oid]
