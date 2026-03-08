"""Position tracker - monitors open positions for SL/TP/trailing stop.

Periodically checks all positions against risk rules and triggers
sell orders when stop-loss, take-profit, or trailing stop conditions are met.
"""

import logging
from dataclasses import dataclass, field

from exchange.base import ExchangeAdapter, Position
from data.market_data_service import MarketDataService
from engine.risk_manager import RiskManager
from engine.order_manager import OrderManager

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    symbol: str
    entry_price: float
    quantity: int
    highest_price: float
    strategy: str = ""
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_activation_pct: float = 0.05
    trailing_stop_pct: float = 0.03


class PositionTracker:
    """Monitor positions and trigger SL/TP/trailing stop sells."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        notification=None,
        market_data: MarketDataService | None = None,
    ):
        self._adapter = adapter
        self._market_data = market_data
        self._risk = risk_manager
        self._orders = order_manager
        self._notification = notification
        self._tracked: dict[str, TrackedPosition] = {}

    def track(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        strategy: str = "",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        """Start tracking a position."""
        self._tracked[symbol] = TrackedPosition(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            highest_price=entry_price,
            strategy=strategy,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        logger.info("Tracking position: %s %d @ $%.2f", symbol, quantity, entry_price)

    def untrack(self, symbol: str) -> None:
        """Stop tracking a position."""
        self._tracked.pop(symbol, None)

    async def check_all(self) -> list[dict]:
        """Check all tracked positions. Returns list of triggered actions."""
        if not self._tracked:
            return []

        try:
            if self._market_data:
                positions = await self._market_data.get_positions()
            else:
                positions = await self._adapter.fetch_positions()
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

        position_map = {p.symbol: p for p in positions}
        triggered = []

        for symbol, tracked in list(self._tracked.items()):
            pos = position_map.get(symbol)
            if not pos:
                logger.info("Position %s no longer held, removing tracker", symbol)
                self.untrack(symbol)
                continue

            current_price = pos.current_price
            if current_price <= 0:
                continue

            # Update highest price for trailing stop
            if current_price > tracked.highest_price:
                tracked.highest_price = current_price

            action = self._evaluate(tracked, current_price)
            if action:
                triggered.append(action)
                await self._execute_sell(tracked, current_price, action["reason"])

        return triggered

    def _evaluate(self, tracked: TrackedPosition, current_price: float) -> dict | None:
        """Evaluate if any exit condition is met."""
        # Stop-loss
        if self._risk.check_stop_loss(
            tracked.entry_price, current_price, tracked.stop_loss_pct
        ):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            return {
                "symbol": tracked.symbol,
                "reason": "stop_loss",
                "entry": tracked.entry_price,
                "current": current_price,
                "pnl": pnl,
            }

        # Take-profit
        if self._risk.check_take_profit(
            tracked.entry_price, current_price, tracked.take_profit_pct
        ):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            return {
                "symbol": tracked.symbol,
                "reason": "take_profit",
                "entry": tracked.entry_price,
                "current": current_price,
                "pnl": pnl,
            }

        # Trailing stop
        if self._risk.check_trailing_stop(
            tracked.entry_price,
            current_price,
            tracked.highest_price,
            tracked.trailing_activation_pct,
            tracked.trailing_stop_pct,
        ):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            return {
                "symbol": tracked.symbol,
                "reason": "trailing_stop",
                "entry": tracked.entry_price,
                "current": current_price,
                "highest": tracked.highest_price,
                "pnl": pnl,
            }

        return None

    async def _execute_sell(
        self, tracked: TrackedPosition, price: float, reason: str
    ) -> None:
        """Execute a sell order and notify."""
        logger.warning(
            "%s triggered for %s: entry=$%.2f current=$%.2f",
            reason.upper(), tracked.symbol, tracked.entry_price, price,
        )

        order = await self._orders.place_sell(
            symbol=tracked.symbol,
            quantity=tracked.quantity,
            price=price,
            strategy_name=f"{tracked.strategy}:{reason}",
            order_type="market",
        )

        if order:
            pnl = (price - tracked.entry_price) * tracked.quantity
            self._risk.update_daily_pnl(pnl)
            self.untrack(tracked.symbol)

            if self._notification:
                await self._notification.notify_stop_loss(
                    tracked.symbol, tracked.quantity,
                    tracked.entry_price, price, pnl,
                )

    @property
    def tracked_symbols(self) -> list[str]:
        return list(self._tracked.keys())

    def get_status(self) -> list[dict]:
        """Get status of all tracked positions."""
        return [
            {
                "symbol": t.symbol,
                "entry_price": t.entry_price,
                "quantity": t.quantity,
                "highest_price": t.highest_price,
                "strategy": t.strategy,
                "stop_loss_pct": t.stop_loss_pct,
                "take_profit_pct": t.take_profit_pct,
            }
            for t in self._tracked.values()
        ]
