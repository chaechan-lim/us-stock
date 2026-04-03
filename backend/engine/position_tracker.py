"""Position tracker - monitors open positions for SL/TP/trailing stop.

Periodically checks all positions against risk rules and triggers
sell orders when stop-loss, take-profit, or trailing stop conditions are met.

On startup, `restore_from_exchange()` fetches current exchange positions
and re-populates the in-memory tracker so SL/TP/trailing stop monitoring
resumes immediately after a restart.

All tracked positions are persisted to the `positions` DB table so the
system can recover even if the exchange API is temporarily unavailable.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from data.market_data_service import MarketDataService
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from exchange.base import ExchangeAdapter
from services.exchange_resolver import ExchangeResolver

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
    trailing_activation_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    tracked_at: float = field(default_factory=time.monotonic)
    partial_profit_taken: bool = False  # True after partial profit sell executed


class PositionTracker:
    """Monitor positions and trigger SL/TP/trailing stop sells."""

    # Minimum interval between auto-recovery attempts (seconds)
    _AUTO_RECOVER_COOLDOWN = 600  # 10 minutes

    def __init__(
        self,
        adapter: ExchangeAdapter,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        notification=None,
        market_data: MarketDataService | None = None,
        event_calendar=None,
        session_factory: "async_sessionmaker[AsyncSession] | None" = None,
        market: str = "US",
        exchange_resolver: ExchangeResolver | None = None,
        account_id: str = "ACC001",
    ):
        self._adapter = adapter
        self._market_data = market_data
        self._risk = risk_manager
        self._orders = order_manager
        self._notification = notification
        self._event_calendar = event_calendar
        self._session_factory = session_factory
        self._market = market
        self._exchange_resolver = exchange_resolver
        self._account_id = account_id
        self._tracked: dict[str, TrackedPosition] = {}
        # Callbacks invoked after a sell is executed (STOCK-43).
        # Signature: callback(symbol: str, sell_timestamp: float) -> None
        # Used by EvaluationLoop to update sell cooldown on PositionTracker-
        # triggered sells (stop-loss, take-profit, trailing stop).
        self._on_sell_callbacks: list[Callable[[str, float], None]] = []
        # Initialize to negative cooldown so the first call always passes,
        # regardless of system uptime (time.monotonic() can be < cooldown
        # on fresh VMs or right after reboot).
        self._last_auto_recover: float = -self._AUTO_RECOVER_COOLDOWN

    def register_on_sell(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback invoked after any sell execution.

        STOCK-43: Allows EvaluationLoop to update its sell cooldown
        when PositionTracker triggers a stop-loss / take-profit / trailing
        stop sell, preventing immediate re-buy of the same symbol.

        Args:
            callback: Called with (symbol, sell_timestamp) after each sell.
        """
        self._on_sell_callbacks.append(callback)

    def track(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        strategy: str = "",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_activation_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        highest_price: float | None = None,  # STOCK-58: Restore from DB
        partial_profit_taken: bool = False,  # STOCK-58: Restore from DB
    ) -> None:
        """Start tracking a position.

        Args:
            trailing_activation_pct: Profit % to activate trailing stop.
                Falls back to RiskParams default if None.
            trailing_stop_pct: Trail % from peak to trigger sell.
                Falls back to RiskParams default if None.
            highest_price: Restore from DB — highest price reached (for trailing stop).
                Defaults to entry_price if not provided.
            partial_profit_taken: Restore from DB — whether partial profit was taken.
        """
        # Apply trailing stop defaults from risk params when not specified
        trail_act = trailing_activation_pct
        trail_pct = trailing_stop_pct
        if trail_act is None:
            trail_act = self._risk.params.default_trailing_activation_pct
        if trail_pct is None:
            trail_pct = self._risk.params.default_trailing_stop_pct

        self._tracked[symbol] = TrackedPosition(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            highest_price=highest_price or entry_price,
            strategy=strategy,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_activation_pct=trail_act,
            trailing_stop_pct=trail_pct,
            partial_profit_taken=partial_profit_taken,
        )
        logger.info(
            "Tracking position: %s %d @ $%.2f (SL=%.1f%% TP=%.1f%% trail=%.1f%%/%.1f%%)",
            symbol,
            quantity,
            entry_price,
            (stop_loss_pct or 0) * 100,
            (take_profit_pct or 0) * 100,
            trail_act * 100,
            trail_pct * 100,
        )
        self._schedule_db_upsert(symbol)

    def untrack(self, symbol: str) -> None:
        """Stop tracking a position."""
        self._tracked.pop(symbol, None)
        self._schedule_db_remove(symbol)

    async def _finalize_sell(
        self,
        symbol: str,
        fill_qty: int,
        fill_price: float,
        tracked: TrackedPosition,
        reason: str,
    ) -> None:
        """Common sell finalization: PnL, untrack, callbacks, notifications.

        Used by both _execute_sell (immediate fills) and handle_sell_fill
        (reconciliation-confirmed fills) to avoid duplicating PnL/callback/
        notification logic across two code paths.
        """
        pnl = (fill_price - tracked.entry_price) * fill_qty
        self._risk.update_daily_pnl(pnl)
        # STOCK-55: Use reason-based is_loss for whipsaw detection.
        # Defensive exits (SL, trailing stop) always count as loss sells
        # even if fill_price > entry (e.g. trailing stop after partial gain).
        # Planned exits (TP, breakeven) never count as loss sells.
        is_loss = reason in ("stop_loss", "trailing_stop", "tiered_trailing_stop")

        self.untrack(symbol)

        # Fire sell callbacks (e.g. EvaluationLoop sell cooldown).
        # Try to pass is_loss for whipsaw detection; fall back to bare
        # (symbol, ts) for callbacks that don't accept the kwarg.
        sell_ts = time.time()
        for cb in self._on_sell_callbacks:
            try:
                cb(symbol, sell_ts, is_loss=is_loss)  # type: ignore[call-arg]
            except TypeError:
                try:
                    cb(symbol, sell_ts)
                except Exception as e:
                    logger.warning("on_sell callback failed for %s: %s", symbol, e)
            except Exception as e:
                logger.warning("on_sell callback failed for %s: %s", symbol, e)

        if self._notification:
            try:
                pnl_pct = (
                    round(
                        (fill_price - tracked.entry_price) / tracked.entry_price * 100,
                        2,
                    )
                    if tracked.entry_price > 0
                    else None
                )
                if reason == "stop_loss":
                    await self._notification.notify_stop_loss(
                        symbol,
                        fill_qty,
                        tracked.entry_price,
                        fill_price,
                        pnl,
                        pnl_pct=pnl_pct,
                    )
                elif reason == "take_profit":
                    await self._notification.notify_take_profit(
                        symbol,
                        fill_qty,
                        tracked.entry_price,
                        fill_price,
                        pnl,
                        pnl_pct=pnl_pct,
                    )
                elif reason in (
                    "trailing_stop",
                    "tiered_trailing_stop",
                    "breakeven_stop",
                ):
                    await self._notification.notify_trailing_stop(
                        symbol,
                        fill_qty,
                        tracked.entry_price,
                        fill_price,
                        tracked.highest_price,
                        pnl,
                        pnl_pct=pnl_pct,
                    )
                else:
                    # Generic sell notification for evaluation_loop paths
                    # (protective sells, signal-based sells) where the
                    # strategy_name lacks a colon separator.
                    await self._notification.notify_trade_executed(
                        symbol,
                        "SELL",
                        fill_qty,
                        fill_price,
                        reason or "sell",
                        market=self._market,
                    )
            except Exception as e:
                logger.error(
                    "Failed to send %s notification for %s: %s",
                    reason or "sell",
                    symbol,
                    e,
                )

    async def handle_sell_fill(
        self,
        symbol: str,
        filled_price: float | None,
        filled_quantity: int | None,
        reason: str = "",
    ) -> None:
        """Handle confirmed sell fill from reconciliation.

        STOCK-52: When a pending sell order is confirmed filled during
        reconciliation, compute PnL, untrack (or reduce qty for partial fills),
        fire sell callbacks, and send notifications. This replaces the
        immediate untrack that was previously done in _execute_sell() for
        pending orders.

        Args:
            symbol: Stock ticker symbol.
            filled_price: Actual fill price from exchange. None if unknown.
            filled_quantity: Number of shares filled. None means full position.
            reason: Sell reason extracted from strategy name (e.g. "stop_loss",
                "take_profit", "trailing_stop", "profit_taking").
        """
        tracked = self._tracked.get(symbol)
        if not tracked:
            # STOCK-60: Already untracked (e.g. by check_all finding position gone
            # before reconciliation arrived). Still fire sell callbacks so that
            # sell cooldown is registered and immediate re-buy is prevented.
            logger.info(
                "handle_sell_fill: %s already untracked — firing callbacks for sell cooldown",
                symbol,
            )
            is_loss = reason in ("stop_loss", "trailing_stop", "tiered_trailing_stop")
            sell_ts = time.time()
            for cb in self._on_sell_callbacks:
                try:
                    cb(symbol, sell_ts, is_loss=is_loss)  # type: ignore[call-arg]
                except TypeError:
                    try:
                        cb(symbol, sell_ts)
                    except Exception as e:
                        logger.warning("on_sell callback failed for %s: %s", symbol, e)
                except Exception as e:
                    logger.warning("on_sell callback failed for %s: %s", symbol, e)
            return

        # Use explicit None checks — 0 and 0.0 should NOT silently fall back.
        # None means "unknown quantity" → assume full position (e.g. KIS didn't
        # return filled_quantity).  0 means "no shares filled" (e.g. cancelled
        # order) → early return to avoid false untrack.
        fill_qty = filled_quantity if filled_quantity is not None else tracked.quantity
        if fill_qty <= 0:
            logger.info(
                "handle_sell_fill: filled_quantity=%s for %s, no shares filled — skipping",
                filled_quantity,
                symbol,
            )
            return

        if filled_price is None or filled_price <= 0:
            logger.warning(
                "handle_sell_fill: no valid fill price for %s (got %s), "
                "using entry price for PnL estimate — accuracy may be off",
                symbol,
                filled_price,
            )
            fill_price = tracked.entry_price
        else:
            fill_price = filled_price

        if fill_qty >= tracked.quantity:
            # Full position sold — delegate to shared finalization
            pnl = (fill_price - tracked.entry_price) * fill_qty
            await self._finalize_sell(symbol, fill_qty, fill_price, tracked, reason)

            logger.info(
                "Reconciliation confirmed sell fill for %s: qty=%d price=$%.2f pnl=$%.2f",
                symbol,
                fill_qty,
                fill_price,
                pnl,
            )
        else:
            # Partial fill — reduce quantity and tighten SL, keep position tracked
            pnl = (fill_price - tracked.entry_price) * fill_qty
            self._risk.update_daily_pnl(pnl)

            tracked.quantity -= fill_qty
            tracked.partial_profit_taken = True
            tracked.stop_loss_pct = max(
                0.01,
                min(tracked.stop_loss_pct or 0.08, 0.03),
            )
            self._schedule_db_upsert(symbol)

            logger.info(
                "Reconciliation confirmed partial sell fill for %s: "
                "qty=%d remaining=%d price=$%.2f pnl=$%.2f SL→%.1f%%",
                symbol,
                fill_qty,
                tracked.quantity,
                fill_price,
                pnl,
                tracked.stop_loss_pct * 100,
            )

            if self._notification:
                try:
                    await self._notification.notify_trade_executed(
                        symbol,
                        "SELL",
                        fill_qty,
                        fill_price,
                        reason or "partial_fill",
                        market=self._market,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to send partial fill notification for %s: %s",
                        symbol,
                        e,
                    )

    async def check_all(self, session: str = "regular") -> list[dict]:
        """Check all tracked positions. Returns list of triggered actions.

        Args:
            session: Trading session for sell execution (regular/pre_market/after_hours).
                     Extended hours sells use limit orders only.
        """
        # Defense-in-depth: if tracker is empty, try auto-recovering from
        # exchange positions.  This prevents silently skipping SL/TP when
        # restore_from_exchange() failed at startup.
        if not self._tracked:
            await self._auto_recover_untracked()
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
                # Grace period: don't remove recently tracked positions
                # (buy order may still be pending/unfilled)
                age = time.monotonic() - tracked.tracked_at
                if age < 300:  # 5 minutes
                    logger.debug(
                        "Position %s not in exchange yet (%.0fs since tracked), keeping",
                        symbol,
                        age,
                    )
                    continue
                logger.info("Position %s no longer held, removing tracker", symbol)
                self.untrack(symbol)
                continue

            current_price = pos.current_price
            if current_price <= 0:
                logger.warning(
                    "Position %s has invalid price (%.2f), skipping",
                    symbol,
                    current_price,
                )
                continue

            # Update highest price for trailing stop — must happen BEFORE
            # the pending sell check so that if a pending sell is later
            # cancelled, check_all resumes with an accurate peak price.
            if current_price > tracked.highest_price:
                tracked.highest_price = current_price

            # STOCK-52: Skip evaluation if there is already a pending sell order
            # for this symbol. This prevents duplicate sell orders while a limit
            # sell is awaiting fill confirmation via reconciliation.
            #
            # Note: if the order fails/cancels, has_pending_order returns False
            # on the next cycle, which naturally retries the sell — this is the
            # correct behavior. Do NOT add an extra guard here.
            if self._orders.has_pending_order(symbol, "SELL"):
                logger.debug(
                    "Position %s has pending sell order, skipping SL/TP evaluation",
                    symbol,
                )
                continue

            action = self._evaluate(tracked, current_price)
            if action:
                triggered.append(action)
                if action["reason"] == "profit_taking":
                    await self._execute_partial_sell(
                        tracked,
                        current_price,
                        action.get("partial_qty", 0),
                        session=session,
                    )
                else:
                    await self._execute_sell(
                        tracked, current_price, action["reason"], session=session
                    )

        return triggered

    def _evaluate(self, tracked: TrackedPosition, current_price: float) -> dict | None:
        """Evaluate if any exit condition is met.

        Check order: stop_loss → profit_taking (partial) → flat trailing_stop
        → tiered_trailing_stop → breakeven_stop → take_profit.

        Tiered trailing and breakeven stops protect large unrealized gains
        without cutting early winners (STOCK-24).
        """
        # STOCK-34: Diagnostic logging for TP/SL evaluation
        gain_pct = (
            (current_price - tracked.entry_price) / tracked.entry_price
            if tracked.entry_price > 0
            else 0.0
        )
        if abs(gain_pct) >= 0.10:  # Log positions with >= 10% gain/loss
            logger.info(
                "Evaluating %s: entry=$%.2f current=$%.2f highest=$%.2f "
                "gain=%.1f%% tp=%.1f%% sl=%.1f%% trail=%.1f%%/%.1f%%",
                tracked.symbol,
                tracked.entry_price,
                current_price,
                tracked.highest_price,
                gain_pct * 100,
                (tracked.take_profit_pct or 0) * 100,
                (tracked.stop_loss_pct or 0) * 100,
                tracked.trailing_activation_pct * 100,
                tracked.trailing_stop_pct * 100,
            )

        # Widen SL if earnings are near
        sl_pct = tracked.stop_loss_pct
        if self._event_calendar and sl_pct:
            sl_mult = self._event_calendar.get_sl_multiplier(tracked.symbol)
            if sl_mult:
                sl_pct = sl_pct * sl_mult

        # Stop-loss
        if self._risk.check_stop_loss(tracked.entry_price, current_price, sl_pct):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            return {
                "symbol": tracked.symbol,
                "reason": "stop_loss",
                "entry": tracked.entry_price,
                "current": current_price,
                "pnl": pnl,
            }

        # Partial profit-taking at intermediate gain level
        if self._check_profit_taking(tracked, current_price):
            gain_pct = (current_price - tracked.entry_price) / tracked.entry_price
            sell_qty = self._calculate_profit_take_qty(tracked)
            if sell_qty > 0:
                pnl = (current_price - tracked.entry_price) * sell_qty
                return {
                    "symbol": tracked.symbol,
                    "reason": "profit_taking",
                    "entry": tracked.entry_price,
                    "current": current_price,
                    "pnl": pnl,
                    "partial_qty": sell_qty,
                    "gain_pct": round(gain_pct * 100, 1),
                }

        # Trailing stop (flat, per-strategy)
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

        # Tiered trailing stop (global, gain-adaptive — STOCK-24)
        if self._risk.check_tiered_trailing_stop(
            tracked.entry_price,
            current_price,
            tracked.highest_price,
        ):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            gain_pct = (tracked.highest_price - tracked.entry_price) / tracked.entry_price
            return {
                "symbol": tracked.symbol,
                "reason": "tiered_trailing_stop",
                "entry": tracked.entry_price,
                "current": current_price,
                "highest": tracked.highest_price,
                "pnl": pnl,
                "peak_gain_pct": round(gain_pct * 100, 1),
            }

        # Breakeven stop (ratcheted SL — STOCK-24)
        if self._risk.check_breakeven_stop(
            tracked.entry_price,
            current_price,
            tracked.highest_price,
            tracked.take_profit_pct,
        ):
            pnl = (current_price - tracked.entry_price) * tracked.quantity
            return {
                "symbol": tracked.symbol,
                "reason": "breakeven_stop",
                "entry": tracked.entry_price,
                "current": current_price,
                "highest": tracked.highest_price,
                "pnl": pnl,
            }

        # Take-profit (full position)
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

        return None

    def _check_profit_taking(self, tracked: TrackedPosition, current_price: float) -> bool:
        """Check if partial profit-taking should be triggered.

        Returns True when:
        - Profit-taking is enabled in risk params
        - Position has not already had partial profit taken
        - Current gain exceeds the profit-taking threshold
        - Current gain is BELOW the take-profit threshold (TP = full sell)
        - Position has enough quantity for partial sell (>= 2 shares)
        """
        if not self._risk.params.profit_taking_enabled:
            return False
        if tracked.partial_profit_taken:
            return False
        if tracked.quantity < 2:
            return False
        if tracked.entry_price <= 0:
            return False

        gain_pct = (current_price - tracked.entry_price) / tracked.entry_price

        # Don't partial-sell if we've already reached full TP level
        tp_pct = tracked.take_profit_pct or self._risk.params.default_take_profit_pct
        if gain_pct >= tp_pct:
            return False

        return gain_pct >= self._risk.params.profit_taking_threshold_pct

    def _calculate_profit_take_qty(self, tracked: TrackedPosition) -> int:
        """Calculate number of shares to sell for partial profit-taking."""
        sell_ratio = self._risk.params.profit_taking_sell_ratio
        sell_qty = max(1, int(tracked.quantity * sell_ratio))
        # Never sell the entire position via profit-taking
        sell_qty = min(sell_qty, tracked.quantity - 1)
        return sell_qty

    async def _execute_partial_sell(
        self,
        tracked: TrackedPosition,
        price: float,
        sell_qty: int,
        session: str = "regular",
    ) -> None:
        """Execute a partial sell (profit-taking) and keep remaining position tracked."""
        gain_pct = (price - tracked.entry_price) / tracked.entry_price * 100
        logger.warning(
            "PROFIT_TAKING for %s: selling %d/%d shares (gain=%.1f%%) entry=$%.2f current=$%.2f",
            tracked.symbol,
            sell_qty,
            tracked.quantity,
            gain_pct,
            tracked.entry_price,
            price,
        )

        # KIS overseas API only supports limit orders (ORD_DVSN="00").
        # Market orders (ORD_DVSN="01") return APBK1269 error.
        order_type = "limit"

        order = await self._orders.place_sell(
            symbol=tracked.symbol,
            quantity=sell_qty,
            price=price,
            strategy_name=f"{tracked.strategy}:profit_taking",
            order_type=order_type,
            exchange=self._resolve_exchange(tracked.symbol),
            entry_price=tracked.entry_price,
            buy_strategy=tracked.strategy,
            session=session,
        )

        if order:
            # STOCK-52: Only update tracked position when order is confirmed filled.
            # Pending partial sells should not modify quantity/SL until fill confirmed.
            # handle_sell_fill() (called by reconciliation) handles the deferred update.
            if order.status != "filled":
                logger.info(
                    "Partial sell order for %s is %s (order_id=%s), "
                    "deferring quantity update until fill confirmed",
                    tracked.symbol,
                    order.status,
                    order.order_id,
                )
                return

            fill_qty = order.filled_quantity or sell_qty
            fill_price = order.filled_price or price
            pnl = (fill_price - tracked.entry_price) * fill_qty
            self._risk.update_daily_pnl(pnl)

            # Update tracked position: reduce quantity, mark partial taken
            tracked.quantity -= fill_qty
            tracked.partial_profit_taken = True

            # Tighten stop-loss to breakeven after profit-taking
            # This protects the remaining position from turning into a loss
            tracked.stop_loss_pct = max(
                0.01,  # minimum 1% SL
                min(tracked.stop_loss_pct or 0.08, 0.03),  # tighten to at most 3%
            )

            logger.info(
                "Profit taken for %s: sold %d, remaining %d, SL tightened to %.1f%%",
                tracked.symbol,
                fill_qty,
                tracked.quantity,
                tracked.stop_loss_pct * 100,
            )
            self._schedule_db_upsert(tracked.symbol)

            if self._notification:
                try:
                    await self._notification.notify_profit_taking(
                        tracked.symbol,
                        fill_qty,
                        tracked.entry_price,
                        price,
                        pnl,
                        tracked.quantity,
                    )
                except AttributeError:
                    # Notification adapter may not have notify_profit_taking yet
                    try:
                        await self._notification.notify_take_profit(
                            tracked.symbol,
                            fill_qty,
                            tracked.entry_price,
                            price,
                            pnl,
                        )
                    except Exception as e:
                        logger.error("Failed to send profit_taking notification: %s", e)
                except Exception as e:
                    logger.error("Failed to send profit_taking notification: %s", e)

    async def _execute_sell(
        self,
        tracked: TrackedPosition,
        price: float,
        reason: str,
        session: str = "regular",
    ) -> None:
        """Execute a full sell order and notify."""
        session_tag = f" [{session}]" if session != "regular" else ""
        logger.warning(
            "%s triggered for %s%s: entry=$%.2f current=$%.2f",
            reason.upper(),
            tracked.symbol,
            session_tag,
            tracked.entry_price,
            price,
        )

        # KIS overseas API only supports limit orders (ORD_DVSN="00").
        # Market orders (ORD_DVSN="01") return APBK1269 error.
        order_type = "limit"

        order = await self._orders.place_sell(
            symbol=tracked.symbol,
            quantity=tracked.quantity,
            price=price,
            strategy_name=f"{tracked.strategy}:{reason}",
            order_type=order_type,
            exchange=self._resolve_exchange(tracked.symbol),
            entry_price=tracked.entry_price,
            buy_strategy=tracked.strategy,
            session=session,
        )

        if order:
            # STOCK-52: Only untrack + update PnL when order is confirmed filled.
            # KIS limit orders return status="pending" — if we untrack immediately,
            # the position is lost from SL/TP monitoring. If the order never fills
            # (price moved away), the position gets abandoned on the exchange.
            # Pending sells are handled by handle_sell_fill() (called from
            # reconciliation when the fill is confirmed).
            if order.status == "filled":
                fill_qty = order.filled_quantity or tracked.quantity
                fill_price = order.filled_price or price
                await self._finalize_sell(
                    tracked.symbol,
                    fill_qty,
                    fill_price,
                    tracked,
                    reason,
                )
            else:
                logger.info(
                    "Sell order for %s is %s (order_id=%s), keeping position tracked "
                    "until fill is confirmed by reconciliation",
                    tracked.symbol,
                    order.status,
                    order.order_id,
                )

    async def restore_from_exchange(self, session_factory=None) -> list[dict]:
        """Restore position tracking from exchange state after restart.

        Fetches current positions from the exchange adapter and looks up
        entry info (price, strategy) from the orders DB table.
        Persists restored positions to the positions DB table.
        Returns a list of restored position summaries.
        """
        # Use provided session_factory or fall back to instance one
        sf = session_factory or self._session_factory

        try:
            if self._market_data:
                positions = await self._market_data.get_positions()
            else:
                positions = await self._adapter.fetch_positions()
        except Exception as e:
            logger.error("Startup position restore failed (fetch): %s", e)
            return []

        if not positions:
            logger.info("No open positions to restore")
            # Clean up any stale DB positions when exchange confirms empty
            if sf:
                await self._clear_all_positions_db(sf)
            return []

        # Look up strategy name per symbol from DB orders.
        # Fallback chain:
        #   1. Latest live BUY order (is_paper=False)
        #   2. Latest BUY order of any type (including paper)
        #   3. Latest SELL order's buy_strategy field
        entry_info: dict[str, dict] = {}
        if sf:
            try:
                from sqlalchemy import desc, select

                from core.models import Order

                async with sf() as session:
                    for pos in positions:
                        # 1) Latest live BUY order
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == pos.symbol,
                                Order.side == "BUY",
                                Order.status.in_(["filled", "submitted"]),
                                Order.is_paper == False,  # noqa: E712
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order and order.strategy_name:
                            entry_info[pos.symbol] = {
                                "strategy": order.strategy_name,
                            }
                            continue

                        # 2) Any BUY order (including paper) as fallback
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == pos.symbol,
                                Order.side == "BUY",
                                Order.status.in_(["filled", "submitted"]),
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order and order.strategy_name:
                            entry_info[pos.symbol] = {
                                "strategy": order.strategy_name,
                            }
                            continue

                        # 3) SELL order's buy_strategy as last resort
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == pos.symbol,
                                Order.side == "SELL",
                                Order.buy_strategy.isnot(None),
                                Order.buy_strategy != "",
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order and order.buy_strategy:
                            entry_info[pos.symbol] = {
                                "strategy": order.buy_strategy,
                            }
            except Exception as e:
                logger.warning("Failed to look up entry info from DB: %s", e)

        restored = []
        for pos in positions:
            if pos.quantity <= 0:
                continue

            # Skip if already tracked (shouldn't happen on fresh start)
            if pos.symbol in self._tracked:
                continue

            info = entry_info.get(pos.symbol, {})
            strategy = info.get("strategy", "unknown")

            # Use avg_price from exchange as entry price
            entry_price = pos.avg_price

            # Dynamic ATR-based SL/TP per stock volatility
            stop_loss_pct = self._risk.params.default_stop_loss_pct
            take_profit_pct = self._risk.params.default_take_profit_pct
            if self._market_data:
                try:
                    ohlcv = await self._market_data.get_ohlcv(pos.symbol, limit=30)
                    if not ohlcv.empty and len(ohlcv) >= 14:
                        import pandas_ta as ta

                        atr_series = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=14)
                        if atr_series is not None and not atr_series.empty:
                            atr_val = float(atr_series.iloc[-1])
                            if atr_val > 0:
                                market = "KR" if pos.symbol.isdigit() else "US"
                                stop_loss_pct, take_profit_pct = self._risk.calculate_dynamic_sl_tp(
                                    entry_price,
                                    atr_val,
                                    market=market,
                                )
                except Exception as e:
                    logger.debug("ATR fetch failed for %s, using defaults: %s", pos.symbol, e)

            self.track(
                symbol=pos.symbol,
                entry_price=entry_price,
                quantity=int(pos.quantity),
                strategy=strategy,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

            pnl_pct = ((pos.current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            restored.append(
                {
                    "symbol": pos.symbol,
                    "quantity": int(pos.quantity),
                    "entry_price": entry_price,
                    "current_price": pos.current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "strategy": strategy,
                }
            )

        if restored:
            logger.info(
                "Restored %d positions from exchange: %s",
                len(restored),
                ", ".join(r["symbol"] for r in restored),
            )

        # Persist all restored positions to DB in bulk
        if sf and restored:
            await self.sync_to_db(sf)

        return restored

    async def restore_from_db(self, session_factory=None) -> list[dict]:
        """Restore position tracking from DB when exchange API is unavailable.

        This is a fallback for restore_from_exchange() — reads the positions
        table (which was populated by sync_to_db) and re-populates the
        in-memory tracker. Preserves SL/TP and strategy from previous session.

        Returns a list of restored position summaries.
        """
        sf = session_factory or self._session_factory
        if not sf:
            logger.warning("restore_from_db: no session_factory available")
            return []

        try:
            positions = await self._load_positions_from_db(sf)
        except Exception as e:
            logger.error("restore_from_db failed: %s", e)
            return []

        if not positions:
            logger.info("restore_from_db: no positions in DB for %s", self._market)
            return []

        restored: list[dict] = []
        for record in positions:
            if record.quantity <= 0:
                continue
            if record.symbol in self._tracked:
                continue

            self.track(
                symbol=record.symbol,
                entry_price=record.avg_price,
                quantity=record.quantity,
                strategy=record.strategy_name or "unknown",
                stop_loss_pct=record.stop_loss,
                take_profit_pct=record.take_profit,
                highest_price=record.highest_price,  # STOCK-58: Restore highest price
                partial_profit_taken=record.partial_profit_taken or False,  # STOCK-58: Restore partial profit flag
            )

            pnl_pct = 0.0
            current_price = record.current_price or record.avg_price
            if record.avg_price > 0:
                pnl_pct = ((current_price / record.avg_price) - 1) * 100

            restored.append(
                {
                    "symbol": record.symbol,
                    "quantity": record.quantity,
                    "entry_price": record.avg_price,
                    "current_price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "strategy": record.strategy_name or "unknown",
                    "source": "db",
                }
            )

        if restored:
            logger.info(
                "Restored %d %s positions from DB: %s",
                len(restored),
                self._market,
                ", ".join(r["symbol"] for r in restored),
            )

        return restored

    async def _load_positions_from_db(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> list:
        """Load all position records for this account+market from DB."""
        from sqlalchemy import select

        from core.models import PositionRecord

        async with session_factory() as session:
            stmt = select(PositionRecord).where(
                PositionRecord.account_id == self._account_id,
                PositionRecord.market == self._market,
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def _auto_recover_untracked(self) -> None:
        """Try to recover positions from exchange when tracker is empty.

        Called by check_all() as defense-in-depth. Has a cooldown to prevent
        hammering the exchange API every 60-second check cycle.
        """
        now = time.monotonic()
        if now - self._last_auto_recover < self._AUTO_RECOVER_COOLDOWN:
            return
        self._last_auto_recover = now

        logger.info(
            "Auto-recovery: %s tracker is empty, attempting exchange fetch",
            self._market,
        )

        try:
            if self._market_data:
                positions = await self._market_data.get_positions()
            else:
                positions = await self._adapter.fetch_positions()
        except Exception as e:
            logger.warning("Auto-recovery exchange fetch failed: %s", e)
            # Fallback: try DB
            try:
                db_restored = await self.restore_from_db()
                if db_restored:
                    logger.info(
                        "Auto-recovery from DB: %d positions",
                        len(db_restored),
                    )
            except Exception as db_err:
                logger.warning("Auto-recovery DB fallback also failed: %s", db_err)
            return

        if not positions:
            logger.debug("Auto-recovery: no exchange positions found")
            return

        # Track any exchange positions not already tracked
        recovered = 0
        for pos in positions:
            if pos.quantity <= 0 or pos.symbol in self._tracked:
                continue
            self.track(
                symbol=pos.symbol,
                entry_price=pos.avg_price,
                quantity=int(pos.quantity),
                strategy="unknown",
                stop_loss_pct=self._risk.params.default_stop_loss_pct,
                take_profit_pct=self._risk.params.default_take_profit_pct,
            )
            recovered += 1

        if recovered:
            logger.warning(
                "Auto-recovered %d untracked %s positions for SL/TP monitoring",
                recovered,
                self._market,
            )

    async def sync_to_db(self, session_factory=None) -> int:
        """Synchronize all in-memory tracked positions to the positions DB table.

        Upserts current tracked positions and removes stale DB rows for
        positions no longer tracked. Fetches current prices from exchange
        to populate current_price and unrealized_pnl. Returns the number
        of positions synced.
        """
        sf = session_factory or self._session_factory
        if not sf:
            return 0

        # Fetch current prices from exchange to populate current_price/unrealized_pnl
        price_map: dict[str, float] = {}
        if self._tracked:
            try:
                if self._market_data:
                    positions = await self._market_data.get_positions()
                else:
                    positions = await self._adapter.fetch_positions()
                price_map = {p.symbol: p.current_price for p in positions if p.current_price > 0}
            except Exception as e:
                logger.debug("Failed to fetch prices for DB sync: %s", e)

        # Re-resolve "unknown" strategies from order history
        unknown_symbols = [sym for sym, t in self._tracked.items() if t.strategy in ("unknown", "")]
        if unknown_symbols:
            await self._resolve_unknown_strategies(sf, unknown_symbols)

        try:
            from sqlalchemy import select

            from core.models import PositionRecord

            async with sf() as session:
                tracked_symbols = set(self._tracked.keys())

                # Remove DB rows for positions no longer tracked in this account+market
                stmt = select(PositionRecord).where(
                    PositionRecord.account_id == self._account_id,
                    PositionRecord.market == self._market,
                )
                result = await session.execute(stmt)
                db_positions = result.scalars().all()

                for db_pos in db_positions:
                    if db_pos.symbol not in tracked_symbols:
                        await session.delete(db_pos)

                # Upsert all currently tracked positions with current prices
                for symbol, tracked in self._tracked.items():
                    current_price = price_map.get(symbol)
                    await self._upsert_position_record(
                        session,
                        symbol,
                        tracked,
                        current_price=current_price,
                    )

                await session.commit()

            synced = len(self._tracked)
            logger.info(
                "Synced %d %s positions to DB",
                synced,
                self._market,
            )
            return synced

        except Exception as e:
            logger.error("Failed to sync positions to DB: %s", e)
            return 0

    def get_buy_strategy(self, symbol: str) -> str:
        """Get the original buy strategy for a tracked position."""
        tracked = self._tracked.get(symbol)
        return tracked.strategy if tracked else ""

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

    # ── DB persistence helpers ──────────────────────────────────────────

    async def _resolve_unknown_strategies(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
        symbols: list[str],
    ) -> None:
        """Try to resolve 'unknown' strategies from order history.

        Updates TrackedPosition.strategy in-memory if a match is found.
        Uses the same fallback chain as restore_from_exchange.
        """
        try:
            from sqlalchemy import desc, select

            from core.models import Order

            async with session_factory() as session:
                for symbol in symbols:
                    strategy: str | None = None

                    # 1) Latest live BUY order
                    stmt = (
                        select(Order)
                        .where(
                            Order.symbol == symbol,
                            Order.side == "BUY",
                            Order.status.in_(["filled", "submitted"]),
                            Order.is_paper == False,  # noqa: E712
                        )
                        .order_by(desc(Order.created_at))
                        .limit(1)
                    )
                    result = await session.execute(stmt)
                    order = result.scalar_one_or_none()
                    if order and order.strategy_name:
                        strategy = order.strategy_name

                    # 2) Any BUY order as fallback
                    if not strategy:
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == symbol,
                                Order.side == "BUY",
                                Order.status.in_(["filled", "submitted"]),
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order and order.strategy_name:
                            strategy = order.strategy_name

                    # 3) SELL order's buy_strategy as last resort
                    if not strategy:
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == symbol,
                                Order.side == "SELL",
                                Order.buy_strategy.isnot(None),
                                Order.buy_strategy != "",
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order and order.buy_strategy:
                            strategy = order.buy_strategy

                    if strategy and symbol in self._tracked:
                        self._tracked[symbol].strategy = strategy
                        logger.info(
                            "Resolved strategy for %s: %s",
                            symbol,
                            strategy,
                        )
        except Exception as e:
            logger.debug("Failed to resolve unknown strategies: %s", e)

    def _schedule_db_upsert(self, symbol: str) -> None:
        """Schedule a fire-and-forget DB upsert for a tracked position."""
        if not self._session_factory:
            return
        tracked = self._tracked.get(symbol)
        if not tracked:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._upsert_position_db(symbol, tracked))
        except RuntimeError:
            # No running event loop (e.g., sync test context) — skip
            pass

    def _schedule_db_remove(self, symbol: str) -> None:
        """Schedule a fire-and-forget DB removal for a position."""
        if not self._session_factory:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._remove_position_db(symbol))
        except RuntimeError:
            pass

    async def _upsert_position_db(
        self,
        symbol: str,
        tracked: TrackedPosition,
    ) -> None:
        """Upsert a single position into the positions DB table."""
        if not self._session_factory:
            return
        try:
            async with self._session_factory() as session:
                await self._upsert_position_record(session, symbol, tracked)
                await session.commit()
        except Exception as e:
            logger.debug("DB upsert failed for %s: %s", symbol, e)

    async def _remove_position_db(self, symbol: str) -> None:
        """Remove a position from the positions DB table."""
        if not self._session_factory:
            return
        try:
            from sqlalchemy import delete

            from core.models import PositionRecord

            async with self._session_factory() as session:
                stmt = delete(PositionRecord).where(
                    PositionRecord.account_id == self._account_id,
                    PositionRecord.market == self._market,
                    PositionRecord.symbol == symbol,
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.debug("DB remove failed for %s: %s", symbol, e)

    async def _lookup_first_buy_time(
        self,
        session: "AsyncSession",
        symbol: str,
    ) -> datetime | None:
        """Look up the earliest live BUY order time for a symbol.

        Queries the orders table for the earliest filled/submitted live
        (is_paper=False) BUY order's created_at. Returns None if no
        matching order is found.
        """
        from sqlalchemy import asc, select

        from core.models import Order

        stmt = (
            select(Order.created_at)
            .where(
                Order.symbol == symbol,
                Order.side == "BUY",
                Order.status.in_(["filled", "submitted"]),
                Order.is_paper == False,  # noqa: E712
            )
            .order_by(asc(Order.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return row

    async def _upsert_position_record(
        self,
        session: "AsyncSession",
        symbol: str,
        tracked: TrackedPosition,
        current_price: float | None = None,
    ) -> None:
        """Upsert a PositionRecord within an existing session (no commit).

        Args:
            current_price: Current market price. When provided, also computes
                and stores unrealized_pnl.
        """
        from sqlalchemy import select

        from core.models import PositionRecord

        stmt = select(PositionRecord).where(
            PositionRecord.account_id == self._account_id,
            PositionRecord.market == self._market,
            PositionRecord.symbol == symbol,
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

        # Calculate unrealized PnL when current price is available
        unrealized_pnl: float | None = None
        if current_price is not None and tracked.entry_price > 0:
            unrealized_pnl = (current_price - tracked.entry_price) * tracked.quantity

        if record:
            record.quantity = tracked.quantity
            record.avg_price = tracked.entry_price
            record.stop_loss = tracked.stop_loss_pct
            record.take_profit = tracked.take_profit_pct
            record.trailing_stop = tracked.trailing_stop_pct
            record.strategy_name = tracked.strategy
            record.highest_price = tracked.highest_price  # STOCK-58: Persist highest price
            record.partial_profit_taken = tracked.partial_profit_taken  # STOCK-58: Persist partial profit flag
            if current_price is not None:
                record.current_price = current_price
            if unrealized_pnl is not None:
                record.unrealized_pnl = unrealized_pnl
            record.updated_at = datetime.utcnow()
        else:
            # Look up actual first buy time from orders table
            first_buy_time = await self._lookup_first_buy_time(session, symbol)
            opened_at = first_buy_time if first_buy_time else datetime.utcnow()

            record = PositionRecord(
                account_id=self._account_id,
                market=self._market,
                symbol=symbol,
                exchange=self._resolve_exchange(symbol),
                quantity=tracked.quantity,
                avg_price=tracked.entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                stop_loss=tracked.stop_loss_pct,
                take_profit=tracked.take_profit_pct,
                trailing_stop=tracked.trailing_stop_pct,
                strategy_name=tracked.strategy,
                highest_price=tracked.highest_price,  # STOCK-58: Persist highest price
                partial_profit_taken=tracked.partial_profit_taken,  # STOCK-58: Persist partial profit flag
                opened_at=opened_at,
                updated_at=datetime.utcnow(),
            )
            session.add(record)

    async def _clear_all_positions_db(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        """Remove all positions for this account+market from DB."""
        try:
            from sqlalchemy import delete

            from core.models import PositionRecord

            async with session_factory() as session:
                stmt = delete(PositionRecord).where(
                    PositionRecord.account_id == self._account_id,
                    PositionRecord.market == self._market,
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.debug("Failed to clear DB positions for %s: %s", self._market, e)

    def _resolve_exchange(self, symbol: str) -> str:
        """Resolve exchange code from symbol.

        KR market always returns 'KRX'.
        US market uses ExchangeResolver (yfinance lookup + cache) when
        available, falling back to 'NASD' for backward compatibility.
        """
        if self._market == "KR":
            return "KRX"
        if self._exchange_resolver is not None:
            return self._exchange_resolver.resolve(symbol)
        return "NASD"
