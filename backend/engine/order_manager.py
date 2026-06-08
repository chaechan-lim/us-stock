"""Order manager - handles order creation, tracking, and lifecycle.

Bridges strategy signals with exchange adapter for order execution.
Includes duplicate order prevention, partial fill tracking, and slippage monitoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from exchange.base import ExchangeAdapter, OrderResult
from engine.risk_manager import RiskManager, PositionSizeResult
from core.timeutil import now_utc_naive

logger = logging.getLogger(__name__)

# Optional trade recorder (set by main.py at startup)
_trade_recorder = None
# STOCK-38: Optional async DB recorder for awaited persistence
_db_recorder = None


def set_trade_recorder(recorder):
    global _trade_recorder
    _trade_recorder = recorder


def set_db_recorder(recorder):
    """Set async DB recorder callback for awaited trade persistence.

    STOCK-38: Ensures filled_price/status are saved to DB at order time,
    rather than relying solely on fire-and-forget or reconciliation.
    """
    global _db_recorder
    _db_recorder = recorder


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
    # FIN-H3 (2026-06-05): tracks how much of `filled_quantity` was
    # ALREADY processed by downstream handlers (handle_sell_fill,
    # position_tracker.quantity -= fill_qty). KIS sometimes reports
    # filled_quantity as the latest tranche rather than cumulative,
    # so subtracting filled_quantity blindly on every reconcile cycle
    # could double-decrement. The new delta we hand to handlers is
    # `filled_quantity - processed_filled_quantity`.
    processed_filled_quantity: int = 0


class OrderManager:
    """Manage order lifecycle: create, track, cancel."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        risk_manager: RiskManager,
        notification=None,
        market_data=None,
        market: str = "US",
        is_paper: bool = False,
        account_id: str = "ACC001",
    ):
        self._adapter = adapter
        self._risk = risk_manager
        self._notification = notification
        self._market_data = market_data
        self._market = market
        self._is_paper = is_paper
        self._account_id = account_id
        self._active_orders: dict[str, ManagedOrder] = {}
        # STOCK-26: Counter for position check failures (API errors causing
        # buy rejections). Allows operations to monitor and distinguish
        # "correctly blocked duplicates" from "missed opportunities due to
        # API issues."
        self._position_check_failures: int = 0

    def has_pending_order(self, symbol: str, side: str | None = None) -> bool:
        """Check if there is already a pending/submitted/open order for this symbol."""
        for o in self._active_orders.values():
            if o.symbol == symbol and o.status in ("pending", "submitted", "open"):
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
        skip_position_limit: bool = False,
        skip_already_held: bool = False,
        session: str = "regular",
    ) -> ManagedOrder | None:
        """Place a buy order after risk checks and deduplication.

        Args:
            sizing_override: Pre-computed sizing (e.g. from Kelly). Skips
                internal sizing calculation when provided.
            skip_position_limit: If True, bypasses max_positions check (used by
                ETF engine which has its own position limits).
            skip_already_held: If True, bypasses the already-held guard. Used
                by continuous-rebalance callers (ETF EW hedge) that legitimately
                top up positions toward a target weight. The default-False
                preserves the fail-closed dedup for every normal BUY path.
        """
        # Duplicate check: prevent double-buying same symbol
        if self.has_pending_order(symbol, "BUY"):
            logger.info("Buy skipped for %s: pending order already exists", symbol)
            return None

        # Defense-in-depth: check exchange positions to prevent buying a
        # symbol we already hold.  This catches cases where in-memory state
        # (position tracker, signal dedup) was lost after a restart.
        # STOCK-26: On failure, reject the buy (fail-safe). Previously this
        # silently swallowed errors, allowing duplicate buys when the API
        # was down.
        # bug_009 (2026-06-09): skip_already_held lets continuous-rebalance
        # callers (EW hedge) top up held positions. Position-check failures
        # still fail closed even when skipping the held guard — a down API
        # must not let an unbounded add-on through.
        if self._market_data and not skip_already_held:
            try:
                exchange_positions = await self._market_data.get_positions()
                if any(p.symbol == symbol and p.quantity > 0 for p in exchange_positions):
                    logger.info(
                        "Buy skipped for %s: already held (exchange positions)",
                        symbol,
                    )
                    return None
            except Exception as e:
                self._position_check_failures += 1
                logger.warning(
                    "Buy rejected for %s: position check failed (%s). "
                    "Refusing buy as safety precaution. "
                    "(total_failures=%d)",
                    symbol,
                    e,
                    self._position_check_failures,
                )
                return None

        # RISK-B3 (2026-06-06): Fat-finger guard. Caller's `price`
        # ultimately becomes a KIS limit. evaluation_loop falls back
        # to df.iloc[-1]['close'] on a fetch error, so a stale/corrupt
        # OHLCV row can hand KIS a zero, negative, or 10× price.
        # A runaway limit on a thin US name fills far above market;
        # KR daily-limit-up still permits +30%. Reject the order
        # before it reaches the broker.
        if price is None or price <= 0:
            logger.error(
                "Buy rejected for %s: invalid price %s — feed corruption?",
                symbol, price,
            )
            if self._notification:
                await self._notification.notify_order_rejected(
                    symbol, f"invalid_price:{price}",
                )
            return None
        if self._market_data is not None:
            try:
                last_close = await self._market_data.get_price(symbol)
                if last_close and last_close > 0:
                    deviation = abs(price - last_close) / last_close
                    # 10% bound covers a normal day's range comfortably
                    # while catching feed corruption / fat-finger.
                    if deviation > 0.10:
                        logger.error(
                            "Buy rejected for %s: limit price %.4f deviates "
                            "%.1f%% from last_close %.4f (>10%% sanity bound)",
                            symbol, price, deviation * 100, last_close,
                        )
                        if self._notification:
                            await self._notification.notify_order_rejected(
                                symbol,
                                f"price_sanity:{price:.4f}_vs_close_{last_close:.4f}"
                                f"_dev_{deviation:.1%}",
                            )
                        return None
            except Exception as e:
                # Don't block the order on a price-lookup failure —
                # this is a guard, not a hard requirement.
                logger.debug("Price sanity check skipped for %s: %s", symbol, e)

        if sizing_override is not None:
            sizing = sizing_override
            # RISK-B4: sizing_override callers (ETFEngine,
            # cash_parking) used to skip RiskManager entirely. Now we
            # still gate them through the safety check — daily loss
            # limit, max positions, max exposure — without
            # re-running sizing math.
            order_value = float(sizing.quantity) * float(price)
            allowed, reason = self._risk.check_safety_gates(
                symbol=symbol,
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                order_value=order_value,
                skip_position_limit=skip_position_limit,
            )
            if not allowed:
                logger.warning(
                    "Buy (sizing_override) rejected for %s by safety gate: %s",
                    symbol, reason,
                )
                if self._notification:
                    await self._notification.notify_order_rejected(
                        symbol, f"safety_gate:{reason}",
                    )
                return None
        else:
            sizing = self._risk.calculate_position_size(
                symbol=symbol,
                price=price,
                portfolio_value=portfolio_value,
                cash_available=cash_available,
                current_positions=0 if skip_position_limit else current_positions,
                atr=atr,
            )

        if not sizing.allowed:
            logger.info("Buy rejected for %s: %s", symbol, sizing.reason)
            if self._notification:
                await self._notification.notify_order_rejected(symbol, sizing.reason)
            return None

        try:
            # Extended hours: force limit order
            if session != "regular":
                order_type = "limit"
            result = await self._adapter.create_buy_order(
                symbol=symbol,
                quantity=sizing.quantity,
                price=price if order_type == "limit" else None,
                order_type=order_type,
                exchange=exchange,
                session=session,
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
                    symbol,
                    sizing.quantity,
                    price,
                    strategy_name,
                )
                if self._notification:
                    await self._notification.notify_order_rejected(
                        symbol,
                        "Order failed at exchange",
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
                created_at=now_utc_naive().isoformat(),
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
                    symbol,
                    sizing.quantity,
                    price,
                    filled_qty,
                    result.filled_price or 0,
                    slippage,
                    strategy_name,
                )
            else:
                logger.info(
                    "Buy order placed: %s %d shares @ $%.2f (%s)",
                    symbol,
                    sizing.quantity,
                    price,
                    strategy_name,
                )

            if self._notification:
                await self._notification.notify_trade_executed(
                    symbol,
                    "BUY",
                    sizing.quantity,
                    price,
                    strategy_name,
                    market=self._market,
                    stop_loss_pct=self._risk.params.default_stop_loss_pct,
                    take_profit_pct=self._risk.params.default_take_profit_pct,
                    filled_qty=filled_qty,
                    filled_price=result.filled_price or 0.0,
                    session=session,
                )
            trade_data = {
                "order_id": result.order_id,
                "symbol": symbol,
                "side": "BUY",
                "quantity": sizing.quantity,
                "price": price,
                "filled_price": result.filled_price,
                "filled_quantity": filled_qty,
                "slippage": slippage,
                "strategy": strategy_name,
                "status": result.status,
                "created_at": order.created_at,
                "market": self._market,
                "exchange": exchange,
                "session": session,
                "is_paper": self._is_paper,
                "account_id": self._account_id,
            }
            if _trade_recorder:
                _trade_recorder(trade_data, skip_db_persist=bool(_db_recorder))
            # STOCK-38: Awaited DB persist ensures filled_price/status
            # are saved immediately, not relying on fire-and-forget
            if _db_recorder:
                try:
                    await _db_recorder(trade_data)
                except Exception:
                    logger.warning(
                        "DB persist failed for %s BUY, will be recovered by reconciliation",
                        symbol,
                    )
            return order

        except Exception as e:
            # LOG-H9 (2026-06-06): exc_info=True so the traceback lands
            # in journald — without it the symptom is harder to chase
            # than the bug.
            logger.error(
                "Failed to place buy order for %s: %s", symbol, e, exc_info=True,
            )
            return None

    async def place_sell(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        strategy_name: str = "",
        order_type: str = "limit",
        exchange: str = "NASD",
        entry_price: float | None = None,
        buy_strategy: str = "",
        session: str = "regular",
    ) -> ManagedOrder | None:
        """Place a sell order. Pass entry_price for PnL, buy_strategy for attribution."""
        # STOCK-52: Prevent duplicate sell orders while a limit sell is pending.
        # Before STOCK-52, immediate untrack in _execute_sell() prevented duplicates.
        # With deferred untrack, the same symbol could be sold again from
        # _check_protective_sells or _execute_signal before reconciliation confirms.
        if self.has_pending_order(symbol, "SELL"):
            logger.info("Sell skipped for %s: pending sell order already exists", symbol)
            return None

        try:
            # Extended hours: force limit order
            if session != "regular":
                order_type = "limit"
            result = await self._adapter.create_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                exchange=exchange,
                session=session,
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
                    symbol,
                    quantity,
                    f"${price:.2f}" if price else "market",
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
                created_at=now_utc_naive().isoformat(),
                exchange=exchange,
            )
            self._active_orders[result.order_id] = order

            # Invalidate balance/positions cache so next fetch gets fresh data
            if self._market_data:
                self._market_data.invalidate_cache()

            logger.info(
                "Sell order placed: %s %d shares @ %s (%s)",
                symbol,
                quantity,
                f"${price:.2f}" if price else "market",
                strategy_name,
            )
            if self._notification:
                await self._notification.notify_trade_executed(
                    symbol,
                    "SELL",
                    quantity,
                    price or 0,
                    strategy_name,
                    market=self._market,
                    filled_qty=filled_qty,
                    filled_price=result.filled_price or 0.0,
                    session=session,
                )
            # Calculate PnL if entry_price is known
            pnl = None
            pnl_pct = None
            if entry_price:
                sell_price = result.filled_price or price or 0
                sell_qty = filled_qty or quantity
                pnl = round((sell_price - entry_price) * sell_qty, 2)
                if entry_price > 0:
                    pnl_pct = round((sell_price - entry_price) / entry_price * 100, 2)

            trade_data = {
                "order_id": result.order_id,
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": price,
                "filled_price": result.filled_price,
                "filled_quantity": filled_qty,
                "slippage": slippage,
                "strategy": strategy_name,
                "status": result.status,
                "buy_strategy": buy_strategy,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "created_at": order.created_at,
                "market": self._market,
                "exchange": exchange,
                "session": session,
                "is_paper": self._is_paper,
                "account_id": self._account_id,
            }
            if _trade_recorder:
                _trade_recorder(trade_data, skip_db_persist=bool(_db_recorder))
            # STOCK-38: Awaited DB persist ensures filled_price/status
            # are saved immediately, not relying on fire-and-forget
            if _db_recorder:
                try:
                    await _db_recorder(trade_data)
                except Exception:
                    logger.warning(
                        "DB persist failed for %s SELL, will be recovered by reconciliation",
                        symbol,
                    )
            return order

        except Exception as e:
            logger.error(
                "Failed to place sell order for %s: %s", symbol, e, exc_info=True,
            )
            return None

    async def cancel(self, order_id: str, symbol: str) -> bool:
        """Cancel an active order."""
        try:
            exchange = "NASD"
            managed = self._active_orders.get(order_id)
            if managed:
                exchange = managed.exchange
            success = await self._adapter.cancel_order(order_id, symbol, exchange=exchange)
            if success and order_id in self._active_orders:
                self._active_orders[order_id].status = "cancelled"
            return success
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    async def cancel_pending_orders(self, symbol: str, side: str) -> int:
        """Cancel all pending orders for a symbol+side. Returns count cancelled."""
        cancelled = 0
        for oid, o in list(self._active_orders.items()):
            if o.symbol == symbol and o.side.upper() == side.upper() and o.status in ("pending", "open", "submitted"):
                try:
                    success = await self.cancel(oid, symbol)
                    if success:
                        cancelled += 1
                except Exception as e:
                    logger.warning("Failed to cancel %s for %s: %s", oid, symbol, e)
        return cancelled

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
            (oid, order)
            for oid, order in self._active_orders.items()
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
        has_new_fill = False
        for order_id, order, result in results:
            if result is None:
                continue
            old_status = order.status

            # STOCK-37: When fetch_order returns "not_found", don't overwrite
            # existing filled_price/filled_quantity with None/0. The order may
            # have been filled but KIS API can't find it (date boundary issue).
            #
            # FIN-H4 (2026-06-05): the position-based "we don't hold it
            # anymore" heuristic was promoting manual SELLs (user clicked
            # in the KIS app) to filled with a synthesized price, which
            # corrupted the audit trail and PnL. Now we cross-check
            # `fetch_executed_orders` for the actual order_id before
            # fabricating a fill — and use the real fill price/qty
            # when found rather than the limit price.
            if result.status == "not_found":
                executed_match = None
                if hasattr(self._adapter, "fetch_executed_orders"):
                    try:
                        execs = await self._adapter.fetch_executed_orders()
                        for ex in execs:
                            if ex.order_id == order_id:
                                executed_match = ex
                                break
                    except Exception as e:
                        logger.debug(
                            "executed-orders lookup failed for %s: %s",
                            order_id, e,
                        )

                if executed_match and executed_match.status == "filled":
                    # Authoritative: KIS confirmed the fill in the
                    # execution log. Use the REAL fill price/qty.
                    logger.info(
                        "Order %s %s %s: not_found in inquire but found in "
                        "executed-orders (filled %s @ %s)",
                        order_id, order.side, order.symbol,
                        executed_match.filled_quantity,
                        executed_match.filled_price,
                    )
                    result = OrderResult(
                        order_id=result.order_id,
                        symbol=result.symbol,
                        side=result.side,
                        order_type=result.order_type,
                        quantity=result.quantity,
                        status="filled",
                        filled_price=executed_match.filled_price,
                        filled_quantity=executed_match.filled_quantity,
                    )
                elif order.side == "BUY" and self._market_data:
                    # Defence-in-depth: position check is a weaker signal
                    # than execution log, but we keep it for BUYs because
                    # the failure mode (treating our own BUY as filled
                    # when we hold the symbol) is less harmful than
                    # missing a fill entirely. Log clearly so any abuse
                    # of the heuristic shows up in journals.
                    try:
                        positions = await self._market_data.get_positions()
                        if any(p.symbol == order.symbol and p.quantity > 0 for p in positions):
                            logger.warning(
                                "Order %s BUY %s: not_found + no exec-log "
                                "confirmation; treating as filled because "
                                "position is held (best-effort price)",
                                order_id, order.symbol,
                            )
                            result = OrderResult(
                                order_id=result.order_id,
                                symbol=result.symbol,
                                side=result.side,
                                order_type=result.order_type,
                                quantity=result.quantity,
                                status="filled",
                                filled_price=order.price,
                                filled_quantity=order.quantity,
                            )
                    except Exception:
                        pass
                elif order.side == "SELL":
                    # FIN-H4: deliberately do NOT promote SELL not_found
                    # based on "position no longer held" alone — that
                    # was the manual-sell fabrication vector. Without
                    # an executed-orders match, leave the status as
                    # not_found so it bubbles up as unresolved. Operator
                    # then sees the stuck order in the dashboard and
                    # can manually reconcile.
                    logger.warning(
                        "Order %s SELL %s: not_found and no exec-log "
                        "confirmation — leaving as not_found (was the "
                        "manual-sell fabrication vector pre-FIN-H4)",
                        order_id, order.symbol,
                    )

                # Preserve any existing fill data on the ManagedOrder
                if result.filled_price is None and order.filled_price is not None:
                    result = OrderResult(
                        order_id=result.order_id,
                        symbol=result.symbol,
                        side=result.side,
                        order_type=result.order_type,
                        quantity=result.quantity,
                        status=result.status,
                        filled_price=order.filled_price,
                        filled_quantity=order.filled_quantity,
                    )

            order.status = result.status
            order.filled_price = result.filled_price
            new_filled = int(result.filled_quantity) if result.filled_quantity else 0
            # FIN-H3 (2026-06-05): emit a change record only when the
            # status flips AND new cumulative fill > already-processed.
            # Without this, the same fill could be handed to
            # handle_sell_fill twice on overlapping reconcile cycles —
            # position_tracker.quantity -= fill_qty would double-
            # decrement and the next SELL would go negative.
            delta_filled = max(0, new_filled - order.processed_filled_quantity)
            order.filled_quantity = new_filled
            if result.filled_price and order.price:
                order.slippage = result.filled_price - order.price

            if old_status != result.status and (delta_filled > 0 or new_filled == 0):
                changes.append(
                    {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "old_status": old_status,
                        "new_status": result.status,
                        # legacy: cumulative (kept so existing callers
                        # that subtract delta on the assumption "this
                        # is the only fill" still see the right value
                        # on a single-tranche fill)
                        "filled_quantity": order.filled_quantity,
                        # FIN-H3: new delta semantic — additional shares
                        # filled since last processed. Downstream code
                        # can use this to avoid double-decrementing.
                        "filled_quantity_delta": delta_filled,
                        "filled_price": order.filled_price,
                        "quantity": order.quantity,
                        "price": order.price,
                        "strategy": order.strategy_name,
                        "market": getattr(order, "exchange", "NASD"),
                    }
                )
                # Mark this much fill as handed off. If the next reconcile
                # cycle sees the same cumulative value, delta will be 0
                # and no duplicate change record is emitted.
                order.processed_filled_quantity = new_filled
                logger.info(
                    "Order %s (%s %s): %s -> %s (cumulative_filled=%d/%d, "
                    "delta=%d)",
                    order_id,
                    order.side,
                    order.symbol,
                    old_status,
                    result.status,
                    order.filled_quantity,
                    order.quantity,
                    delta_filled,
                )
                if result.status == "filled":
                    has_new_fill = True

        # Invalidate balance/positions cache when fills detected so next
        # dashboard fetch returns fresh data instead of stale cached values
        if has_new_fill and self._market_data:
            self._market_data.invalidate_balance_cache()

        # Clean up completed orders
        self.clear_completed()
        return changes

    async def cancel_stale_orders(self, ttl_minutes: int = 15) -> list[dict]:
        """Cancel orders that have been pending/open longer than ttl_minutes.

        Returns list of cancelled order info dicts for notification.
        """
        if not self._active_orders or ttl_minutes <= 0:
            return []

        now = now_utc_naive()
        cutoff = now - timedelta(minutes=ttl_minutes)
        cancelled = []

        for oid, order in list(self._active_orders.items()):
            if order.status not in ("pending", "submitted", "open"):
                continue
            if not order.created_at:
                continue
            try:
                created = datetime.fromisoformat(order.created_at)
            except (ValueError, TypeError):
                continue
            if created >= cutoff:
                continue

            # Order is stale — cancel it
            age_min = (now - created).total_seconds() / 60
            success = await self.cancel(oid, order.symbol)
            if success:
                logger.info(
                    "Stale order cancelled: %s %s %s %d shares @ %.0f (age=%.0fmin, ttl=%dmin)",
                    oid,
                    order.side,
                    order.symbol,
                    order.quantity,
                    order.price or 0,
                    age_min,
                    ttl_minutes,
                )
                cancelled.append(
                    {
                        "order_id": oid,
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "price": order.price,
                        "strategy": order.strategy_name,
                        "age_min": round(age_min, 1),
                    }
                )
            else:
                # STOCK-78: Force-remove if order is very old (>3x TTL)
                # to prevent permanent dedup blocking
                if age_min > ttl_minutes * 3:
                    order.status = "cancelled"
                    logger.warning(
                        "Force-cancelled stuck order %s (%s %s, age=%.0fmin): "
                        "cancel API failed but removing from active to unblock",
                        oid,
                        order.side,
                        order.symbol,
                        age_min,
                    )
                else:
                    logger.warning(
                        "Failed to cancel stale order %s (%s %s)",
                        oid,
                        order.side,
                        order.symbol,
                    )

        return cancelled

    @property
    def active_orders(self) -> dict[str, ManagedOrder]:
        return dict(self._active_orders)

    @property
    def position_check_failures(self) -> int:
        """Number of buy rejections caused by position-check API failures."""
        return self._position_check_failures

    def clear_completed(self) -> None:
        """Remove completed/cancelled orders from tracking."""
        to_remove = [
            oid for oid, o in self._active_orders.items() if o.status in ("filled", "cancelled")
        ]
        for oid in to_remove:
            del self._active_orders[oid]
