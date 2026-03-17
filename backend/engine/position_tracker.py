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
from typing import TYPE_CHECKING

from data.market_data_service import MarketDataService
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from exchange.base import ExchangeAdapter

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
    trailing_activation_pct: float = 0.0  # disabled: cuts winners short
    trailing_stop_pct: float = 0.0
    tracked_at: float = field(default_factory=time.monotonic)


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
    ):
        self._adapter = adapter
        self._market_data = market_data
        self._risk = risk_manager
        self._orders = order_manager
        self._notification = notification
        self._event_calendar = event_calendar
        self._session_factory = session_factory
        self._market = market
        self._tracked: dict[str, TrackedPosition] = {}
        self._last_auto_recover: float = 0.0

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
        self._schedule_db_upsert(symbol)

    def untrack(self, symbol: str) -> None:
        """Stop tracking a position."""
        self._tracked.pop(symbol, None)
        self._schedule_db_remove(symbol)

    async def check_all(self, session: str = "regular") -> list[dict]:
        """Check all tracked positions. Returns list of triggered actions.

        Defense-in-depth: if the in-memory tracker is empty but the exchange
        reports open positions, auto-recover them so SL/TP monitoring is not
        silently disabled.  Recovery is rate-limited to avoid hammering the
        exchange/DB on every 60-second check cycle.

        Args:
            session: Trading session for sell execution (regular/pre_market/after_hours).
                     Extended hours sells use limit orders only.
        """
        try:
            if self._market_data:
                positions = await self._market_data.get_positions()
            else:
                positions = await self._adapter.fetch_positions()
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

        # Defense-in-depth: auto-recover untracked exchange positions
        if positions:
            await self._auto_recover_untracked(positions)

        if not self._tracked:
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

            # Update highest price for trailing stop
            if current_price > tracked.highest_price:
                tracked.highest_price = current_price

            action = self._evaluate(tracked, current_price)
            if action:
                triggered.append(action)
                await self._execute_sell(tracked, current_price, action["reason"], session=session)

        return triggered

    def _evaluate(self, tracked: TrackedPosition, current_price: float) -> dict | None:
        """Evaluate if any exit condition is met."""
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
        self,
        tracked: TrackedPosition,
        price: float,
        reason: str,
        session: str = "regular",
    ) -> None:
        """Execute a sell order and notify."""
        session_tag = f" [{session}]" if session != "regular" else ""
        logger.warning(
            "%s triggered for %s%s: entry=$%.2f current=$%.2f",
            reason.upper(),
            tracked.symbol,
            session_tag,
            tracked.entry_price,
            price,
        )

        # Extended hours: force limit order (market orders not supported)
        order_type = "limit" if session != "regular" else "market"

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
            # Use actual filled quantity for PnL (handles partial fills)
            fill_qty = order.filled_quantity or tracked.quantity
            fill_price = order.filled_price or price
            pnl = (fill_price - tracked.entry_price) * fill_qty
            self._risk.update_daily_pnl(pnl)
            self.untrack(tracked.symbol)

            if self._notification:
                try:
                    if reason == "stop_loss":
                        await self._notification.notify_stop_loss(
                            tracked.symbol,
                            tracked.quantity,
                            tracked.entry_price,
                            price,
                            pnl,
                        )
                    elif reason == "take_profit":
                        await self._notification.notify_take_profit(
                            tracked.symbol,
                            tracked.quantity,
                            tracked.entry_price,
                            price,
                            pnl,
                        )
                    elif reason == "trailing_stop":
                        await self._notification.notify_trailing_stop(
                            tracked.symbol,
                            tracked.quantity,
                            tracked.entry_price,
                            price,
                            tracked.highest_price,
                            pnl,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to send %s notification for %s: %s",
                        reason,
                        tracked.symbol,
                        e,
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

        # Look up latest live BUY order per symbol from DB to get entry info
        # (paper orders excluded to prevent position quantity distortion)
        entry_info: dict[str, dict] = {}
        if sf:
            try:
                from sqlalchemy import desc, select

                from core.models import Order

                async with sf() as session:
                    for pos in positions:
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
                        if order:
                            entry_info[pos.symbol] = {
                                "strategy": order.strategy_name or "",
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

    async def restore_from_db(self, session_factory=None) -> int:
        """Restore tracked positions from the positions DB table.

        Fallback for when ``restore_from_exchange()`` fails or is incomplete.
        Only restores positions not already tracked in memory.
        Returns the number of positions restored.
        """
        sf = session_factory or self._session_factory
        if not sf:
            return 0

        try:
            from sqlalchemy import select

            from core.models import PositionRecord

            async with sf() as session:
                stmt = select(PositionRecord).where(
                    PositionRecord.market == self._market,
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

            if not records:
                return 0

            restored = 0
            for record in records:
                if record.symbol in self._tracked:
                    continue
                if record.quantity is not None and record.quantity <= 0:
                    continue

                self.track(
                    symbol=record.symbol,
                    entry_price=record.avg_price or 0.0,
                    quantity=int(record.quantity or 0),
                    strategy=record.strategy_name or "db_restored",
                    stop_loss_pct=record.stop_loss,
                    take_profit_pct=record.take_profit,
                )
                restored += 1

            if restored:
                logger.info(
                    "Restored %d %s positions from DB: %s",
                    restored,
                    self._market,
                    ", ".join(r.symbol for r in records),
                )
            return restored

        except Exception as e:
            logger.error("Failed to restore positions from DB: %s", e)
            return 0

    async def _auto_recover_untracked(self, exchange_positions: list) -> None:
        """Auto-track exchange positions missing from the in-memory tracker.

        Defense-in-depth: ensures SL/TP monitoring is not silently disabled
        when the tracker is empty (e.g. after a failed restore or restart).

        Recovery is rate-limited to ``_AUTO_RECOVER_COOLDOWN`` seconds to
        avoid hammering the DB/exchange every check cycle.
        """
        untracked = [
            p for p in exchange_positions if p.quantity > 0 and p.symbol not in self._tracked
        ]
        if not untracked:
            return

        # Rate-limit recovery attempts
        now = time.monotonic()
        if now - self._last_auto_recover < self._AUTO_RECOVER_COOLDOWN:
            return
        self._last_auto_recover = now

        logger.warning(
            "Auto-recovering %d untracked %s exchange positions: %s",
            len(untracked),
            self._market,
            ", ".join(p.symbol for p in untracked),
        )

        # Try loading entry info from positions DB table first
        db_info = await self._load_positions_from_db()

        for pos in untracked:
            db_record = db_info.get(pos.symbol)
            if db_record:
                # Use DB info (preserves SL/TP, strategy from previous session)
                self.track(
                    symbol=pos.symbol,
                    entry_price=db_record.get("avg_price", pos.avg_price),
                    quantity=int(pos.quantity),
                    strategy=db_record.get("strategy", "auto_recovered"),
                    stop_loss_pct=db_record.get("stop_loss"),
                    take_profit_pct=db_record.get("take_profit"),
                )
            else:
                # Fall back to exchange info with default SL/TP
                self.track(
                    symbol=pos.symbol,
                    entry_price=pos.avg_price,
                    quantity=int(pos.quantity),
                    strategy="auto_recovered",
                    stop_loss_pct=self._risk.params.default_stop_loss_pct,
                    take_profit_pct=self._risk.params.default_take_profit_pct,
                )

    async def _load_positions_from_db(self) -> dict[str, dict]:
        """Load position records from DB for this market.

        Returns a dict mapping symbol -> {avg_price, stop_loss, take_profit, strategy}.
        """
        sf = self._session_factory
        if not sf:
            return {}

        try:
            from sqlalchemy import select

            from core.models import PositionRecord

            async with sf() as session:
                stmt = select(PositionRecord).where(
                    PositionRecord.market == self._market,
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

            return {
                r.symbol: {
                    "avg_price": r.avg_price,
                    "stop_loss": r.stop_loss,
                    "take_profit": r.take_profit,
                    "strategy": r.strategy_name or "",
                }
                for r in records
            }

        except Exception as e:
            logger.debug("Failed to load positions from DB: %s", e)
            return {}

    async def sync_to_db(self, session_factory=None) -> int:
        """Synchronize all in-memory tracked positions to the positions DB table.

        Upserts current tracked positions and removes stale DB rows for
        positions no longer tracked. Returns the number of positions synced.
        """
        sf = session_factory or self._session_factory
        if not sf:
            return 0

        try:
            from sqlalchemy import select

            from core.models import PositionRecord

            async with sf() as session:
                tracked_symbols = set(self._tracked.keys())

                # Remove DB rows for positions no longer tracked in this market
                stmt = select(PositionRecord).where(
                    PositionRecord.market == self._market,
                )
                result = await session.execute(stmt)
                db_positions = result.scalars().all()

                for db_pos in db_positions:
                    if db_pos.symbol not in tracked_symbols:
                        await session.delete(db_pos)

                # Upsert all currently tracked positions
                for symbol, tracked in self._tracked.items():
                    await self._upsert_position_record(
                        session,
                        symbol,
                        tracked,
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
                    PositionRecord.market == self._market,
                    PositionRecord.symbol == symbol,
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.debug("DB remove failed for %s: %s", symbol, e)

    async def _upsert_position_record(
        self,
        session: "AsyncSession",
        symbol: str,
        tracked: TrackedPosition,
    ) -> None:
        """Upsert a PositionRecord within an existing session (no commit)."""
        from sqlalchemy import select

        from core.models import PositionRecord

        stmt = select(PositionRecord).where(
            PositionRecord.market == self._market,
            PositionRecord.symbol == symbol,
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            record.quantity = tracked.quantity
            record.avg_price = tracked.entry_price
            record.stop_loss = tracked.stop_loss_pct
            record.take_profit = tracked.take_profit_pct
            record.trailing_stop = tracked.trailing_stop_pct
            record.strategy_name = tracked.strategy
            record.updated_at = datetime.utcnow()
        else:
            record = PositionRecord(
                market=self._market,
                symbol=symbol,
                exchange=self._resolve_exchange(symbol),
                quantity=tracked.quantity,
                avg_price=tracked.entry_price,
                stop_loss=tracked.stop_loss_pct,
                take_profit=tracked.take_profit_pct,
                trailing_stop=tracked.trailing_stop_pct,
                strategy_name=tracked.strategy,
                opened_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(record)

    async def _clear_all_positions_db(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        """Remove all positions for this market from DB."""
        try:
            from sqlalchemy import delete

            from core.models import PositionRecord

            async with session_factory() as session:
                stmt = delete(PositionRecord).where(
                    PositionRecord.market == self._market,
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.debug("Failed to clear DB positions for %s: %s", self._market, e)

    def _resolve_exchange(self, symbol: str) -> str:
        """Resolve exchange code from symbol."""
        if self._market == "KR":
            return "KRX"
        # US: default NASD, could be improved with exchange resolver
        return "NASD"
