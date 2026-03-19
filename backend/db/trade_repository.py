"""Trade and order persistence using SQLAlchemy async."""

import logging
from datetime import datetime, timedelta

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import Order, Watchlist

logger = logging.getLogger(__name__)


class TradeRepository:
    """CRUD operations for trades/orders and watchlist."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # --- Orders / Trade History ---

    async def save_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        filled_quantity: float = 0,
        filled_price: float | None = None,
        status: str = "pending",
        strategy_name: str = "",
        buy_strategy: str = "",
        kis_order_id: str = "",
        pnl: float | None = None,
        pnl_pct: float | None = None,
        exchange: str = "NASD",
        market: str = "US",
        session: str = "regular",
        is_paper: bool = False,
    ) -> Order:
        # UPSERT: if kis_order_id already exists, update instead of inserting
        # Prevents duplicate rows when both order placement and reconciliation
        # call save_order() for the same KIS order
        if kis_order_id:
            existing = await self.find_by_kis_order_id(kis_order_id)
            if existing:
                # Update with newer/more complete data
                if filled_quantity and filled_quantity > (existing.filled_quantity or 0):
                    existing.filled_quantity = filled_quantity
                if filled_price is not None:
                    existing.filled_price = filled_price
                if status == "filled":
                    existing.status = status
                    if not existing.filled_at:
                        existing.filled_at = datetime.utcnow()
                elif status and existing.status != "filled":
                    # STOCK-37: Don't set "not_found" when PnL exists —
                    # the order was actually filled, KIS API just can't find it.
                    if status == "not_found" and existing.pnl is not None:
                        existing.status = "filled"
                        if not existing.filled_at:
                            existing.filled_at = existing.created_at or datetime.utcnow()
                    else:
                        existing.status = status
                if pnl is not None:
                    existing.pnl = pnl
                if pnl_pct is not None:
                    existing.pnl_pct = pnl_pct
                if buy_strategy:
                    existing.buy_strategy = buy_strategy
                await self._session.commit()
                await self._session.refresh(existing)
                logger.debug(
                    "Updated existing order kis_id=%s (id=%d) instead of inserting duplicate",
                    kis_order_id,
                    existing.id,
                )
                return existing

        order = Order(
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            status=status,
            strategy_name=strategy_name,
            buy_strategy=buy_strategy or None,
            kis_order_id=kis_order_id,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_paper=is_paper,
            market=market,
            session=session,
        )
        self._session.add(order)
        await self._session.commit()
        await self._session.refresh(order)
        return order

    async def update_order_status(
        self,
        order_id: int,
        status: str,
        filled_price: float | None = None,
        filled_quantity: float | None = None,
        pnl: float | None = None,
    ) -> Order | None:
        result = await self._session.get(Order, order_id)
        if result is None:
            return None
        result.status = status
        if filled_price is not None:
            result.filled_price = filled_price
        if filled_quantity is not None:
            result.filled_quantity = filled_quantity
        if pnl is not None:
            result.pnl = pnl
        if status == "filled":
            result.filled_at = datetime.utcnow()
        await self._session.commit()
        return result

    async def get_trade_history(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: str | None = None,
        exclude_paper: bool = False,
    ) -> list[Order]:
        stmt = select(Order).order_by(desc(Order.created_at))
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        if exclude_paper:
            stmt = stmt.where(Order.is_paper == False)  # noqa: E712
        if offset > 0:
            stmt = stmt.offset(offset)
        stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_open_orders(self, exclude_paper: bool = False) -> list[Order]:
        stmt = select(Order).where(Order.status.in_(["pending", "open", "submitted", "not_found"]))
        if exclude_paper:
            stmt = stmt.where(Order.is_paper == False)  # noqa: E712
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_kis_order_id(self, kis_order_id: str) -> Order | None:
        """Find an order by KIS exchange order ID."""
        if not kis_order_id:
            return None
        stmt = select(Order).where(Order.kis_order_id == kis_order_id).limit(1)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent_trades(
        self,
        hours: int = 24,
        exclude_paper: bool = False,
    ) -> list[Order]:
        """Get filled trades from the last N hours.

        STOCK-37: Also includes 'not_found' orders that have PnL, since these
        were actually filled but KIS API couldn't locate them during
        reconciliation (date boundary / API delay).
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        stmt = (
            select(Order)
            .where(
                or_(
                    # Normal filled orders with filled_at timestamp
                    and_(Order.status == "filled", Order.filled_at >= cutoff),
                    # STOCK-37: not_found orders with PnL — use created_at as
                    # fallback since filled_at is NULL for these
                    and_(
                        Order.status == "not_found",
                        Order.pnl.isnot(None),
                        Order.created_at >= cutoff,
                    ),
                )
            )
            .order_by(desc(Order.filled_at.isnot(None)), desc(Order.created_at))
        )
        if exclude_paper:
            stmt = stmt.where(Order.is_paper == False)  # noqa: E712
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def cleanup_duplicate_orders(self) -> int:
        """Remove duplicate orders with the same kis_order_id.

        Keeps the row with the most complete data (filled_at set, or highest id).
        Returns count of deleted duplicate rows.
        """
        from sqlalchemy import func

        # Find kis_order_ids that appear more than once
        stmt = (
            select(Order.kis_order_id, func.count(Order.id).label("cnt"))
            .where(Order.kis_order_id.isnot(None), Order.kis_order_id != "")
            .group_by(Order.kis_order_id)
            .having(func.count(Order.id) > 1)
        )
        result = await self._session.execute(stmt)
        duplicates = result.all()

        deleted = 0
        for kis_id, _count in duplicates:
            # Get all orders with this kis_order_id, ordered to keep the best one
            dup_stmt = (
                select(Order)
                .where(Order.kis_order_id == kis_id)
                .order_by(
                    # Prefer: filled_at set > higher filled_quantity > higher id
                    desc(Order.filled_at.isnot(None)),
                    desc(Order.filled_quantity),
                    desc(Order.id),
                )
            )
            dup_result = await self._session.execute(dup_stmt)
            orders = list(dup_result.scalars().all())

            # Keep first (best), delete rest
            for dup_order in orders[1:]:
                await self._session.delete(dup_order)
                deleted += 1
                logger.info(
                    "Deleted duplicate order id=%d kis_id=%s symbol=%s",
                    dup_order.id,
                    kis_id,
                    dup_order.symbol,
                )

        if deleted:
            await self._session.commit()
            logger.info("Cleaned up %d duplicate orders", deleted)
        return deleted

    # --- Watchlist ---

    async def get_watchlist(
        self,
        active_only: bool = True,
        market: str | None = None,
    ) -> list[Watchlist]:
        stmt = select(Watchlist).order_by(Watchlist.added_at)
        if active_only:
            stmt = stmt.where(Watchlist.is_active == True)
        if market:
            stmt = stmt.where(Watchlist.market == market)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add_to_watchlist(
        self,
        symbol: str,
        exchange: str = "NASD",
        name: str | None = None,
        sector: str | None = None,
        source: str = "manual",
        market: str = "US",
    ) -> Watchlist:
        # Check if already exists (within same market)
        stmt = select(Watchlist).where(
            Watchlist.symbol == symbol,
            Watchlist.market == market,
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.is_active = True
            existing.updated_at = datetime.utcnow()
            await self._session.commit()
            return existing

        item = Watchlist(
            symbol=symbol,
            exchange=exchange,
            market=market,
            name=name,
            sector=sector,
            source=source,
            is_active=True,
        )
        self._session.add(item)
        await self._session.commit()
        await self._session.refresh(item)
        return item

    async def update_watchlist_name(
        self,
        symbol: str,
        name: str,
        market: str = "US",
    ) -> bool:
        """Update the cached name for a watchlist symbol."""
        stmt = select(Watchlist).where(
            Watchlist.symbol == symbol,
            Watchlist.market == market,
        )
        result = await self._session.execute(stmt)
        item = result.scalar_one_or_none()
        if item and name:
            item.name = name
            await self._session.commit()
            return True
        return False

    async def remove_from_watchlist(self, symbol: str, market: str = "US") -> bool:
        stmt = select(Watchlist).where(
            Watchlist.symbol == symbol,
            Watchlist.market == market,
        )
        result = await self._session.execute(stmt)
        item = result.scalar_one_or_none()
        if item:
            item.is_active = False
            await self._session.commit()
            return True
        return False
