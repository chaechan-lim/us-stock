"""Trade and order persistence using SQLAlchemy async."""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select, desc
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
        exchange: str = "NASD",
        market: str = "US",
        session: str = "regular",
        is_paper: bool = False,
    ) -> Order:
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
        symbol: str | None = None,
        exclude_paper: bool = False,
    ) -> list[Order]:
        stmt = select(Order).order_by(desc(Order.created_at)).limit(limit)
        if symbol:
            stmt = stmt.where(Order.symbol == symbol)
        if exclude_paper:
            stmt = stmt.where(Order.is_paper == False)  # noqa: E712
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
        """Get filled trades from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        stmt = (
            select(Order)
            .where(Order.status == "filled", Order.filled_at >= cutoff)
            .order_by(desc(Order.filled_at))
        )
        if exclude_paper:
            stmt = stmt.where(Order.is_paper == False)  # noqa: E712
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

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
