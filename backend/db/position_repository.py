"""Position persistence using SQLAlchemy async.

Provides CRUD operations for the positions table so that
PositionTracker state survives server restarts without relying
solely on the exchange API.
"""

import logging
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import PositionRecord

logger = logging.getLogger(__name__)


class PositionRepository:
    """CRUD operations for the positions table."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert_position(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        current_price: float | None = None,
        unrealized_pnl: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop: float | None = None,
        strategy_name: str = "",
        exchange: str = "NASD",
        market: str = "US",
    ) -> PositionRecord:
        """Insert or update a position record."""
        stmt = select(PositionRecord).where(
            PositionRecord.symbol == symbol,
            PositionRecord.market == market,
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.quantity = quantity
            existing.avg_price = avg_price
            existing.current_price = current_price
            existing.unrealized_pnl = unrealized_pnl
            existing.stop_loss = stop_loss
            existing.take_profit = take_profit
            existing.trailing_stop = trailing_stop
            existing.strategy_name = strategy_name or existing.strategy_name
            existing.exchange = exchange
            existing.updated_at = datetime.utcnow()
            await self._session.commit()
            return existing

        record = PositionRecord(
            market=market,
            symbol=symbol,
            exchange=exchange,
            quantity=quantity,
            avg_price=avg_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            strategy_name=strategy_name,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def remove_position(self, symbol: str, market: str = "US") -> bool:
        """Remove a position record by symbol and market."""
        stmt = select(PositionRecord).where(
            PositionRecord.symbol == symbol,
            PositionRecord.market == market,
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            await self._session.delete(existing)
            await self._session.commit()
            return True
        return False

    async def get_all_positions(self, market: str | None = None) -> list[PositionRecord]:
        """Get all position records, optionally filtered by market."""
        stmt = select(PositionRecord)
        if market:
            stmt = stmt.where(PositionRecord.market == market)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def remove_all(self, market: str | None = None) -> int:
        """Remove all position records, optionally filtered by market.

        Returns the number of records deleted.
        """
        stmt = delete(PositionRecord)
        if market:
            stmt = stmt.where(PositionRecord.market == market)
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount
