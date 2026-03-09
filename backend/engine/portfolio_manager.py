"""Portfolio state tracker with DB snapshots.

Tracks balance, positions, equity, and PnL using cached market data.
Saves periodic snapshots to the portfolio_snapshots table for
equity curve tracking and daily PnL calculation.
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from data.market_data_service import MarketDataService
from core.models import PortfolioSnapshot

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Track portfolio state and persist snapshots to DB."""

    def __init__(
        self,
        market_data: MarketDataService,
        session_factory: async_sessionmaker[AsyncSession],
        market: str = "US",
    ):
        self._market_data = market_data
        self._session_factory = session_factory
        self._market = market

    async def get_summary(self) -> dict:
        """Get current portfolio state: balance, positions, total equity, unrealized PnL."""
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()

        invested = sum(p.quantity * p.avg_price for p in positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_equity = balance.total + unrealized_pnl

        return {
            "cash": balance.available,
            "invested": invested,
            "total_equity": total_equity,
            "unrealized_pnl": unrealized_pnl,
            "position_count": len(positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ],
        }

    async def save_snapshot(self) -> None:
        """Save current portfolio state to portfolio_snapshots table."""
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()

        invested = sum(p.quantity * p.avg_price for p in positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_equity = balance.total + unrealized_pnl

        daily_pnl = await self._calculate_daily_pnl(total_equity)

        snapshot = PortfolioSnapshot(
            market=self._market,
            total_value_usd=total_equity,
            cash_usd=balance.available,
            invested_usd=invested,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            recorded_at=datetime.utcnow(),
        )

        async with self._session_factory() as session:
            session.add(snapshot)
            await session.commit()

        logger.info(
            "Portfolio snapshot saved: equity=%.2f, cash=%.2f, pnl=%.2f",
            total_equity, balance.available, daily_pnl or 0.0,
        )

    async def _calculate_daily_pnl(self, current_equity: float) -> float | None:
        """Calculate PnL vs the first snapshot of today."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.recorded_at >= today_start)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(PortfolioSnapshot.recorded_at.asc())
                .limit(1)
            )
            result = await session.execute(stmt)
            first_today = result.scalar_one_or_none()

        if first_today is None:
            return None

        return current_equity - first_today.total_value_usd

    async def get_daily_pnl(self) -> float:
        """Calculate today's PnL from snapshots."""
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        current_equity = balance.total + unrealized_pnl

        pnl = await self._calculate_daily_pnl(current_equity)
        return pnl if pnl is not None else 0.0

    async def get_equity_history(self, days: int = 30) -> list[dict]:
        """Get equity curve from snapshots."""
        since = datetime.utcnow() - timedelta(days=days)

        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.recorded_at >= since)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(PortfolioSnapshot.recorded_at.asc())
            )
            result = await session.execute(stmt)
            snapshots = result.scalars().all()

        return [
            {
                "date": s.recorded_at.strftime("%Y-%m-%d %H:%M") if s.recorded_at else None,
                "total_value_usd": s.total_value_usd,
                "cash_usd": s.cash_usd,
                "invested_usd": s.invested_usd,
                "unrealized_pnl": s.unrealized_pnl,
                "daily_pnl": s.daily_pnl,
            }
            for s in snapshots
        ]
