"""Portfolio state tracker with DB snapshots.

Tracks balance, positions, equity, and PnL using cached market data.
Saves periodic snapshots to the portfolio_snapshots table for
equity curve tracking and daily PnL calculation.
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models import PortfolioSnapshot
from data.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

# Anomaly detection: skip snapshot if total_value drops more than this
# fraction vs the previous snapshot (e.g. 0.5 = 50% drop).
ANOMALY_DROP_THRESHOLD = 0.5


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
        total_equity = balance.total  # already includes position market value

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
        """Save current portfolio state to portfolio_snapshots table.

        Includes anomaly detection: if total_value drops more than
        ANOMALY_DROP_THRESHOLD (50%) vs the previous snapshot, the
        snapshot is skipped and a warning is logged.  This guards
        against timing issues where balance.total does not yet include
        position market value (STOCK-45).
        """
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()

        invested = sum(p.quantity * p.avg_price for p in positions)
        position_market_value = sum(
            p.quantity * p.current_price for p in positions if p.current_price > 0
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_equity = balance.total  # already includes position market value

        # STOCK-45: Detect when balance.total excludes position value.
        # If balance.total < cash + 50% of position market value,
        # positions are likely missing from the total.
        cash_plus_half_pos = balance.available + position_market_value * 0.5
        if position_market_value > 0 and total_equity < cash_plus_half_pos:
            logger.warning(
                "[%s] Snapshot anomaly: total_equity=%.2f < cash=%.2f + "
                "50%% position_value=%.2f — positions may be excluded from "
                "balance.total. Skipping snapshot.",
                self._market,
                total_equity,
                balance.available,
                position_market_value,
            )
            return

        # STOCK-45: Compare with previous snapshot — skip on anomalous drop.
        prev = await self._get_last_snapshot()
        if prev is not None and prev.total_value_usd > 0:
            drop_ratio = 1.0 - total_equity / prev.total_value_usd
            if drop_ratio > ANOMALY_DROP_THRESHOLD:
                logger.warning(
                    "[%s] Snapshot anomaly: total_equity=%.2f vs "
                    "previous=%.2f (%.1f%% drop). Skipping snapshot.",
                    self._market,
                    total_equity,
                    prev.total_value_usd,
                    drop_ratio * 100,
                )
                return

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
            total_equity,
            balance.available,
            daily_pnl or 0.0,
        )

    async def _get_last_snapshot(self) -> PortfolioSnapshot | None:
        """Fetch the most recent snapshot for this market."""
        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(desc(PortfolioSnapshot.recorded_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def delete_snapshots_by_ids(self, ids: list[int]) -> int:
        """Delete snapshots by ID list. Returns count of deleted rows.

        Admin utility for correcting bad snapshots (e.g. STOCK-45
        anomalous data from timing issues).
        """
        if not ids:
            return 0

        async with self._session_factory() as session:
            stmt = (
                delete(PortfolioSnapshot)
                .where(PortfolioSnapshot.id.in_(ids))
                .where(PortfolioSnapshot.market == self._market)
            )
            result = await session.execute(stmt)
            await session.commit()
            deleted = result.rowcount

        logger.info(
            "[%s] Deleted %d anomalous snapshots (ids=%s)",
            self._market,
            deleted,
            ids,
        )
        return deleted

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
        current_equity = balance.total  # already includes position market value

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
