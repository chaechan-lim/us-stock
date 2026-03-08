"""Tests for PortfolioManager."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.models import Base, PortfolioSnapshot
from engine.portfolio_manager import PortfolioManager
from exchange.base import Balance, Position


@pytest_asyncio.fixture
async def db_setup():
    """Create in-memory SQLite engine and session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def mock_market_data():
    svc = AsyncMock()
    svc.get_balance = AsyncMock(return_value=Balance(
        currency="USD", total=100_000, available=80_000,
    ))
    svc.get_positions = AsyncMock(return_value=[
        Position(
            symbol="AAPL", exchange="NASD",
            quantity=10, avg_price=150.0,
            current_price=160.0,
            unrealized_pnl=100.0, unrealized_pnl_pct=6.67,
        ),
        Position(
            symbol="MSFT", exchange="NASD",
            quantity=5, avg_price=400.0,
            current_price=410.0,
            unrealized_pnl=50.0, unrealized_pnl_pct=2.5,
        ),
    ])
    return svc


@pytest.fixture
def manager(mock_market_data, db_setup):
    return PortfolioManager(market_data=mock_market_data, session_factory=db_setup)


class TestGetSummary:
    async def test_returns_correct_structure(self, manager):
        summary = await manager.get_summary()

        assert "cash" in summary
        assert "invested" in summary
        assert "total_equity" in summary
        assert "unrealized_pnl" in summary
        assert "position_count" in summary
        assert "positions" in summary

    async def test_calculates_values_correctly(self, manager):
        summary = await manager.get_summary()

        assert summary["cash"] == 80_000
        # invested = 10*150 + 5*400 = 1500 + 2000 = 3500
        assert summary["invested"] == 3500.0
        # unrealized_pnl = 100 + 50 = 150
        assert summary["unrealized_pnl"] == 150.0
        # total_equity = 100_000 + 150 = 100_150
        assert summary["total_equity"] == 100_150.0
        assert summary["position_count"] == 2
        assert len(summary["positions"]) == 2

    async def test_empty_positions(self, mock_market_data, db_setup):
        mock_market_data.get_positions.return_value = []
        mgr = PortfolioManager(market_data=mock_market_data, session_factory=db_setup)
        summary = await mgr.get_summary()

        assert summary["invested"] == 0
        assert summary["unrealized_pnl"] == 0
        assert summary["total_equity"] == 100_000
        assert summary["position_count"] == 0


class TestSaveSnapshot:
    async def test_persists_to_db(self, manager, db_setup):
        await manager.save_snapshot()

        async with db_setup() as session:
            from sqlalchemy import select
            stmt = select(PortfolioSnapshot)
            result = await session.execute(stmt)
            snapshots = result.scalars().all()

        assert len(snapshots) == 1
        s = snapshots[0]
        assert s.total_value_usd == 100_150.0
        assert s.cash_usd == 80_000
        assert s.invested_usd == 3500.0
        assert s.unrealized_pnl == 150.0

    async def test_multiple_snapshots(self, manager, db_setup):
        await manager.save_snapshot()
        await manager.save_snapshot()

        async with db_setup() as session:
            from sqlalchemy import select, func
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            count = result.scalar()

        assert count == 2


class TestGetDailyPnl:
    async def test_no_previous_snapshot(self, manager):
        pnl = await manager.get_daily_pnl()
        assert pnl == 0.0

    async def test_calculates_from_first_snapshot(self, manager, db_setup):
        async with db_setup() as session:
            snapshot = PortfolioSnapshot(
                total_value_usd=99_000.0,
                cash_usd=79_000.0,
                invested_usd=3000.0,
                unrealized_pnl=100.0,
                recorded_at=datetime.utcnow(),
            )
            session.add(snapshot)
            await session.commit()

        # Current equity = 100_000 + 150 = 100_150
        pnl = await manager.get_daily_pnl()
        assert pnl == pytest.approx(100_150.0 - 99_000.0)


class TestGetEquityHistory:
    async def test_returns_ordered_data(self, manager, db_setup):
        now = datetime.utcnow()
        async with db_setup() as session:
            for i in range(5):
                session.add(PortfolioSnapshot(
                    total_value_usd=100_000 + i * 100,
                    cash_usd=80_000,
                    invested_usd=3500.0,
                    unrealized_pnl=float(i * 100),
                    recorded_at=now - timedelta(days=4 - i),
                ))
            await session.commit()

        history = await manager.get_equity_history(days=30)

        assert len(history) == 5
        # Check ascending order
        values = [h["total_value_usd"] for h in history]
        assert values == sorted(values)

    async def test_respects_days_parameter(self, manager, db_setup):
        now = datetime.utcnow()
        async with db_setup() as session:
            # Add snapshot from 60 days ago (should be excluded with days=30)
            session.add(PortfolioSnapshot(
                total_value_usd=95_000,
                cash_usd=75_000,
                invested_usd=3000.0,
                unrealized_pnl=0.0,
                recorded_at=now - timedelta(days=60),
            ))
            # Add snapshot from 5 days ago (should be included)
            session.add(PortfolioSnapshot(
                total_value_usd=100_000,
                cash_usd=80_000,
                invested_usd=3500.0,
                unrealized_pnl=150.0,
                recorded_at=now - timedelta(days=5),
            ))
            await session.commit()

        history = await manager.get_equity_history(days=30)
        assert len(history) == 1
        assert history[0]["total_value_usd"] == 100_000

    async def test_empty_history(self, manager):
        history = await manager.get_equity_history(days=30)
        assert history == []
