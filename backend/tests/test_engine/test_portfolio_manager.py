"""Tests for PortfolioManager."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base, PortfolioSnapshot
from engine.portfolio_manager import ANOMALY_DROP_THRESHOLD, PortfolioManager
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
    svc.get_balance = AsyncMock(
        return_value=Balance(
            currency="USD",
            total=100_000,
            available=80_000,
        )
    )
    svc.get_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=10,
                avg_price=150.0,
                current_price=160.0,
                unrealized_pnl=100.0,
                unrealized_pnl_pct=6.67,
            ),
            Position(
                symbol="MSFT",
                exchange="NASD",
                quantity=5,
                avg_price=400.0,
                current_price=410.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=2.5,
            ),
        ]
    )
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
        # total_equity = balance.total (already includes position market value)
        assert summary["total_equity"] == 100_000
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
            stmt = select(PortfolioSnapshot)
            result = await session.execute(stmt)
            snapshots = result.scalars().all()

        assert len(snapshots) == 1
        s = snapshots[0]
        assert s.total_value_usd == 100_000  # balance.total already includes positions
        assert s.cash_usd == 80_000
        assert s.invested_usd == 3500.0
        assert s.unrealized_pnl == 150.0

    async def test_multiple_snapshots(self, manager, db_setup):
        await manager.save_snapshot()
        await manager.save_snapshot()

        async with db_setup() as session:
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

        # Current equity = balance.total = 100_000 (already includes positions)
        pnl = await manager.get_daily_pnl()
        assert pnl == pytest.approx(100_000.0 - 99_000.0)


class TestGetEquityHistory:
    async def test_returns_ordered_data(self, manager, db_setup):
        now = datetime.utcnow()
        async with db_setup() as session:
            for i in range(5):
                session.add(
                    PortfolioSnapshot(
                        total_value_usd=100_000 + i * 100,
                        cash_usd=80_000,
                        invested_usd=3500.0,
                        unrealized_pnl=float(i * 100),
                        recorded_at=now - timedelta(days=4 - i),
                    )
                )
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
            session.add(
                PortfolioSnapshot(
                    total_value_usd=95_000,
                    cash_usd=75_000,
                    invested_usd=3000.0,
                    unrealized_pnl=0.0,
                    recorded_at=now - timedelta(days=60),
                )
            )
            # Add snapshot from 5 days ago (should be included)
            session.add(
                PortfolioSnapshot(
                    total_value_usd=100_000,
                    cash_usd=80_000,
                    invested_usd=3500.0,
                    unrealized_pnl=150.0,
                    recorded_at=now - timedelta(days=5),
                )
            )
            await session.commit()

        history = await manager.get_equity_history(days=30)
        assert len(history) == 1
        assert history[0]["total_value_usd"] == 100_000

    async def test_empty_history(self, manager):
        history = await manager.get_equity_history(days=30)
        assert history == []


class TestSnapshotAnomalyDetection:
    """STOCK-45: Anomaly detection prevents saving bad snapshots."""

    async def test_skips_when_positions_excluded_from_balance(self, db_setup):
        """When balance.total is lower than cash + position value, skip."""
        # Simulate the STOCK-45 bug: balance.total = cash only (no positions)
        svc = AsyncMock()
        svc.get_balance = AsyncMock(
            return_value=Balance(
                currency="KRW",
                total=4_763_401,
                available=4_763_401,
            )
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="005930",
                    exchange="KRX",
                    quantity=100,
                    avg_price=75_000,
                    current_price=80_000,
                    unrealized_pnl=500_000,
                    unrealized_pnl_pct=6.67,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        # balance.total (4.7M) < cash (4.7M) + 50% * position_value (8M * 0.5 = 4M)
        # → anomaly detected, snapshot skipped
        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 0

    async def test_skips_on_large_drop_vs_previous(self, db_setup):
        """When total_equity drops >50% from previous snapshot, skip."""
        svc = AsyncMock()
        # Previous snapshot had 13.3M, now balance reports 6M (no position mismatch
        # because positions are empty — pure balance drop scenario)
        svc.get_balance = AsyncMock(
            return_value=Balance(
                currency="KRW",
                total=6_000_000,
                available=6_000_000,
            )
        )
        svc.get_positions = AsyncMock(return_value=[])
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        # Seed a previous snapshot with much higher value
        async with db_setup() as session:
            session.add(
                PortfolioSnapshot(
                    market="KR",
                    total_value_usd=13_315_617,
                    cash_usd=5_000_000,
                    invested_usd=8_000_000,
                    unrealized_pnl=315_617,
                    recorded_at=datetime.utcnow() - timedelta(hours=1),
                )
            )
            await session.commit()

        # Drop: 1 - 6M/13.3M ≈ 55% > 50% threshold → skip
        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            # Only the seeded snapshot, no new one added
            assert result.scalar() == 1

    async def test_saves_normally_when_no_anomaly(self, manager, db_setup):
        """Normal case: balance includes positions, no previous → saves fine."""
        await manager.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 1

    async def test_saves_when_drop_below_threshold(self, db_setup):
        """A moderate drop (<50%) should still save."""
        svc = AsyncMock()
        svc.get_balance = AsyncMock(
            return_value=Balance(
                currency="KRW",
                total=8_000_000,
                available=8_000_000,
            )
        )
        svc.get_positions = AsyncMock(return_value=[])
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        # Seed previous at 10M → current 8M = 20% drop < 50% threshold
        async with db_setup() as session:
            session.add(
                PortfolioSnapshot(
                    market="KR",
                    total_value_usd=10_000_000,
                    cash_usd=10_000_000,
                    invested_usd=0,
                    unrealized_pnl=0,
                    recorded_at=datetime.utcnow() - timedelta(hours=1),
                )
            )
            await session.commit()

        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            # Previous + new snapshot = 2
            assert result.scalar() == 2

    async def test_first_snapshot_saves_without_previous(self, db_setup):
        """First ever snapshot (no previous) should save normally."""
        svc = AsyncMock()
        svc.get_balance = AsyncMock(
            return_value=Balance(
                currency="KRW",
                total=13_000_000,
                available=5_000_000,
            )
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="005930",
                    exchange="KRX",
                    quantity=100,
                    avg_price=75_000,
                    current_price=80_000,
                    unrealized_pnl=500_000,
                    unrealized_pnl_pct=6.67,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 1

    async def test_position_cross_check_passes_when_total_includes_positions(self, db_setup):
        """When balance.total properly includes position value, save normally."""
        svc = AsyncMock()
        # total (13M) > cash (5M) + 50% * positions (8M * 0.5 = 4M) → OK
        svc.get_balance = AsyncMock(
            return_value=Balance(
                currency="KRW",
                total=13_000_000,
                available=5_000_000,
            )
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="005930",
                    exchange="KRX",
                    quantity=100,
                    avg_price=75_000,
                    current_price=80_000,
                    unrealized_pnl=500_000,
                    unrealized_pnl_pct=6.67,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 1

    async def test_anomaly_threshold_constant(self):
        """Verify threshold is set to 50%."""
        assert ANOMALY_DROP_THRESHOLD == 0.5


class TestDeleteSnapshotsByIds:
    """STOCK-45: Admin method to delete bad snapshots."""

    async def test_deletes_matching_ids(self, db_setup):
        """Delete specific snapshots by ID."""
        svc = AsyncMock()
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        # Seed 5 snapshots
        async with db_setup() as session:
            for i in range(5):
                session.add(
                    PortfolioSnapshot(
                        market="KR",
                        total_value_usd=float(10_000_000 + i * 100_000),
                        cash_usd=5_000_000,
                        invested_usd=5_000_000,
                        unrealized_pnl=0,
                        recorded_at=datetime.utcnow() - timedelta(hours=5 - i),
                    )
                )
            await session.commit()

        # Fetch actual IDs
        async with db_setup() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.id)
            result = await session.execute(stmt)
            all_ids = [s.id for s in result.scalars().all()]

        # Delete 3 of 5
        delete_ids = all_ids[1:4]  # ids at index 1,2,3
        deleted = await mgr.delete_snapshots_by_ids(delete_ids)

        assert deleted == 3

        # Verify only 2 remain
        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 2

    async def test_only_deletes_own_market(self, db_setup):
        """delete_snapshots_by_ids only deletes snapshots for its market."""
        svc = AsyncMock()
        kr_mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        # Seed US + KR snapshots
        async with db_setup() as session:
            session.add(
                PortfolioSnapshot(
                    market="US",
                    total_value_usd=50_000,
                    cash_usd=30_000,
                    invested_usd=20_000,
                    unrealized_pnl=0,
                    recorded_at=datetime.utcnow(),
                )
            )
            session.add(
                PortfolioSnapshot(
                    market="KR",
                    total_value_usd=10_000_000,
                    cash_usd=5_000_000,
                    invested_usd=5_000_000,
                    unrealized_pnl=0,
                    recorded_at=datetime.utcnow(),
                )
            )
            await session.commit()

        # Get all IDs
        async with db_setup() as session:
            stmt = select(PortfolioSnapshot)
            result = await session.execute(stmt)
            all_snapshots = result.scalars().all()
            us_id = [s.id for s in all_snapshots if s.market == "US"][0]
            kr_id = [s.id for s in all_snapshots if s.market == "KR"][0]

        # Try to delete both IDs via KR manager — should only delete KR
        deleted = await kr_mgr.delete_snapshots_by_ids([us_id, kr_id])
        assert deleted == 1  # Only KR deleted

        # US snapshot still exists
        async with db_setup() as session:
            stmt = select(func.count()).select_from(PortfolioSnapshot)
            result = await session.execute(stmt)
            assert result.scalar() == 1

    async def test_empty_ids_returns_zero(self, db_setup):
        """Passing empty list returns 0."""
        svc = AsyncMock()
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        deleted = await mgr.delete_snapshots_by_ids([])
        assert deleted == 0

    async def test_nonexistent_ids_returns_zero(self, db_setup):
        """Deleting non-existent IDs returns 0."""
        svc = AsyncMock()
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup, market="KR")

        deleted = await mgr.delete_snapshots_by_ids([999, 1000, 1001])
        assert deleted == 0
