"""Tests for portfolio snapshot admin endpoints — STOCK-45.

Validates:
- DELETE /portfolio/snapshots removes anomalous snapshots by ID
- Proper HTTP status codes for error cases
- Market parameter validation
- Both US and KR market paths
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.router import api_router
from core.models import Base, PortfolioSnapshot
from engine.portfolio_manager import PortfolioManager
from exchange.base import Balance


@pytest_asyncio.fixture
async def db_factory():
    """Create in-memory SQLite engine and session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_portfolio_manager(factory, market: str = "KR") -> PortfolioManager:
    mock_md = AsyncMock()
    mock_md.get_balance = AsyncMock(
        return_value=Balance(currency="KRW", total=10_000_000, available=10_000_000)
    )
    mock_md.get_positions = AsyncMock(return_value=[])
    return PortfolioManager(market_data=mock_md, session_factory=factory, market=market)


class TestDeleteSnapshotsEndpoint:
    """STOCK-45: DELETE /portfolio/snapshots endpoint."""

    async def test_deletes_by_ids(self, db_factory):
        kr_pm = _make_portfolio_manager(db_factory, "KR")
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        app.state.kr_portfolio_manager = kr_pm
        app.state.portfolio_manager = None

        # Seed snapshots
        async with db_factory() as session:
            for i in range(5):
                session.add(
                    PortfolioSnapshot(
                        market="KR",
                        total_value_usd=float(10_000_000 + i),
                        cash_usd=5_000_000,
                        invested_usd=5_000_000,
                        unrealized_pnl=0,
                        recorded_at=datetime.utcnow() - timedelta(hours=5 - i),
                    )
                )
            await session.commit()

        # Get IDs
        async with db_factory() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.id)
            result = await session.execute(stmt)
            all_ids = [s.id for s in result.scalars().all()]

        delete_ids = all_ids[1:4]  # Delete 3 of 5

        client = TestClient(app)
        ids_str = ",".join(str(i) for i in delete_ids)
        resp = client.delete(f"/api/v1/portfolio/snapshots?ids={ids_str}&market=KR")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 3
        assert data["market"] == "KR"

    async def test_deletes_us_market(self, db_factory):
        """US market path uses portfolio_manager (not kr_portfolio_manager)."""
        us_pm = _make_portfolio_manager(db_factory, "US")
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        app.state.kr_portfolio_manager = None
        app.state.portfolio_manager = us_pm

        # Seed US snapshots
        async with db_factory() as session:
            for i in range(3):
                session.add(
                    PortfolioSnapshot(
                        market="US",
                        total_value_usd=float(50_000 + i),
                        cash_usd=30_000,
                        invested_usd=20_000,
                        unrealized_pnl=0,
                        recorded_at=datetime.utcnow() - timedelta(hours=3 - i),
                    )
                )
            await session.commit()

        async with db_factory() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.id)
            result = await session.execute(stmt)
            all_ids = [s.id for s in result.scalars().all()]

        client = TestClient(app)
        ids_str = ",".join(str(i) for i in all_ids[:2])
        resp = client.delete(f"/api/v1/portfolio/snapshots?ids={ids_str}&market=US")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 2
        assert data["market"] == "US"

    def test_empty_ids_returns_400(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.delete("/api/v1/portfolio/snapshots?ids=&market=KR")
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_invalid_ids_format_returns_400(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.delete("/api/v1/portfolio/snapshots?ids=abc,def&market=KR")
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_invalid_market_returns_400(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.delete("/api/v1/portfolio/snapshots?ids=1,2,3&market=JP")
        assert resp.status_code == 400
        assert "Invalid market" in resp.json()["error"]

    def test_no_portfolio_manager_returns_503(self):
        """Returns 503 when portfolio manager not configured."""
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        app.state.kr_portfolio_manager = None
        app.state.portfolio_manager = None

        client = TestClient(app)
        resp = client.delete("/api/v1/portfolio/snapshots?ids=1,2,3&market=KR")
        assert resp.status_code == 503
        assert "error" in resp.json()
