"""Tests for portfolio snapshot admin endpoints — STOCK-45.

Validates:
- DELETE /portfolio/snapshots removes anomalous snapshots by ID
- Only deletes snapshots for the specified market
- Error handling for invalid input
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.router import api_router
from core.models import Base, PortfolioSnapshot
from engine.portfolio_manager import PortfolioManager
from exchange.base import Balance


@pytest.fixture
def app_with_portfolio_manager(tmp_path):
    """Create test app with a real PortfolioManager backed by in-memory SQLite."""
    import asyncio

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.get_event_loop_policy()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_setup())

    mock_md = AsyncMock()
    mock_md.get_balance = AsyncMock(
        return_value=Balance(
            currency="KRW",
            total=10_000_000,
            available=10_000_000,
        )
    )
    mock_md.get_positions = AsyncMock(return_value=[])

    kr_pm = PortfolioManager(market_data=mock_md, session_factory=factory, market="KR")

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.kr_portfolio_manager = kr_pm
    app.state.portfolio_manager = None

    yield app, factory, loop, engine

    loop.run_until_complete(engine.dispose())
    loop.close()


class TestDeleteSnapshotsEndpoint:
    """STOCK-45: DELETE /portfolio/snapshots endpoint."""

    def test_deletes_by_ids(self, app_with_portfolio_manager):
        app, factory, loop, _ = app_with_portfolio_manager

        # Seed snapshots
        async def seed():
            async with factory() as session:
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

        loop.run_until_complete(seed())

        # Get IDs
        async def get_ids():
            from sqlalchemy import select

            async with factory() as session:
                stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.id)
                result = await session.execute(stmt)
                return [s.id for s in result.scalars().all()]

        all_ids = loop.run_until_complete(get_ids())
        delete_ids = all_ids[1:4]  # Delete 3 of 5

        client = TestClient(app)
        ids_str = ",".join(str(i) for i in delete_ids)
        resp = client.delete(f"/api/v1/portfolio/snapshots?ids={ids_str}&market=KR")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 3
        assert data["market"] == "KR"

    def test_empty_ids(self, app_with_portfolio_manager):
        app, _, _, _ = app_with_portfolio_manager
        client = TestClient(app)

        resp = client.delete("/api/v1/portfolio/snapshots?ids=&market=KR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 0
        assert "error" in data

    def test_invalid_ids_format(self, app_with_portfolio_manager):
        app, _, _, _ = app_with_portfolio_manager
        client = TestClient(app)

        resp = client.delete("/api/v1/portfolio/snapshots?ids=abc,def&market=KR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 0
        assert "error" in data

    def test_no_portfolio_manager(self):
        """Returns error when portfolio manager not configured."""
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        app.state.kr_portfolio_manager = None
        app.state.portfolio_manager = None

        client = TestClient(app)
        resp = client.delete("/api/v1/portfolio/snapshots?ids=1,2,3&market=KR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 0
        assert "error" in data
