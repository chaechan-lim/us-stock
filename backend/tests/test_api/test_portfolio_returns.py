"""Tests for portfolio returns endpoint — STOCK-41.

Validates:
- /portfolio/returns returns daily, weekly, monthly equity changes
- Correct change and percentage calculation from snapshots
- Handles missing snapshots gracefully (returns None)
- Handles single-market-only snapshots
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.portfolio import init_portfolio
from api.router import api_router
from core.models import Base, PortfolioSnapshot


@pytest.fixture
def _setup_db():
    """Create async session factory with in-memory SQLite for snapshot tests."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    import asyncio

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_create_tables(engine))

    yield async_session, engine

    loop.run_until_complete(engine.dispose())
    loop.close()


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _make_app(session_factory) -> FastAPI:
    """Create test app with portfolio router and session factory."""
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    init_portfolio(session_factory)
    return app


def _make_snapshot(
    market: str,
    total_value_usd: float,
    recorded_at: datetime,
    cash_usd: float = 0,
    invested_usd: float = 0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        market=market,
        total_value_usd=total_value_usd,
        cash_usd=cash_usd,
        invested_usd=invested_usd,
        recorded_at=recorded_at,
    )


class TestPortfolioReturnsNoData:
    """Returns None for all periods when no snapshots exist."""

    def test_returns_none_when_no_session_factory(self):
        """Returns all None when session factory is not configured."""
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        init_portfolio(None)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"] is None
        assert data["weekly"] is None
        assert data["monthly"] is None

    def test_returns_none_when_no_snapshots(self, _setup_db):
        """Returns None for all periods when DB has no snapshots."""
        session_factory, _ = _setup_db
        app = _make_app(session_factory)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"] is None
        assert data["weekly"] is None
        assert data["monthly"] is None


class TestPortfolioReturnsWithSnapshots:
    """Correct change/pct calculation from equity snapshots."""

    def test_daily_return_calculated(self, _setup_db):
        """Daily return shows change from yesterday's equity to today's."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                # US snapshots: old (12h ago) and new (1h ago)
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=12)))
                session.add(_make_snapshot("US", 10500.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        # Patch _cached_usd_krw to a known value
        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        daily = data["daily"]
        # US equity: old=10000*1400=14_000_000, new=10500*1400=14_700_000
        expected_change = (10500 - 10000) * 1400
        assert daily["change"] == expected_change
        expected_pct = round((500 / 10000) * 100, 2)
        assert daily["pct"] == expected_pct

        loop.close()

    def test_weekly_return_calculated(self, _setup_db):
        """Weekly return uses snapshots from up to 7 days ago."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                # US snapshots: 5 days ago and now
                session.add(_make_snapshot("US", 9000.0, now - timedelta(days=5)))
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["weekly"] is not None
        weekly = data["weekly"]
        expected_change = (10000 - 9000) * 1400
        assert weekly["change"] == expected_change
        expected_pct = round((1000 / 9000) * 100, 2)
        assert weekly["pct"] == expected_pct

        loop.close()

    def test_monthly_return_calculated(self, _setup_db):
        """Monthly return uses snapshots from up to 30 days ago."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                # US snapshots: 20 days ago and now
                session.add(_make_snapshot("US", 8000.0, now - timedelta(days=20)))
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["monthly"] is not None
        monthly = data["monthly"]
        expected_change = (10000 - 8000) * 1400
        assert monthly["change"] == expected_change
        expected_pct = round((2000 / 8000) * 100, 2)
        assert monthly["pct"] == expected_pct

        loop.close()

    def test_dual_market_returns(self, _setup_db):
        """Returns combine US and KR equity snapshots."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                # US snapshots (value in USD, multiplied by rate)
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=12)))
                session.add(_make_snapshot("US", 11000.0, now - timedelta(hours=1)))
                # KR snapshots (total_value_usd stores KRW directly for KR market)
                session.add(_make_snapshot("KR", 5_000_000.0, now - timedelta(hours=12)))
                session.add(_make_snapshot("KR", 5_500_000.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        daily = data["daily"]
        # old: US 10000*1400 + KR 5_000_000 = 19_000_000
        # new: US 11000*1400 + KR 5_500_000 = 20_900_000
        old_equity = 10000 * 1400 + 5_000_000
        new_equity = 11000 * 1400 + 5_500_000
        expected_change = new_equity - old_equity
        assert daily["change"] == expected_change

        loop.close()

    def test_negative_return(self, _setup_db):
        """Correctly handles negative equity changes."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=12)))
                session.add(_make_snapshot("US", 9500.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        daily = data["daily"]
        expected_change = (9500 - 10000) * 1400
        assert daily["change"] == expected_change
        assert daily["pct"] < 0

        loop.close()

    def test_returns_include_base_equity(self, _setup_db):
        """Each period result includes base_equity field."""
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                session.add(_make_snapshot("US", 10000.0, now - timedelta(hours=12)))
                session.add(_make_snapshot("US", 10500.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        assert "base_equity" in data["daily"]
        assert data["daily"]["base_equity"] == 10000 * 1400

        loop.close()

    def test_daily_zero_change_when_only_recent_snapshot(self, _setup_db):
        """Daily shows zero change when only recent snapshot exists within 24h.

        _get_oldest_snapshot finds the 1h-ago snapshot (oldest after now-1day),
        and _get_latest_snapshot also returns the same snapshot → change=0.
        """
        session_factory, engine = _setup_db
        now = datetime.utcnow()

        import asyncio

        async def _seed():
            async with session_factory() as session:
                # Old snapshot outside daily window, recent snapshot within
                session.add(_make_snapshot("US", 10000.0, now - timedelta(days=3)))
                session.add(_make_snapshot("US", 10500.0, now - timedelta(hours=1)))
                await session.commit()

        loop = asyncio.get_event_loop_policy().new_event_loop()
        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        # Daily: oldest after now-1day = 1h-ago, latest = 1h-ago → same → change=0
        assert data["daily"] is not None
        assert data["daily"]["change"] == 0.0
        assert data["daily"]["pct"] == 0.0
        # Weekly should find the 3-day-old snapshot as base
        assert data["weekly"] is not None
        assert data["weekly"]["change"] != 0

        loop.close()

    def test_returns_none_when_no_snapshots_for_market(self, _setup_db):
        """Returns None when no snapshots exist at all."""
        session_factory, _ = _setup_db
        app = _make_app(session_factory)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"] is None
        assert data["weekly"] is None
        assert data["monthly"] is None

    def test_response_structure(self, _setup_db):
        """Response has daily, weekly, monthly keys."""
        session_factory, _ = _setup_db
        app = _make_app(session_factory)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert "daily" in data
        assert "weekly" in data
        assert "monthly" in data
