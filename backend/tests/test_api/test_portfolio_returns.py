"""Tests for /portfolio/returns — realized P&L based (2026-04-17 rewrite).

Previously: snapshot-equity-diff based, broke whenever the equity formula
changed or snapshot timing produced phantom swings.

Now: aggregates Order.pnl from filled SELL trades within each window.
Deterministic — does NOT depend on snapshots, equity formula, or KIS API.
"""

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.portfolio import init_portfolio
from api.router import api_router
from core.models import Base, Order


@pytest.fixture
def _setup_db():
    """Create async session factory with in-memory SQLite for trade tests."""
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
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    init_portfolio(session_factory)
    return app


def _make_sell_order(
    *,
    market: str,
    pnl: float,
    recorded_at: datetime,
    symbol: str = "TST",
    is_paper: bool = False,
    status: str = "filled",
) -> Order:
    return Order(
        kis_order_id=f"O{recorded_at.timestamp()}-{symbol}",
        symbol=symbol,
        market=market,
        side="SELL",
        order_type="limit",
        quantity=1,
        price=100.0,
        filled_price=100.0,
        filled_quantity=1,
        status=status,
        strategy_name="test",
        pnl=pnl,
        is_paper=is_paper,
        created_at=recorded_at,
    )


async def _seed_orders(session_factory, orders: list[Order]):
    async with session_factory() as session:
        for o in orders:
            session.add(o)
        await session.commit()


def _run(coro):
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestPortfolioReturnsNoData:
    def test_returns_none_when_no_session_factory(self):
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        init_portfolio(None)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"] is None
        assert data["weekly"] is None
        assert data["monthly"] is None

    def test_returns_zero_when_no_trades(self, _setup_db):
        """No trades in window → realized=0 for all periods."""
        session_factory, _ = _setup_db
        app = _make_app(session_factory)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["change"] == 0
        assert data["daily"]["realized_kr"] == 0
        assert data["daily"]["realized_us"] == 0


class TestRealizedAggregation:
    def test_daily_sum_kr(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=5000, recorded_at=now - timedelta(hours=1)),
            _make_sell_order(market="KR", pnl=3000, recorded_at=now - timedelta(hours=2)),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == 8000
        assert data["daily"]["realized_us"] == 0

    def test_us_pnl_converted_to_krw(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="US", pnl=100, recorded_at=now - timedelta(hours=1)),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        # change = us_pnl × rate (rate >= 1450 fallback)
        assert data["daily"]["realized_us"] == 100
        assert data["daily"]["change"] >= 100 * 1450

    def test_combined_kr_and_us(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=10_000, recorded_at=now - timedelta(hours=1)),
            _make_sell_order(market="US", pnl=50, recorded_at=now - timedelta(hours=2)),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == 10_000
        assert data["daily"]["realized_us"] == 50

    def test_excludes_old_trades_outside_period(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=999_999, recorded_at=now - timedelta(days=10)),  # outside daily/weekly
            _make_sell_order(market="KR", pnl=1000, recorded_at=now - timedelta(hours=1)),     # daily only
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == 1000
        assert data["weekly"]["realized_kr"] == 1000  # 10 days ago is excluded
        assert data["monthly"]["realized_kr"] == 1_000_999  # both included

    def test_excludes_paper_trades(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=5000, recorded_at=now - timedelta(hours=1), is_paper=True),
            _make_sell_order(market="KR", pnl=2000, recorded_at=now - timedelta(hours=2), is_paper=False),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == 2000

    def test_excludes_unfilled(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=5000, recorded_at=now - timedelta(hours=1), status="cancelled"),
            _make_sell_order(market="KR", pnl=2000, recorded_at=now - timedelta(hours=2)),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == 2000

    def test_handles_negative_pnl(self, _setup_db):
        session_factory, _ = _setup_db
        now = datetime.utcnow()
        _run(_seed_orders(session_factory, [
            _make_sell_order(market="KR", pnl=-3000, recorded_at=now - timedelta(hours=1)),
            _make_sell_order(market="KR", pnl=1000, recorded_at=now - timedelta(hours=2)),
        ]))
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"]["realized_kr"] == -2000


class TestResponseShape:
    def test_response_keys(self, _setup_db):
        session_factory, _ = _setup_db
        client = TestClient(_make_app(session_factory))
        data = client.get("/api/v1/portfolio/returns").json()
        for period in ("daily", "weekly", "monthly"):
            r = data[period]
            assert "change" in r
            assert "realized_kr" in r
            assert "realized_us" in r
            assert "pct" in r  # legacy field
