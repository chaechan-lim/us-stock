"""Tests for TWR (Time-Weighted Return) in portfolio returns (STOCK-46).

Validates:
- TWR calculation produces correct results with deposits/withdrawals
- TWR matches simple return when no cash flows exist
- API endpoint returns both pct (TWR) and simple_pct
- has_cash_flows flag is correctly set
- Backward compatibility with snapshots missing cash_flow column
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.portfolio import (
    _build_equity_timeline,
    _calculate_twr,
    _has_cash_flows,
    init_portfolio,
)
from api.router import api_router
from core.models import Base, PortfolioSnapshot

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def _setup_db():
    """Create async session factory with in-memory SQLite for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_create_tables(engine))
    yield async_session, engine
    loop.run_until_complete(engine.dispose())
    loop.close()


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _make_app(session_factory) -> FastAPI:
    """Create test app with portfolio router."""
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
    cash_flow: float = 0.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        market=market,
        total_value_usd=total_value_usd,
        cash_usd=cash_usd,
        invested_usd=invested_usd,
        cash_flow=cash_flow,
        recorded_at=recorded_at,
    )


# ── _has_cash_flows() tests ─────────────────────────────────────────────


class TestHasCashFlows:
    """Unit tests for _has_cash_flows() helper."""

    def test_empty_list(self):
        assert _has_cash_flows([]) is False

    def test_no_cash_flows(self):
        now = datetime.utcnow()
        snapshots = [
            _make_snapshot("US", 100_000, now, cash_flow=0.0),
            _make_snapshot("US", 101_000, now + timedelta(hours=1), cash_flow=0.0),
        ]
        assert _has_cash_flows(snapshots) is False

    def test_has_deposit(self):
        now = datetime.utcnow()
        snapshots = [
            _make_snapshot("US", 100_000, now, cash_flow=0.0),
            _make_snapshot("US", 150_000, now + timedelta(hours=1), cash_flow=50_000.0),
        ]
        assert _has_cash_flows(snapshots) is True

    def test_backward_compat_missing_attr(self):
        """Snapshots without cash_flow attribute (legacy) treated as no cash flow."""
        snap = PortfolioSnapshot(
            market="US",
            total_value_usd=100_000,
            cash_usd=80_000,
            invested_usd=20_000,
        )
        # Remove cash_flow to simulate old snapshot
        assert _has_cash_flows([snap]) is False


# ── _build_equity_timeline() tests ──────────────────────────────────────


class TestBuildEquityTimeline:
    """Unit tests for _build_equity_timeline() helper."""

    def test_us_only_timeline(self):
        now = datetime.utcnow()
        us = [_make_snapshot("US", 10_000, now, cash_flow=0.0)]
        timeline = _build_equity_timeline(us, [], 1400.0)
        assert len(timeline) == 1
        assert timeline[0][1] == pytest.approx(14_000_000.0)  # 10k * 1400
        assert timeline[0][2] == 0.0

    def test_kr_only_timeline(self):
        now = datetime.utcnow()
        kr = [_make_snapshot("KR", 10_000_000, now, cash_flow=500_000.0)]
        timeline = _build_equity_timeline([], kr, 1400.0)
        assert len(timeline) == 1
        assert timeline[0][1] == 10_000_000.0
        assert timeline[0][2] == 500_000.0

    def test_mixed_timeline_sorted(self):
        """Carry-forward: first US event is skipped until KR arrives.

        The US snapshot at T comes in before KR, but since both markets are
        active we wait for each market's first reading before emitting a combined
        entry.  The single aggregated entry reflects both markets' equity.
        """
        now = datetime.utcnow()
        # US arrives at T, KR arrives 2 seconds later
        us = [_make_snapshot("US", 10_000, now)]
        kr = [_make_snapshot("KR", 10_000_000, now + timedelta(seconds=2))]
        timeline = _build_equity_timeline(us, kr, 1400.0)
        # The US event is skipped (no KR data yet); the KR event produces the
        # first combined entry because US equity is now carried forward.
        assert len(timeline) == 1
        assert timeline[0][1] == pytest.approx(10_000 * 1400.0 + 10_000_000)


# ── _calculate_twr() tests ──────────────────────────────────────────────


class TestCalculateTWR:
    """Unit tests for TWR calculation logic."""

    def test_no_snapshots_returns_zero(self):
        assert _calculate_twr([], [], 1400.0) == 0.0

    def test_single_snapshot_returns_zero(self):
        now = datetime.utcnow()
        us = [_make_snapshot("US", 10_000, now)]
        assert _calculate_twr(us, [], 1400.0) == 0.0

    def test_simple_return_no_cash_flows(self):
        """With no cash flows, TWR equals simple return."""
        now = datetime.utcnow()
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 10_500, now + timedelta(hours=1), cash_flow=0.0),
        ]
        twr = _calculate_twr(us, [], 1400.0)
        expected = (10_500 - 10_000) / 10_000 * 100  # 5%
        assert twr == pytest.approx(expected, abs=0.01)

    def test_deposit_excluded_from_return(self):
        """A deposit should be excluded from the return calculation.

        Scenario:
        - Start: equity = 10,000 USD
        - End: equity = 15,500 USD (deposit 5,000 + gain 500)
        - Simple return: (15500-10000)/10000 = 55% (wrong, includes deposit)
        - TWR: (15500 - 5000 - 10000) / 10000 = 5% (correct)
        """
        now = datetime.utcnow()
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 15_500, now + timedelta(hours=1), cash_flow=5_000.0),
        ]
        twr = _calculate_twr(us, [], 1400.0)
        expected = (15_500 - 5_000 - 10_000) / 10_000 * 100  # 5%
        assert twr == pytest.approx(expected, abs=0.01)

    def test_withdrawal_excluded_from_return(self):
        """A withdrawal should be excluded from the return calculation.

        Scenario:
        - Start: equity = 10,000
        - End: equity = 7,500 (withdrawal 3,000, gain 500)
        - Simple return: (7500-10000)/10000 = -25%
        - TWR: (7500 - (-3000) - 10000) / 10000 = 5%
        """
        now = datetime.utcnow()
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 7_500, now + timedelta(hours=1), cash_flow=-3_000.0),
        ]
        twr = _calculate_twr(us, [], 1400.0)
        expected = (7_500 - (-3_000) - 10_000) / 10_000 * 100  # 5%
        assert twr == pytest.approx(expected, abs=0.01)

    def test_multi_period_twr_chain(self):
        """TWR chains multiple sub-period returns geometrically.

        Period 1: 10000 → 10500 (5% gain)
        Period 2: 10500 + 5000 deposit → 16275 (i.e. 15500 after deposit grew by 5%)
        Wait, let me recalculate:
        Period 2: start=10500, end=16275, cf=5000
        sub_return = (16275-5000-10500)/10500 = 775/10500 ≈ 7.38%

        TWR = (1.05) * (1 + 775/10500) - 1
        """
        now = datetime.utcnow()
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 10_500, now + timedelta(hours=1), cash_flow=0.0),
            _make_snapshot("US", 16_275, now + timedelta(hours=2), cash_flow=5_000.0),
        ]
        twr = _calculate_twr(us, [], 1400.0)
        # Period 1: (10500-0-10000)/10000 = 0.05
        # Period 2: (16275-5000-10500)/10500 = 775/10500 ≈ 0.07381
        r1 = (10_500 - 10_000) / 10_000
        r2 = (16_275 - 5_000 - 10_500) / 10_500
        expected = ((1 + r1) * (1 + r2) - 1) * 100
        assert twr == pytest.approx(expected, abs=0.01)

    def test_dual_market_twr(self):
        """TWR with both US and KR snapshots using realistic offset timestamps.

        In production, save_snapshot() is called independently for each market
        so their recorded_at values differ by at least milliseconds.  Using
        slightly offset timestamps validates that the carry-forward aggregation
        handles this correctly rather than producing nonsensical per-market
        comparisons.
        """
        now = datetime.utcnow()
        # US snapshots arrive at T+0 and T+1h; KR snapshots arrive 5 seconds later.
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 10_500, now + timedelta(hours=1), cash_flow=0.0),
        ]
        kr = [
            _make_snapshot("KR", 5_000_000, now + timedelta(seconds=5), cash_flow=0.0),
            _make_snapshot(
                "KR", 5_250_000, now + timedelta(hours=1, seconds=5), cash_flow=0.0
            ),
        ]
        rate = 1400.0
        twr = _calculate_twr(us, kr, rate)
        # Combined: old = 10000*1400 + 5M = 19M, new = 10500*1400 + 5.25M = 19.95M
        # Carry-forward chaining still produces (19.95/19 - 1) * 100 = 5%
        old_eq = 10_000 * rate + 5_000_000
        new_eq = 10_500 * rate + 5_250_000
        expected = (new_eq - old_eq) / old_eq * 100
        assert twr == pytest.approx(expected, abs=0.1)

    def test_negative_return_with_deposit(self):
        """Even with deposit, a net loss is correctly negative.

        Start: 10,000
        Deposit: 5,000 → subtotal 15,000
        End: 14,000 (lost 1,000 trading)
        TWR: (14000-5000-10000)/10000 = -10%
        """
        now = datetime.utcnow()
        us = [
            _make_snapshot("US", 10_000, now, cash_flow=0.0),
            _make_snapshot("US", 14_000, now + timedelta(hours=1), cash_flow=5_000.0),
        ]
        twr = _calculate_twr(us, [], 1400.0)
        expected = (14_000 - 5_000 - 10_000) / 10_000 * 100  # -10%
        assert twr == pytest.approx(expected, abs=0.01)


# ── Endpoint integration tests ──────────────────────────────────────────


class TestPortfolioReturnsTWR:
    """Integration tests for /portfolio/returns with TWR."""

    def test_response_includes_twr_fields(self, _setup_db):
        """Response includes pct (TWR), simple_pct, and has_cash_flows."""
        session_factory, _ = _setup_db
        now = datetime.utcnow()

        loop = asyncio.new_event_loop()

        async def _seed():
            async with session_factory() as session:
                session.add(_make_snapshot("US", 10_000, now - timedelta(hours=12), cash_flow=0.0))
                session.add(_make_snapshot("US", 10_500, now - timedelta(hours=1), cash_flow=0.0))
                await session.commit()

        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        daily = data["daily"]
        assert "pct" in daily
        assert "simple_pct" in daily
        assert "has_cash_flows" in daily
        assert daily["has_cash_flows"] is False
        # Without cash flows, pct == simple_pct
        assert daily["pct"] == daily["simple_pct"]

        loop.close()

    def test_twr_applied_with_deposit(self, _setup_db):
        """When deposit detected, pct uses TWR (differs from simple_pct)."""
        session_factory, _ = _setup_db
        now = datetime.utcnow()

        loop = asyncio.new_event_loop()

        async def _seed():
            async with session_factory() as session:
                session.add(
                    _make_snapshot(
                        "US",
                        10_000,
                        now - timedelta(hours=12),
                        cash_usd=8_000,
                        invested_usd=2_000,
                        cash_flow=0.0,
                    )
                )
                session.add(
                    _make_snapshot(
                        "US",
                        15_500,
                        now - timedelta(hours=1),
                        cash_usd=13_000,
                        invested_usd=2_500,
                        cash_flow=5_000.0,
                    )
                )
                await session.commit()

        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        daily = data["daily"]
        assert daily["has_cash_flows"] is True
        # Simple: (15500-10000)/10000 = 55%
        assert daily["simple_pct"] == pytest.approx(55.0, abs=0.1)
        # TWR: (15500-5000-10000)/10000 = 5%
        assert daily["pct"] == pytest.approx(5.0, abs=0.1)

        loop.close()

    def test_backward_compat_no_cash_flow_column(self, _setup_db):
        """Old snapshots without cash_flow value work fine (treated as 0)."""
        session_factory, _ = _setup_db
        now = datetime.utcnow()

        loop = asyncio.new_event_loop()

        async def _seed():
            async with session_factory() as session:
                # Insert without explicitly setting cash_flow
                snap_old = PortfolioSnapshot(
                    market="US",
                    total_value_usd=10_000,
                    cash_usd=8_000,
                    invested_usd=2_000,
                    recorded_at=now - timedelta(hours=12),
                )
                snap_new = PortfolioSnapshot(
                    market="US",
                    total_value_usd=10_500,
                    cash_usd=8_500,
                    invested_usd=2_000,
                    recorded_at=now - timedelta(hours=1),
                )
                session.add(snap_old)
                session.add(snap_new)
                await session.commit()

        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        assert data["daily"] is not None
        daily = data["daily"]
        assert daily["has_cash_flows"] is False
        # pct == simple_pct when no cash flows
        assert daily["pct"] == daily["simple_pct"]

        loop.close()

    def test_weekly_twr_with_multiple_deposits(self, _setup_db):
        """Weekly period with multiple deposits uses TWR correctly."""
        session_factory, _ = _setup_db
        now = datetime.utcnow()

        loop = asyncio.new_event_loop()

        async def _seed():
            async with session_factory() as session:
                # Day 1: Start at 10,000
                session.add(_make_snapshot("US", 10_000, now - timedelta(days=5), cash_flow=0.0))
                # Day 3: Grew to 10,500 (5%), then deposit 5,000
                session.add(
                    _make_snapshot("US", 15_500, now - timedelta(days=3), cash_flow=5_000.0)
                )
                # Day 5: Grew to 16,275 (5% gain on 15,500)
                session.add(_make_snapshot("US", 16_275, now - timedelta(hours=1), cash_flow=0.0))
                await session.commit()

        loop.run_until_complete(_seed())

        with patch("api.portfolio._cached_usd_krw", 1400.0):
            app = _make_app(session_factory)
            client = TestClient(app)
            data = client.get("/api/v1/portfolio/returns").json()

        weekly = data["weekly"]
        assert weekly is not None
        assert weekly["has_cash_flows"] is True
        # Simple: (16275-10000)/10000 = 62.75%
        assert weekly["simple_pct"] == pytest.approx(62.75, abs=0.1)
        # TWR: R1 = (15500-5000-10000)/10000 = 0.05
        #       R2 = (16275-0-15500)/15500 = 0.05
        #       TWR = (1.05 * 1.05 - 1) * 100 = 10.25%
        assert weekly["pct"] == pytest.approx(10.25, abs=0.1)

        loop.close()

    def test_no_data_returns_none(self, _setup_db):
        """No snapshots → all periods None (unchanged behavior)."""
        session_factory, _ = _setup_db
        app = _make_app(session_factory)
        client = TestClient(app)
        data = client.get("/api/v1/portfolio/returns").json()
        assert data["daily"] is None
        assert data["weekly"] is None
        assert data["monthly"] is None
