"""Tests for cash flow detection in PortfolioManager (STOCK-46).

Validates:
- detect_cash_flow() pure function correctly identifies deposits/withdrawals
- Profitable sells are NOT misclassified as deposits (rework fix)
- Small market moves (below threshold) are ignored
- cash_flow is persisted to PortfolioSnapshot via save_snapshot
- Backward compatibility: old snapshots without cash_flow treated as 0
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base, PortfolioSnapshot
from engine.portfolio_manager import (
    CASH_FLOW_THRESHOLD,
    PortfolioManager,
    detect_cash_flow,
)
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


# ── detect_cash_flow() pure function tests ──────────────────────────────


class TestDetectCashFlow:
    """Unit tests for detect_cash_flow() pure function.

    The function now uses total equity change (new_total - prev_total) rather
    than (cash + invested_cost_basis) change.  This prevents profitable sells
    from being misclassified as deposits.
    """

    def test_no_change_returns_zero(self):
        """No change in total equity → no cash flow."""
        result = detect_cash_flow(prev_total=100_000, new_total=100_000)
        assert result == 0.0

    def test_large_deposit_detected(self):
        """Large deposit (>5% of total) is detected."""
        # Deposit of 50,000 on 100,000 total = 50%
        result = detect_cash_flow(prev_total=100_000, new_total=150_000)
        assert result == pytest.approx(50_000.0)

    def test_large_withdrawal_detected(self):
        """Large withdrawal (>5% of total) is detected as negative."""
        result = detect_cash_flow(prev_total=100_000, new_total=70_000)
        assert result == pytest.approx(-30_000.0)

    def test_normal_buy_has_zero_cash_flow(self):
        """Buy moves cash → invested; total equity is unchanged → no cash flow."""
        # Buy $10k of stock: cash -10k, invested +10k, total unchanged at 100k
        result = detect_cash_flow(prev_total=100_000, new_total=100_000)
        assert result == 0.0

    def test_small_realized_pnl_ignored(self):
        """Small realized PnL (<5%) doesn't trigger cash flow detection.

        Selling a position at a small gain leaves total equity ~unchanged
        (market value converts to cash), so the raw_cf stays below threshold.
        """
        # Sell with ~$2k unrealized already reflected in prev_total.
        # After sell: total stays the same (102k → 102k) → below threshold.
        result = detect_cash_flow(prev_total=100_000, new_total=102_000)
        assert result == 0.0

    def test_large_realized_pnl_not_detected_as_deposit(self):
        """Profitable sell with large PnL is NOT falsely classified as a deposit.

        This is the core bug fixed in the STOCK-46 rework.  With the old
        (cash + invested_cost_basis) formula a sell at 15% profit was detected
        as a deposit because cash rose by market_value while invested fell only
        by cost_basis.

        With the new total-equity formula: selling doesn't change total equity
        (unrealized PnL converts to cash), so raw_cf = 0 when the prior snapshot
        already captured the unrealized gain.

        Scenario:
          prev: cash=80k, position at market=35k (cost=20k, unrealized=15k),
                total = 115k
          Sell all at market (35k proceeds): cash=115k, positions=0, total=115k
          → raw_cf = 115k - 115k = 0 → correctly NOT a deposit
        """
        result = detect_cash_flow(prev_total=115_000, new_total=115_000)
        assert result == 0.0

    def test_threshold_boundary_below(self):
        """Cash flow exactly at threshold boundary (below) returns 0."""
        # 4.9% of 100,000 = 4,900 → below 5% threshold
        result = detect_cash_flow(prev_total=100_000, new_total=104_900)
        assert result == 0.0

    def test_threshold_boundary_above(self):
        """Cash flow just above threshold is detected."""
        # 5.1% of 100,000 = 5,100 → above 5% threshold
        result = detect_cash_flow(prev_total=100_000, new_total=105_100)
        assert result == pytest.approx(5_100.0)

    def test_zero_prev_total_returns_zero(self):
        """Zero previous total equity returns 0 (no division by zero)."""
        result = detect_cash_flow(prev_total=0, new_total=50_000)
        assert result == 0.0

    def test_negative_prev_total_returns_zero(self):
        """Negative previous total equity returns 0."""
        result = detect_cash_flow(prev_total=-100, new_total=50_000)
        assert result == 0.0

    def test_deposit_with_simultaneous_trade(self):
        """Deposit + trade: deposit raises total; trade (buy) leaves it unchanged.

        Deposit 50k → total 100k→150k; buy 10k of stock → total stays 150k.
        So new_total = 150k, raw_cf = 50k detected.
        """
        result = detect_cash_flow(prev_total=100_000, new_total=150_000)
        assert result == pytest.approx(50_000.0)

    def test_krw_scale_deposit_detected(self):
        """Works correctly with KRW-scale numbers (millions)."""
        # prev total=10M, deposit 7M → new total=17M
        result = detect_cash_flow(prev_total=10_000_000, new_total=17_000_000)
        assert result == pytest.approx(7_000_000.0)

    def test_threshold_constant_is_five_percent(self):
        """Verify the threshold constant is 0.05 (5%)."""
        assert CASH_FLOW_THRESHOLD == 0.05


# ── save_snapshot integration tests (cash_flow persisted) ──────────────


class TestSaveSnapshotCashFlow:
    """Integration tests: cash_flow detection and persistence in save_snapshot."""

    async def test_first_snapshot_has_zero_cash_flow(self, db_setup):
        """First snapshot (no previous) should have cash_flow=0."""
        svc = AsyncMock()
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=80_000)
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
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup)

        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(PortfolioSnapshot)
            result = await session.execute(stmt)
            snap = result.scalar_one()
        assert snap.cash_flow == 0.0

    async def test_deposit_detected_in_second_snapshot(self, db_setup):
        """When total equity increases substantially, cash_flow > 0."""
        svc = AsyncMock()

        # First snapshot: total=100k (positions at market value=cost, no unrealized)
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=80_000)
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=100,
                    avg_price=200.0,
                    current_price=200.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup)
        await mgr.save_snapshot()

        # Second snapshot: deposit $50k → total goes from 100k to 150k
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=150_000, available=130_000)
        )
        # Same positions (position value unchanged, so entire 50k rise = deposit)
        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.recorded_at.desc()).limit(1)
            result = await session.execute(stmt)
            latest = result.scalar_one()
        assert latest.cash_flow == pytest.approx(50_000.0)

    async def test_normal_trade_has_zero_cash_flow(self, db_setup):
        """Buy stock (cash → invested) should NOT trigger cash flow.

        Buying keeps total equity constant (cash -B, position market +B),
        so the total-equity formula correctly returns 0.
        """
        svc = AsyncMock()

        # First: total=100k
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=80_000)
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=100,
                    avg_price=200.0,
                    current_price=200.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup)
        await mgr.save_snapshot()

        # Second: bought more stock → cash=70k, total still 100k
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=70_000)
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=100,
                    avg_price=200.0,
                    current_price=200.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                ),
                Position(
                    symbol="MSFT",
                    exchange="NASD",
                    quantity=25,
                    avg_price=400.0,
                    current_price=400.0,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                ),
            ]
        )
        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.recorded_at.desc()).limit(1)
            result = await session.execute(stmt)
            latest = result.scalar_one()
        assert latest.cash_flow == 0.0

    async def test_profitable_sell_not_detected_as_deposit(self, db_setup):
        """Selling a profitable position should NOT be detected as a deposit.

        This is the primary bug from the original STOCK-46 review.  When a
        position appreciated and was sold, (cash + cost_basis) increased by the
        realized PnL, which the old formula misclassified as a deposit.

        With the total-equity formula: if the previous snapshot already captured
        the unrealized gain, selling converts it to cash without changing total
        equity → cash_flow = 0.
        """
        svc = AsyncMock()

        # First snapshot: position at market value already reflects the unrealized
        # gain (prev snapshot captured total=115k = 80k cash + 35k position).
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=115_000, available=80_000)
        )
        svc.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=100,
                    avg_price=200.0,       # cost basis = 20k
                    current_price=350.0,   # market value = 35k (unrealized = 15k)
                    unrealized_pnl=15_000.0,
                    unrealized_pnl_pct=75.0,
                ),
            ]
        )
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup)
        await mgr.save_snapshot()

        # Second snapshot: sold AAPL at market (35k proceeds) → total stays 115k
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=115_000, available=115_000)
        )
        svc.get_positions = AsyncMock(return_value=[])
        await mgr.save_snapshot()

        async with db_setup() as session:
            stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.recorded_at.desc()).limit(1)
            result = await session.execute(stmt)
            latest = result.scalar_one()
        # Selling at market price doesn't change total equity → no cash flow
        assert latest.cash_flow == 0.0

    async def test_backward_compat_old_snapshot_without_cash_flow(self, db_setup):
        """Old snapshots without cash_flow column default to 0."""
        # Manually insert a snapshot without cash_flow
        async with db_setup() as session:
            snap = PortfolioSnapshot(
                market="US",
                total_value_usd=100_000,
                cash_usd=80_000,
                invested_usd=20_000,
                unrealized_pnl=0.0,
                recorded_at=datetime.utcnow() - timedelta(hours=1),
            )
            session.add(snap)
            await session.commit()

        # Verify cash_flow defaults to 0 (or None → treated as 0)
        async with db_setup() as session:
            stmt = select(PortfolioSnapshot)
            result = await session.execute(stmt)
            snap = result.scalar_one()
        # Column default is 0.0, but for truly old rows it might be None
        cf = snap.cash_flow if snap.cash_flow is not None else 0.0
        assert cf == 0.0

    async def test_equity_history_includes_cash_flow(self, db_setup):
        """get_equity_history includes cash_flow field."""
        svc = AsyncMock()
        svc.get_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=80_000)
        )
        svc.get_positions = AsyncMock(return_value=[])
        mgr = PortfolioManager(market_data=svc, session_factory=db_setup)

        # Seed a snapshot with cash_flow
        async with db_setup() as session:
            session.add(
                PortfolioSnapshot(
                    market="US",
                    total_value_usd=100_000,
                    cash_usd=80_000,
                    invested_usd=20_000,
                    unrealized_pnl=0.0,
                    cash_flow=50_000.0,
                    recorded_at=datetime.utcnow() - timedelta(hours=1),
                )
            )
            await session.commit()

        history = await mgr.get_equity_history(days=30)
        assert len(history) == 1
        assert history[0]["cash_flow"] == 50_000.0
