"""Tests for Hermes Phase 3 C2 — funnel_replay counterfactual."""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.models import Base, FunnelEvent, Order
from services.funnel_replay import (
    is_replayable,
    replay_recommendation,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


class TestIsReplayable:
    def test_known_paths(self):
        assert is_replayable("markets.KR.evaluation_loop.daily_buy_limit")
        assert is_replayable("markets.US.evaluation_loop.opening_avoidance_minutes")
        assert is_replayable("markets.KR.evaluation_loop.sell_cooldown_days")

    def test_unknown_paths(self):
        assert not is_replayable("markets.KR.risk.max_positions")
        assert not is_replayable("markets.KR.sizing_up.threshold")


class TestDailyLimitReplay:
    async def test_higher_limit_unlocks_rejections(self, session):
        today = datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0)
        # Day 1: 4 daily_limit rejections (limit was 5; raising to 8 should
        # let 3 of these 4 pass, since headroom is +3)
        for i in range(4):
            session.add(FunnelEvent(
                ts=today - timedelta(days=1, minutes=i),
                market="KR", symbol=f"SYM{i:03d}", strategy_name="test",
                signal_confidence=0.6, decision="rejected",
                reject_reason="daily_limit", price=100.0,
            ))
        # Day 2: 2 daily_limit rejections (both pass at limit 8)
        for i in range(2):
            session.add(FunnelEvent(
                ts=today - timedelta(days=2, minutes=i),
                market="KR", symbol=f"SYM{i+10:03d}", strategy_name="test",
                signal_confidence=0.6, decision="rejected",
                reject_reason="daily_limit", price=100.0,
            ))
        await session.commit()

        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.daily_buy_limit",
            current_value=5, proposed_value=8,
            lookback_days=30,
        )
        assert result["replayed_rejections"] == 6
        # Day 1: min(4, headroom=3) = 3; Day 2: min(2, 3) = 2 → 5
        assert result["would_pass_under_proposed"] == 5
        assert result["trading_days_with_rejections"] == 2

    async def test_lower_proposed_does_nothing(self, session):
        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.daily_buy_limit",
            current_value=10, proposed_value=5,
        )
        assert result["would_pass"] == 0

    async def test_no_rejections_in_window(self, session):
        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.daily_buy_limit",
            current_value=5, proposed_value=10,
        )
        assert result["replayed_rejections"] == 0
        assert result["would_pass_under_proposed"] == 0


class TestOpeningAvoidanceReplay:
    async def test_shorter_window_passes_more(self, session):
        base = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        # KR open = 09:00. Event at 09:10 (10 min in) and 09:25 (25 min in)
        for minute_offset in (10, 25, 40):
            session.add(FunnelEvent(
                ts=base + timedelta(hours=9, minutes=minute_offset),
                market="KR", symbol=f"S{minute_offset}", strategy_name="t",
                signal_confidence=0.6, decision="rejected",
                reject_reason="opening_avoidance", price=100.0,
            ))
        await session.commit()

        # Current 30min → shortening to 15min: 09:25 + 09:40 pass (2/3)
        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.opening_avoidance_minutes",
            current_value=30, proposed_value=15,
        )
        assert result["replayed_rejections"] == 3
        assert result["would_pass_under_proposed"] == 2  # only 09:25 + 09:40

    async def test_longer_window_passes_fewer(self, session):
        base = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        session.add(FunnelEvent(
            ts=base + timedelta(hours=9, minutes=35),
            market="KR", symbol="X", strategy_name="t",
            signal_confidence=0.6, decision="rejected",
            reject_reason="opening_avoidance", price=100.0,
        ))
        await session.commit()

        # 30min → 60min: this event (35 min in) would now still be blocked
        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.opening_avoidance_minutes",
            current_value=30, proposed_value=60,
        )
        assert result["replayed_rejections"] == 1
        assert result["would_pass_under_proposed"] == 0


class TestSellCooldownReplay:
    async def test_shorter_cooldown_unlocks_recent_rejections(self, session):
        today = datetime.utcnow().replace(hour=14, minute=0, second=0, microsecond=0)
        # SELL 1.5 days ago
        session.add(Order(
            market="KR", symbol="005930", side="SELL", order_type="market",
            quantity=10, price=80000.0, status="filled",
            filled_price=80000.0, filled_at=today - timedelta(days=1, hours=12),
        ))
        # Rejection now (1.5 days after SELL) — under cd=3, blocked;
        # under cd=1, would pass.
        session.add(FunnelEvent(
            ts=today,
            market="KR", symbol="005930", strategy_name="test",
            signal_confidence=0.7, decision="rejected",
            reject_reason="sell_cooldown", price=82000.0,
        ))
        await session.commit()

        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.sell_cooldown_days",
            current_value=3, proposed_value=1,
        )
        assert result["replayed_rejections"] == 1
        assert result["would_pass_under_proposed"] == 1
        assert result["no_sell_history"] == 0

    async def test_no_sell_history_tracked(self, session):
        """Event with no prior SELL → not_resolved bucket."""
        today = datetime.utcnow().replace(hour=14)
        session.add(FunnelEvent(
            ts=today, market="KR", symbol="XYZ", strategy_name="t",
            signal_confidence=0.6, decision="rejected",
            reject_reason="sell_cooldown", price=100.0,
        ))
        await session.commit()

        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.evaluation_loop.sell_cooldown_days",
            current_value=3, proposed_value=1,
        )
        assert result["replayed_rejections"] == 1
        assert result["no_sell_history"] == 1
        assert result["would_pass_under_proposed"] == 0


class TestNotReplayable:
    async def test_unknown_path_returns_message(self, session):
        result = await replay_recommendation(
            session=session,
            param_path="markets.KR.risk.kelly_fraction",
            current_value=0.4, proposed_value=0.5,
        )
        assert "not_replayable" in result
