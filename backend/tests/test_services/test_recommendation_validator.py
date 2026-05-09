"""Tests for the auto-backtest validator (B1)."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.models import AgentRecommendation, Base
from services.recommendation_validator import (
    _BACKTEST_PARAM_MAP,
    _is_validatable,
    _passes_floor,
    validate_recommendation,
)


@pytest_asyncio.fixture
async def db_setup():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


class TestPathMapping:
    def test_known_paths_validatable(self):
        assert _is_validatable("markets.KR.risk.max_positions")
        assert _is_validatable("markets.US.evaluation_loop.daily_buy_limit")
        assert _is_validatable("markets.KR.disabled_strategies")

    def test_unknown_paths_not_validatable(self):
        # Live-only param (timing) — not in backtest config
        assert not _is_validatable(
            "markets.KR.evaluation_loop.opening_avoidance_minutes"
        )
        # Restart-required param
        assert not _is_validatable("markets.KR.risk.kelly_fraction")


class TestFloorCheck:
    def test_passes_when_no_regression(self):
        baseline = {"ret": 10.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2}
        proposed = {"ret": 12.0, "sharpe": 0.6, "mdd": -9.0, "pf": 1.3}
        assert _passes_floor(baseline, proposed) is True

    def test_fails_on_ret_drop(self):
        baseline = {"ret": 10.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2}
        proposed = {"ret": 7.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2}
        assert _passes_floor(baseline, proposed) is False

    def test_fails_on_sharpe_drop(self):
        baseline = {"ret": 10.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2}
        proposed = {"ret": 10.0, "sharpe": 0.1, "mdd": -10.0, "pf": 1.2}
        assert _passes_floor(baseline, proposed) is False

    def test_fails_on_mdd_blowout(self):
        baseline = {"ret": 10.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2}
        proposed = {"ret": 10.0, "sharpe": 0.5, "mdd": -16.0, "pf": 1.2}
        assert _passes_floor(baseline, proposed) is False

    def test_handles_missing_metrics(self):
        assert _passes_floor({}, {}) is False
        assert _passes_floor(None, {"ret": 10}) is False


@pytest.mark.asyncio
async def test_validate_skips_unvalidatable_path(db_setup):
    """Recommendations that point at non-backtest params record a skip
    reason instead of running a backtest."""
    async with db_setup() as session:
        rec = AgentRecommendation(
            agent_type="trade_review",
            param_path="markets.KR.evaluation_loop.opening_avoidance_minutes",
            current_value=30,
            proposed_value=60,
            status="pending",
        )
        session.add(rec)
        await session.commit()
        rec_id = rec.id

    await validate_recommendation(rec_id, db_setup)

    async with db_setup() as session:
        loaded = await session.get(AgentRecommendation, rec_id)
        assert loaded.backtest_result is not None
        assert "skip" in loaded.backtest_result


@pytest.mark.asyncio
async def test_validate_skips_already_decided(db_setup):
    """Already-accepted/rejected recommendations don't re-run backtest."""
    async with db_setup() as session:
        rec = AgentRecommendation(
            agent_type="trade_review",
            param_path="markets.KR.risk.max_positions",
            current_value=18,
            proposed_value=22,
            status="accepted",
        )
        session.add(rec)
        await session.commit()
        rec_id = rec.id

    # Should silently no-op (no backtest, no DB write)
    await validate_recommendation(rec_id, db_setup)

    async with db_setup() as session:
        loaded = await session.get(AgentRecommendation, rec_id)
        assert loaded.backtest_result is None


@pytest.mark.asyncio
async def test_validate_persists_metrics_on_success(db_setup):
    """Happy path: backtest runs, baseline + proposed + delta + pass flag
    all stored on the recommendation."""
    async with db_setup() as session:
        rec = AgentRecommendation(
            agent_type="trade_review",
            param_path="markets.KR.risk.max_positions",
            current_value=18,
            proposed_value=22,
            status="pending",
        )
        session.add(rec)
        await session.commit()
        rec_id = rec.id

    fake_baseline = {"ret": 10.0, "sharpe": 0.5, "mdd": -10.0, "pf": 1.2, "trades": 100}
    fake_proposed = {"ret": 12.0, "sharpe": 0.6, "mdd": -9.0, "pf": 1.3, "trades": 105}

    with patch(
        "services.recommendation_validator._run_backtest",
        new=AsyncMock(side_effect=[fake_baseline, fake_proposed]),
    ):
        await validate_recommendation(rec_id, db_setup)

    async with db_setup() as session:
        loaded = await session.get(AgentRecommendation, rec_id)
        result = loaded.backtest_result
        assert result["baseline"] == fake_baseline
        assert result["proposed"] == fake_proposed
        assert result["delta"]["ret"] == 2.0
        assert result["passes_floor"] is True
