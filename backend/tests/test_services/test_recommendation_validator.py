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


class TestCoerce:
    """Direct unit tests for the _coerce helper."""

    def test_int(self):
        from services.recommendation_validator import _coerce
        assert _coerce("18", int) == 18
        assert _coerce(18.7, int) == 18

    def test_float(self):
        from services.recommendation_validator import _coerce
        assert _coerce("0.15", float) == pytest.approx(0.15)
        assert _coerce(15, float) == pytest.approx(15.0)

    def test_list(self):
        from services.recommendation_validator import _coerce
        assert _coerce(["a", "b"], list) == ["a", "b"]
        assert _coerce(("a",), list) == ["a"]

    def test_list_none(self):
        from services.recommendation_validator import _coerce
        assert _coerce(None, list) == []

    def test_unknown_type_returns_as_is(self):
        from services.recommendation_validator import _coerce
        # No coercer registered → return value untouched
        assert _coerce("foo", None) == "foo"
        assert _coerce(42, str) == 42  # str isn't one of the handled branches


class TestBuildBaselineConfig:
    def test_us_branch(self):
        from services.recommendation_validator import _build_baseline_config
        kw = _build_baseline_config("US")
        assert kw["market"] == "US"
        assert kw["initial_equity"] == 100_000
        assert kw["max_positions"] == 20

    def test_kr_branch(self):
        from services.recommendation_validator import _build_baseline_config
        kw = _build_baseline_config("KR")
        assert kw["market"] == "KR"
        assert kw["initial_equity"] == 100_000_000


class TestRunBacktestPure:
    """Cover the success and exception branches of _run_backtest."""

    async def test_success_returns_metrics_dict(self):
        from services.recommendation_validator import _run_backtest

        class _Metrics:
            total_return_pct = 12.34
            sharpe_ratio = 1.23
            max_drawdown_pct = -5.67
            profit_factor = 1.45
            total_trades = 87

        class _Result:
            metrics = _Metrics()

        class _Engine:
            def __init__(self, cfg):
                self.cfg = cfg
            async def run(self, period):
                assert period == "2y"
                return _Result()

        with patch(
            "backtest.full_pipeline.FullPipelineBacktest",
            new=_Engine,
        ):
            out = await _run_backtest({
                "market": "US", "initial_equity": 100_000,
                "default_stop_loss_pct": 0.08, "default_take_profit_pct": 0.20,
                "max_positions": 20, "max_position_pct": 0.10, "min_position_pct": 0.05,
                "sell_cooldown_days": 1, "whipsaw_max_losses": 2, "min_hold_days": 1,
                "slippage_pct": 0.05, "volume_adjusted_slippage": True,
                "min_confidence": 0.30, "sector_boost_weight": 0.2,
                "disabled_strategies": [],
            })
        assert out == {
            "ret": 12.34, "sharpe": 1.23, "mdd": -5.67, "pf": 1.45, "trades": 87,
        }

    async def test_exception_returns_none(self):
        from services.recommendation_validator import _run_backtest

        class _Engine:
            def __init__(self, cfg):
                raise RuntimeError("config invalid")

        with patch(
            "backtest.full_pipeline.FullPipelineBacktest",
            new=_Engine,
        ):
            out = await _run_backtest({
                "market": "US", "initial_equity": 100_000,
                "default_stop_loss_pct": 0.08, "default_take_profit_pct": 0.20,
                "max_positions": 20, "max_position_pct": 0.10, "min_position_pct": 0.05,
                "sell_cooldown_days": 1, "whipsaw_max_losses": 2, "min_hold_days": 1,
                "slippage_pct": 0.05, "volume_adjusted_slippage": True,
                "min_confidence": 0.30, "sector_boost_weight": 0.2,
                "disabled_strategies": [],
            })
        assert out is None


@pytest.mark.asyncio
async def test_validate_missing_rec_returns(db_setup):
    """Passing a non-existent rec_id is a no-op."""
    await validate_recommendation(rec_id=999_999, session_factory=db_setup)
    # No write, no raise.


@pytest.mark.asyncio
async def test_validate_coerce_failure_records_skip(db_setup):
    """proposed_value that can't coerce to expected type → 'skip: coerce failed'."""
    async with db_setup() as session:
        rec = AgentRecommendation(
            agent_type="trade_review",
            param_path="markets.KR.risk.max_positions",
            current_value=18,
            proposed_value="not_a_number",  # can't int()
            status="pending",
        )
        session.add(rec)
        await session.commit()
        rec_id = rec.id

    await validate_recommendation(rec_id, db_setup)

    async with db_setup() as session:
        loaded = await session.get(AgentRecommendation, rec_id)
        assert "coerce failed" in (loaded.backtest_result or {}).get("skip", "")


@pytest.mark.asyncio
async def test_validate_backtest_run_failure_records_error(db_setup):
    """When _run_backtest returns None → backtest_result = {'error': …}."""
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

    with patch(
        "services.recommendation_validator._run_backtest",
        new=AsyncMock(return_value=None),
    ):
        await validate_recommendation(rec_id, db_setup)

    async with db_setup() as session:
        loaded = await session.get(AgentRecommendation, rec_id)
        assert loaded.backtest_result == {"error": "backtest run failed"}
