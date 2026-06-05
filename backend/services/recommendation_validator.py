"""B1 — auto-backtest validation for agent recommendations.

When trade_review proposes `markets.KR.risk.max_positions: 18 → 22`, this
service runs a 2y FullPipelineBacktest with the change applied (in-memory
override) and saves the resulting metrics + a pass/fail flag onto the
recommendation row. Operator sees the projected impact alongside the
proposal in the dashboard.

Validation is a background task — backtest is slow (30-90s per market),
must not block the trade_review scheduler. Each validator runs
sequentially (cap-1 worker) to avoid CPU contention with the live engines.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from analytics.performance_metrics import compute_equity_metrics

logger = logging.getLogger(__name__)


# Path → (market, PipelineConfig field, type-coercer or None)
# Only params that the backtest config actually consumes can be validated.
# 2026-06-05: expanded from 11 to 33 paths — LLM rec coverage went from
# ~15% (most got skip) to ~80%. Mirror per-market for both KR and US.
def _both_markets(suffix: str, field: str, coercer: Any) -> dict[str, tuple[str, str, Any]]:
    """Helper: build {KR, US} entries for a yaml suffix→PipelineConfig field."""
    return {
        f"markets.KR.{suffix}": ("KR", field, coercer),
        f"markets.US.{suffix}": ("US", field, coercer),
    }


_BACKTEST_PARAM_MAP: dict[str, tuple[str, str, Any]] = {
    # ── risk params ─────────────────────────────────────────
    **_both_markets("risk.max_positions", "max_positions", int),
    **_both_markets("risk.max_position_pct", "max_position_pct", float),
    **_both_markets("risk.min_position_pct", "min_position_pct", float),
    **_both_markets("risk.default_stop_loss_pct", "default_stop_loss_pct", float),
    **_both_markets("risk.default_take_profit_pct", "default_take_profit_pct", float),
    **_both_markets("risk.kelly_fraction", "kelly_fraction", float),
    **_both_markets("risk.confidence_exponent", "confidence_exponent", float),
    **_both_markets("risk.hard_sl_pct", "hard_sl_pct", float),
    # ── evaluation_loop: anti-churn ─────────────────────────
    **_both_markets("evaluation_loop.sell_cooldown_days", "sell_cooldown_days", int),
    **_both_markets("evaluation_loop.min_hold_days", "min_hold_days", int),
    **_both_markets("evaluation_loop.whipsaw_max_losses", "whipsaw_max_losses", int),
    # ── evaluation_loop: stale exit ─────────────────────────
    **_both_markets("evaluation_loop.stale_time_days", "stale_time_days", int),
    **_both_markets(
        "evaluation_loop.stale_time_pnl_threshold", "stale_time_pnl_threshold", float
    ),
    # ── evaluation_loop: held-position bias ─────────────────
    **_both_markets("evaluation_loop.held_sell_bias", "held_sell_bias", float),
    **_both_markets("evaluation_loop.held_min_confidence", "held_min_confidence", float),
    # ── evaluation_loop: combiner thresholds ────────────────
    **_both_markets("evaluation_loop.min_confidence", "min_confidence", float),
    **_both_markets("evaluation_loop.min_active_ratio", "min_active_ratio", float),
    # ── evaluation_loop: scoring + budget ───────────────────
    **_both_markets("evaluation_loop.sector_boost_weight", "sector_boost_weight", float),
    **_both_markets("evaluation_loop.daily_buy_limit", "daily_buy_limit", int),
    # ── strategy-list overrides ─────────────────────────────
    "markets.KR.disabled_strategies": ("KR", "disabled_strategies", list),
    "markets.US.disabled_strategies": ("US", "disabled_strategies", list),
}


# Single-worker queue so backtests don't pile up
_validation_lock = asyncio.Lock()


def _coerce(value: Any, target_type: Any) -> Any:
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list:
        return list(value) if value is not None else []
    return value


def _is_validatable(param_path: str) -> bool:
    return param_path in _BACKTEST_PARAM_MAP


def _build_baseline_config(market: str) -> dict:
    """Return PipelineConfig kwargs matching the current live config."""
    from strategies.config_loader import StrategyConfigLoader

    loader = StrategyConfigLoader()
    disabled = loader.get_market_disabled_strategies(market)
    if market == "KR":
        kw = dict(
            market="KR", initial_equity=100_000_000,
            default_stop_loss_pct=0.12, default_take_profit_pct=0.20,
            max_positions=18, max_position_pct=0.20, min_position_pct=0.04,
            sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
            slippage_pct=0.08, volume_adjusted_slippage=True,
            min_confidence=0.30, sector_boost_weight=0.3,
            disabled_strategies=disabled,
        )
    else:
        kw = dict(
            market="US", initial_equity=100_000,
            default_stop_loss_pct=0.08, default_take_profit_pct=0.20,
            max_positions=20, max_position_pct=0.10, min_position_pct=0.05,
            sell_cooldown_days=1, whipsaw_max_losses=2, min_hold_days=1,
            slippage_pct=0.05, volume_adjusted_slippage=True,
            min_confidence=0.30, sector_boost_weight=0.2,
            disabled_strategies=disabled,
        )
    return kw


async def _run_backtest(kw: dict) -> dict | None:
    """Run a 2y pipeline backtest, return summary metrics dict or None."""
    from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig

    try:
        cfg = PipelineConfig(**kw)
        eng = FullPipelineBacktest(cfg)
        res = await eng.run(period="2y")
        m = res.metrics
        return {
            "ret": round(m.total_return_pct, 2),
            "sharpe": round(m.sharpe_ratio, 2),
            "mdd": round(m.max_drawdown_pct, 2),
            "pf": round(m.profit_factor, 2),
            "trades": m.total_trades,
        }
    except Exception as e:
        logger.warning("backtest validation failed: %s", e)
        return None


def _passes_floor(baseline: dict, proposed: dict) -> bool:
    """Floor: proposed must not regress any of {ret, sharpe, mdd, pf} by
    more than the gate's tolerances. Used to flag whether the change is
    backtest-safe."""
    if not baseline or not proposed:
        return False
    return (
        proposed["ret"] - baseline["ret"] >= -2.0
        and proposed["sharpe"] - baseline["sharpe"] >= -0.30
        and proposed["mdd"] - baseline["mdd"] >= -5.0
        and proposed["pf"] - baseline["pf"] >= -0.20
    )


async def validate_recommendation(rec_id: int, session_factory) -> None:
    """Compute backtest metrics for the recommendation's proposed change
    and persist the result. Runs sequentially via a process-wide lock so
    we never have two backtests competing for the cache + CPU at once.

    Stores `backtest_result = {baseline, proposed, delta, pass}` on the
    row. If the path isn't validatable (e.g. opening_avoidance_minutes —
    affects timing only, not backtest), records `{skip: "reason"}` instead.
    """
    from core.models import AgentRecommendation

    async with _validation_lock:
        async with session_factory() as session:
            rec = await session.get(AgentRecommendation, rec_id)
            if not rec:
                return
            if rec.status != "pending":
                return  # operator already decided

            if not _is_validatable(rec.param_path):
                # Hermes Phase 3 C2: fall through to funnel replay for
                # paths that affect rejection logic but no backtest knob.
                # Replay simulates the proposed change against the last
                # 30 days of FunnelEvent rows and reports would-pass
                # counts. Falls through to skip if path not replayable.
                from services.funnel_replay import (
                    is_replayable,
                    replay_recommendation,
                )
                if is_replayable(rec.param_path):
                    replay_result = await replay_recommendation(
                        session=session,
                        param_path=rec.param_path,
                        current_value=rec.current_value,
                        proposed_value=rec.proposed_value,
                    )
                    rec.backtest_result = replay_result
                else:
                    rec.backtest_result = {
                        "skip": f"path {rec.param_path!r} not in backtest config map",
                    }
                await session.commit()
                return

            market, field, coercer = _BACKTEST_PARAM_MAP[rec.param_path]
            try:
                proposed_value = _coerce(rec.proposed_value, coercer)
            except (TypeError, ValueError) as e:
                rec.backtest_result = {"skip": f"coerce failed: {e}"}
                await session.commit()
                return

            baseline_kw = _build_baseline_config(market)
            proposed_kw = dict(baseline_kw)
            proposed_kw[field] = proposed_value

            logger.info(
                "Validating recommendation #%d: %s = %r (vs baseline)",
                rec.id, rec.param_path, proposed_value,
            )
            baseline_metrics = await _run_backtest(baseline_kw)
            proposed_metrics = await _run_backtest(proposed_kw)

            if baseline_metrics and proposed_metrics:
                delta = {
                    k: round(proposed_metrics[k] - baseline_metrics[k], 2)
                    for k in ("ret", "sharpe", "mdd", "pf")
                }
                rec.backtest_result = {
                    "baseline": baseline_metrics,
                    "proposed": proposed_metrics,
                    "delta": delta,
                    "passes_floor": _passes_floor(baseline_metrics, proposed_metrics),
                }
            else:
                rec.backtest_result = {"error": "backtest run failed"}

            await session.commit()
            logger.info(
                "Validated #%d: %s",
                rec.id, rec.backtest_result,
            )
