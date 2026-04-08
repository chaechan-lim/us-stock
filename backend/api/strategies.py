"""Strategy API endpoints."""

import logging

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_registry
from strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("/")
async def list_strategies(registry: StrategyRegistry = Depends(get_registry)):
    """List all strategies and their status."""
    strategies = registry.get_all()
    return [
        {
            "name": s.name,
            "display_name": s.display_name,
            "applicable_market_types": s.applicable_market_types,
            "timeframe": s.required_timeframe,
            "params": s.get_params(),
        }
        for s in strategies.values()
    ]


@router.get("/{name}/params")
async def get_strategy_params(
    name: str,
    registry: StrategyRegistry = Depends(get_registry),
):
    """Get parameters for a specific strategy."""
    strategy = registry.get(name)
    if not strategy:
        return {"error": f"Strategy '{name}' not found"}
    return {"name": name, "params": strategy.get_params()}


@router.post("/reload")
async def reload_config(
    request: Request,
    registry: StrategyRegistry = Depends(get_registry),
):
    """Hot-reload strategy configuration from YAML."""
    registry.reload_config()

    # STOCK-61: Also reload hard_sl_pct on evaluation loops
    hard_sl_pct = registry.config_loader.get_hard_sl_pct()
    if hasattr(request.app.state, "evaluation_loop"):
        request.app.state.evaluation_loop.reload_hard_sl_pct(hard_sl_pct)
    if hasattr(request.app.state, "kr_evaluation_loop"):
        request.app.state.kr_evaluation_loop.reload_hard_sl_pct(hard_sl_pct)

    # STOCK-65: Re-apply KR market-specific eval overrides so disabled_strategies,
    # min_confidence, and min_active_ratio stay in sync with the reloaded YAML.
    if hasattr(request.app.state, "apply_kr_eval_overrides"):
        request.app.state.apply_kr_eval_overrides()
        logger.info(
            "KR eval-loop config reloaded; KR risk params (kelly_fraction, "
            "dynamic_sl_tp, etc.) require a restart to take effect — they are "
            "NOT updated by hot-reload."
        )

    # 2026-04-09: Same for US — previously the hot-reload endpoint silently
    # ignored markets.US.disabled_strategies + cash_parking changes,
    # requiring a backend restart for every yaml tweak. The 4-08 trade
    # history showed donchian_breakout still buying after a yaml-only
    # disable + reload_strategies call. Fixed by extracting US setup into
    # _apply_us_eval_overrides.
    if hasattr(request.app.state, "apply_us_eval_overrides"):
        request.app.state.apply_us_eval_overrides()
        logger.info(
            "US eval-loop config reloaded (disabled_strategies + cash_parking)."
        )

    return {"status": "ok", "strategies": registry.get_names()}
