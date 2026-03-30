"""Strategy API endpoints."""

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_registry
from strategies.registry import StrategyRegistry

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
    hard_sl_pct = registry._config_loader.get_hard_sl_pct()
    if hasattr(request.app.state, "evaluation_loop"):
        request.app.state.evaluation_loop.reload_hard_sl_pct(hard_sl_pct)
    if hasattr(request.app.state, "kr_evaluation_loop"):
        request.app.state.kr_evaluation_loop.reload_hard_sl_pct(hard_sl_pct)

    # STOCK-65: Re-apply KR market-specific eval overrides so disabled_strategies,
    # min_confidence, and min_active_ratio stay in sync with the reloaded YAML.
    if hasattr(request.app.state, "apply_kr_eval_overrides"):
        request.app.state.apply_kr_eval_overrides()

    return {"status": "ok", "strategies": registry.get_names()}
