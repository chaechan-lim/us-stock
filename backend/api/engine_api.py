"""Engine control API endpoints."""

import asyncio

from fastapi import APIRouter, Request

from engine.scheduler import get_market_phase

router = APIRouter(prefix="/engine", tags=["engine"])


@router.get("/status")
async def engine_status(request: Request):
    """Get current engine status including scheduler info."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler:
        return scheduler.get_status()
    return {"running": False, "market_phase": "unknown"}


@router.post("/start")
async def start_engine(request: Request):
    """Start the trading scheduler."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        return {"status": "error", "detail": "Scheduler not initialized"}
    if scheduler.running:
        return {"status": "already_running"}
    asyncio.create_task(scheduler.start())
    return {"status": "started"}


@router.post("/stop")
async def stop_engine(request: Request):
    """Stop the trading scheduler."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        return {"status": "error", "detail": "Scheduler not initialized"}
    if not scheduler.running:
        return {"status": "already_stopped"}
    await scheduler.stop()
    return {"status": "stopped"}


@router.get("/market-state")
async def market_state(request: Request):
    """Get current market state assessment."""
    phase = get_market_phase()
    result = {"market_phase": phase.value}

    detector = getattr(request.app.state, "market_state_detector", None)
    if detector and detector.last_state:
        state = detector.last_state
        result.update({
            "regime": state.regime.value,
            "spy_price": state.spy_price,
            "spy_sma200": state.spy_sma200,
            "spy_above_sma200": state.spy_above_sma200,
            "spy_distance_pct": state.spy_distance_pct,
            "vix_level": state.vix_level,
            "confidence": state.confidence,
        })

    return result


@router.get("/recovery")
async def recovery_status(request: Request):
    """Get circuit breaker and recovery status for all tasks."""
    recovery = getattr(request.app.state, "recovery", None)
    if not recovery:
        return {"status": "not_configured"}
    return recovery.get_status()


@router.post("/recovery/reset/{task_name}")
async def reset_circuit(task_name: str, request: Request):
    """Manually reset a circuit breaker for a specific task."""
    recovery = getattr(request.app.state, "recovery", None)
    if not recovery:
        return {"status": "error", "detail": "Recovery not configured"}
    if recovery.reset_circuit(task_name):
        return {"status": "reset", "task": task_name}
    return {"status": "error", "detail": f"Task '{task_name}' not found"}


@router.post("/recovery/reset-all")
async def reset_all_circuits(request: Request):
    """Reset all circuit breakers."""
    recovery = getattr(request.app.state, "recovery", None)
    if not recovery:
        return {"status": "error", "detail": "Recovery not configured"}
    recovery.reset_all()
    return {"status": "all_reset"}


@router.get("/macro")
async def macro_indicators(request: Request):
    """Get latest FRED macroeconomic indicators."""
    fred = getattr(request.app.state, "fred_service", None)
    if not fred:
        return {"status": "not_configured"}
    if not fred.available:
        return {"status": "no_api_key"}
    indicators = fred.last_indicators
    if indicators:
        return indicators.to_dict()
    return {"status": "not_fetched_yet"}


@router.get("/adaptive-weights")
async def adaptive_weights_status(request: Request):
    """Get per-stock adaptive weight status and classifications."""
    eval_loop = getattr(request.app.state, "evaluation_loop", None)
    if not eval_loop or not hasattr(eval_loop, "_adaptive"):
        return {"status": "not_configured"}

    adaptive = eval_loop._adaptive
    categories = {
        sym: cat.value for sym, cat in adaptive._categories.items()
    }
    performance = adaptive.get_all_summaries()
    return {
        "categories": categories,
        "performance": performance,
        "config": {
            "alpha": adaptive._alpha,
            "min_signals": adaptive._min_signals,
        },
    }


@router.get("/websocket")
async def websocket_status(request: Request):
    """Get KIS WebSocket connection status."""
    ws = getattr(request.app.state, "kis_ws", None)
    if not ws:
        return {"status": "not_configured"}
    return ws.get_status()
