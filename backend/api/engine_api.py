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
    notification = getattr(request.app.state, "notification", None)
    if notification:
        await notification.notify_system_event("engine_start", "Trading engine started")
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
    notification = getattr(request.app.state, "notification", None)
    if notification:
        await notification.notify_system_event("engine_stop", "Trading engine stopped")
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


@router.post("/evaluate")
async def run_evaluation(request: Request):
    """Manually trigger one evaluation cycle (for testing outside market hours)."""
    eval_loop = getattr(request.app.state, "evaluation_loop", None)
    if not eval_loop:
        return {"status": "error", "detail": "Evaluation loop not initialized"}

    # Load watchlist from DB if eval_loop's watchlist is empty
    if not eval_loop._watchlist:
        from db.session import get_session_factory
        from db.trade_repository import TradeRepository
        try:
            factory = get_session_factory()
            async with factory() as session:
                repo = TradeRepository(session)
                items = await repo.get_watchlist(active_only=True)
                eval_loop.set_watchlist([w.symbol for w in items])
        except Exception:
            pass

    if not eval_loop._watchlist:
        return {"status": "error", "detail": "Watchlist is empty. Add symbols via /watchlist/ first."}

    try:
        await eval_loop._evaluate_all()
        return {
            "status": "ok",
            "symbols_evaluated": eval_loop._watchlist,
            "market_state": eval_loop._market_state,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/run-task/{task_name}")
async def run_task(task_name: str, request: Request):
    """Manually trigger a scheduler task by name."""
    scheduler = getattr(request.app.state, "scheduler", None)
    if not scheduler:
        return {"status": "error", "detail": "Scheduler not initialized"}

    task = next((t for t in scheduler._tasks if t.name == task_name), None)
    if not task:
        available = [t.name for t in scheduler._tasks]
        return {"status": "error", "detail": f"Task '{task_name}' not found", "available": available}

    try:
        await task.fn()
        return {"status": "ok", "task": task_name}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/websocket")
async def websocket_status(request: Request):
    """Get KIS WebSocket connection status."""
    ws = getattr(request.app.state, "kis_ws", None)
    if not ws:
        return {"status": "not_configured"}
    return ws.get_status()


@router.get("/etf")
async def etf_engine_status(request: Request):
    """Get ETF engine status: regime, sector rotation, managed positions."""
    etf_engine = getattr(request.app.state, "etf_engine", None)
    if not etf_engine:
        return {"status": "not_configured"}
    status = etf_engine.get_status()

    # If engine hasn't evaluated yet, detect current state from market data
    market_data = getattr(request.app.state, "market_data", None)

    if status["last_regime"] is None and market_data:
        detector = getattr(request.app.state, "market_state_detector", None)
        if detector:
            try:
                spy_df = await market_data.get_ohlcv("SPY", limit=250)
                if not spy_df.empty:
                    state = detector.detect(spy_df)
                    status["last_regime"] = state.regime.value
            except Exception:
                pass

    if not status["top_sectors"] and market_data:
        external_data = getattr(request.app.state, "external_data", None)
        sector_analyzer = getattr(request.app.state, "sector_analyzer", None)
        if external_data and sector_analyzer:
            try:
                sector_data = await external_data.get_sector_performance()
                if sector_data:
                    scores = sector_analyzer.analyze(sector_data)
                    top = sector_analyzer.get_top_sectors(scores, n=3, min_score=0)
                    status["top_sectors"] = [s.name for s in top]
            except Exception:
                pass

    return status


@router.get("/analytics/factors")
async def factor_scores(request: Request):
    """Get current factor model scores for watchlist stocks."""
    eval_loop = getattr(request.app.state, "evaluation_loop", None)
    if not eval_loop:
        return {"status": "not_configured"}

    scores = eval_loop.factor_scores
    return {
        "count": len(scores),
        "scores": [
            {
                "symbol": s.symbol,
                "growth": s.growth,
                "profitability": s.profitability,
                "garp": s.garp,
                "momentum": s.momentum,
                "composite": s.composite,
                "rank": s.rank,
            }
            for s in sorted(scores.values(), key=lambda x: x.rank)
        ],
    }


@router.get("/analytics/signal-quality")
async def signal_quality(request: Request):
    """Get strategy signal quality metrics."""
    eval_loop = getattr(request.app.state, "evaluation_loop", None)
    if not eval_loop:
        return {"status": "not_configured"}

    tracker = eval_loop.signal_quality
    all_metrics = tracker.get_all_metrics()
    return {
        "strategies": {
            name: {
                "win_rate": m.win_rate,
                "avg_win": m.avg_win,
                "avg_loss": m.avg_loss,
                "profit_factor": m.profit_factor,
                "total_trades": m.total_trades,
                "quality_score": m.quality_score,
                "has_edge": m.has_edge,
            }
            for name, m in all_metrics.items()
        },
        "active": tracker.get_active_strategies(),
        "gated": tracker.get_gated_strategies(),
    }
