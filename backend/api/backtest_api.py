"""Backtest API endpoints."""

import logging
import math

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backtest.engine import BacktestEngine
from backtest.simulator import SimConfig
from backtest.result_store import BacktestResultStore
from backtest.adaptive_backtest import AdaptiveBacktestEngine
from backtest.full_pipeline import FullPipelineBacktest, PipelineConfig, DEFAULT_UNIVERSE
from backtest.optimize_all import (
    optimize_strategy, optimize_all, PARAM_GRIDS, SL_TP_GRIDS,
    optimize_sl_tp, run_walk_forward,
)

router = APIRouter(prefix="/backtest", tags=["backtest"])
logger = logging.getLogger(__name__)

_result_store = BacktestResultStore()


def safe(v: float) -> float:
    """Replace inf/nan with 0 for JSON serialization."""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return 0.0
    return v


def _format_metrics(m) -> dict:
    return {
        "total_return_pct": round(safe(m.total_return_pct), 2),
        "cagr": round(safe(m.cagr * 100), 2),
        "sharpe_ratio": round(safe(m.sharpe_ratio), 2),
        "sortino_ratio": round(safe(m.sortino_ratio), 2),
        "max_drawdown_pct": round(safe(m.max_drawdown_pct), 2),
        "max_drawdown_days": m.max_drawdown_days,
        "total_trades": m.total_trades,
        "win_rate": round(safe(m.win_rate), 1),
        "profit_factor": round(safe(m.profit_factor), 2),
        "final_equity": round(safe(m.final_equity), 2),
        "start_date": m.start_date,
        "end_date": m.end_date,
        "trading_days": m.trading_days,
    }


def _format_trades(trades) -> list[dict]:
    return [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": round(safe(t.pnl), 2),
            "pnl_pct": round(safe(t.pnl_pct), 2),
            "holding_days": t.holding_days,
        }
        for t in trades[-50:]
    ]


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    period: str = "3y"
    start: str | None = None
    end: str | None = None
    initial_equity: float = 100_000
    force: bool = False


@router.post("/run")
async def run_backtest(req: BacktestRequest, request: Request):
    """Run a backtest for a strategy on a symbol. Skips if already stored."""
    registry = request.app.state.registry
    strategies = registry.get_enabled()

    strategy = next((s for s in strategies if s.name == req.strategy_name), None)
    if not strategy:
        return {"error": f"Strategy '{req.strategy_name}' not found or not enabled"}

    # Check dedup
    params = strategy.get_params()
    if not req.force and _result_store.exists(
        req.strategy_name, req.symbol, req.period, params
    ):
        stored = _result_store.get(
            req.strategy_name, req.symbol, req.period, params
        )
        if stored:
            logger.info(
                "Returning cached result for %s/%s", req.strategy_name, req.symbol
            )
            return {**stored["result"], "cached": True}

    engine = BacktestEngine(sim_config=SimConfig(initial_equity=req.initial_equity))

    try:
        result = await engine.run(
            strategy=strategy,
            symbol=req.symbol,
            period=req.period,
            start=req.start,
            end=req.end,
        )
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        return {"error": str(e)}

    m = result.metrics

    response = {
        "strategy": result.strategy_name,
        "symbol": result.symbol,
        "passed": bool(result.passed),
        "metrics": _format_metrics(m),
        "trades": _format_trades(result.trades),
        "equity_curve": [
            {"date": str(date), "equity": round(safe(float(val)), 2)}
            for date, val in result.equity_curve.items()
        ][-200:],
    }

    # Save to store
    _result_store.save(
        strategy_name=req.strategy_name,
        symbol=req.symbol,
        period=req.period,
        result_data=response,
        params=params,
    )

    return {**response, "cached": False}


@router.get("/strategies")
async def backtest_strategies(request: Request):
    """List available strategies for backtesting."""
    registry = request.app.state.registry
    return [
        {"name": s.name, "params": s.get_params()}
        for s in registry.get_enabled()
    ]


# --- Result Store endpoints ---


@router.get("/results")
async def list_results(
    strategy: str | None = None,
    symbol: str | None = None,
    mode: str | None = None,
    include_stale: bool = True,
):
    """List all stored backtest results.

    Each result includes a ``stale`` boolean. When ``include_stale=False``,
    results produced before the reliability cutoff (commit ff6279f) are
    excluded — use this when basing live decisions on the response.
    """
    results = _result_store.list_results(
        strategy_name=strategy, symbol=symbol, mode=mode,
        include_stale=include_stale,
    )
    return {
        "count": len(results),
        "stale_count": sum(1 for r in results if r.get("stale")),
        "results": results,
    }


@router.get("/results/{key}")
async def get_result(key: str):
    """Get a stored backtest result by key."""
    result = _result_store.get_by_key(key)
    if not result:
        return {"error": "Result not found"}
    return result


@router.delete("/results/{key}")
async def delete_result(key: str):
    """Delete a stored backtest result."""
    if _result_store.delete(key):
        return {"status": "deleted", "key": key}
    return {"error": "Result not found"}


# --- Adaptive Backtest endpoints ---


class AdaptiveBacktestRequest(BaseModel):
    symbols: list[str] = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ"]
    period: str = "3y"
    modes: list[str] | None = None
    strategies: list[str] | None = None
    market_state: str = "uptrend"
    force: bool = False


@router.post("/adaptive")
async def run_adaptive_backtest(req: AdaptiveBacktestRequest):
    """Run adaptive weight comparison backtest."""
    store_key_symbol = ",".join(sorted(req.symbols))
    store_key_strats = ",".join(sorted(req.strategies)) if req.strategies else "all"

    # Check dedup
    if not req.force and _result_store.exists(
        strategy_name=store_key_strats,
        symbol=store_key_symbol,
        period=req.period,
        params={"market_state": req.market_state, "modes": req.modes},
        mode="adaptive_comparison",
    ):
        stored = _result_store.get(
            strategy_name=store_key_strats,
            symbol=store_key_symbol,
            period=req.period,
            params={"market_state": req.market_state, "modes": req.modes},
            mode="adaptive_comparison",
        )
        if stored:
            logger.info("Returning cached adaptive backtest result")
            return {**stored["result"], "cached": True}

    engine = AdaptiveBacktestEngine(market_state=req.market_state)

    try:
        result = await engine.run(
            symbols=req.symbols,
            period=req.period,
            modes=req.modes,
            strategy_names=req.strategies,
        )
    except Exception as e:
        logger.error("Adaptive backtest failed: %s", e)
        return {"error": str(e)}

    response = result.to_dict()

    # Save
    _result_store.save(
        strategy_name=store_key_strats,
        symbol=store_key_symbol,
        period=req.period,
        result_data=response,
        params={"market_state": req.market_state, "modes": req.modes},
        mode="adaptive_comparison",
    )

    return {**response, "cached": False}


# --- Optimization endpoints ---


class OptimizeRequest(BaseModel):
    strategy_name: str | None = None  # None = all strategies
    symbols: list[str] = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ"]
    period: str = "3y"
    metric: str = "sharpe_ratio"
    force: bool = False


@router.post("/optimize")
async def run_optimization(req: OptimizeRequest):
    """Run parameter optimization for one or all strategies."""
    if req.strategy_name:
        # Single strategy
        if req.strategy_name not in PARAM_GRIDS:
            return {"error": f"No param grid for '{req.strategy_name}'"}
        try:
            result = await optimize_strategy(
                strategy_name=req.strategy_name,
                symbols=req.symbols,
                period=req.period,
                metric=req.metric,
                result_store=_result_store,
                force=req.force,
            )
            return {
                "strategy": result.strategy_name,
                "best_params": result.best_params,
                "avg_sharpe": round(safe(result.avg_sharpe), 3),
                "avg_cagr": round(safe(result.avg_cagr * 100), 2),
                "grid_combos": result.grid_combos,
                "elapsed_sec": round(result.elapsed_sec, 1),
                "per_symbol": {
                    sym: {k: round(safe(v), 3) if isinstance(v, float) else v
                          for k, v in m.items()}
                    for sym, m in result.per_symbol.items()
                },
            }
        except Exception as e:
            logger.error("Optimization failed: %s", e)
            return {"error": str(e)}
    else:
        # All strategies
        try:
            full = await optimize_all(
                symbols=req.symbols,
                period=req.period,
                metric=req.metric,
                force=req.force,
            )
            return full.to_dict()
        except Exception as e:
            logger.error("Full optimization failed: %s", e)
            return {"error": str(e)}


@router.get("/optimize/grids")
async def get_param_grids():
    """Get parameter grids for all strategies."""
    return PARAM_GRIDS


@router.post("/optimize/apply")
async def apply_optimized_params(request: Request):
    """Apply optimized parameters to the running strategy registry.

    Reads best_params from stored optimization results and updates
    the live strategy instances.
    """
    registry = request.app.state.registry
    stored = _result_store.list_results(mode="optimization")
    if not stored:
        return {"error": "No optimization results found. Run /optimize first."}

    applied = {}
    for entry in stored:
        data = _result_store.get_by_key(entry["key"])
        if not data or "result" not in data:
            continue
        result = data["result"]
        strat_name = result.get("strategy_name", "")
        best_params = result.get("best_params", {})
        if not strat_name or not best_params:
            continue

        # Find strategy in registry
        for s in registry.get_enabled():
            if s.name == strat_name:
                old_params = s.get_params()
                s.set_params(best_params)
                applied[strat_name] = {
                    "old": old_params,
                    "new": best_params,
                }
                logger.info("Applied optimized params for %s: %s", strat_name, best_params)
                break

    return {
        "status": "applied",
        "count": len(applied),
        "strategies": applied,
    }


# --- SL/TP Optimization ---


class SlTpOptimizeRequest(BaseModel):
    strategy_name: str
    symbols: list[str] = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ"]
    period: str = "3y"
    metric: str = "sharpe_ratio"
    force: bool = False


@router.post("/optimize/sl-tp")
async def run_sl_tp_optimization(req: SlTpOptimizeRequest):
    """Optimize stop-loss / take-profit / trailing stop for a strategy."""
    try:
        result = await optimize_sl_tp(
            strategy_name=req.strategy_name,
            symbols=req.symbols,
            period=req.period,
            metric=req.metric,
            result_store=_result_store,
            force=req.force,
        )
        return result
    except Exception as e:
        logger.error("SL/TP optimization failed: %s", e)
        return {"error": str(e)}


@router.get("/optimize/sl-tp/presets")
async def get_sl_tp_presets():
    """Get available SL/TP presets."""
    return SL_TP_GRIDS


# --- Walk-Forward Validation ---


class WalkForwardRequest(BaseModel):
    strategy_name: str
    symbol: str = "AAPL"
    period: str = "5y"
    n_splits: int = 3
    metric: str = "sharpe_ratio"
    force: bool = False


@router.post("/optimize/walk-forward")
async def run_walk_forward_validation(req: WalkForwardRequest):
    """Run walk-forward analysis to check for overfitting."""
    try:
        result = await run_walk_forward(
            strategy_name=req.strategy_name,
            symbol=req.symbol,
            period=req.period,
            n_splits=req.n_splits,
            metric=req.metric,
            result_store=_result_store,
            force=req.force,
        )
        return result
    except Exception as e:
        logger.error("Walk-forward failed: %s", e)
        return {"error": str(e)}


# --- Full Pipeline Backtest ---


class PipelineBacktestRequest(BaseModel):
    universe: list[str] | None = None  # None = default 55-stock universe
    period: str = "3y"
    initial_equity: float = 100_000
    max_positions: int = 20
    max_watchlist: int = 30
    screen_interval: int = 20
    dynamic_sl_tp: bool = True


@router.post("/pipeline")
async def run_pipeline_backtest(req: PipelineBacktestRequest):
    """Run full pipeline backtest (scanner → combiner → Kelly → SL/TP).

    Simulates the complete live trading system on historical data.
    WARNING: This is computationally intensive (5-15 min for 3y × 55 stocks).
    """
    config = PipelineConfig(
        universe=req.universe or list(DEFAULT_UNIVERSE),
        initial_equity=req.initial_equity,
        max_positions=req.max_positions,
        max_watchlist=req.max_watchlist,
        screen_interval=req.screen_interval,
        dynamic_sl_tp=req.dynamic_sl_tp,
    )

    engine = FullPipelineBacktest(config)

    try:
        result = await engine.run(period=req.period)
    except Exception as e:
        logger.error("Pipeline backtest failed: %s", e)
        return {"error": str(e)}

    m = result.metrics
    return {
        "passed": bool(m.passes_minimum()),
        "metrics": {
            **_format_metrics(m),
            "benchmark_return_pct": round(safe(m.benchmark_return_pct), 2),
            "alpha": round(safe(m.alpha), 2),
            "avg_holding_days": round(safe(m.avg_holding_days), 1),
            "avg_win_pct": round(safe(m.avg_win_pct), 2),
            "avg_loss_pct": round(safe(m.avg_loss_pct), 2),
        },
        "strategy_breakdown": result.strategy_stats,
        "equity_curve": [
            {"date": str(date), "equity": round(safe(float(val)), 2)}
            for date, val in result.equity_curve.items()
        ][-500:],
        "trades_count": len(result.trades),
        "trades": _format_trades(result.trades),
        "config": {
            "universe_size": len(config.universe),
            "max_positions": config.max_positions,
            "dynamic_sl_tp": config.dynamic_sl_tp,
            "screen_interval": config.screen_interval,
        },
    }
