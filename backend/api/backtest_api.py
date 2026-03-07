"""Backtest API endpoints."""

import logging
import math

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backtest.engine import BacktestEngine
from backtest.simulator import SimConfig
from backtest.result_store import BacktestResultStore
from backtest.adaptive_backtest import AdaptiveBacktestEngine

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
):
    """List all stored backtest results."""
    return {
        "count": _result_store.count,
        "results": _result_store.list_results(
            strategy_name=strategy, symbol=symbol, mode=mode,
        ),
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
