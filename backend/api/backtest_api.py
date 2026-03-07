"""Backtest API endpoints."""

import logging
import math

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backtest.engine import BacktestEngine
from backtest.simulator import SimConfig

router = APIRouter(prefix="/backtest", tags=["backtest"])
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    period: str = "3y"
    start: str | None = None
    end: str | None = None
    initial_equity: float = 100_000


@router.post("/run")
async def run_backtest(req: BacktestRequest, request: Request):
    """Run a backtest for a strategy on a symbol."""
    registry = request.app.state.registry
    strategies = registry.get_enabled()

    strategy = next((s for s in strategies if s.name == req.strategy_name), None)
    if not strategy:
        return {"error": f"Strategy '{req.strategy_name}' not found or not enabled"}

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

    def safe(v: float) -> float:
        """Replace inf/nan with 0 for JSON serialization."""
        if math.isinf(v) or math.isnan(v):
            return 0.0
        return v

    return {
        "strategy": result.strategy_name,
        "symbol": result.symbol,
        "passed": bool(result.passed),
        "metrics": {
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
        },
        "trades": [
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
            for t in result.trades[-50:]  # last 50 trades
        ],
        "equity_curve": [
            {"date": str(date), "equity": round(safe(float(val)), 2)}
            for date, val in result.equity_curve.items()
        ][-200:],  # last 200 data points
    }


@router.get("/strategies")
async def backtest_strategies(request: Request):
    """List available strategies for backtesting."""
    registry = request.app.state.registry
    return [
        {"name": s.name, "params": s.get_params()}
        for s in registry.get_enabled()
    ]
