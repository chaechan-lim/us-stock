"""US Stock Trading MCP Server.

Exposes the trading system's REST API as MCP tools so Claude Desktop/Code
can control it via natural language using the user's Max plan.

Run: python mcp_server.py
Or configure in Claude Desktop's claude_desktop_config.json.
"""

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Backend API base URL and auth
API_BASE = os.getenv("USSTOCK_API_URL", "http://localhost:8001")
API_TOKEN = os.getenv("AUTH_API_TOKEN", "")

mcp = FastMCP(
    "US Stock Trading",
    instructions=(
        "US stock auto-trading system controller. "
        "Use these tools to check portfolio, market state, scanner results, "
        "and manage the trading engine. Read-only tools are safe to call anytime. "
        "Engine control tools (start/stop/evaluate) affect live trading."
    ),
)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _api(
    method: str, path: str, params: dict | None = None, body: dict | None = None,
) -> dict[str, Any]:
    """Call the FastAPI backend."""
    url = f"{API_BASE}{path}"
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        if method == "GET":
            resp = await client.get(url, params=params)
        elif method == "POST":
            resp = await client.post(url, json=body)
        elif method == "DELETE":
            resp = await client.delete(url)
        else:
            return {"error": f"Unsupported method: {method}"}

        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        return resp.json()


# ===========================================================================
# Portfolio & Account Tools
# ===========================================================================

@mcp.tool()
async def get_portfolio_summary() -> str:
    """Get portfolio summary: balance, positions, unrealized P&L.

    Returns total equity, available cash, all open positions with
    current prices and unrealized profit/loss.
    """
    data = await _api("GET", "/api/v1/portfolio/summary")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_positions() -> str:
    """Get all current stock positions.

    Shows symbol, quantity, avg price, current price, unrealized P&L
    for each position.
    """
    data = await _api("GET", "/api/v1/portfolio/positions")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_equity_history(days: int = 30) -> str:
    """Get portfolio equity curve history.

    Args:
        days: Number of days to look back (default 30).
    """
    data = await _api("GET", "/api/v1/portfolio/equity-history", params={"days": days})
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_trade_history(limit: int = 20, symbol: str | None = None) -> str:
    """Get recent trade history (executed orders).

    Args:
        limit: Max trades to return (default 20).
        symbol: Optional filter by symbol (e.g. "AAPL").
    """
    params: dict[str, Any] = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    data = await _api("GET", "/api/v1/trades/", params=params)
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_trade_summary() -> str:
    """Get aggregated trade statistics: win rate, total P&L, trade count."""
    data = await _api("GET", "/api/v1/trades/summary")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Market Data Tools
# ===========================================================================

@mcp.tool()
async def get_price(symbol: str) -> str:
    """Get current price, change %, and volume for a stock.

    Args:
        symbol: Stock ticker (e.g. "AAPL", "NVDA", "TSLA").
    """
    data = await _api("GET", f"/api/v1/market/price/{symbol}")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_market_state() -> str:
    """Get current market regime and conditions.

    Returns market state (uptrend/downtrend/choppy), VIX level,
    SPY trend, and regime confidence.
    """
    data = await _api("GET", "/api/v1/engine/market-state")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_macro_indicators() -> str:
    """Get FRED macro economic indicators.

    Returns Fed Funds Rate, 10Y/2Y Treasury yields, CPI,
    unemployment rate, and yield curve status.
    """
    data = await _api("GET", "/api/v1/engine/macro")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Scanner & Universe Tools
# ===========================================================================

@mcp.tool()
async def scan_stocks(symbols: list[str], min_grade: str = "B") -> str:
    """Run technical indicator screening on stocks.

    Grades each stock A-F based on technical indicators (RSI, MACD,
    Bollinger, volume, etc.). Only returns stocks at or above min_grade.

    Args:
        symbols: List of stock tickers to scan (e.g. ["AAPL", "NVDA", "TSLA"]).
        min_grade: Minimum grade to include ("A", "B", "C", "D", "F").
    """
    data = await _api("POST", "/api/v1/scanner/run", body={
        "symbols": symbols, "min_grade": min_grade,
    })
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def discover_universe() -> str:
    """Discover stocks dynamically using screeners.

    Combines yfinance screeners (most active, gainers, growth tech),
    sector rotation analysis, and KIS ranking APIs to find tradeable
    stocks. Returns discovered symbols with source tracking.
    """
    data = await _api("GET", "/api/v1/scanner/universe")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_sector_performance() -> str:
    """Get sector ETF performance data.

    Shows 1-week and 1-month returns for each sector ETF
    (XLK, XLF, XLE, etc.) to identify sector rotation trends.
    """
    data = await _api("GET", "/api/v1/scanner/sectors")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Strategy & Analytics Tools
# ===========================================================================

@mcp.tool()
async def get_strategies() -> str:
    """List all 13 trading strategies with their parameters.

    Shows strategy name, type (trend/reversal/momentum), applicable
    market states, current weight, and key parameters.
    """
    data = await _api("GET", "/api/v1/strategies/")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_signal_quality() -> str:
    """Get strategy signal quality metrics.

    Shows win rate, profit factor, average win/loss, and gating
    status for each strategy. Strategies below quality threshold
    are automatically gated (disabled).
    """
    data = await _api("GET", "/api/v1/engine/analytics/signal-quality")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_factor_scores() -> str:
    """Get multi-factor analysis scores for watchlist stocks.

    Returns growth, profitability, GARP, and momentum factor
    z-scores for each stock in the watchlist.
    """
    data = await _api("GET", "/api/v1/engine/analytics/factors")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_etf_status() -> str:
    """Get ETF Engine status.

    Shows current regime, leveraged pair positions (TQQQ/SQQQ etc.),
    sector ETF rotation status, and exposure limits.
    """
    data = await _api("GET", "/api/v1/engine/etf")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Watchlist Tools
# ===========================================================================

@mcp.tool()
async def get_watchlist() -> str:
    """Get the current watchlist of monitored stocks."""
    data = await _api("GET", "/api/v1/watchlist/")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def add_to_watchlist(symbol: str) -> str:
    """Add a stock symbol to the watchlist.

    The symbol will be monitored by the evaluation loop and scanner.

    Args:
        symbol: Stock ticker to add (e.g. "AAPL").
    """
    data = await _api("POST", "/api/v1/watchlist/", body={
        "symbol": symbol.upper(), "exchange": "NASD",
    })
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def remove_from_watchlist(symbol: str) -> str:
    """Remove a stock symbol from the watchlist.

    Args:
        symbol: Stock ticker to remove (e.g. "AAPL").
    """
    data = await _api("DELETE", f"/api/v1/watchlist/{symbol.upper()}")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Engine Control Tools
# ===========================================================================

@mcp.tool()
async def get_engine_status() -> str:
    """Get trading engine and scheduler status.

    Shows if the scheduler is running, which tasks are active,
    last execution times, and any circuit breaker states.
    """
    data = await _api("GET", "/api/v1/engine/status")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def start_engine() -> str:
    """Start the trading scheduler.

    This activates all scheduled tasks: evaluation loop, scanners,
    position checks, etc. Only works during market hours for trading tasks.
    """
    data = await _api("POST", "/api/v1/engine/start")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def stop_engine() -> str:
    """Stop the trading scheduler.

    Gracefully stops all scheduled tasks. Open positions are NOT
    automatically closed — they remain until manually managed.
    """
    data = await _api("POST", "/api/v1/engine/stop")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def run_evaluation() -> str:
    """Manually trigger one evaluation cycle.

    Runs all 13 strategies on all watchlist stocks, generates
    signals, and executes trades if conditions are met.
    """
    data = await _api("POST", "/api/v1/engine/evaluate")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def run_task(task_name: str) -> str:
    """Manually trigger a specific scheduler task.

    Args:
        task_name: Task to run. Available tasks:
            - health_check: System health check
            - position_check: Check all positions for SL/TP
            - daily_scan: Full stock screening (Layer 1+2+3)
            - intraday_hot_scan: Quick intraday momentum scan
            - sector_analysis: Sector rotation analysis
            - market_state_update: Update market regime (SPY/VIX)
            - etf_evaluation: ETF engine evaluation cycle
            - portfolio_snapshot: Save portfolio snapshot
            - macro_update: Fetch FRED macro indicators
            - daily_briefing: Generate daily summary
    """
    data = await _api("POST", f"/api/v1/engine/run-task/{task_name}")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def reload_strategies() -> str:
    """Hot-reload strategy configuration from YAML.

    Reloads config/strategies.yaml without restarting the backend.
    Use after manually editing strategy parameters or weights.
    """
    data = await _api("POST", "/api/v1/strategies/reload")
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_recovery_status() -> str:
    """Get circuit breaker status for all scheduler tasks.

    Shows which tasks have tripped circuit breakers due to
    repeated failures, and their recovery state.
    """
    data = await _api("GET", "/api/v1/engine/recovery")
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Backtest Tools
# ===========================================================================

@mcp.tool()
async def run_backtest(
    strategy_name: str, symbol: str, period: str = "3y",
) -> str:
    """Run a backtest for a strategy on a stock.

    Returns performance metrics: CAGR, Sharpe, max drawdown,
    win rate, trade count, and equity curve.

    Args:
        strategy_name: Strategy to test (e.g. "rsi_divergence", "supertrend").
        symbol: Stock ticker (e.g. "AAPL").
        period: Backtest period ("1y", "2y", "3y", "5y").
    """
    data = await _api("POST", "/api/v1/backtest/run", body={
        "strategy_name": strategy_name,
        "symbol": symbol,
        "period": period,
    })
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_backtest_results(
    strategy: str | None = None,
    symbol: str | None = None,
    include_stale: bool = False,
) -> str:
    """Get stored backtest results.

    Each result includes a ``stale`` flag — True means the result was
    produced before commit ff6279f (2026-04-07 09:37 KST) which fixed
    look-ahead bias, Kelly param mismatch, asymmetric slippage, and the
    KR currency bug. Stale results should not be used for live decisions.

    Args:
        strategy: Optional filter by strategy name.
        symbol: Optional filter by stock ticker.
        include_stale: When True, also returns pre-cutoff results (default
            False — only fresh results, safer for decision-making).
    """
    params: dict[str, str | bool] = {"include_stale": include_stale}
    if strategy:
        params["strategy"] = strategy
    if symbol:
        params["symbol"] = symbol
    data = await _api("GET", "/api/v1/backtest/results", params=params)
    return json.dumps(data, indent=2, ensure_ascii=False)


@mcp.tool()
async def run_pipeline_backtest(
    period: str = "3y",
    max_positions: int = 20,
    initial_equity: float = 100_000,
) -> str:
    """Run full pipeline backtest simulating the complete live trading system.

    Tests the entire pipeline: screener → 14 strategies → combiner →
    Kelly sizing → dynamic SL/TP on a 55-stock universe.
    WARNING: Takes 5-15 minutes to complete.

    Args:
        period: Backtest period ("2y", "3y", "5y").
        max_positions: Maximum concurrent positions (default 20).
        initial_equity: Starting capital (default $100,000).
    """
    data = await _api("POST", "/api/v1/backtest/pipeline", body={
        "period": period,
        "max_positions": max_positions,
        "initial_equity": initial_equity,
    })
    return json.dumps(data, indent=2, ensure_ascii=False)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run()
