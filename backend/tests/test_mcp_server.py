"""Tests for MCP server tool definitions and API calls."""

import json
import pytest
from unittest.mock import AsyncMock, patch

import mcp_server
from mcp_server import _api


class TestToolRegistration:
    """Verify all expected tools are registered."""

    def test_total_tool_count(self):
        tools = mcp_server.mcp._tool_manager.list_tools()
        assert len(tools) == 27

    def test_portfolio_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "get_portfolio_summary" in names
        assert "get_positions" in names
        assert "get_equity_history" in names
        assert "get_trade_history" in names
        assert "get_trade_summary" in names

    def test_market_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "get_price" in names
        assert "get_market_state" in names
        assert "get_macro_indicators" in names

    def test_scanner_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "scan_stocks" in names
        assert "discover_universe" in names
        assert "get_sector_performance" in names

    def test_strategy_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "get_strategies" in names
        assert "get_signal_quality" in names
        assert "get_factor_scores" in names
        assert "get_etf_status" in names

    def test_watchlist_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "get_watchlist" in names
        assert "add_to_watchlist" in names
        assert "remove_from_watchlist" in names

    def test_engine_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "get_engine_status" in names
        assert "start_engine" in names
        assert "stop_engine" in names
        assert "run_evaluation" in names
        assert "run_task" in names
        assert "reload_strategies" in names
        assert "get_recovery_status" in names

    def test_backtest_tools(self):
        names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        assert "run_backtest" in names
        assert "get_backtest_results" in names


class TestApiHelper:
    """Test the HTTP helper function."""

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_get_request(self, mock_api):
        mock_api.return_value = {"status": "ok", "data": {"balance": 10000}}
        result = await mcp_server.get_portfolio_summary()
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        mock_api.assert_called_with("GET", "/api/v1/portfolio/summary")

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_post_request(self, mock_api):
        mock_api.return_value = {"status": "ok"}
        result = await mcp_server.start_engine()
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        mock_api.assert_called_with("POST", "/api/v1/engine/start")

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_delete_request(self, mock_api):
        mock_api.return_value = {"status": "ok"}
        result = await mcp_server.remove_from_watchlist("AAPL")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        mock_api.assert_called_with("DELETE", "/api/v1/watchlist/AAPL")

    @pytest.mark.asyncio
    async def test_error_response_from_api(self):
        result = await _api("PUT", "/api/v1/foo")
        assert "error" in result
        assert "Unsupported" in result["error"]

    @pytest.mark.asyncio
    async def test_unsupported_method(self):
        result = await _api("PATCH", "/api/v1/foo")
        assert "error" in result
        assert "Unsupported" in result["error"]


class TestToolFunctions:
    """Test individual tool functions with mocked API."""

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_get_price(self, mock_api):
        mock_api.return_value = {"symbol": "AAPL", "price": 185.5, "change_pct": 1.2}
        result = await mcp_server.get_price("AAPL")
        parsed = json.loads(result)
        assert parsed["symbol"] == "AAPL"
        mock_api.assert_called_with("GET", "/api/v1/market/price/AAPL")

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_scan_stocks(self, mock_api):
        mock_api.return_value = {"results": [{"symbol": "AAPL", "grade": "A"}]}
        result = await mcp_server.scan_stocks(["AAPL", "NVDA"], min_grade="B")
        mock_api.assert_called_with("POST", "/api/v1/scanner/run", body={
            "symbols": ["AAPL", "NVDA"], "min_grade": "B",
        })

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_add_to_watchlist_uppercases(self, mock_api):
        mock_api.return_value = {"status": "ok"}
        await mcp_server.add_to_watchlist("aapl")
        mock_api.assert_called_with("POST", "/api/v1/watchlist/", body={
            "symbol": "AAPL", "exchange": "NASD",
        })

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_remove_from_watchlist_uppercases(self, mock_api):
        mock_api.return_value = {"status": "ok"}
        await mcp_server.remove_from_watchlist("tsla")
        mock_api.assert_called_with("DELETE", "/api/v1/watchlist/TSLA")

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_run_backtest(self, mock_api):
        mock_api.return_value = {"metrics": {"cagr": 15.2, "sharpe": 1.5}}
        result = await mcp_server.run_backtest("supertrend", "NVDA", "3y")
        mock_api.assert_called_with("POST", "/api/v1/backtest/run", body={
            "strategy_name": "supertrend", "symbol": "NVDA", "period": "3y",
        })

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_run_task(self, mock_api):
        mock_api.return_value = {"status": "ok"}
        await mcp_server.run_task("daily_scan")
        mock_api.assert_called_with("POST", "/api/v1/engine/run-task/daily_scan")

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_get_trade_history_with_filters(self, mock_api):
        mock_api.return_value = {"trades": []}
        await mcp_server.get_trade_history(limit=10, symbol="AAPL")
        mock_api.assert_called_with("GET", "/api/v1/trades/", params={
            "limit": 10, "symbol": "AAPL",
        })

    @pytest.mark.asyncio
    @patch("mcp_server._api")
    async def test_get_trade_history_no_symbol(self, mock_api):
        mock_api.return_value = {"trades": []}
        await mcp_server.get_trade_history(limit=5)
        mock_api.assert_called_with("GET", "/api/v1/trades/", params={"limit": 5})
