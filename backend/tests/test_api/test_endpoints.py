"""Tests for API endpoints using TestClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from exchange.base import Balance, Position, Ticker, OrderResult
from exchange.paper_adapter import PaperAdapter


@pytest.fixture
def app():
    """Create a test app with mocked state."""
    from fastapi import FastAPI
    from api.router import api_router
    from data.market_data_service import MarketDataService
    from strategies.registry import StrategyRegistry
    from strategies.config_loader import StrategyConfigLoader
    from engine.risk_manager import RiskManager
    from engine.order_manager import OrderManager
    from services.rate_limiter import RateLimiter

    test_app = FastAPI()
    test_app.include_router(api_router, prefix="/api/v1")

    # Mock adapter
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(return_value=Balance(
        currency="USD", total=100_000, available=80_000, locked=20_000,
    ))
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(
            symbol="AAPL", exchange="NASD", quantity=50,
            avg_price=150.0, current_price=160.0,
            unrealized_pnl=500.0, unrealized_pnl_pct=6.67,
        ),
    ])
    adapter.fetch_ticker = AsyncMock(return_value=Ticker(
        symbol="AAPL", price=160.0, volume=50_000_000,
    ))
    adapter.fetch_ohlcv = AsyncMock(return_value=[])

    test_app.state.adapter = adapter
    test_app.state.market_data = MarketDataService(
        adapter=adapter, rate_limiter=RateLimiter(max_per_second=100),
    )

    # Registry with no config file
    with patch.object(StrategyConfigLoader, '__init__', lambda self, **kw: None):
        loader = StrategyConfigLoader.__new__(StrategyConfigLoader)
        loader._path = "nonexistent.yaml"
        loader._config = {
            "strategies": {
                "trend_following": {"enabled": True, "params": {}},
            },
            "profiles": {"uptrend": {"trend_following": 1.0}},
        }
        loader.reload = lambda: None
        registry = StrategyRegistry(config_loader=loader)
    test_app.state.registry = registry

    risk = RiskManager()
    test_app.state.risk_manager = risk
    test_app.state.order_manager = OrderManager(adapter=adapter, risk_manager=risk)

    from engine.scheduler import TradingScheduler
    test_app.state.scheduler = TradingScheduler()

    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestPortfolioAPI:
    def test_portfolio_summary(self, client):
        resp = client.get("/api/v1/portfolio/summary", params={"market": "US"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"]["total"] == 100_000
        assert data["positions_count"] == 1

    def test_portfolio_summary_all(self, client):
        resp = client.get("/api/v1/portfolio/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["market"] == "ALL"

    def test_list_positions(self, client):
        resp = client.get("/api/v1/portfolio/positions", params={"market": "US"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"
        assert data[0]["quantity"] == 50


class TestMarketAPI:
    def test_get_price(self, client):
        resp = client.get("/api/v1/market/price/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["price"] == 160.0

    @patch("data.market_data_service.MarketDataService._fetch_yfinance", return_value=pd.DataFrame())
    def test_get_chart_empty(self, mock_yf, client):
        resp = client.get("/api/v1/market/chart/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["data"] == []


class TestStrategyAPI:
    def test_list_strategies(self, client):
        resp = client.get("/api/v1/strategies/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["name"] == "trend_following"

    def test_get_strategy_params(self, client):
        resp = client.get("/api/v1/strategies/trend_following/params")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "trend_following"

    def test_get_unknown_strategy(self, client):
        resp = client.get("/api/v1/strategies/nonexistent/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data


class TestEngineAPI:
    def test_engine_status(self, client):
        resp = client.get("/api/v1/engine/status")
        assert resp.status_code == 200

    def test_start_stop_engine(self, client):
        resp = client.post("/api/v1/engine/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        resp = client.post("/api/v1/engine/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


class TestWatchlistAPI:
    @pytest.fixture(autouse=True)
    def _mock_db(self):
        """Mock get_session_factory to use in-memory SQLite."""
        import asyncio
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from core.models import Base

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")

        loop = asyncio.new_event_loop()
        try:
            async def _setup():
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            loop.run_until_complete(_setup())
        finally:
            loop.close()

        factory = async_sessionmaker(engine, expire_on_commit=False)
        with patch("api.watchlist.get_session_factory", return_value=factory):
            yield

        loop2 = asyncio.new_event_loop()
        try:
            loop2.run_until_complete(engine.dispose())
        finally:
            loop2.close()

    def test_watchlist_crud(self, client):
        # Get empty
        resp = client.get("/api/v1/watchlist/")
        assert resp.status_code == 200

        # Add
        resp = client.post("/api/v1/watchlist/", json={"symbol": "AAPL"})
        assert resp.status_code == 200
        assert "AAPL" in resp.json()["symbols"]

        # Add duplicate
        resp = client.post("/api/v1/watchlist/", json={"symbol": "AAPL"})
        assert resp.json()["symbols"].count("AAPL") == 1

        # Delete
        resp = client.delete("/api/v1/watchlist/AAPL")
        assert resp.status_code == 200
        assert "AAPL" not in resp.json()["symbols"]
