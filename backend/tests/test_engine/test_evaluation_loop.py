"""Tests for Evaluation Loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from engine.evaluation_loop import EvaluationLoop
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from exchange.base import Balance, Position, OrderResult
from strategies.base import Signal
from core.enums import SignalType


def _make_ohlcv_df(n=50):
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": np.random.randint(100000, 500000, n).astype(float),
    })


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(return_value=Balance(
        currency="USD", total=100_000, available=80_000,
    ))
    adapter.fetch_positions = AsyncMock(return_value=[])
    adapter.create_buy_order = AsyncMock(return_value=OrderResult(
        order_id="O1", symbol="AAPL", side="BUY",
        order_type="limit", quantity=10, price=150.0,
        status="filled", filled_price=150.0,
    ))
    return adapter


@pytest.fixture
def mock_market_data():
    svc = AsyncMock()
    svc.get_ohlcv = AsyncMock(return_value=_make_ohlcv_df())
    svc.get_balance = AsyncMock(return_value=Balance(
        currency="USD", total=100_000, available=80_000,
    ))
    svc.get_positions = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    mock_strategy = AsyncMock()
    mock_strategy.name = "trend_following"
    mock_strategy.analyze = AsyncMock(return_value=Signal(
        signal_type=SignalType.BUY, confidence=0.8,
        strategy_name="trend_following", reason="test",
    ))
    registry.get_enabled.return_value = [mock_strategy]
    registry.get_profile_weights.return_value = {"trend_following": 1.0}
    return registry


@pytest.fixture
def eval_loop(mock_adapter, mock_market_data, mock_registry):
    from data.indicator_service import IndicatorService
    from strategies.combiner import SignalCombiner

    risk = RiskManager()
    order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

    return EvaluationLoop(
        adapter=mock_adapter,
        market_data=mock_market_data,
        indicator_svc=IndicatorService(),
        registry=mock_registry,
        combiner=SignalCombiner(),
        order_manager=order_mgr,
        risk_manager=risk,
        watchlist=["AAPL", "TSLA"],
        market_state="uptrend",
        interval_sec=1,
    )


class TestEvaluationLoop:
    async def test_evaluate_symbol_buy(self, eval_loop, mock_adapter):
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_evaluate_symbol_hold(self, eval_loop, mock_registry, mock_adapter):
        # Change strategy to return HOLD
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.HOLD, confidence=0.3,
            strategy_name="trend_following", reason="hold",
        )
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_evaluate_symbol_sell(self, eval_loop, mock_adapter, mock_registry, mock_market_data):
        # Strategy says SELL
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL, confidence=0.8,
            strategy_name="trend_following", reason="sell",
        )
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O2", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=150.0,
            status="filled", filled_price=150.0,
        ))
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_sell_order.assert_called_once()

    async def test_evaluate_empty_ohlcv(self, eval_loop, mock_market_data, mock_adapter):
        mock_market_data.get_ohlcv.return_value = pd.DataFrame()
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_set_watchlist(self, eval_loop):
        eval_loop.set_watchlist(["MSFT", "GOOG"])
        assert eval_loop._watchlist == ["MSFT", "GOOG"]

    async def test_set_market_state(self, eval_loop):
        eval_loop.set_market_state("downtrend")
        assert eval_loop._market_state == "downtrend"

    async def test_start_stop(self, eval_loop):
        assert eval_loop.running is False
        # Start in background, stop immediately
        import asyncio
        task = asyncio.create_task(eval_loop.start())
        await asyncio.sleep(0.1)
        assert eval_loop.running is True
        await eval_loop.stop()
        await asyncio.sleep(0.1)
        assert eval_loop.running is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
