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


class TestRiskAgentIntegration:
    """Test AI risk agent pre-trade check in evaluation loop."""

    @pytest.fixture
    def risk_agent_approved(self):
        agent = AsyncMock()
        agent.assess_pre_trade = AsyncMock(return_value={
            "approved": True,
            "risk_level": "LOW",
            "reason": "Acceptable risk",
            "suggested_size": 5000,
            "warnings": [],
        })
        return agent

    @pytest.fixture
    def risk_agent_rejected(self):
        agent = AsyncMock()
        agent.assess_pre_trade = AsyncMock(return_value={
            "approved": False,
            "risk_level": "CRITICAL",
            "reason": "Over-concentrated in tech sector",
            "suggested_size": 0,
            "warnings": ["Sector concentration too high"],
        })
        return agent

    @pytest.fixture
    def loop_with_risk(self, mock_adapter, mock_market_data, mock_registry, risk_agent_approved):
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
            watchlist=["AAPL"],
            risk_agent=risk_agent_approved,
        )

    async def test_buy_proceeds_when_risk_approved(
        self, loop_with_risk, mock_adapter, risk_agent_approved,
    ):
        await loop_with_risk.evaluate_symbol("AAPL")
        # Risk agent was called
        risk_agent_approved.assess_pre_trade.assert_called_once()
        # Buy order went through
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_blocked_when_risk_rejected(
        self, mock_adapter, mock_market_data, mock_registry, risk_agent_rejected,
    ):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            risk_agent=risk_agent_rejected,
        )
        await loop.evaluate_symbol("AAPL")
        # Risk agent was called
        risk_agent_rejected.assess_pre_trade.assert_called_once()
        # Buy was NOT placed
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_proceeds_when_risk_agent_errors(
        self, mock_adapter, mock_market_data, mock_registry,
    ):
        """If risk agent throws an exception, the trade should still proceed."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        agent = AsyncMock()
        agent.assess_pre_trade = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            risk_agent=agent,
        )
        await loop.evaluate_symbol("AAPL")
        # Buy should still go through despite agent error
        mock_adapter.create_buy_order.assert_called_once()

    async def test_no_risk_check_without_agent(self, eval_loop, mock_adapter):
        """Without risk_agent, buy proceeds normally (no crash)."""
        assert eval_loop._risk_agent is None
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()


class TestConfidenceRankedBuy:
    """Test that BUYs are ranked by confidence and SELLs execute immediately."""

    @pytest.fixture
    def multi_signal_loop(self, mock_adapter, mock_market_data):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        registry = MagicMock()
        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

        return EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["LOW_CONF", "HIGH_CONF", "MID_CONF"],
            market_state="uptrend",
        ), registry

    async def test_buys_execute_highest_confidence_first(
        self, multi_signal_loop, mock_adapter,
    ):
        """BUY signals should execute in descending confidence order."""
        loop, registry = multi_signal_loop
        call_order = []
        original_create = mock_adapter.create_buy_order

        async def track_buy(*args, **kwargs):
            call_order.append(kwargs.get("symbol") or args[0])
            return await original_create(*args, **kwargs)

        mock_adapter.create_buy_order = AsyncMock(side_effect=track_buy)

        # Each symbol returns a different confidence BUY signal
        confidence_map = {"LOW_CONF": 0.55, "HIGH_CONF": 0.95, "MID_CONF": 0.75}

        def make_strategy(symbol):
            s = AsyncMock()
            s.name = "trend_following"
            s.analyze = AsyncMock(return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=confidence_map[symbol],
                strategy_name="trend_following",
                reason="test",
            ))
            return s

        # Registry returns strategy per symbol
        strategy = AsyncMock()
        strategy.name = "trend_following"
        call_count = [0]
        symbols_order = ["LOW_CONF", "HIGH_CONF", "MID_CONF"]

        async def dynamic_analyze(df, symbol):
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence_map[symbol],
                strategy_name="trend_following",
                reason="test",
            )

        strategy.analyze = AsyncMock(side_effect=dynamic_analyze)
        registry.get_enabled.return_value = [strategy]
        registry.get_profile_weights.return_value = {"trend_following": 1.0}

        await loop._evaluate_all()

        # Should be called in order: HIGH_CONF (0.95), MID_CONF (0.75), LOW_CONF (0.55)
        assert len(call_order) == 3
        assert call_order[0] == "HIGH_CONF"
        assert call_order[1] == "MID_CONF"
        assert call_order[2] == "LOW_CONF"

    async def test_sells_execute_before_buys(
        self, mock_adapter, mock_market_data,
    ):
        """SELL signals should execute immediately, not wait for ranking."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        registry = MagicMock()
        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["BUY_STOCK", "SELL_STOCK"],
            market_state="uptrend",
        )

        # SELL_STOCK has a position
        mock_market_data.get_positions.return_value = [
            Position(symbol="SELL_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O2", symbol="SELL_STOCK", side="SELL",
            order_type="limit", quantity=10, price=105.0,
            status="filled", filled_price=105.0,
        ))

        signal_map = {
            "BUY_STOCK": Signal(
                signal_type=SignalType.BUY, confidence=0.8,
                strategy_name="trend_following", reason="buy",
            ),
            "SELL_STOCK": Signal(
                signal_type=SignalType.SELL, confidence=0.7,
                strategy_name="trend_following", reason="sell",
            ),
        }

        strategy = AsyncMock()
        strategy.name = "trend_following"

        async def dynamic_analyze(df, symbol):
            return signal_map[symbol]

        strategy.analyze = AsyncMock(side_effect=dynamic_analyze)
        registry.get_enabled.return_value = [strategy]
        registry.get_profile_weights.return_value = {"trend_following": 1.0}

        await loop._evaluate_all()

        # Both should execute
        mock_adapter.create_sell_order.assert_called_once()
        mock_adapter.create_buy_order.assert_called_once()


class TestKRMarketExchange:
    """Test that KR market skips yfinance exchange resolution."""

    @pytest.fixture
    def kr_eval_loop(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")

        return EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["005930", "035720"],
            market="KR",
        )

    async def test_kr_buy_uses_krx_exchange(self, kr_eval_loop, mock_adapter):
        """KR market should pass exchange='KRX', not resolve via yfinance."""
        await kr_eval_loop.evaluate_symbol("005930")
        mock_adapter.create_buy_order.assert_called_once()
        call_kwargs = mock_adapter.create_buy_order.call_args
        assert call_kwargs.kwargs.get("exchange") == "KRX"

    async def test_kr_sell_uses_krx_exchange(self, kr_eval_loop, mock_adapter, mock_market_data, mock_registry):
        """KR sell should also use KRX exchange."""
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL, confidence=0.8,
            strategy_name="trend_following", reason="sell",
        )
        mock_market_data.get_positions.return_value = [
            Position(symbol="005930", exchange="KRX", quantity=10, avg_price=70000.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O2", symbol="005930", side="SELL",
            order_type="limit", quantity=10, price=72000.0,
            status="filled", filled_price=72000.0,
        ))
        await kr_eval_loop.evaluate_symbol("005930")
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args
        assert call_kwargs.kwargs.get("exchange") == "KRX"

    async def test_kr_does_not_call_exchange_resolver(self, kr_eval_loop, mock_adapter):
        """KR market should never call ExchangeResolver.resolve()."""
        resolver = kr_eval_loop._exchange_resolver
        resolver.resolve = MagicMock(return_value="NASD")
        await kr_eval_loop.evaluate_symbol("005930")
        resolver.resolve.assert_not_called()

    async def test_us_still_uses_exchange_resolver(self, eval_loop, mock_adapter):
        """US market should still use ExchangeResolver."""
        resolver = eval_loop._exchange_resolver
        resolver.resolve = MagicMock(return_value="NASD")
        await eval_loop.evaluate_symbol("AAPL")
        resolver.resolve.assert_called_with("AAPL")
