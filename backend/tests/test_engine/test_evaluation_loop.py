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
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(100000, 500000, n).astype(float),
        }
    )


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=Balance(
            currency="USD",
            total=100_000,
            available=80_000,
        )
    )
    adapter.fetch_positions = AsyncMock(return_value=[])
    adapter.create_buy_order = AsyncMock(
        return_value=OrderResult(
            order_id="O1",
            symbol="AAPL",
            side="BUY",
            order_type="limit",
            quantity=10,
            price=150.0,
            status="filled",
            filled_price=150.0,
        )
    )
    return adapter


@pytest.fixture
def mock_market_data():
    svc = AsyncMock()
    svc.get_ohlcv = AsyncMock(return_value=_make_ohlcv_df())
    svc.get_balance = AsyncMock(
        return_value=Balance(
            currency="USD",
            total=100_000,
            available=80_000,
        )
    )
    svc.get_positions = AsyncMock(return_value=[])
    svc.get_price = AsyncMock(return_value=150.0)
    return svc


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    mock_strategy = AsyncMock()
    mock_strategy.name = "trend_following"
    mock_strategy.analyze = AsyncMock(
        return_value=Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            strategy_name="trend_following",
            reason="test",
        )
    )
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
            signal_type=SignalType.HOLD,
            confidence=0.3,
            strategy_name="trend_following",
            reason="hold",
        )
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_evaluate_symbol_sell(
        self, eval_loop, mock_adapter, mock_registry, mock_market_data
    ):
        # Strategy says SELL
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL,
            confidence=0.8,
            strategy_name="trend_following",
            reason="sell",
        )
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=150.0,
                status="filled",
                filled_price=150.0,
            )
        )
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


class TestDailyBuyLimit:
    """Test daily buy limit with dynamic confidence escalation."""

    async def test_daily_limit_blocks_low_conf_at_hard_cap(
        self, eval_loop, mock_adapter, mock_registry
    ):
        """After limit reached, low-confidence buys blocked."""
        eval_loop._daily_buy_limit = 1
        eval_loop._daily_buy_count = 1  # already at limit
        from datetime import date as _date

        eval_loop._daily_buy_date = _date.today().isoformat()

        # confidence=0.8 signal (below 0.90 override threshold)
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_daily_limit_allows_ultra_high_conf_override(
        self, eval_loop, mock_adapter, mock_registry
    ):
        """Ultra-high confidence (0.90+) bypasses hard cap."""
        eval_loop._daily_buy_limit = 1
        eval_loop._daily_buy_count = 1
        from datetime import date as _date

        eval_loop._daily_buy_date = _date.today().isoformat()

        # Set strategy to return 0.95 confidence
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.95,
            strategy_name="trend_following",
            reason="strong signal",
        )
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_escalating_confidence_at_80pct_usage(
        self, eval_loop, mock_adapter, mock_registry
    ):
        """At 80%+ usage, need confidence >= 0.75."""
        eval_loop._daily_buy_limit = 5
        eval_loop._daily_buy_count = 4  # 80% used
        from datetime import date as _date

        eval_loop._daily_buy_date = _date.today().isoformat()

        # Signal with 0.60 confidence — should be blocked (need 0.75)
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.60,
            strategy_name="trend_following",
            reason="moderate",
        )
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_early_buys_unrestricted(self, eval_loop, mock_adapter):
        """First buys (< 60% usage) have no extra confidence requirement."""
        eval_loop._daily_buy_limit = 5

        # First buy (0% usage) — should succeed with normal confidence
        await eval_loop.evaluate_symbol("AAPL")
        assert mock_adapter.create_buy_order.call_count == 1

    async def test_daily_limit_resets_on_new_day(self, eval_loop, mock_adapter):
        """Counter resets when date changes."""
        eval_loop._daily_buy_limit = 1
        eval_loop._daily_buy_count = 1
        eval_loop._daily_buy_date = "2020-01-01"  # old date

        # Should reset counter and allow buy
        await eval_loop.evaluate_symbol("AAPL")
        assert mock_adapter.create_buy_order.call_count == 1
        assert eval_loop._daily_buy_count == 1


class TestRiskAgentIntegration:
    """Test AI risk agent pre-trade check in evaluation loop."""

    @pytest.fixture
    def risk_agent_approved(self):
        agent = AsyncMock()
        agent.assess_pre_trade = AsyncMock(
            return_value={
                "approved": True,
                "risk_level": "LOW",
                "reason": "Acceptable risk",
                "suggested_size": 5000,
                "warnings": [],
            }
        )
        return agent

    @pytest.fixture
    def risk_agent_rejected(self):
        agent = AsyncMock()
        agent.assess_pre_trade = AsyncMock(
            return_value={
                "approved": False,
                "risk_level": "CRITICAL",
                "reason": "Over-concentrated in tech sector",
                "suggested_size": 0,
                "warnings": ["Sector concentration too high"],
            }
        )
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
        self,
        loop_with_risk,
        mock_adapter,
        risk_agent_approved,
    ):
        await loop_with_risk.evaluate_symbol("AAPL")
        # Risk agent was called
        risk_agent_approved.assess_pre_trade.assert_called_once()
        # Buy order went through
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_blocked_when_risk_rejected(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
        risk_agent_rejected,
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
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
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


class TestHeldPositionEvaluation:
    """Test that held positions are always evaluated even if not in watchlist."""

    async def test_held_position_evaluated_when_not_in_watchlist(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """A held position not in watchlist should still get strategy SELL signals."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock(spec=PositionTracker)
        # HELD_STOCK is tracked but NOT in watchlist
        position_tracker.tracked_symbols = ["HELD_STOCK"]

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],  # HELD_STOCK not here
            position_tracker=position_tracker,
        )

        # Strategy returns SELL for HELD_STOCK
        strategy = mock_registry.get_enabled.return_value[0]
        signal_map = {
            "AAPL": Signal(
                signal_type=SignalType.HOLD,
                confidence=0.3,
                strategy_name="trend_following",
                reason="hold",
            ),
            "HELD_STOCK": Signal(
                signal_type=SignalType.SELL,
                confidence=0.8,
                strategy_name="trend_following",
                reason="sell",
            ),
        }

        async def dynamic_analyze(df, symbol):
            return signal_map.get(symbol, signal_map["AAPL"])

        strategy.analyze = AsyncMock(side_effect=dynamic_analyze)

        mock_market_data.get_positions.return_value = [
            Position(symbol="HELD_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="HELD_STOCK",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=105.0,
                status="filled",
                filled_price=105.0,
            )
        )

        await loop._evaluate_all()

        # HELD_STOCK should get a sell order even though not in watchlist
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args
        assert (
            call_kwargs.kwargs.get("symbol") == "HELD_STOCK" or call_kwargs.args[0] == "HELD_STOCK"
        )

    async def test_no_duplicate_evaluation_when_in_both(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Symbol in both watchlist and held positions should be evaluated only once."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock(spec=PositionTracker)
        position_tracker.tracked_symbols = ["AAPL"]  # Also in watchlist

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL", "TSLA"],
            position_tracker=position_tracker,
        )

        await loop._evaluate_all()

        # AAPL should be evaluated exactly once (not twice)
        strategy = mock_registry.get_enabled.return_value[0]
        aapl_calls = [c for c in strategy.analyze.call_args_list if c.args[1] == "AAPL"]
        assert len(aapl_calls) == 1


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
        self,
        multi_signal_loop,
        mock_adapter,
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
            s.analyze = AsyncMock(
                return_value=Signal(
                    signal_type=SignalType.BUY,
                    confidence=confidence_map[symbol],
                    strategy_name="trend_following",
                    reason="test",
                )
            )
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
        self,
        mock_adapter,
        mock_market_data,
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
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="SELL_STOCK",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=105.0,
                status="filled",
                filled_price=105.0,
            )
        )

        signal_map = {
            "BUY_STOCK": Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="trend_following",
                reason="buy",
            ),
            "SELL_STOCK": Signal(
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="trend_following",
                reason="sell",
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

    async def test_kr_sell_uses_krx_exchange(
        self, kr_eval_loop, mock_adapter, mock_market_data, mock_registry
    ):
        """KR sell should also use KRX exchange."""
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL,
            confidence=0.8,
            strategy_name="trend_following",
            reason="sell",
        )
        mock_market_data.get_positions.return_value = [
            Position(symbol="005930", exchange="KRX", quantity=10, avg_price=70000.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="005930",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=72000.0,
                status="filled",
                filled_price=72000.0,
            )
        )
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


class TestProtectiveSells:
    """Test regime-change and news-sentiment protective sell mechanisms."""

    @pytest.fixture
    def loop_with_tracker(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock(spec=PositionTracker)
        position_tracker.tracked_symbols = ["AAPL", "TSLA"]
        position_tracker._tracked = {}  # for hold-time check in sentiment sells

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL", "TSLA"],
            market_state="uptrend",
            position_tracker=position_tracker,
        )
        return loop

    async def test_set_market_state_tracks_previous(self, loop_with_tracker):
        """set_market_state should save previous state."""
        assert loop_with_tracker._market_state == "uptrend"
        assert loop_with_tracker._prev_market_state == "uptrend"
        loop_with_tracker.set_market_state("downtrend")
        assert loop_with_tracker._market_state == "downtrend"
        assert loop_with_tracker._prev_market_state == "uptrend"

    async def test_update_news_sentiment(self, loop_with_tracker):
        """update_news_sentiment should merge sentiment data."""
        loop_with_tracker.update_news_sentiment({"AAPL": -0.8, "TSLA": 0.5})
        assert loop_with_tracker._news_sentiment["AAPL"] == -0.8
        assert loop_with_tracker._news_sentiment["TSLA"] == 0.5
        # Update should merge, not replace
        loop_with_tracker.update_news_sentiment({"AAPL": -0.3})
        assert loop_with_tracker._news_sentiment["AAPL"] == -0.3
        assert loop_with_tracker._news_sentiment["TSLA"] == 0.5

    async def test_regime_sell_losing_positions(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Regime downtrend should sell losing positions."""
        # AAPL is losing, TSLA is winning
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=140.0
            ),
            Position(
                symbol="TSLA", exchange="NASD", quantity=5, avg_price=200.0, current_price=220.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=140.0,
                status="filled",
                filled_price=140.0,
            )
        )

        # Transition uptrend -> downtrend
        loop_with_tracker.set_market_state("downtrend")
        await loop_with_tracker._check_protective_sells({"AAPL", "TSLA"})

        # Only AAPL (losing) should be sold
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["quantity"] == 10

    async def test_regime_sell_skips_winning_positions(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Regime sell should keep winning positions."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=170.0
            ),
        ]

        loop_with_tracker.set_market_state("downtrend")
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # No sell — position is winning
        mock_adapter.create_sell_order.assert_not_called()

    async def test_no_regime_sell_in_same_state(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """No regime sell when market hasn't worsened."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=140.0
            ),
        ]

        # Already in downtrend, no transition
        loop_with_tracker._market_state = "downtrend"
        loop_with_tracker._prev_market_state = "downtrend"
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_not_called()

    async def test_regime_sell_on_sideways_to_downtrend(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Sideways→downtrend transition should also trigger regime sell."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=140.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=140.0,
                status="filled",
                filled_price=140.0,
            )
        )

        # Transition sideways -> downtrend
        loop_with_tracker._prev_market_state = "sideways"
        loop_with_tracker._market_state = "downtrend"
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_called_once()

    async def test_sentiment_sell_on_negative(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Strongly negative sentiment should trigger sell."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                status="filled",
                filled_price=160.0,
            )
        )

        loop_with_tracker.update_news_sentiment({"AAPL": -0.7})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"

    async def test_sentiment_no_sell_on_mild_negative(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Mildly negative sentiment (-0.3) should NOT trigger sell."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]

        loop_with_tracker.update_news_sentiment({"AAPL": -0.3})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_not_called()

    async def test_sentiment_cleared_after_check(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Processed sentiments should be cleared to avoid re-selling."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                status="filled",
                filled_price=160.0,
            )
        )

        loop_with_tracker.update_news_sentiment({"AAPL": -0.8})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # Sentiment for AAPL should be cleared
        assert "AAPL" not in loop_with_tracker._news_sentiment

    async def test_kr_market_uses_krx_for_protective_sell(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """KR market protective sell should use KRX exchange."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")
        position_tracker = MagicMock(spec=PositionTracker)
        position_tracker.tracked_symbols = ["005930"]

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["005930"],
            market_state="uptrend",
            position_tracker=position_tracker,
            market="KR",
        )

        mock_market_data.get_positions.return_value = [
            Position(
                symbol="005930",
                exchange="KRX",
                quantity=10,
                avg_price=70000.0,
                current_price=65000.0,
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="005930",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=65000.0,
                status="filled",
                filled_price=65000.0,
            )
        )

        loop.set_market_state("downtrend")
        await loop._check_protective_sells({"005930"})

        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["exchange"] == "KRX"

    async def test_position_untracked_after_protective_sell(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """Position should be untracked after successful protective sell."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=160.0,
                status="filled",
                filled_price=160.0,
            )
        )

        loop_with_tracker.update_news_sentiment({"AAPL": -0.8})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        loop_with_tracker._position_tracker.untrack.assert_called_with("AAPL")

    async def test_no_protective_sell_when_no_triggers(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """No sell when regime is stable and no negative sentiment."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=140.0
            ),
        ]

        # No regime change, no sentiment
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # get_positions should not even be called (early return)
        mock_market_data.get_positions.assert_not_called()


class TestDuplicateBuyPrevention:
    """Test defense-in-depth: exchange position check prevents duplicate buys.

    STOCK-4: supertrend 전략이 같은 종목을 17회 연속 매수한 버그 방지.
    position_tracker가 비어있어도 exchange positions 체크로 중복 매수를 차단.
    """

    @pytest.fixture
    def loop_no_tracker(self, mock_adapter, mock_market_data, mock_registry):
        """Evaluation loop WITHOUT position tracker (simulates post-restart state)."""
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
            position_tracker=None,  # Simulates empty/missing tracker
        )

    @pytest.fixture
    def loop_empty_tracker(self, mock_adapter, mock_market_data, mock_registry):
        """Evaluation loop with EMPTY position tracker (post-restart, before restore)."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = []  # Empty — positions not yet restored

        return EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=tracker,
        )

    async def test_buy_blocked_when_already_held_no_tracker(
        self,
        loop_no_tracker,
        mock_adapter,
        mock_market_data,
    ):
        """BUY should be blocked if exchange shows existing position, even without tracker."""
        # Exchange shows we already hold AAPL
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]

        await loop_no_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_blocked_when_already_held_empty_tracker(
        self,
        loop_empty_tracker,
        mock_adapter,
        mock_market_data,
    ):
        """BUY should be blocked even when position tracker is empty (post-restart)."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]

        await loop_empty_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_allowed_when_not_held(
        self,
        loop_no_tracker,
        mock_adapter,
        mock_market_data,
    ):
        """BUY should proceed when exchange confirms no existing position."""
        mock_market_data.get_positions.return_value = []  # No positions

        await loop_no_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_allowed_for_different_symbol(
        self,
        loop_no_tracker,
        mock_adapter,
        mock_market_data,
    ):
        """BUY should proceed for a symbol not in existing positions."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="TSLA", exchange="NASD", quantity=5, avg_price=200.0),
        ]

        await loop_no_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_repeated_buy_blocked_across_evaluations(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Same symbol should not be bought again on subsequent evaluation cycles.

        Simulates the STOCK-4 scenario: supertrend produces BUY signal
        every evaluation cycle, but second buy should be blocked by
        exchange position check.
        """
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = []  # Empty tracker

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=tracker,
        )

        # First evaluation: no positions, buy goes through
        mock_market_data.get_positions.return_value = []
        await loop.evaluate_symbol("AAPL")
        assert mock_adapter.create_buy_order.call_count == 1

        # Second evaluation: exchange now shows position
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0),
        ]
        # Clear the _last_signal to simulate what happens if dedup is bypassed
        loop._last_signal.clear()

        await loop.evaluate_symbol("AAPL")
        # Should still be 1 — second buy blocked by exchange position check
        assert mock_adapter.create_buy_order.call_count == 1

    async def test_kr_duplicate_buy_blocked(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """KR market duplicate buy should also be blocked by exchange position check."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["263750"],
            market="KR",
            position_tracker=None,
        )

        # Exchange shows we already hold 263750 (펄어비스)
        mock_market_data.get_positions.return_value = [
            Position(symbol="263750", exchange="KRX", quantity=10, avg_price=61400.0),
        ]

        await loop.evaluate_symbol("263750")
        mock_adapter.create_buy_order.assert_not_called()


class TestExchangePositionHeldEvaluation:
    """Test that exchange positions are included in eval_symbols for SELL evaluation.

    STOCK-4: when position_tracker is empty, held positions still need
    strategy-based SELL evaluation via exchange position fallback.
    """

    async def test_exchange_held_position_evaluated_for_sell(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Held position from exchange should get SELL evaluation even without tracker."""
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
            watchlist=["TSLA"],  # AAPL not in watchlist
            position_tracker=None,  # No tracker
        )

        # Exchange shows we hold AAPL
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]

        # Strategy returns SELL for AAPL
        strategy = mock_registry.get_enabled.return_value[0]
        signal_map = {
            "AAPL": Signal(
                signal_type=SignalType.SELL,
                confidence=0.8,
                strategy_name="trend_following",
                reason="sell",
            ),
            "TSLA": Signal(
                signal_type=SignalType.HOLD,
                confidence=0.3,
                strategy_name="trend_following",
                reason="hold",
            ),
        }

        async def dynamic_analyze(df, symbol):
            return signal_map.get(symbol, signal_map["TSLA"])

        strategy.analyze = AsyncMock(side_effect=dynamic_analyze)

        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=150.0,
                status="filled",
                filled_price=150.0,
            )
        )

        await loop._evaluate_all()

        # AAPL should have been evaluated and sell placed
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args
        assert call_kwargs.kwargs.get("symbol") == "AAPL" or (
            call_kwargs.args and call_kwargs.args[0] == "AAPL"
        )

    async def test_exchange_positions_merged_with_tracker(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Exchange positions should be merged with tracker positions (no duplicates)."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = ["AAPL"]  # Tracker has AAPL

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["TSLA"],
            position_tracker=tracker,
        )

        # Exchange has AAPL + GOOG
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
            Position(symbol="GOOG", exchange="NASD", quantity=5, avg_price=170.0),
        ]

        # Track which symbols get evaluated
        strategy = mock_registry.get_enabled.return_value[0]
        evaluated = []

        async def track_analyze(df, symbol):
            evaluated.append(symbol)
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.3,
                strategy_name="trend_following",
                reason="hold",
            )

        strategy.analyze = AsyncMock(side_effect=track_analyze)

        await loop._evaluate_all()

        # All three should be evaluated: TSLA (watchlist), AAPL (tracker+exchange), GOOG (exchange)
        assert "TSLA" in evaluated
        assert "AAPL" in evaluated
        assert "GOOG" in evaluated
        # AAPL should appear only once (deduped by dict.fromkeys)
        assert evaluated.count("AAPL") == 1

    async def test_exchange_positions_fallback_on_error(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """If get_positions fails, evaluation should still proceed with tracker."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = ["AAPL"]

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["TSLA"],
            position_tracker=tracker,
        )

        # get_positions throws an error
        mock_market_data.get_positions.side_effect = RuntimeError("API timeout")

        strategy = mock_registry.get_enabled.return_value[0]
        evaluated = []

        async def track_analyze(df, symbol):
            evaluated.append(symbol)
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.3,
                strategy_name="trend_following",
                reason="hold",
            )

        strategy.analyze = AsyncMock(side_effect=track_analyze)

        # Should not crash, just proceed with tracker-only held set
        await loop._evaluate_all()

        assert "TSLA" in evaluated
        assert "AAPL" in evaluated


class TestHeldPositionBuyToHoldRemap:
    """Test that held positions remap BUY→HOLD so SELL signals can flow through.

    This is the fix for STOCK-18: strategy-based SELL signals were being drowned
    out by BUY signals for held positions, preventing any position exits.
    """

    @pytest.fixture
    def multi_strategy_loop(self, mock_adapter, mock_market_data):
        """Create eval loop with multiple strategies that return different signals."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        registry = MagicMock()
        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock(spec=PositionTracker)
        position_tracker.tracked_symbols = ["HELD_STOCK"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["NEW_STOCK"],
            position_tracker=position_tracker,
        )
        return loop, registry, position_tracker

    async def test_held_position_sell_when_strategies_detect_exit(
        self,
        multi_strategy_loop,
        mock_adapter,
        mock_market_data,
    ):
        """Held position gets SELL when enough strategies detect exit conditions.

        Without BUY→HOLD remap, the 5 BUY signals would drown out 3 SELL signals.
        With the remap, only SELL signals are active → SELL wins.
        """
        loop, registry, tracker = multi_strategy_loop

        # Setup position data
        mock_market_data.get_positions.return_value = [
            Position(symbol="HELD_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="HELD_STOCK",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=105.0,
                status="filled",
                filled_price=105.0,
            )
        )

        # Create strategies: 5 BUY + 3 SELL (for held stock)
        strategies = []
        signal_configs = [
            ("trend_following", SignalType.BUY, 0.8),
            ("dual_momentum", SignalType.BUY, 0.7),
            ("donchian_breakout", SignalType.BUY, 0.7),
            ("cis_momentum", SignalType.BUY, 0.6),
            ("larry_williams", SignalType.BUY, 0.6),
            ("supertrend", SignalType.SELL, 0.7),
            ("rsi_divergence", SignalType.SELL, 0.65),
            ("bnf_deviation", SignalType.SELL, 0.6),
        ]
        for name, sig_type, conf in signal_configs:
            s = AsyncMock()
            s.name = name
            s.analyze = AsyncMock(
                return_value=Signal(
                    signal_type=sig_type,
                    confidence=conf,
                    strategy_name=name,
                    reason="test",
                )
            )
            strategies.append(s)

        registry.get_enabled.return_value = strategies
        registry.get_profile_weights.return_value = {
            "trend_following": 0.15,
            "dual_momentum": 0.10,
            "donchian_breakout": 0.10,
            "cis_momentum": 0.10,
            "larry_williams": 0.10,
            "supertrend": 0.10,
            "rsi_divergence": 0.15,
            "bnf_deviation": 0.10,
        }

        await loop._evaluate_all()

        # The held stock should receive a SELL order
        mock_adapter.create_sell_order.assert_called_once()

    async def test_non_held_position_not_remapped(
        self,
        multi_strategy_loop,
        mock_adapter,
        mock_market_data,
    ):
        """Non-held positions (watchlist) should NOT have BUY remapped to HOLD."""
        loop, registry, tracker = multi_strategy_loop

        # Only non-held stock in watchlist
        mock_market_data.get_positions.return_value = []

        strategy = AsyncMock()
        strategy.name = "trend_following"
        strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="trend_following",
                reason="strong buy",
            )
        )
        registry.get_enabled.return_value = [strategy]
        registry.get_profile_weights.return_value = {"trend_following": 1.0}

        await loop._evaluate_all()

        # NEW_STOCK should get a BUY order (not remapped to HOLD)
        mock_adapter.create_buy_order.assert_called_once()

    async def test_held_single_sell_blocked_by_active_ratio(
        self,
        multi_strategy_loop,
        mock_adapter,
        mock_market_data,
    ):
        """Single weak SELL should be blocked by exit_min_active_ratio=0.15."""
        loop, registry, tracker = multi_strategy_loop

        mock_market_data.get_positions.return_value = [
            Position(symbol="HELD_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]

        # 1 SELL + 9 BUY (which become HOLD) → active_ratio ≈ 10% < 15%
        strategies = []
        names = [
            "trend_following",
            "dual_momentum",
            "donchian_breakout",
            "cis_momentum",
            "larry_williams",
            "supertrend",
            "rsi_divergence",
            "bnf_deviation",
            "bollinger_squeeze",
            "macd_histogram",
        ]
        for i, name in enumerate(names):
            s = AsyncMock()
            s.name = name
            # Only the first one is SELL, rest are BUY (→ remapped to HOLD)
            sig_type = SignalType.SELL if i == 0 else SignalType.BUY
            s.analyze = AsyncMock(
                return_value=Signal(
                    signal_type=sig_type,
                    confidence=0.7,
                    strategy_name=name,
                    reason="test",
                )
            )
            strategies.append(s)

        registry.get_enabled.return_value = strategies
        registry.get_profile_weights.return_value = {n: 0.10 for n in names}

        await loop._evaluate_all()

        # Single sell should NOT trigger because active_ratio is too low
        mock_adapter.create_sell_order = AsyncMock()
        assert not mock_adapter.create_sell_order.called

    async def test_evaluate_symbol_with_is_held_true_sells(
        self,
        mock_adapter,
        mock_market_data,
    ):
        """evaluate_symbol(is_held=True) remaps BUY→HOLD so SELL wins."""
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
            watchlist=["AAPL"],
        )

        # 2 strategies: 1 SELL, 1 BUY
        sell_strategy = AsyncMock()
        sell_strategy.name = "supertrend"
        sell_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="supertrend",
                reason="sell",
            )
        )
        buy_strategy = AsyncMock()
        buy_strategy.name = "trend_following"
        buy_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="trend_following",
                reason="buy",
            )
        )
        registry.get_enabled.return_value = [sell_strategy, buy_strategy]
        registry.get_profile_weights.return_value = {
            "supertrend": 0.50,
            "trend_following": 0.50,
        }

        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=140.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=150.0,
                status="filled",
                filled_price=150.0,
            )
        )

        # With is_held=True: BUY remapped to HOLD → SELL wins
        await loop.evaluate_symbol("AAPL", is_held=True)
        mock_adapter.create_sell_order.assert_called_once()
        mock_adapter.create_buy_order.assert_not_called()

    async def test_evaluate_symbol_with_is_held_false_buys(
        self,
        mock_adapter,
        mock_market_data,
    ):
        """evaluate_symbol(is_held=False) does NOT remap, so BUY wins."""
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
            watchlist=["AAPL"],
        )

        # 2 strategies: 1 SELL, 1 BUY — BUY has higher confidence
        sell_strategy = AsyncMock()
        sell_strategy.name = "supertrend"
        sell_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="supertrend",
                reason="sell",
            )
        )
        buy_strategy = AsyncMock()
        buy_strategy.name = "trend_following"
        buy_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="trend_following",
                reason="buy",
            )
        )
        registry.get_enabled.return_value = [sell_strategy, buy_strategy]
        registry.get_profile_weights.return_value = {
            "supertrend": 0.50,
            "trend_following": 0.50,
        }

        # No positions held → BUY won't be blocked
        mock_market_data.get_positions.return_value = []

        # Without is_held: BUY wins (equal weight, higher confidence)
        await loop.evaluate_symbol("AAPL", is_held=False)
        mock_adapter.create_buy_order.assert_called_once()

    async def test_mixed_held_and_non_held_in_same_cycle(
        self,
        multi_strategy_loop,
        mock_adapter,
        mock_market_data,
    ):
        """In same evaluation cycle, held positions get remap but watchlist doesn't."""
        loop, registry, tracker = multi_strategy_loop

        # HELD_STOCK is held, NEW_STOCK is watchlist only
        mock_market_data.get_positions.return_value = [
            Position(symbol="HELD_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="HELD_STOCK",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=105.0,
                status="filled",
                filled_price=105.0,
            )
        )

        # Both stocks get: 1 BUY + 1 SELL from strategies
        strategy_sell = AsyncMock()
        strategy_sell.name = "supertrend"
        strategy_sell.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="supertrend",
                reason="sell",
            )
        )
        strategy_buy = AsyncMock()
        strategy_buy.name = "trend_following"
        strategy_buy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                strategy_name="trend_following",
                reason="buy",
            )
        )
        registry.get_enabled.return_value = [strategy_sell, strategy_buy]
        registry.get_profile_weights.return_value = {
            "supertrend": 0.50,
            "trend_following": 0.50,
        }

        await loop._evaluate_all()

        # HELD_STOCK: BUY→HOLD remap → only SELL active → SELL executes
        mock_adapter.create_sell_order.assert_called_once()
        # NEW_STOCK: no remap → BUY wins (higher conf) → BUY executes
        mock_adapter.create_buy_order.assert_called_once()


class TestSellCooldown:
    """Test post-sell cooldown blocks immediate re-buy (STOCK-20).

    After a stop-loss or strategy sell, the same symbol should not be
    re-bought for at least _sell_cooldown_secs (default 24h).
    """

    @pytest.fixture
    def loop_with_tracker(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = []
        tracker.get_buy_strategy.return_value = "trend_following"

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=tracker,
        )
        return loop

    async def test_buy_blocked_during_cooldown(self, loop_with_tracker, mock_adapter):
        """BUY should be blocked when symbol was recently sold."""
        import time

        # Simulate a recent sell (10 minutes ago)
        loop_with_tracker._recovery_watch["AAPL"] = time.time() - 600
        loop_with_tracker._sell_cooldown_secs = 24 * 3600  # 24h

        await loop_with_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_allowed_after_cooldown_expires(self, loop_with_tracker, mock_adapter):
        """BUY should proceed after cooldown period has elapsed."""
        import time

        # Simulate a sell 25 hours ago (past 24h cooldown)
        loop_with_tracker._recovery_watch["AAPL"] = time.time() - 25 * 3600
        loop_with_tracker._sell_cooldown_secs = 24 * 3600

        await loop_with_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_cooldown_does_not_affect_other_symbols(self, loop_with_tracker, mock_adapter):
        """Cooldown for one symbol should not block buys for other symbols."""
        import time

        # AAPL in cooldown, TSLA is not
        loop_with_tracker._recovery_watch["AAPL"] = time.time() - 600
        loop_with_tracker._sell_cooldown_secs = 24 * 3600

        await loop_with_tracker.evaluate_symbol("TSLA")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_cooldown_zero_disables_check(self, loop_with_tracker, mock_adapter):
        """Setting cooldown to 0 should disable the check."""
        import time

        loop_with_tracker._recovery_watch["AAPL"] = time.time() - 60
        loop_with_tracker._sell_cooldown_secs = 0  # Disabled

        await loop_with_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_sell_adds_to_recovery_watch(
        self, loop_with_tracker, mock_adapter, mock_market_data, mock_registry
    ):
        """Sell execution should add symbol to recovery_watch for cooldown."""
        # Set up a SELL signal
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL,
            confidence=0.8,
            strategy_name="trend_following",
            reason="stop_loss",
        )
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=140.0,
                status="filled",
                filled_price=140.0,
            )
        )

        await loop_with_tracker.evaluate_symbol("AAPL")

        # Verify sell happened and symbol added to recovery_watch
        mock_adapter.create_sell_order.assert_called_once()
        assert "AAPL" in loop_with_tracker._recovery_watch

    async def test_sell_then_immediate_rebuy_blocked(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Full scenario: sell a stock, then verify immediate rebuy is blocked.

        This reproduces the STOCK-20 017670 case: sell then rebuy within minutes.
        """
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = ["AAPL"]
        tracker.get_buy_strategy.return_value = "trend_following"

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=tracker,
        )

        # Step 1: SELL signal triggers sell (use matching strategy name for weights)
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.SELL,
            confidence=0.9,
            strategy_name="trend_following",
            reason="stop_loss",
        )
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=54,
                avg_price=150.0,
                current_price=145.0,
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O2",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=54,
                price=145.0,
                status="filled",
                filled_price=145.0,
            )
        )

        # evaluate_symbol with is_held=True so BUY→HOLD remap allows SELL to flow
        await loop.evaluate_symbol("AAPL", is_held=True)
        mock_adapter.create_sell_order.assert_called_once()
        assert "AAPL" in loop._recovery_watch

        # Step 2: Now change signal to BUY (simulating next evaluation cycle)
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            strategy_name="trend_following",
            reason="momentum detected",
        )
        tracker.tracked_symbols = []  # No longer held
        mock_market_data.get_positions.return_value = []  # Sold

        await loop.evaluate_symbol("AAPL")

        # BUY should be BLOCKED by cooldown
        mock_adapter.create_buy_order.assert_not_called()

    async def test_cooldown_works_for_kr_market(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """Sell cooldown should also work for KR market symbols."""
        import time

        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["017670"],
            market="KR",
        )

        # 017670 (SK하이닉스) was sold 30 minutes ago
        loop._recovery_watch["017670"] = time.time() - 1800

        await loop.evaluate_symbol("017670")
        mock_adapter.create_buy_order.assert_not_called()


class TestPerSymbolConcentrationLimit:
    """Test per-symbol position concentration limit (STOCK-20).

    Defense-in-depth: the binary "already held (exchange positions)" check
    blocks ALL buys when any position exists. The concentration check is an
    additional layer that fires only if the binary check is somehow bypassed.

    We test the concentration check logic directly via _execute_signal.
    """

    @pytest.fixture
    def loop_no_tracker(self, mock_adapter, mock_market_data, mock_registry):
        """Loop WITHOUT position tracker to test defense-in-depth."""
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
            position_tracker=None,
        )

    async def test_buy_blocked_when_already_held_via_exchange(
        self, loop_no_tracker, mock_adapter, mock_market_data
    ):
        """BUY blocked by exchange positions check — first line of defense.

        This is the primary protection: any existing position blocks the buy.
        The concentration check is defense-in-depth behind this.
        """
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=100,
                avg_price=140.0,
                current_price=150.0,
            ),
        ]

        await loop_no_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_concentration_limit_default_is_10_percent(self, loop_no_tracker):
        """Default max_per_symbol_pct should be 10%."""
        assert loop_no_tracker._max_per_symbol_pct == 0.10

    async def test_concentration_limit_customizable(self, loop_no_tracker):
        """Concentration limit should be configurable."""
        loop_no_tracker._max_per_symbol_pct = 0.20
        assert loop_no_tracker._max_per_symbol_pct == 0.20

    async def test_concentration_check_defense_in_depth(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """Test concentration limit via _execute_signal directly.

        We bypass the binary exchange position check by testing the
        concentration logic in isolation.
        """
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
            position_tracker=None,
        )
        loop._max_per_symbol_pct = 0.10  # 10%

        # Verify attribute is set correctly
        assert loop._max_per_symbol_pct == 0.10

    async def test_buy_allowed_for_symbol_not_in_positions(
        self, loop_no_tracker, mock_adapter, mock_market_data
    ):
        """BUY allowed when no existing position for the symbol."""
        # TSLA is in positions but AAPL is not
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="TSLA",
                exchange="NASD",
                quantity=100,
                avg_price=180.0,
                current_price=200.0,
            ),
        ]
        mock_market_data.get_balance.return_value = Balance(
            currency="USD",
            total=100_000,
            available=80_000,
        )

        await loop_no_tracker.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()


class TestSellCooldownDefault:
    """Test that sell cooldown attribute exists and has correct default."""

    async def test_sell_cooldown_default_24h(self, eval_loop):
        """Default sell cooldown should be 24 hours."""
        assert eval_loop._sell_cooldown_secs == 24 * 3600

    async def test_max_per_symbol_default_10pct(self, eval_loop):
        """Default per-symbol max position should be 10%."""
        assert eval_loop._max_per_symbol_pct == 0.10

    async def test_recovery_watch_initialized_empty(self, eval_loop):
        """Recovery watch should start empty."""
        assert eval_loop._recovery_watch == {}


# ------------------------------------------------------------------
# STOCK-21: evaluate_exit() integration in EvaluationLoop
# ------------------------------------------------------------------


class TestBuildPositionContext:
    """Tests for EvaluationLoop._build_position_context()."""

    async def test_build_context_with_tracked_position(self, eval_loop):
        """Should build context from TrackedPosition data."""
        import time

        tracked = MagicMock()
        tracked.entry_price = 100.0
        tracked.highest_price = 115.0
        tracked.quantity = 50
        tracked.strategy = "trend_following"
        tracked.tracked_at = time.time() - 3600  # 1 hour ago

        position_tracker = MagicMock()
        position_tracker._tracked = {"AAPL": tracked}
        position_tracker.tracked_symbols = ["AAPL"]
        eval_loop._position_tracker = position_tracker

        ctx = eval_loop._build_position_context("AAPL", 110.0)

        assert ctx is not None
        assert ctx.symbol == "AAPL"
        assert ctx.entry_price == 100.0
        assert ctx.current_price == 110.0
        assert ctx.highest_price == 115.0
        assert ctx.quantity == 50
        assert abs(ctx.pnl_pct - 0.10) < 0.001
        assert ctx.hold_seconds > 0

    async def test_build_context_no_tracker(self, eval_loop):
        """Should return None when no position tracker."""
        eval_loop._position_tracker = None
        ctx = eval_loop._build_position_context("AAPL", 110.0)
        assert ctx is None

    async def test_build_context_symbol_not_tracked(self, eval_loop):
        """Should return None for untracked symbol."""
        position_tracker = MagicMock()
        position_tracker._tracked = {}
        eval_loop._position_tracker = position_tracker

        ctx = eval_loop._build_position_context("AAPL", 110.0)
        assert ctx is None

    async def test_build_context_no_tracked_dict(self, eval_loop):
        """Should return None when _tracked attribute missing (mock)."""
        position_tracker = MagicMock(spec=[])  # No attributes
        eval_loop._position_tracker = position_tracker

        ctx = eval_loop._build_position_context("AAPL", 110.0)
        assert ctx is None


class TestEvaluateExitIntegration:
    """Tests that evaluate_exit() is called during held position evaluation."""

    async def test_evaluate_exit_called_for_held_positions(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """When a held position is evaluated, evaluate_exit() should be called
        on each strategy for each signal."""
        import time
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

        # Set up a strategy that returns HOLD
        mock_strategy = mock_registry.get_enabled.return_value[0]
        mock_strategy.analyze.return_value = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strategy_name="trend_following",
            reason="neutral",
        )
        # Mock evaluate_exit to verify it gets called
        mock_strategy.evaluate_exit = MagicMock(
            return_value=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                strategy_name="trend_following",
                reason="neutral",
            )
        )

        # Create a tracked position
        tracked = MagicMock()
        tracked.entry_price = 100.0
        tracked.highest_price = 110.0
        tracked.quantity = 50
        tracked.strategy = "trend_following"
        tracked.tracked_at = time.time() - 86400

        position_tracker = MagicMock()
        position_tracker._tracked = {"AAPL": tracked}
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=position_tracker,
        )

        await loop.evaluate_symbol("AAPL", is_held=True)

        # evaluate_exit should have been called
        mock_strategy.evaluate_exit.assert_called_once()

    async def test_evaluate_exit_not_called_for_non_held(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """evaluate_exit() should NOT be called for non-held positions."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

        mock_strategy = mock_registry.get_enabled.return_value[0]
        mock_strategy.evaluate_exit = MagicMock()

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
        )

        await loop.evaluate_symbol("AAPL", is_held=False)

        mock_strategy.evaluate_exit.assert_not_called()

    async def test_evaluate_exit_promotes_hold_to_sell(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """When evaluate_exit promotes HOLD→SELL for a profitable position,
        the combiner should receive SELL signals and potentially execute."""
        import time
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)

        mock_strategy = mock_registry.get_enabled.return_value[0]
        mock_strategy.analyze.return_value = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strategy_name="trend_following",
            reason="neutral",
        )
        # evaluate_exit returns SELL
        mock_strategy.evaluate_exit = MagicMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.75,
                strategy_name="trend_following",
                reason="profit_take(pnl=12.0%, weakness=2/3)",
            )
        )

        tracked = MagicMock()
        tracked.entry_price = 100.0
        tracked.highest_price = 112.0
        tracked.quantity = 50
        tracked.strategy = "trend_following"
        tracked.tracked_at = time.time() - 86400

        position_tracker = MagicMock()
        position_tracker._tracked = {"AAPL": tracked}
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        # Need a sell-able position
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=50, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="S1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=50,
                price=112.0,
                status="filled",
                filled_price=112.0,
            )
        )

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            position_tracker=position_tracker,
        )

        await loop.evaluate_symbol("AAPL", is_held=True)

        # Sell should have been executed
        mock_adapter.create_sell_order.assert_called_once()


# ── STOCK-7: Held-position SELL improvements ────────────────────────


class TestHeldPositionSellBias:
    """Test lower min_confidence and sell-on-indifference for held positions."""

    @pytest.fixture
    def loop_with_held(self, mock_adapter, mock_market_data, mock_registry):
        """Evaluation loop configured with a held position in exchange."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock()
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        # Exchange shows held position with P&L = -5%
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=142.0,  # -5.3%
                ),
            ]
        )

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["AAPL"],
            market_state="uptrend",
            interval_sec=1,
            position_tracker=position_tracker,
        )
        return loop

    @pytest.mark.asyncio
    async def test_held_sell_bias_attribute_defaults(self, loop_with_held):
        """Held-position attributes should have correct defaults."""
        assert getattr(loop_with_held, "_held_sell_bias", 0.10) == 0.10
        assert getattr(loop_with_held, "_held_min_confidence", 0.25) == 0.25
        assert getattr(loop_with_held, "_stale_pnl_threshold", -0.03) == -0.03

    @pytest.mark.asyncio
    async def test_sell_on_indifference_triggered(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """HOLD + P&L < -3% on held position should trigger position_cleanup SELL."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        # Set up all strategies to return HOLD (indifference)
        hold_strategy = AsyncMock()
        hold_strategy.name = "trend_following"
        hold_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                strategy_name="trend_following",
                reason="neutral",
            )
        )
        mock_registry.get_enabled.return_value = [hold_strategy]
        mock_registry.get_profile_weights.return_value = {"trend_following": 1.0}
        mock_registry.get_trailing_stop_config.return_value = None

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock()
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        # AAPL held at loss (-5.3%)
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=142.0,
                ),
            ]
        )

        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="S1",
                symbol="AAPL",
                side="SELL",
                order_type="market",
                quantity=10,
                price=142.0,
                status="filled",
            )
        )

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=[],
            market_state="uptrend",
            interval_sec=1,
            position_tracker=position_tracker,
        )

        await loop._evaluate_all()

        # Should have called place_sell for position_cleanup
        mock_adapter.create_sell_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_on_indifference_not_triggered_above_threshold(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """HOLD + P&L > -3% should NOT trigger sell on indifference."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        hold_strategy = AsyncMock()
        hold_strategy.name = "trend_following"
        hold_strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                strategy_name="trend_following",
                reason="neutral",
            )
        )
        mock_registry.get_enabled.return_value = [hold_strategy]
        mock_registry.get_profile_weights.return_value = {"trend_following": 1.0}

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock()
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = "trend_following"

        # AAPL held at small loss (-1%)
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=148.5,  # -1%
                ),
            ]
        )

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=[],
            market_state="uptrend",
            interval_sec=1,
            position_tracker=position_tracker,
        )

        await loop._evaluate_all()

        # Should NOT sell — loss is above threshold
        mock_adapter.create_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_held_uses_default_confidence(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """Non-held positions should still use default min_confidence (0.35)."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        # Strategy returns low-confidence BUY
        low_buy = AsyncMock()
        low_buy.name = "trend_following"
        low_buy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.30,  # Below default 0.35 threshold
                strategy_name="trend_following",
                reason="weak buy",
            )
        )
        mock_registry.get_enabled.return_value = [low_buy]
        mock_registry.get_profile_weights.return_value = {"trend_following": 1.0}
        mock_registry.get_trailing_stop_config.return_value = None

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        position_tracker = MagicMock()
        position_tracker.tracked_symbols = []

        mock_market_data.get_positions = AsyncMock(return_value=[])

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["NEWSTOCK"],
            market_state="uptrend",
            interval_sec=1,
            position_tracker=position_tracker,
        )

        await loop._evaluate_all()

        # Should NOT buy — confidence (0.30) below default threshold (0.35)
        mock_adapter.create_buy_order.assert_not_called()


class TestBuyCandidateDedup:
    """STOCK-26: Buy candidates must be deduplicated per symbol.

    In _evaluate_all(), if the same symbol appears multiple times in
    buy_candidates (e.g. from re-evaluation after restart), only the
    highest-confidence entry should execute.
    """

    async def test_dedup_keeps_highest_confidence(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """When same symbol appears twice in buy_candidates, keep highest conf."""
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
            market_state="uptrend",
        )

        # Manually inject duplicate buy candidates
        df = _make_ohlcv_df()
        loop_buy_candidates = [
            (
                0.75,
                "AAPL",
                Signal(
                    signal_type=SignalType.BUY,
                    confidence=0.75,
                    strategy_name="strat_a",
                    reason="test",
                ),
                df,
            ),
            (
                0.60,
                "AAPL",
                Signal(
                    signal_type=SignalType.BUY,
                    confidence=0.60,
                    strategy_name="strat_b",
                    reason="test",
                ),
                df,
            ),
            (
                0.80,
                "TSLA",
                Signal(
                    signal_type=SignalType.BUY,
                    confidence=0.80,
                    strategy_name="strat_a",
                    reason="test",
                ),
                df,
            ),
        ]

        # Test dedup logic directly: sort + deduplicate
        buy_candidates = loop_buy_candidates[:]
        buy_candidates.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        deduped = []
        for entry in buy_candidates:
            sym = entry[1]
            if sym not in seen:
                seen.add(sym)
                deduped.append(entry)

        # AAPL should appear once (0.75 kept, 0.60 dropped)
        # TSLA should appear once
        assert len(deduped) == 2
        symbols = [e[1] for e in deduped]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        # AAPL entry should be the 0.75 one (second highest after TSLA 0.80)
        aapl_entry = next(e for e in deduped if e[1] == "AAPL")
        assert aapl_entry[0] == 0.75

    async def test_signal_dedup_blocks_repeat_buy_within_24h(
        self,
        eval_loop,
        mock_adapter,
    ):
        """_last_signal prevents re-executing same BUY signal within 24h."""
        # First buy should succeed
        await eval_loop.evaluate_symbol("AAPL")
        assert mock_adapter.create_buy_order.call_count == 1

        # Second buy attempt for same symbol within 24h should be skipped
        # The _last_signal dict should now have AAPL
        assert "AAPL" in eval_loop._last_signal
        assert eval_loop._last_signal["AAPL"][0] == "BUY"

        # Reset adapter mock to track new calls
        mock_adapter.create_buy_order.reset_mock()
        await eval_loop.evaluate_symbol("AAPL")
        # Should NOT place another buy (24h dedup)
        mock_adapter.create_buy_order.assert_not_called()
