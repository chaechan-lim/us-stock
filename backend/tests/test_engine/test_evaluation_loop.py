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


class TestDailyBuyLimit:
    """Test daily buy limit with dynamic confidence escalation."""

    async def test_daily_limit_blocks_low_conf_at_hard_cap(self, eval_loop, mock_adapter, mock_registry):
        """After limit reached, low-confidence buys blocked."""
        eval_loop._daily_buy_limit = 1
        eval_loop._daily_buy_count = 1  # already at limit
        from datetime import date as _date
        eval_loop._daily_buy_date = _date.today().isoformat()

        # confidence=0.8 signal (below 0.90 override threshold)
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_daily_limit_allows_ultra_high_conf_override(self, eval_loop, mock_adapter, mock_registry):
        """Ultra-high confidence (0.90+) bypasses hard cap."""
        eval_loop._daily_buy_limit = 1
        eval_loop._daily_buy_count = 1
        from datetime import date as _date
        eval_loop._daily_buy_date = _date.today().isoformat()

        # Set strategy to return 0.95 confidence
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY, confidence=0.95,
            strategy_name="trend_following", reason="strong signal",
        )
        await eval_loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_escalating_confidence_at_80pct_usage(self, eval_loop, mock_adapter, mock_registry):
        """At 80%+ usage, need confidence >= 0.75."""
        eval_loop._daily_buy_limit = 5
        eval_loop._daily_buy_count = 4  # 80% used
        from datetime import date as _date
        eval_loop._daily_buy_date = _date.today().isoformat()

        # Signal with 0.60 confidence — should be blocked (need 0.75)
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY, confidence=0.60,
            strategy_name="trend_following", reason="moderate",
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


class TestHeldPositionEvaluation:
    """Test that held positions are always evaluated even if not in watchlist."""

    async def test_held_position_evaluated_when_not_in_watchlist(
        self, mock_adapter, mock_market_data, mock_registry,
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
            "AAPL": Signal(signal_type=SignalType.HOLD, confidence=0.3,
                           strategy_name="trend_following", reason="hold"),
            "HELD_STOCK": Signal(signal_type=SignalType.SELL, confidence=0.8,
                                 strategy_name="trend_following", reason="sell"),
        }

        async def dynamic_analyze(df, symbol):
            return signal_map.get(symbol, signal_map["AAPL"])

        strategy.analyze = AsyncMock(side_effect=dynamic_analyze)

        mock_market_data.get_positions.return_value = [
            Position(symbol="HELD_STOCK", exchange="NASD", quantity=10, avg_price=100.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O2", symbol="HELD_STOCK", side="SELL",
            order_type="limit", quantity=10, price=105.0,
            status="filled", filled_price=105.0,
        ))

        await loop._evaluate_all()

        # HELD_STOCK should get a sell order even though not in watchlist
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args
        assert call_kwargs.kwargs.get("symbol") == "HELD_STOCK" or call_kwargs.args[0] == "HELD_STOCK"

    async def test_no_duplicate_evaluation_when_in_both(
        self, mock_adapter, mock_market_data, mock_registry,
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
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Regime downtrend should sell losing positions."""
        # AAPL is losing, TSLA is winning
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=140.0),
            Position(symbol="TSLA", exchange="NASD", quantity=5,
                     avg_price=200.0, current_price=220.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=140.0,
            status="filled", filled_price=140.0,
        ))

        # Transition uptrend -> downtrend
        loop_with_tracker.set_market_state("downtrend")
        await loop_with_tracker._check_protective_sells({"AAPL", "TSLA"})

        # Only AAPL (losing) should be sold
        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["quantity"] == 10

    async def test_regime_sell_skips_winning_positions(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Regime sell should keep winning positions."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=170.0),
        ]

        loop_with_tracker.set_market_state("downtrend")
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # No sell — position is winning
        mock_adapter.create_sell_order.assert_not_called()

    async def test_no_regime_sell_in_same_state(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """No regime sell when market hasn't worsened."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=140.0),
        ]

        # Already in downtrend, no transition
        loop_with_tracker._market_state = "downtrend"
        loop_with_tracker._prev_market_state = "downtrend"
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_not_called()

    async def test_regime_sell_on_sideways_to_downtrend(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Sideways→downtrend transition should also trigger regime sell."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=140.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=140.0,
            status="filled", filled_price=140.0,
        ))

        # Transition sideways -> downtrend
        loop_with_tracker._prev_market_state = "sideways"
        loop_with_tracker._market_state = "downtrend"
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_called_once()

    async def test_sentiment_sell_on_negative(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Strongly negative sentiment should trigger sell."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=160.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=160.0,
            status="filled", filled_price=160.0,
        ))

        loop_with_tracker.update_news_sentiment({"AAPL": -0.7})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"

    async def test_sentiment_no_sell_on_mild_negative(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Mildly negative sentiment (-0.3) should NOT trigger sell."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=160.0),
        ]

        loop_with_tracker.update_news_sentiment({"AAPL": -0.3})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        mock_adapter.create_sell_order.assert_not_called()

    async def test_sentiment_cleared_after_check(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Processed sentiments should be cleared to avoid re-selling."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=160.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=160.0,
            status="filled", filled_price=160.0,
        ))

        loop_with_tracker.update_news_sentiment({"AAPL": -0.8})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # Sentiment for AAPL should be cleared
        assert "AAPL" not in loop_with_tracker._news_sentiment

    async def test_kr_market_uses_krx_for_protective_sell(
        self, mock_adapter, mock_market_data, mock_registry,
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
            Position(symbol="005930", exchange="KRX", quantity=10,
                     avg_price=70000.0, current_price=65000.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="005930", side="SELL",
            order_type="limit", quantity=10, price=65000.0,
            status="filled", filled_price=65000.0,
        ))

        loop.set_market_state("downtrend")
        await loop._check_protective_sells({"005930"})

        mock_adapter.create_sell_order.assert_called_once()
        call_kwargs = mock_adapter.create_sell_order.call_args.kwargs
        assert call_kwargs["exchange"] == "KRX"

    async def test_position_untracked_after_protective_sell(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """Position should be untracked after successful protective sell."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=160.0),
        ]
        mock_adapter.create_sell_order = AsyncMock(return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="SELL",
            order_type="limit", quantity=10, price=160.0,
            status="filled", filled_price=160.0,
        ))

        loop_with_tracker.update_news_sentiment({"AAPL": -0.8})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        loop_with_tracker._position_tracker.untrack.assert_called_with("AAPL")

    async def test_no_protective_sell_when_no_triggers(
        self, loop_with_tracker, mock_market_data, mock_adapter,
    ):
        """No sell when regime is stable and no negative sentiment."""
        mock_market_data.get_positions.return_value = [
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=140.0),
        ]

        # No regime change, no sentiment
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # get_positions should not even be called (early return)
        mock_market_data.get_positions.assert_not_called()
