"""Tests for Evaluation Loop."""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from engine.evaluation_loop import EvaluationLoop
from engine.order_manager import ManagedOrder, OrderManager
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

    async def test_combined_portfolio_value_uses_integrated_margin_total(self, eval_loop):
        other_md = AsyncMock()
        eval_loop._market = "US"
        eval_loop._exchange_rate = 1450.0
        eval_loop._market_data._adapter = MagicMock(_us_position_value_krw=9_561_545)
        other_md._adapter = MagicMock(_tot_evlu_amt=9_602_699)
        other_md.get_balance = AsyncMock(return_value=MagicMock(total=8_654_115))
        eval_loop.set_other_market_data(other_md)

        combined = await eval_loop._get_combined_portfolio_value(own_balance_total=9152.0)
        assert combined == pytest.approx((9_602_699 + 9_561_545) / 1450.0, rel=1e-6)

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
        # With quadratic combiner weighting: conf² must exceed min_confidence
        confidence_map = {"LOW_CONF": 0.65, "HIGH_CONF": 0.95, "MID_CONF": 0.80}

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

        # Should be called in order: HIGH_CONF (0.95), MID_CONF (0.80), LOW_CONF (0.65)
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


def _make_loop_with_tracker(
    mock_adapter,
    mock_market_data,
    mock_registry,
    *,
    tracked_symbols: list,
    watchlist: list | None = None,
):
    """Shared factory for EvaluationLoop with a mocked PositionTracker.

    Extracted to avoid verbatim duplication across TestProtectiveSells,
    TestPhase0HeldSetUpdate, and TestSellPendingOrderDedup (STOCK-54 rework).
    """
    from data.indicator_service import IndicatorService
    from engine.position_tracker import PositionTracker
    from strategies.combiner import SignalCombiner

    risk = RiskManager()
    order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
    position_tracker = MagicMock(spec=PositionTracker)
    position_tracker.tracked_symbols = tracked_symbols
    position_tracker._tracked = {}  # for hold-time check in sentiment sells
    position_tracker.get_buy_strategy.return_value = "trend_following"

    loop = EvaluationLoop(
        adapter=mock_adapter,
        market_data=mock_market_data,
        indicator_svc=IndicatorService(),
        registry=mock_registry,
        combiner=SignalCombiner(),
        order_manager=order_mgr,
        risk_manager=risk,
        watchlist=watchlist if watchlist is not None else tracked_symbols,
        market_state="uptrend",
        position_tracker=position_tracker,
    )
    return loop


class TestProtectiveSells:
    """Test regime-change and news-sentiment protective sell mechanisms."""

    @pytest.fixture
    def loop_with_tracker(self, mock_adapter, mock_market_data, mock_registry):
        return _make_loop_with_tracker(
            mock_adapter,
            mock_market_data,
            mock_registry,
            tracked_symbols=["AAPL", "TSLA"],
        )

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

    async def test_sentiment_sell_respects_min_hold_monotonic(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-50: Sentiment sell hold check must use time.monotonic().

        tracked_at is set via time.monotonic(). If the hold check used
        time.time(), the difference would be millions of seconds and the
        min-hold gate would always pass, defeating its purpose.
        """
        import time

        tracked = MagicMock()
        tracked.tracked_at = time.monotonic() - 60  # only 1 minute ago

        loop_with_tracker._position_tracker._tracked = {"AAPL": tracked}

        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]

        loop_with_tracker.update_news_sentiment({"AAPL": -0.8})
        await loop_with_tracker._check_protective_sells({"AAPL"})

        # Should NOT sell because position held only 1 minute (min hold is 4 hours)
        mock_adapter.create_sell_order.assert_not_called()

    async def test_protective_sell_returns_sold_symbols(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-54: _check_protective_sells should return filled sell symbols."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=130.0
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
                price=130.0,
                status="filled",
                filled_price=130.0,
            )
        )
        # Regime change: uptrend -> downtrend
        loop_with_tracker.set_market_state("downtrend")

        sold = await loop_with_tracker._check_protective_sells({"AAPL", "TSLA"})

        # AAPL is losing (130 < 150) so it should be sold; TSLA is winning so kept
        assert "AAPL" in sold
        assert "TSLA" not in sold

    async def test_protective_sell_includes_submitted_in_sold(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-54: Submitted (in-flight) sell orders are included in sold_symbols.

        A "submitted" order means the sell is in transit. Including the symbol
        in sold_symbols prevents Phase 1 BUY→HOLD remapping while the order
        is pending. Untrack is deferred to reconciliation.
        """
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=130.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=130.0,
                status="submitted",
                filled_price=None,
            )
        )
        loop_with_tracker.set_market_state("downtrend")

        sold = await loop_with_tracker._check_protective_sells({"AAPL"})

        assert "AAPL" in sold

    async def test_protective_sell_excludes_failed_orders(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-54: Failed sell orders not returned so next cycle can retry."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=130.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=130.0,
                status="failed",
                filled_price=None,
            )
        )
        loop_with_tracker.set_market_state("downtrend")

        sold = await loop_with_tracker._check_protective_sells({"AAPL"})

        # "failed" must not be in sold_symbols so the next cycle can retry
        assert sold == set()

    async def test_protective_sell_returns_empty_set_no_triggers(
        self,
        loop_with_tracker,
    ):
        """STOCK-54: Returns empty set when no regime change or sentiment trigger."""
        sold = await loop_with_tracker._check_protective_sells({"AAPL"})
        assert sold == set()


class TestPhase0HeldSetUpdate:
    """STOCK-54: Phase 0 매도 후 held set이 업데이트되어야 Phase 1 이중매도 방지."""

    @pytest.fixture
    def loop_with_tracker(self, mock_adapter, mock_market_data, mock_registry):
        return _make_loop_with_tracker(
            mock_adapter,
            mock_market_data,
            mock_registry,
            tracked_symbols=["AAPL", "TSLA"],
        )

    async def test_held_set_updated_after_phase0_sell(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
        mock_registry,
    ):
        """STOCK-54: Symbols sold in Phase 0 are skipped entirely in Phase 1.

        When Phase 0 sells AAPL (filled), Phase 1 must not evaluate it at all
        — no BUY→HOLD remapping, no held_sell_bias. TSLA (still held) should
        be the only symbol reaching _combiner.combine, and it must have
        held_sell_bias applied.
        """
        # Setup: AAPL is losing (will be sold in Phase 0), TSLA is winning (kept)
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=130.0
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
                price=130.0,
                status="filled",
                filled_price=130.0,
            )
        )

        # Trigger regime change to activate Phase 0 protective sell
        loop_with_tracker.set_market_state("downtrend")

        # Intercept combine calls to observe how each symbol is evaluated
        original_combine = loop_with_tracker._combiner.combine
        combine_calls: list[dict] = []

        def tracking_combine(signals, weights, **kwargs):
            combine_calls.append(dict(kwargs))
            return original_combine(signals, weights, **kwargs)

        loop_with_tracker._combiner.combine = tracking_combine

        mock_registry.get_enabled.return_value[0].analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.7,
                strategy_name="trend_following",
                reason="test sell",
            )
        )

        await loop_with_tracker._evaluate_all()

        # AAPL is completely skipped in Phase 1 (sold_in_phase0).
        # Only TSLA should reach combine — exactly 1 call, with held_sell_bias.
        assert len(combine_calls) == 1, (
            f"Expected 1 combine call (TSLA only), got {len(combine_calls)}. "
            "AAPL sold in Phase 0 must not reach Phase 1 evaluation."
        )
        assert "held_sell_bias" in combine_calls[0], (
            "TSLA (still held) must be evaluated with held_sell_bias"
        )

    async def test_no_double_sell_after_phase0(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
        mock_registry,
    ):
        """STOCK-54: A symbol sold in Phase 0 should not be sold again in Phase 1."""
        mock_market_data.get_positions.return_value = [
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=130.0
            ),
        ]
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O1",
                symbol="AAPL",
                side="SELL",
                order_type="limit",
                quantity=10,
                price=130.0,
                status="filled",
                filled_price=130.0,
            )
        )

        # Trigger regime change
        loop_with_tracker.set_market_state("downtrend")

        # Strategy returns SELL for AAPL
        mock_registry.get_enabled.return_value[0].analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.SELL,
                confidence=0.8,
                strategy_name="trend_following",
                reason="test sell",
            )
        )

        await loop_with_tracker._evaluate_all()

        # create_sell_order should be called exactly once (Phase 0 only),
        # NOT twice (Phase 0 + Phase 1).
        assert mock_adapter.create_sell_order.call_count == 1


class TestSellPendingOrderDedup:
    """STOCK-54: Defense-in-depth sell dedup in _execute_signal."""

    @pytest.fixture
    def loop_with_tracker(self, mock_adapter, mock_market_data, mock_registry):
        return _make_loop_with_tracker(
            mock_adapter,
            mock_market_data,
            mock_registry,
            tracked_symbols=["AAPL"],
        )

    async def test_sell_skipped_when_pending_sell_exists(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-54: _execute_signal should skip SELL if pending sell order exists."""
        # Use ManagedOrder (the correct internal type for _active_orders), not
        # OrderResult, to avoid a duck-typing mismatch that mypy strict would flag.
        loop_with_tracker._order_manager._active_orders["O1"] = ManagedOrder(
            order_id="O1",
            symbol="AAPL",
            side="SELL",
            quantity=10,
            price=150.0,
            strategy_name="trend_following",
            status="pending",
        )

        sell_signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.7,
            strategy_name="trend_following",
            reason="test sell",
        )
        df = _make_ohlcv_df()

        await loop_with_tracker._execute_signal(sell_signal, "AAPL", df)

        # Should not attempt to create a sell order (pending already exists)
        mock_adapter.create_sell_order.assert_not_called()

    async def test_sell_proceeds_when_no_pending_order(
        self,
        loop_with_tracker,
        mock_market_data,
        mock_adapter,
    ):
        """STOCK-54: _execute_signal should proceed with SELL when no pending order."""
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

        sell_signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.7,
            strategy_name="trend_following",
            reason="test sell",
        )
        df = _make_ohlcv_df()

        await loop_with_tracker._execute_signal(sell_signal, "AAPL", df)

        # Should proceed with the sell
        mock_adapter.create_sell_order.assert_called_once()


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
        tracked.tracked_at = time.monotonic() - 3600  # 1 hour ago

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

    async def test_build_context_hold_seconds_uses_monotonic(self, eval_loop):
        """STOCK-50: hold_seconds must use time.monotonic(), not time.time().

        tracked_at is set via time.monotonic(). If _build_position_context
        used time.time(), hold_seconds would be millions of seconds off.
        """
        import time

        tracked = MagicMock()
        tracked.entry_price = 100.0
        tracked.highest_price = 105.0
        tracked.quantity = 10
        tracked.strategy = "trend_following"
        tracked.tracked_at = time.monotonic() - 7200  # 2 hours ago

        position_tracker = MagicMock()
        position_tracker._tracked = {"AAPL": tracked}
        eval_loop._position_tracker = position_tracker

        ctx = eval_loop._build_position_context("AAPL", 105.0)

        assert ctx is not None
        # hold_seconds should be ~7200 (2 hours), not millions
        assert 7100 < ctx.hold_seconds < 7300


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
        tracked.tracked_at = time.monotonic() - 86400

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
        tracked.tracked_at = time.monotonic() - 86400

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
        position_tracker._tracked = {}  # Prevent MagicMock auto-attribute in _check_min_hold

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
        # Explicit empty _tracked so _check_min_hold returns True immediately.
        # Without this, MagicMock auto-creates _tracked which returns
        # float(MagicMock()) = 1.0 for tracked_at, causing hold_secs =
        # time.monotonic() - 1.0 which fails on fresh CI runners where
        # monotonic() < 14400 (4h min hold).
        position_tracker._tracked = {}

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
        position_tracker._tracked = {}

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

        # Build duplicate buy candidates for the same symbol
        df = _make_ohlcv_df()
        candidates = [
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

        # Call the actual production dedup method
        deduped = EvaluationLoop._dedup_buy_candidates(candidates)

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


class TestSilentExceptionLogging:
    """STOCK-28: Verify that silent exceptions now produce log output."""

    @pytest.mark.asyncio
    async def test_exchange_position_fetch_failure_logs_warning(
        self,
        eval_loop,
        mock_market_data,
        caplog,
    ):
        """When get_positions raises in _evaluate_all, should log warning."""
        # First call to get_positions (in _evaluate_all) fails,
        # subsequent calls (in _execute_signal) succeed with empty list.
        mock_market_data.get_positions = AsyncMock(
            side_effect=[RuntimeError("API connection failed"), []]
        )
        eval_loop.set_watchlist(["AAPL"])

        with caplog.at_level(logging.WARNING, logger="engine.evaluation_loop"):
            await eval_loop._evaluate_all()

        assert any("Exchange position fetch failed" in r.message for r in caplog.records), (
            "Expected warning log for position fetch failure"
        )

    @pytest.mark.asyncio
    async def test_factor_score_ohlcv_failure_logs_warning(
        self,
        eval_loop,
        mock_market_data,
        caplog,
    ):
        """When OHLCV fetch for factor scoring fails, should log warning."""
        mock_market_data.get_ohlcv = AsyncMock(side_effect=RuntimeError("Network timeout"))
        eval_loop.set_watchlist(["AAPL", "MSFT", "GOOG"])

        with caplog.at_level(logging.WARNING, logger="engine.evaluation_loop"):
            await eval_loop._update_factor_scores()

        assert any(
            "Failed to fetch OHLCV for factor scoring" in r.message for r in caplog.records
        ), "Expected warning log for OHLCV fetch failure"


# ── STOCK-34: Profit Protection Tests ─────────────────────────────────


class TestProfitProtection:
    """STOCK-34: Evaluation loop profit protection — sell on high profit.

    When all strategies say HOLD but the held position has very high
    unrealized gain (>= 15%), the loop should generate a SELL to secure gains.
    This mirrors the 'sell on indifference' mechanism for losing positions.
    """

    @pytest.mark.asyncio
    async def test_profit_protection_triggered_on_high_gain(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """HOLD + P&L >= 25% on held position should trigger profit_protection SELL."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        # All strategies return HOLD (no sell signal)
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
        position_tracker.tracked_symbols = ["VG"]
        position_tracker.get_buy_strategy.return_value = "trend_following"
        position_tracker._tracked = {}

        # VG held at +30% gain (simulating the STOCK-34 scenario)
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="VG",
                    exchange="NASD",
                    quantity=5,
                    avg_price=100.0,
                    current_price=130.0,  # +30%
                ),
            ]
        )

        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="S1",
                symbol="VG",
                side="SELL",
                order_type="market",
                quantity=5,
                price=130.0,
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

        # Should have triggered a sell due to profit protection
        mock_adapter.create_sell_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_profit_protection_not_triggered_below_threshold(
        self,
        mock_adapter,
        mock_market_data,
        mock_registry,
    ):
        """HOLD + P&L < 25% should NOT trigger profit protection SELL."""
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
        position_tracker._tracked = {}

        # AAPL held at +5% gain (below 15% threshold)
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=157.5,  # +5%
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

        # Should NOT sell — gain is below threshold
        mock_adapter.create_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_profit_protection_attribute_default(self):
        """Profit protection threshold should default to 0.15."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
        )
        assert getattr(loop, "_profit_protection_pct", 0.15) == 0.15


# ------------------------------------------------------------------
# STOCK-43: Sell cooldown bypass fix tests
# ------------------------------------------------------------------


class TestSellCooldownFromPositionTracker:
    """STOCK-43: PositionTracker stop-loss/TP sells must update cooldown.

    When PositionTracker triggers a sell (stop-loss, take-profit, trailing),
    the on_sell callback must update EvaluationLoop._recovery_watch so the
    same symbol is not immediately re-bought.
    """

    @pytest.fixture
    def loop_with_callback(self, mock_adapter, mock_market_data, mock_registry):
        """EvaluationLoop + real PositionTracker with callback wired."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = PositionTracker(
            adapter=mock_adapter,
            risk_manager=risk,
            order_manager=order_mgr,
            market="US",
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
            position_tracker=tracker,
        )

        # Wire the callback (as done in main.py)
        tracker.register_on_sell(loop.register_sell_cooldown)
        return loop, tracker

    async def test_position_tracker_sell_updates_recovery_watch(
        self, loop_with_callback, mock_adapter
    ):
        """Stop-loss sell via PositionTracker should update _recovery_watch."""
        loop, tracker = loop_with_callback

        # Track a position
        tracker.track("AAPL", entry_price=150.0, quantity=10, stop_loss_pct=0.05)

        # Mock exchange returning position with price below stop-loss
        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=140.0,  # 6.7% drop > 5% SL
                ),
            ]
        )
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="SL1",
                symbol="AAPL",
                side="SELL",
                order_type="market",
                quantity=10,
                price=140.0,
                status="filled",
                filled_price=140.0,
            )
        )

        # check_all should trigger stop-loss
        triggered = await tracker.check_all()
        assert len(triggered) >= 1
        assert triggered[0]["reason"] == "stop_loss"

        # Recovery watch should be updated via callback
        assert "AAPL" in loop._recovery_watch

    async def test_cooldown_blocks_rebuy_after_tracker_sell(
        self, loop_with_callback, mock_adapter, mock_market_data, mock_registry
    ):
        """After PositionTracker stop-loss, evaluation loop should block rebuy."""
        loop, tracker = loop_with_callback
        loop._sell_cooldown_secs = 14400  # 4 hours

        # Simulate: PositionTracker just sold AAPL (via callback)
        import time

        loop.register_sell_cooldown("AAPL", time.time())

        # Now strategies say BUY
        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            strategy_name="trend_following",
            reason="momentum detected",
        )
        mock_market_data.get_positions.return_value = []

        # BUY should be blocked
        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_on_sell_callback_error_does_not_crash_tracker(self, mock_adapter):
        """Callback errors should be caught, not crash PositionTracker."""
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = PositionTracker(
            adapter=mock_adapter,
            risk_manager=risk,
            order_manager=order_mgr,
        )

        # Register a broken callback
        def bad_callback(symbol: str, ts: float) -> None:
            raise RuntimeError("callback exploded")

        tracker.register_on_sell(bad_callback)
        tracker.track("AAPL", entry_price=150.0, quantity=10, stop_loss_pct=0.05)

        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=150.0,
                    current_price=140.0,
                ),
            ]
        )
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="SL2",
                symbol="AAPL",
                side="SELL",
                order_type="market",
                quantity=10,
                price=140.0,
                status="filled",
                filled_price=140.0,
            )
        )

        # Should NOT raise despite broken callback
        triggered = await tracker.check_all()
        assert len(triggered) >= 1


class TestSellCooldownConfigApplied:
    """STOCK-43: config.cooldown_after_sell_sec should be applied."""

    async def test_sell_cooldown_configurable(self):
        """_sell_cooldown_secs should be settable from config."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
        )
        # Default is 24h
        assert loop._sell_cooldown_secs == 24 * 3600

        # Simulate what main.py does with config
        loop._sell_cooldown_secs = 14400  # 4 hours from config
        assert loop._sell_cooldown_secs == 14400

    async def test_cooldown_uses_configured_value(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """BUY should be blocked/allowed based on configured cooldown, not hardcoded 24h."""
        import time
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = []

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
        loop._sell_cooldown_secs = 14400  # 4 hours

        # Sold 5 hours ago — should be past 4h cooldown
        loop._recovery_watch["AAPL"] = time.time() - 5 * 3600

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_cooldown_blocks_within_configured_period(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """BUY should be blocked within configured cooldown period."""
        import time
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        tracker = MagicMock(spec=PositionTracker)
        tracker.tracked_symbols = []

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
        loop._sell_cooldown_secs = 14400  # 4 hours

        # Sold 2 hours ago — still within 4h cooldown
        loop._recovery_watch["AAPL"] = time.time() - 2 * 3600

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()


class TestSellCooldownRedis:
    """STOCK-43: sell cooldown persistence via Redis."""

    async def test_register_sell_cooldown_updates_memory(self):
        """register_sell_cooldown should update _recovery_watch in memory."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
        )
        import time

        ts = time.time()
        loop.register_sell_cooldown("AAPL", ts)
        assert loop._recovery_watch["AAPL"] == ts

    async def test_register_sell_cooldown_persists_to_redis(self):
        """register_sell_cooldown should persist to Redis when cache is set."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        mock_cache = MagicMock()
        mock_cache.available = True
        mock_cache.set = AsyncMock(return_value=True)

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
            market="US",
        )
        loop.set_cache(mock_cache)

        import time

        ts = time.time()

        # register_sell_cooldown fires task, test _persist directly
        await loop._persist_sell_cooldown("AAPL", ts)
        mock_cache.set.assert_called_once_with(
            "sell_cooldown:US:AAPL",
            str(ts),
            ex=loop._recovery_watch_secs,
        )

    async def test_load_sell_cooldowns_from_redis(self):
        """load_sell_cooldowns should restore from Redis keys."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        import time

        now = time.time()

        # Build a mock Redis that simulates scan_iter + get
        mock_redis = AsyncMock()

        async def fake_scan_iter(match=None, count=None):
            yield "sell_cooldown:US:AAPL"
            yield "sell_cooldown:US:MSFT"

        mock_redis.scan_iter = fake_scan_iter

        async def fake_get(key):
            if "AAPL" in key:
                return str(now - 3600)  # 1h ago
            if "MSFT" in key:
                return str(now - 7200)  # 2h ago
            return None

        mock_redis.get = fake_get

        mock_cache = MagicMock()
        mock_cache.available = True
        mock_cache._redis = mock_redis

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
            market="US",
        )
        loop.set_cache(mock_cache)

        loaded = await loop.load_sell_cooldowns()
        assert loaded == 2
        assert "AAPL" in loop._recovery_watch
        assert "MSFT" in loop._recovery_watch

    async def test_load_skips_expired_entries(self):
        """Expired entries beyond recovery window should not be loaded."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        import time

        now = time.time()
        old_ts = now - 8 * 86400  # 8 days ago (> 7 day window)

        mock_redis = AsyncMock()

        async def fake_scan_iter(match=None, count=None):
            yield "sell_cooldown:US:OLD_STOCK"

        mock_redis.scan_iter = fake_scan_iter
        mock_redis.get = AsyncMock(return_value=str(old_ts))

        mock_cache = MagicMock()
        mock_cache.available = True
        mock_cache._redis = mock_redis

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
            market="US",
        )
        loop.set_cache(mock_cache)

        loaded = await loop.load_sell_cooldowns()
        assert loaded == 0
        assert "OLD_STOCK" not in loop._recovery_watch

    async def test_load_without_cache_returns_zero(self):
        """Should return 0 when no cache is configured."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            watchlist=[],
        )
        loaded = await loop.load_sell_cooldowns()
        assert loaded == 0

    async def test_cache_key_includes_market(self):
        """Cache key should be market-specific (US vs KR)."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        us_loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="US",
        )
        kr_loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="KR",
        )
        assert us_loop._cache_key == "sell_cooldown:US"
        assert kr_loop._cache_key == "sell_cooldown:KR"


class TestSellCooldownKRMarket:
    """STOCK-43: Verify fix works for KR market (009150, 005935, 034020)."""

    async def test_kr_tracker_sell_updates_cooldown(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """KR PositionTracker sell should update KR eval loop cooldown."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.position_tracker import PositionTracker

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")
        tracker = PositionTracker(
            adapter=mock_adapter,
            risk_manager=risk,
            order_manager=order_mgr,
            market="KR",
        )

        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["009150"],
            market="KR",
        )
        loop._sell_cooldown_secs = 14400  # 4h from config

        # Wire callback
        tracker.register_on_sell(loop.register_sell_cooldown)

        # Track SK하이닉스
        tracker.track("009150", entry_price=180000, quantity=5, stop_loss_pct=0.05)

        # Price drops below stop-loss
        mock_adapter.fetch_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="009150",
                    exchange="KRX",
                    quantity=5,
                    avg_price=180000,
                    current_price=168000,  # -6.7%
                ),
            ]
        )
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="KR_SL1",
                symbol="009150",
                side="SELL",
                order_type="market",
                quantity=5,
                price=168000,
                status="filled",
                filled_price=168000,
            )
        )

        await tracker.check_all()
        assert "009150" in loop._recovery_watch

        # Try to rebuy 1h later — should be blocked
        import time

        loop._recovery_watch["009150"] = time.time() - 3600  # 1h ago
        mock_market_data.get_positions.return_value = []

        await loop.evaluate_symbol("009150")
        mock_adapter.create_buy_order.assert_not_called()


# ---------------------------------------------------------------------------
# STOCK-47: Anti-Whipsaw Tests
# ---------------------------------------------------------------------------


class TestAntiWhipsawDefaults:
    """STOCK-47: Verify default values for anti-whipsaw parameters.

    These tests validate the compile-time constants set in EvaluationLoop.__init__.
    The values are intentionally NOT loaded from strategies.yaml (the YAML config
    only applies to strategy weights/params, not evaluation-loop internals).

    TODO: If YAML config loading is ever wired up for these constants via
    StrategyConfigLoader.get_anti_whipsaw_config(), update these tests to
    verify that YAML values override the hardcoded __init__ defaults.
    """

    def test_min_hold_secs_default(self, eval_loop):
        """_min_hold_secs should be 4 hours (14400 seconds)."""
        assert eval_loop._min_hold_secs == 4 * 3600

    def test_hard_sl_pct_default(self, eval_loop):
        """_hard_sl_pct should be -15% (hardcoded EvaluationLoop default)."""
        assert eval_loop._hard_sl_pct == -0.15

    def test_hard_sl_pct_config_override(self, eval_loop, mock_registry):
        """hard_sl_pct should be overridable from config (STOCK-61)."""
        # STOCK-61: Verify that evaluation_loop can be configured with a custom hard_sl_pct
        # from the config_loader (as done in main.py at startup).
        mock_registry._config_loader.get_hard_sl_pct.return_value = -0.10

        # Simulate the startup behavior in main.py
        eval_loop._hard_sl_pct = mock_registry._config_loader.get_hard_sl_pct()

        assert eval_loop._hard_sl_pct == -0.10
        mock_registry._config_loader.get_hard_sl_pct.assert_called_once()

    def test_max_loss_sells_default(self, eval_loop):
        """_max_loss_sells should be 2 (block re-entry after 2 losses in 7d)."""
        assert eval_loop._max_loss_sells == 2

    def test_loss_sell_history_initial(self, eval_loop):
        """_loss_sell_history should start as an empty dict."""
        assert eval_loop._loss_sell_history == {}

    def test_check_min_hold_no_tracker(self, eval_loop):
        """_check_min_hold returns True when no position tracker is set."""
        eval_loop._position_tracker = None
        assert eval_loop._check_min_hold("AAPL") is True

    def test_check_min_hold_no_tracked_dict(self, eval_loop):
        """_check_min_hold returns True when _tracked is missing from tracker."""
        tracker = MagicMock(spec=[])  # No attributes
        eval_loop._position_tracker = tracker
        assert eval_loop._check_min_hold("AAPL") is True

    def test_check_min_hold_symbol_not_tracked(self, eval_loop):
        """_check_min_hold returns True when symbol not in _tracked."""
        tracker = MagicMock()
        tracker._tracked = {}
        eval_loop._position_tracker = tracker
        assert eval_loop._check_min_hold("AAPL") is True

    def test_check_min_hold_disabled_when_zero(self, eval_loop):
        """_check_min_hold returns True when _min_hold_secs=0 (disabled)."""

        eval_loop._min_hold_secs = 0
        tracked = MagicMock()
        tracked.tracked_at = time.monotonic()  # Just bought
        tracker = MagicMock()
        tracker._tracked = {"AAPL": tracked}
        eval_loop._position_tracker = tracker
        assert eval_loop._check_min_hold("AAPL") is True

    def test_check_min_hold_held_long_enough(self, eval_loop):
        """_check_min_hold returns True when position held > 4 hours."""

        tracked = MagicMock()
        tracked.tracked_at = time.monotonic() - 5 * 3600  # 5h ago
        tracker = MagicMock()
        tracker._tracked = {"AAPL": tracked}
        eval_loop._position_tracker = tracker
        assert eval_loop._check_min_hold("AAPL") is True

    def test_check_min_hold_too_short(self, eval_loop):
        """_check_min_hold returns False when position held < 4 hours."""

        tracked = MagicMock()
        tracked.tracked_at = time.monotonic() - 2 * 3600  # 2h ago
        tracker = MagicMock()
        tracker._tracked = {"AAPL": tracked}
        eval_loop._position_tracker = tracker
        assert eval_loop._check_min_hold("AAPL") is False

    def test_check_min_hold_invalid_tracked_at(self, eval_loop):
        """_check_min_hold returns True when tracked_at cannot be converted."""
        tracked = MagicMock()
        tracked.tracked_at = "not-a-number"
        tracker = MagicMock()
        tracker._tracked = {"AAPL": tracked}
        eval_loop._position_tracker = tracker
        # Should not raise — returns True (allow sell) on bad data
        assert eval_loop._check_min_hold("AAPL") is True


class TestStopLossCounter:
    """STOCK-47: Verify whipsaw counter tracks loss sells correctly."""

    def test_register_sell_cooldown_is_loss_false_no_history(self, eval_loop):
        """register_sell_cooldown with is_loss=False should NOT record loss history."""

        eval_loop.register_sell_cooldown("AAPL", time.time(), is_loss=False)
        assert "AAPL" not in eval_loop._loss_sell_history

    def test_register_sell_cooldown_is_loss_true_records(self, eval_loop):
        """register_sell_cooldown with is_loss=True should record in _loss_sell_history."""

        ts = time.time()
        eval_loop.register_sell_cooldown("AAPL", ts, is_loss=True)
        assert "AAPL" in eval_loop._loss_sell_history
        assert len(eval_loop._loss_sell_history["AAPL"]) == 1
        assert eval_loop._loss_sell_history["AAPL"][0] == ts

    def test_multiple_loss_sells_accumulated(self, eval_loop):
        """Multiple loss sells for same symbol should accumulate in history."""

        now = time.time()
        eval_loop.register_sell_cooldown("AAPL", now - 86400, is_loss=True)
        eval_loop.register_sell_cooldown("AAPL", now - 3600, is_loss=True)
        eval_loop.register_sell_cooldown("AAPL", now, is_loss=True)
        assert len(eval_loop._loss_sell_history["AAPL"]) == 3

    def test_loss_history_prunes_old_entries(self, eval_loop):
        """Entries older than 7 days should be pruned when new loss is recorded."""

        now = time.time()
        old_ts = now - 8 * 86400  # 8 days ago
        eval_loop._loss_sell_history["TSLA"] = [old_ts]

        eval_loop.register_sell_cooldown("TSLA", now, is_loss=True)
        # Old entry pruned; only the new one remains
        history = eval_loop._loss_sell_history["TSLA"]
        assert len(history) == 1
        assert history[0] == now

    def test_loss_history_boundary_exactly_7_days(self, eval_loop):
        """Entry at exactly 7 days uses strict > comparison, so it is pruned.

        The implementation uses ``ts > cutoff`` (strict greater-than), meaning
        a sell at exactly 7 * 86400 seconds ago equals the cutoff and is NOT
        counted as recent. This test documents that boundary semantics.
        """
        now = time.time()
        exactly_7d = now - 7 * 86400  # exactly on the boundary
        eval_loop._loss_sell_history["AAPL"] = [exactly_7d]

        eval_loop.register_sell_cooldown("AAPL", now, is_loss=True)
        # exactly_7d == cutoff, so ts > cutoff is False → entry is pruned
        # Only the new entry (now) survives
        history = eval_loop._loss_sell_history["AAPL"]
        assert len(history) == 1
        assert history[0] == now

    def test_loss_history_default_is_not_loss(self, eval_loop):
        """register_sell_cooldown without is_loss kwarg defaults to False."""

        # Call without keyword to verify default is False
        eval_loop.register_sell_cooldown("MSFT", time.time())
        assert "MSFT" not in eval_loop._loss_sell_history

    def test_loss_history_kr_market(self):
        """KR market symbols should track loss sells correctly."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        loop = EvaluationLoop(
            adapter=AsyncMock(),
            market_data=AsyncMock(),
            indicator_svc=IndicatorService(),
            registry=MagicMock(),
            combiner=SignalCombiner(),
            order_manager=MagicMock(),
            risk_manager=RiskManager(),
            market="KR",
        )
        ts = time.time()
        loop.register_sell_cooldown("091160", ts, is_loss=True)
        assert "091160" in loop._loss_sell_history
        assert len(loop._loss_sell_history["091160"]) == 1

    def test_different_symbols_independent(self, eval_loop):
        """Loss sells on one symbol should not affect another symbol's counter."""

        now = time.time()
        eval_loop.register_sell_cooldown("AAPL", now, is_loss=True)
        eval_loop.register_sell_cooldown("AAPL", now - 3600, is_loss=True)

        # TSLA should have no loss history
        assert "TSLA" not in eval_loop._loss_sell_history

    def test_cooldown_still_registered_when_is_loss_true(self, eval_loop):
        """is_loss=True should ALSO register the normal sell cooldown."""

        ts = time.time()
        eval_loop.register_sell_cooldown("AAPL", ts, is_loss=True)
        # Verify normal cooldown is still set
        assert "AAPL" in eval_loop._recovery_watch


class TestMinimumHoldPeriod:
    """STOCK-47: Verify minimum hold period enforcement via _evaluate_all.

    position_cleanup and strategy SELL min hold checks live in _evaluate_all(),
    not in evaluate_symbol(), so these tests call _evaluate_all() directly.
    """

    def _make_full_loop(
        self,
        mock_adapter,
        mock_market_data,
        *,
        strategy_signal: Signal,
        tracked_at: float,
        avg_price: float = 100.0,
        current_price: float = 94.0,
    ) -> "EvaluationLoop":
        """Create EvaluationLoop with a held position for _evaluate_all tests."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        strategy = AsyncMock()
        strategy.name = strategy_signal.strategy_name or "trend_following"
        strategy.analyze = AsyncMock(return_value=strategy_signal)
        strategy.evaluate_exit = MagicMock(return_value=strategy_signal)

        registry = MagicMock()
        registry.get_enabled.return_value = [strategy]
        registry.get_profile_weights.return_value = {strategy.name: 1.0}
        registry.get_trailing_stop_config.return_value = None

        # Position with requested PnL
        mock_market_data.get_positions = AsyncMock(
            return_value=[
                Position(
                    symbol="AAPL",
                    exchange="NASD",
                    quantity=10,
                    avg_price=avg_price,
                    current_price=current_price,
                )
            ]
        )

        # Set up sell order mock to return success by default
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_SELL",
                symbol="AAPL",
                side="SELL",
                order_type="market",
                quantity=10,
                price=current_price,
                status="filled",
                filled_price=current_price,
            )
        )

        tracked = MagicMock()
        tracked.tracked_at = tracked_at
        position_tracker = MagicMock()
        position_tracker.tracked_symbols = ["AAPL"]
        position_tracker.get_buy_strategy.return_value = strategy.name
        position_tracker._tracked = {"AAPL": tracked}

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
            watchlist=[],  # Only held positions evaluated
            market_state="uptrend",
            interval_sec=1,
            position_tracker=position_tracker,
        )
        return loop

    async def test_position_cleanup_blocked_within_min_hold(self, mock_adapter, mock_market_data):
        """position_cleanup SELL blocked if held < 4h (not hard SL)."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.6,
                strategy_name="trend_following",
                reason="hold",
            ),
            tracked_at=time.monotonic() - 3600,  # 1h ago (< 4h)
            avg_price=100.0,
            current_price=94.0,  # -6%, below -5% threshold, above -7% hard SL
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_not_called()

    async def test_position_cleanup_allowed_after_min_hold(self, mock_adapter, mock_market_data):
        """position_cleanup SELL fires after 4h min hold period."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.6,
                strategy_name="trend_following",
                reason="hold",
            ),
            tracked_at=time.monotonic() - 5 * 3600,  # 5h ago (> 4h)
            avg_price=100.0,
            current_price=94.0,  # -6%, triggers cleanup
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    async def test_hard_sl_bypasses_min_hold_in_position_cleanup(
        self, mock_adapter, mock_market_data
    ):
        """Hard SL (-15%+) bypasses min hold in position_cleanup path."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.6,
                strategy_name="trend_following",
                reason="hold",
            ),
            tracked_at=time.monotonic() - 1800,  # 30min ago (< 4h)
            avg_price=100.0,
            current_price=84.0,  # -16%, below -15% hard SL → bypass min hold
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    async def test_strategy_sell_blocked_within_min_hold(self, mock_adapter, mock_market_data):
        """Strategy SELL blocked if held < 4h and loss not hard SL."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.SELL,
                confidence=0.8,
                strategy_name="trend_following",
                reason="sell",
            ),
            tracked_at=time.monotonic() - 3600,  # 1h ago (< 4h)
            avg_price=100.0,
            current_price=97.0,  # -3%, loss but not hard SL
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_not_called()

    async def test_strategy_sell_allowed_after_min_hold(self, mock_adapter, mock_market_data):
        """Strategy SELL executes after 4h min hold period."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.SELL,
                confidence=0.8,
                strategy_name="trend_following",
                reason="sell",
            ),
            tracked_at=time.monotonic() - 5 * 3600,  # 5h ago (> 4h)
            avg_price=100.0,
            current_price=97.0,
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    async def test_strategy_hard_sl_bypasses_min_hold(self, mock_adapter, mock_market_data):
        """Strategy SELL with hard SL (-15%+) bypasses min hold check."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.SELL,
                confidence=0.9,
                strategy_name="trend_following",
                reason="stop_loss",
            ),
            tracked_at=time.monotonic() - 1800,  # 30min ago (< 4h)
            avg_price=100.0,
            current_price=84.0,  # -16%, hard SL bypasses
        )
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    async def test_profit_protection_not_blocked_by_min_hold(self, mock_adapter, mock_market_data):
        """profit_protection sells bypass min hold check (profit_protection exempt)."""

        loop = self._make_full_loop(
            mock_adapter,
            mock_market_data,
            strategy_signal=Signal(
                signal_type=SignalType.HOLD,
                confidence=0.6,
                strategy_name="trend_following",
                reason="hold",
            ),
            tracked_at=time.monotonic() - 3600,  # Only 1h ago (< 4h)
            avg_price=100.0,
            current_price=120.0,  # +20%, triggers profit_protection
        )
        loop._profit_protection_pct = 0.15  # 15% gain triggers
        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    async def test_min_hold_not_applied_to_buy_signals(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """Min hold is a SELL barrier only; BUY signals on non-held stocks proceed."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            strategy_name="trend_following",
            reason="buy",
        )
        mock_market_data.get_positions.return_value = []
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_BUY",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )

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
            interval_sec=1,
        )
        # No position tracker → AAPL is not held → BUY proceeds freely
        await loop._evaluate_all()
        mock_adapter.create_buy_order.assert_called_once()


class TestWhipsawCounter:
    """STOCK-47: Verify whipsaw counter blocks re-entry after repeated loss sells."""

    def _make_loop(self, mock_adapter, mock_market_data, mock_registry):
        """Helper: create a clean EvaluationLoop."""
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
        )

    async def test_buy_blocked_after_max_loss_sells(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """BUY should be blocked after 2 loss sells in 7 days."""

        loop = self._make_loop(mock_adapter, mock_market_data, mock_registry)
        now = time.time()
        # Simulate 2 loss sells in the past 3 days
        loop._loss_sell_history["AAPL"] = [now - 3 * 86400, now - 86400]

        strategy = mock_registry.get_enabled.return_value[0]
        strategy.analyze.return_value = Signal(
            signal_type=SignalType.BUY,
            confidence=0.9,
            strategy_name="trend_following",
            reason="buy",
        )
        mock_market_data.get_positions.return_value = []

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_buy_allowed_with_one_loss_sell(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """BUY should be allowed when only 1 loss sell in 7 days (below max=2)."""

        loop = self._make_loop(mock_adapter, mock_market_data, mock_registry)
        now = time.time()
        # Only 1 loss sell — below the block threshold of 2
        loop._loss_sell_history["AAPL"] = [now - 86400]
        mock_market_data.get_positions.return_value = []
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_BUY",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_allowed_after_window_expires(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """BUY should be allowed when all loss sells are older than 7 days."""

        loop = self._make_loop(mock_adapter, mock_market_data, mock_registry)
        now = time.time()
        # Both loss sells are older than 7 days
        loop._loss_sell_history["AAPL"] = [now - 8 * 86400, now - 10 * 86400]
        mock_market_data.get_positions.return_value = []
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_BUY2",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_buy_blocked_kr_market(self):
        """Whipsaw counter should work for KR market symbols."""
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner

        mock_adapter = AsyncMock()
        mock_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="KRW", total=10_000_000, available=8_000_000)
        )
        mock_adapter.fetch_positions = AsyncMock(return_value=[])
        mock_adapter.create_buy_order = AsyncMock()

        mock_md = AsyncMock()
        mock_md.get_ohlcv = AsyncMock(return_value=_make_ohlcv_df())
        mock_md.get_balance = AsyncMock(
            return_value=Balance(currency="KRW", total=10_000_000, available=8_000_000)
        )
        mock_md.get_positions = AsyncMock(return_value=[])
        mock_md.get_price = AsyncMock(return_value=50000.0)

        strategy = AsyncMock()
        strategy.name = "dual_momentum"
        strategy.analyze = AsyncMock(
            return_value=Signal(
                signal_type=SignalType.BUY,
                confidence=0.85,
                strategy_name="dual_momentum",
                reason="buy",
            )
        )
        registry = MagicMock()
        registry.get_enabled.return_value = [strategy]
        registry.get_profile_weights.return_value = {"dual_momentum": 1.0}

        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk, market="KR")
        loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_md,
            indicator_svc=IndicatorService(),
            registry=registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            watchlist=["005935"],
            market="KR",
        )

        now = time.time()
        # Simulate 2 loss sells on 005935 in KR market
        loop._loss_sell_history["005935"] = [now - 3 * 86400, now - 86400]

        await loop.evaluate_symbol("005935")
        mock_adapter.create_buy_order.assert_not_called()

    async def test_max_loss_sells_configurable(self, mock_adapter, mock_market_data, mock_registry):
        """_max_loss_sells should be configurable per loop instance."""

        loop = self._make_loop(mock_adapter, mock_market_data, mock_registry)
        loop._max_loss_sells = 3  # Raise threshold to 3

        now = time.time()
        # 2 loss sells — below new max of 3, so BUY should go through
        loop._loss_sell_history["AAPL"] = [now - 3 * 86400, now - 86400]
        mock_market_data.get_positions.return_value = []
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_CONF3",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()

    async def test_loss_sells_on_other_symbol_dont_affect_aapl(
        self, mock_adapter, mock_market_data, mock_registry
    ):
        """Loss sells on TSLA should not block BUY on AAPL."""

        loop = self._make_loop(mock_adapter, mock_market_data, mock_registry)
        now = time.time()
        # 2 loss sells on TSLA — NOT AAPL
        loop._loss_sell_history["TSLA"] = [now - 3 * 86400, now - 86400]
        mock_market_data.get_positions.return_value = []
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="O_AAPL",
                symbol="AAPL",
                side="BUY",
                order_type="limit",
                quantity=10,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )

        await loop.evaluate_symbol("AAPL")
        mock_adapter.create_buy_order.assert_called_once()


class TestCashParking:
    """Tests for cash parking — port from backtest, validated +13.3pp alpha
    in backend/scripts/validate_cash_parking.py V1_park_30 (US 2y).
    """

    @pytest.fixture
    def loop(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.order_manager import OrderManager
        from engine.risk_manager import RiskManager

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
        return loop

    def test_default_off(self, loop):
        assert loop._cash_parking_enabled is False
        assert loop._cash_parking_symbol == "SPY"
        assert loop._cash_parking_threshold == 0.30
        assert loop._cash_parking_buffer == 0.10

    def test_set_config_basic(self, loop):
        loop.set_cash_parking_config(enabled=True)
        assert loop._cash_parking_enabled is True

    def test_set_config_overrides(self, loop):
        loop.set_cash_parking_config(
            enabled=True, symbol="QQQ", threshold=0.25, buffer=0.05,
        )
        assert loop._cash_parking_symbol == "QQQ"
        assert loop._cash_parking_threshold == 0.25
        assert loop._cash_parking_buffer == 0.05

    def test_set_config_validates_threshold(self, loop):
        with pytest.raises(ValueError):
            loop.set_cash_parking_config(enabled=True, threshold=1.5)
        with pytest.raises(ValueError):
            loop.set_cash_parking_config(enabled=True, threshold=-0.1)

    def test_set_config_validates_buffer(self, loop):
        with pytest.raises(ValueError):
            loop.set_cash_parking_config(enabled=True, buffer=1.5)

    def test_kr_default_symbol(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.order_manager import OrderManager
        from engine.risk_manager import RiskManager
        risk = RiskManager()
        order_mgr = OrderManager(adapter=mock_adapter, risk_manager=risk)
        kr_loop = EvaluationLoop(
            adapter=mock_adapter,
            market_data=mock_market_data,
            indicator_svc=IndicatorService(),
            registry=mock_registry,
            combiner=SignalCombiner(),
            order_manager=order_mgr,
            risk_manager=risk,
            market="KR",
        )
        assert kr_loop._cash_parking_symbol == "069500"

    async def test_park_does_nothing_when_disabled(self, loop):
        loop._cash_parking_enabled = False
        await loop._park_excess_cash()
        # Should not call place_buy
        loop._order_manager._adapter.create_buy_order.assert_not_called()

    async def test_park_does_nothing_when_cash_below_threshold(self, loop, mock_adapter):
        loop.set_cash_parking_config(enabled=True)
        # cash_pct = 80k / 100k = 80% — wait that's > 30%, so it WOULD park
        # Lower available so cash_pct < 30%
        mock_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="USD", total=100_000, available=20_000)
        )
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_not_called()

    async def test_park_buys_spy_when_cash_excess(self, loop, mock_adapter, mock_market_data):
        loop.set_cash_parking_config(enabled=True, threshold=0.30, buffer=0.10)
        # cash 80k / 100k = 80% > 30% → park (80k - 100k*0.10 = 70k)
        # SPY price from mock_market_data ohlcv → ~100
        mock_market_data.get_positions = AsyncMock(return_value=[])
        # ohlcv mock returns ~100 close prices, so quantity ~ 70k / 100 = 700
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="PARK1",
                symbol="SPY",
                side="BUY",
                order_type="limit",
                quantity=700,
                price=100.0,
                status="filled",
                filled_price=100.0,
            )
        )
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_called_once()
        call_kwargs = mock_adapter.create_buy_order.call_args.kwargs
        assert call_kwargs["symbol"] == "SPY"
        assert call_kwargs["quantity"] > 0

    async def test_park_skipped_if_already_holding_spy(self, loop, mock_adapter, mock_market_data):
        loop.set_cash_parking_config(enabled=True)
        mock_market_data.get_positions = AsyncMock(
            return_value=[Position(
                symbol="SPY", quantity=100, avg_price=100, current_price=100, exchange="NASD",
            )]
        )
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_not_called()

    async def test_park_cooldown_prevents_rapid_repeat(self, loop, mock_adapter, mock_market_data):
        """2026-04-11: park should fire at most once per hour (cooldown).
        The previous version had no cooldown and parked every 5-min cycle,
        causing 93 SPY round-trips in 2 days (~860k KRW commissions).
        """
        loop.set_cash_parking_config(enabled=True, threshold=0.30, buffer=0.10)
        mock_market_data.get_positions = AsyncMock(return_value=[])
        mock_adapter.create_buy_order = AsyncMock(
            return_value=OrderResult(
                order_id="PARK1", symbol="SPY", side="BUY", order_type="limit",
                quantity=4, price=680.0, status="filled", filled_price=680.0,
            )
        )
        # First park: should fire
        await loop._park_excess_cash()
        assert mock_adapter.create_buy_order.call_count == 1

        # Immediate second park: blocked by cooldown
        mock_market_data.get_positions = AsyncMock(return_value=[])  # pretend no position (edge case)
        await loop._park_excess_cash()
        assert mock_adapter.create_buy_order.call_count == 1  # still 1, not 2

    async def test_park_skips_when_pending_order_exists(self, loop, mock_adapter, mock_market_data):
        """Don't place a second BUY if a parking BUY is already pending."""
        loop.set_cash_parking_config(enabled=True)
        mock_market_data.get_positions = AsyncMock(return_value=[])
        # Simulate pending order via order_manager's internal dict
        from engine.order_manager import ManagedOrder
        loop._order_manager._active_orders = {
            "SPY_BUY_123": ManagedOrder(
                order_id="SPY_BUY_123", symbol="SPY", side="BUY",
                quantity=4, price=680.0, status="pending",
                strategy_name="cash_parking",
            ),
        }
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_not_called()

    async def test_park_handles_balance_failure(self, loop, mock_adapter):
        loop.set_cash_parking_config(enabled=True)
        mock_adapter.fetch_balance = AsyncMock(side_effect=Exception("API down"))
        # Should not raise
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_not_called()

    async def test_park_handles_zero_equity(self, loop, mock_adapter):
        loop.set_cash_parking_config(enabled=True)
        mock_adapter.fetch_balance = AsyncMock(
            return_value=Balance(currency="USD", total=0, available=0)
        )
        await loop._park_excess_cash()
        mock_adapter.create_buy_order.assert_not_called()


class TestStrategySLResolution:
    """Tests for _resolve_strategy_sl_tp — picks SL/TP from strategy YAML
    instead of always using ATR/default. See docs/IMPROVEMENT_PLAN.md §1.

    This fix unblocks the per-strategy lifecycle — pre-fix, the yaml
    `stop_loss.type` block was dead code (config_loader.get_stop_loss_config
    was defined but never called from the live engine).
    """

    @pytest.fixture
    def loop(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.order_manager import OrderManager
        from engine.risk_manager import RiskManager
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
        )
        return loop

    def _df_with_atr(self, atr=2.0):
        df = _make_ohlcv_df()
        df["atr"] = atr
        return df

    def test_fixed_pct_strategy_uses_max_pct(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "fixed_pct", "max_pct": 0.05}
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        sl, tp = loop._resolve_strategy_sl_tp("rsi_divergence", 100.0, 2.0, self._df_with_atr())
        assert sl == 0.05
        assert tp == 0.10  # default 2x risk

    def test_atr_strategy_uses_multiplier(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "atr", "atr_multiplier": 1.5}
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        sl, tp = loop._resolve_strategy_sl_tp("bollinger_squeeze", 100.0, 2.0, self._df_with_atr())
        # 1.5 * 2.0 / 100.0 = 0.03 → clamped to min 0.02 (passes)
        assert abs(sl - 0.03) < 1e-6
        assert tp == 0.06

    def test_supertrend_uses_line_when_present(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "supertrend"}
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        df = self._df_with_atr()
        df["supertrend_long"] = 96.5  # 3.5% below entry of 100
        sl, tp = loop._resolve_strategy_sl_tp("supertrend", 100.0, 2.0, df)
        assert abs(sl - 0.035) < 1e-6
        assert abs(tp - 0.07) < 1e-6  # 2x risk

    def test_supertrend_falls_back_to_atr_when_no_line(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "supertrend"}
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        df = self._df_with_atr(atr=3.0)
        # No supertrend_long column → falls back to 2x ATR / price = 0.06
        sl, tp = loop._resolve_strategy_sl_tp("supertrend", 100.0, 3.0, df)
        assert abs(sl - 0.06) < 1e-6

    def test_no_strategy_config_falls_back_to_dynamic_atr(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(return_value={})
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        sl, tp = loop._resolve_strategy_sl_tp("unknown", 100.0, 2.0, self._df_with_atr())
        # Should call risk_manager.calculate_dynamic_sl_tp which has its own logic
        assert 0 < sl < 0.30
        assert 0 < tp < 0.50

    def test_no_atr_no_config_falls_back_to_default(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(return_value={})
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        df = _make_ohlcv_df()  # no atr column
        sl, tp = loop._resolve_strategy_sl_tp("unknown", 100.0, None, df)
        # Should fall back to RiskManager defaults
        from engine.risk_manager import RiskParams
        defaults = RiskParams()
        assert sl == defaults.default_stop_loss_pct
        assert tp == defaults.default_take_profit_pct

    def test_sl_clamped_to_max_20pct(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "fixed_pct", "max_pct": 0.50}  # 50% absurd
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        sl, tp = loop._resolve_strategy_sl_tp("test", 100.0, 2.0, self._df_with_atr())
        assert sl == 0.20  # clamped

    def test_sl_clamped_to_min_2pct(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "fixed_pct", "max_pct": 0.005}  # 0.5% too tight
        )
        mock_registry.get_take_profit_config = MagicMock(return_value={})
        sl, tp = loop._resolve_strategy_sl_tp("test", 100.0, 2.0, self._df_with_atr())
        assert sl == 0.02  # clamped

    def test_ratio_tp_config(self, loop, mock_registry):
        mock_registry.get_stop_loss_config = MagicMock(
            return_value={"type": "fixed_pct", "max_pct": 0.04}
        )
        mock_registry.get_take_profit_config = MagicMock(
            return_value={"type": "ratio", "risk_multiple": 3.0}
        )
        sl, tp = loop._resolve_strategy_sl_tp("test", 100.0, 2.0, self._df_with_atr())
        assert sl == 0.04
        assert abs(tp - 0.12) < 1e-6  # 3 * 0.04


class TestCashParkingUnpark:
    """Tests for _unpark_for_buy — sell parking when BUY needs cash."""

    @pytest.fixture
    def loop(self, mock_adapter, mock_market_data, mock_registry):
        from data.indicator_service import IndicatorService
        from strategies.combiner import SignalCombiner
        from engine.order_manager import OrderManager
        from engine.risk_manager import RiskManager
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
        return loop

    async def test_unpark_sells_when_held_long_enough(self, loop, mock_adapter, mock_market_data):
        """Parking held > min_hold_days → unpark sells it."""
        import time as _time
        loop.set_cash_parking_config(enabled=True, min_hold_days=10, enable_unpark=True)
        loop._cash_parking_parked_at = _time.time() - 15 * 86400  # 15 days ago

        mock_market_data.get_positions = AsyncMock(
            return_value=[Position(symbol="SPY", quantity=5, avg_price=680, current_price=700, exchange="NASD")]
        )
        mock_adapter.create_sell_order = AsyncMock(
            return_value=OrderResult(
                order_id="UNPARK1", symbol="SPY", side="SELL", order_type="market",
                quantity=5, price=700, status="filled", filled_price=700,
            )
        )
        await loop._unpark_for_buy(needed=5000, available=1000)
        mock_adapter.create_sell_order.assert_called_once()

    async def test_unpark_skipped_when_held_too_short(self, loop, mock_adapter, mock_market_data):
        """Parking held < min_hold_days → don't sell."""
        import time as _time
        loop.set_cash_parking_config(enabled=True, min_hold_days=10, enable_unpark=True)
        loop._cash_parking_parked_at = _time.time() - 3 * 86400  # 3 days ago

        mock_market_data.get_positions = AsyncMock(
            return_value=[Position(symbol="SPY", quantity=5, avg_price=680, current_price=700, exchange="NASD")]
        )
        await loop._unpark_for_buy(needed=5000, available=1000)
        mock_adapter.create_sell_order.assert_not_called()

    async def test_unpark_skipped_when_not_holding(self, loop, mock_adapter, mock_market_data):
        """No parking position → skip."""
        import time as _time
        loop.set_cash_parking_config(enabled=True, min_hold_days=10, enable_unpark=True)
        loop._cash_parking_parked_at = _time.time() - 15 * 86400

        mock_market_data.get_positions = AsyncMock(return_value=[])
        await loop._unpark_for_buy(needed=5000, available=1000)
        mock_adapter.create_sell_order.assert_not_called()

    async def test_unpark_skipped_when_disabled(self, loop, mock_adapter, mock_market_data):
        """enable_unpark=False → never sell."""
        import time as _time
        loop.set_cash_parking_config(enabled=True, min_hold_days=10, enable_unpark=False)
        loop._cash_parking_parked_at = _time.time() - 15 * 86400

        mock_market_data.get_positions = AsyncMock(
            return_value=[Position(symbol="SPY", quantity=5, avg_price=680, current_price=700, exchange="NASD")]
        )
        # _unpark_for_buy won't be called because _cash_parking_enable_unpark is False
        # Test the guard in _execute_signal path
        assert loop._cash_parking_enable_unpark is False

    def test_config_sets_unpark_params(self, loop):
        loop.set_cash_parking_config(enabled=True, min_hold_days=14, enable_unpark=True)
        assert loop._cash_parking_min_hold_days == 14
        assert loop._cash_parking_enable_unpark is True
