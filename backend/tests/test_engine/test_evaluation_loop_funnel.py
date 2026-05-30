"""F1 attribution funnel tests — BUY-flow rejection counters."""

from datetime import date as _date
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from engine.evaluation_loop import EvaluationLoop
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from exchange.base import Balance, OrderResult, Position
from strategies.base import Signal
from core.enums import SignalType


def _df(n=50):
    np.random.seed(7)
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(100_000, 500_000, n).astype(float),
        }
    )


@pytest.fixture
def buy_signal():
    return Signal(
        signal_type=SignalType.BUY,
        confidence=0.8,
        strategy_name="trend_following",
        reason="test",
    )


@pytest.fixture
def loop():
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=Balance(currency="USD", total=100_000, available=80_000)
    )
    adapter.fetch_positions = AsyncMock(return_value=[])
    adapter.create_buy_order = AsyncMock(
        return_value=OrderResult(
            order_id="O1", symbol="AAPL", side="BUY", order_type="limit",
            quantity=10, price=150.0, status="filled", filled_price=150.0,
        )
    )

    md = AsyncMock()
    md.get_ohlcv = AsyncMock(return_value=_df())
    md.get_balance = AsyncMock(
        return_value=Balance(currency="USD", total=100_000, available=80_000)
    )
    md.get_positions = AsyncMock(return_value=[])
    md.get_price = AsyncMock(return_value=150.0)

    registry = MagicMock()
    strat = AsyncMock()
    strat.name = "trend_following"
    strat.analyze = AsyncMock(
        return_value=Signal(
            signal_type=SignalType.BUY, confidence=0.8,
            strategy_name="trend_following", reason="test",
        )
    )
    registry.get_enabled.return_value = [strat]
    registry.get_profile_weights.return_value = {"trend_following": 1.0}

    from data.indicator_service import IndicatorService
    from strategies.combiner import SignalCombiner

    risk = RiskManager()
    return EvaluationLoop(
        adapter=adapter,
        market_data=md,
        indicator_svc=IndicatorService(),
        registry=registry,
        combiner=SignalCombiner(),
        order_manager=OrderManager(adapter=adapter, risk_manager=risk),
        risk_manager=risk,
        watchlist=["AAPL"],
        market_state="uptrend",
        interval_sec=1,
    )


class TestFunnelInit:
    def test_counters_start_empty(self, loop):
        assert loop._reject_counters == {}
        assert loop._buy_flow_counters == {"buy_signals_total": 0, "buys_placed": 0}

    def test_bump_reject_increments(self, loop):
        loop._bump_reject("foo")
        loop._bump_reject("foo")
        loop._bump_reject("bar")
        assert loop._reject_counters == {"foo": 2, "bar": 1}


class TestDailyResetIsolation:
    """The reset method must run at signal start, before any counter touch."""

    def test_reset_clears_all_counters_on_date_change(self, loop):
        loop._daily_buy_date = "2020-01-01"
        loop._daily_buy_count = 3
        loop._reject_counters = {"opening_avoidance": 5}
        loop._buy_flow_counters = {"buy_signals_total": 9, "buys_placed": 3}

        loop._reset_daily_counters_if_needed()

        today = _date.today().isoformat()
        assert loop._daily_buy_date == today
        assert loop._daily_buy_count == 0
        assert loop._reject_counters == {}
        assert loop._buy_flow_counters == {"buy_signals_total": 0, "buys_placed": 0}

    def test_reset_no_op_when_date_same(self, loop):
        today = _date.today().isoformat()
        loop._daily_buy_date = today
        loop._daily_buy_count = 2
        loop._reject_counters = {"foo": 1}
        loop._buy_flow_counters = {"buy_signals_total": 4, "buys_placed": 2}

        loop._reset_daily_counters_if_needed()

        assert loop._daily_buy_count == 2
        assert loop._reject_counters == {"foo": 1}
        assert loop._buy_flow_counters == {"buy_signals_total": 4, "buys_placed": 2}


class TestRejectionCounters:
    """Verify each rejection site bumps its dedicated counter."""

    async def test_opening_avoidance_bump(self, loop, buy_signal, monkeypatch):
        loop._opening_avoidance_minutes = 30
        loop._market = "US"
        # Force is_opening_minutes True
        monkeypatch.setattr(
            "engine.scheduler.is_opening_minutes", lambda _m, _n: True
        )
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("opening_avoidance") == 1
        assert loop._buy_flow_counters["buy_signals_total"] == 1
        assert loop._buy_flow_counters["buys_placed"] == 0

    async def test_daily_limit_bump(self, loop, buy_signal):
        loop._daily_buy_limit = 1
        loop._daily_buy_count = 1
        loop._daily_buy_date = _date.today().isoformat()
        # confidence 0.8 < override 0.9 → blocked
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("daily_limit") == 1

    async def test_pending_order_bump(self, loop, buy_signal):
        loop._daily_buy_date = _date.today().isoformat()
        loop._order_manager.has_pending_order = MagicMock(return_value=True)
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("pending_order") == 1

    async def test_already_held_bump(self, loop, buy_signal):
        loop._daily_buy_date = _date.today().isoformat()
        tracker = MagicMock()
        tracker.tracked_symbols = {"AAPL"}
        loop._position_tracker = tracker
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("already_held") == 1

    async def test_sell_cooldown_bump(self, loop, buy_signal):
        import time as _time

        loop._daily_buy_date = _date.today().isoformat()
        loop._sell_cooldown_secs = 3600
        loop._recovery_watch["AAPL"] = _time.time() - 10  # just sold
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("sell_cooldown") == 1

    async def test_sell_cooldown_bypassed_for_sizing_up_held(self, loop, buy_signal):
        """When the symbol is still held (partial sell left a placeholder)
        AND sizing_up is enabled, the BUY signal is an add-on, not a re-
        entry. sell_cooldown must NOT fire. whipsaw_block further down
        still catches genuine loser-symbol cases."""
        import time as _time

        loop._daily_buy_date = _date.today().isoformat()
        loop._sell_cooldown_secs = 3600
        loop._recovery_watch["AAPL"] = _time.time() - 10
        tracker = MagicMock()
        tracker.tracked_symbols = {"AAPL"}
        loop._position_tracker = tracker
        loop.set_sizing_up_config(enabled=True, threshold=0.5, min_confidence=0.5)

        await loop._execute_signal(buy_signal, "AAPL", _df())

        assert loop._reject_counters.get("sell_cooldown") is None

    async def test_whipsaw_block_bump(self, loop, buy_signal):
        import time as _time

        loop._daily_buy_date = _date.today().isoformat()
        loop._max_loss_sells = 2
        loop._loss_sell_history["AAPL"] = [_time.time() - 100, _time.time() - 200]
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("whipsaw_block") == 1

    async def test_same_signal_24h_bump(self, loop, buy_signal):
        import time as _time

        loop._daily_buy_date = _date.today().isoformat()
        loop._last_signal["AAPL"] = ("BUY", _time.time() - 100)
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("same_signal_24h") == 1

    async def test_same_signal_24h_bypassed_for_sizing_up_held(self, loop, buy_signal):
        """If symbol is held and sizing_up is enabled, same_signal_24h must
        NOT fire — the repeated daily BUY signal is required to reach the
        sizing-up branch. Without this bypass, 1-share placeholders stay
        1-share forever."""
        import time as _time

        loop._daily_buy_date = _date.today().isoformat()
        loop._last_signal["AAPL"] = ("BUY", _time.time() - 100)
        # Held + sizing_up enabled
        tracker = MagicMock()
        tracker.tracked_symbols = {"AAPL"}
        loop._position_tracker = tracker
        loop.set_sizing_up_config(enabled=True, threshold=0.5, min_confidence=0.5)

        await loop._execute_signal(buy_signal, "AAPL", _df())

        assert loop._reject_counters.get("same_signal_24h") is None


class TestSuccessCounter:
    async def test_buys_placed_increments_on_successful_order(self, loop, buy_signal):
        loop._daily_buy_date = _date.today().isoformat()
        await loop._execute_signal(buy_signal, "AAPL", _df())
        # signal counted at top, order placed → both counters move
        assert loop._buy_flow_counters["buy_signals_total"] == 1
        assert loop._buy_flow_counters["buys_placed"] == 1
        assert loop._reject_counters == {}


class TestSizingTokenCategorization:
    """sizing.reason like 'max_exposure: 80%' should bucket into sizing_max_exposure."""

    async def test_sizing_reason_prefix_extracted(self, loop, buy_signal):
        loop._daily_buy_date = _date.today().isoformat()
        # Force sizing rejection with structured reason
        from engine.risk_manager import PositionSizeResult

        loop._risk_manager.calculate_kelly_position_size = MagicMock(
            return_value=PositionSizeResult(
                allowed=False, quantity=0, allocation_usd=0, risk_per_share=0,
                reason="max_exposure: 80% of portfolio",
            )
        )
        await loop._execute_signal(buy_signal, "AAPL", _df())
        assert loop._reject_counters.get("sizing_max_exposure") == 1
