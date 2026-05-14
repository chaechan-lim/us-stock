"""P1 (#55) — time-based stale exit unit tests.

Covers:
1. set_stale_time_exit setter validation + state
2. cleanup fires when held >= N days + pnl < threshold (no -5% needed)
3. cleanup does NOT fire when conditions partial
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
import time as time_mod

from engine.evaluation_loop import EvaluationLoop
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from exchange.base import Balance, OrderResult, Position
from strategies.base import Signal
from core.enums import SignalType


def _df(n=50):
    np.random.seed(13)
    close = 150 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    })


@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.fetch_balance = AsyncMock(
        return_value=Balance(currency="USD", total=100_000, available=80_000)
    )
    return adapter


@pytest.fixture
def mock_market_data():
    svc = AsyncMock()
    svc.get_ohlcv = AsyncMock(return_value=_df())
    svc.get_balance = AsyncMock(
        return_value=Balance(currency="USD", total=100_000, available=80_000)
    )
    svc.get_positions = AsyncMock(return_value=[])
    svc.get_price = AsyncMock(return_value=147.0)
    return svc


@pytest.fixture
def mock_registry():
    return MagicMock()


class TestSetStaleTimeExit:
    """Setter validation."""

    @pytest.fixture
    def loop(self):
        loop = EvaluationLoop.__new__(EvaluationLoop)
        loop._stale_time_days = 0
        loop._stale_time_pnl_threshold = 0.0
        loop._market = "US"
        return loop

    def test_disable_default(self, loop):
        loop.set_stale_time_exit(days=0)
        assert loop._stale_time_days == 0
        assert loop._stale_time_pnl_threshold == 0.0

    def test_enable_with_threshold(self, loop):
        loop.set_stale_time_exit(days=2, pnl_threshold=-0.02)
        assert loop._stale_time_days == 2
        assert loop._stale_time_pnl_threshold == pytest.approx(-0.02)

    def test_none_days_treated_as_zero(self, loop):
        loop.set_stale_time_exit(days=None)
        assert loop._stale_time_days == 0

    def test_rejects_negative_days(self, loop):
        with pytest.raises(ValueError):
            loop.set_stale_time_exit(days=-1)

    def test_rejects_extreme_threshold(self, loop):
        with pytest.raises(ValueError):
            loop.set_stale_time_exit(days=2, pnl_threshold=-0.50)
        with pytest.raises(ValueError):
            loop.set_stale_time_exit(days=2, pnl_threshold=0.50)


def _build_loop(mock_adapter, mock_market_data, mock_registry, stale_days, stale_thr):
    """Build an EvaluationLoop with stale_time_exit configured."""
    from data.indicator_service import IndicatorService
    from strategies.combiner import SignalCombiner

    # All HOLD strategies
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
    loop.set_stale_time_exit(days=stale_days, pnl_threshold=stale_thr)
    return loop, position_tracker


def _setup_held_position(mock_market_data, mock_adapter, current_price=147.0):
    """AAPL held at 150, current_price=147 → −2% loss (below -2% threshold)."""
    mock_market_data.get_positions = AsyncMock(
        return_value=[
            Position(symbol="AAPL", exchange="NASD", quantity=10,
                     avg_price=150.0, current_price=current_price),
        ]
    )
    mock_adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="S1", symbol="AAPL", side="SELL", order_type="market",
            quantity=10, price=current_price, status="filled",
        )
    )


def _set_hold_secs(position_tracker, seconds: float) -> None:
    """Pretend the AAPL position has been held for `seconds` (monotonic)."""
    tracked = MagicMock()
    tracked.tracked_at = time_mod.monotonic() - seconds
    position_tracker._tracked = {"AAPL": tracked}


class TestStaleTimeTrigger:
    @pytest.mark.asyncio
    async def test_fires_when_held_long_and_below_threshold(
        self, mock_adapter, mock_market_data, mock_registry,
    ):
        loop, pt = _build_loop(
            mock_adapter, mock_market_data, mock_registry,
            stale_days=2, stale_thr=-0.02,
        )
        # current_price 145 → −3.3% loss → below -2% threshold
        _setup_held_position(mock_market_data, mock_adapter, current_price=145.0)
        # Held for 3 days (well above 2-day stale_time)
        _set_hold_secs(pt, 3 * 86400)
        # Bypass min_hold for cleanup
        loop._min_hold_secs = 0

        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_fire_when_held_short(
        self, mock_adapter, mock_market_data, mock_registry,
    ):
        loop, pt = _build_loop(
            mock_adapter, mock_market_data, mock_registry,
            stale_days=2, stale_thr=-0.02,
        )
        _setup_held_position(mock_market_data, mock_adapter, current_price=145.0)
        # Held for only 1 day (below 2-day stale_time)
        _set_hold_secs(pt, 1 * 86400)
        loop._min_hold_secs = 0
        # Loss is -3.3% which is NOT below -5% loss_trigger either
        loop._stale_pnl_threshold = -0.05

        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_fire_when_pnl_above_threshold(
        self, mock_adapter, mock_market_data, mock_registry,
    ):
        loop, pt = _build_loop(
            mock_adapter, mock_market_data, mock_registry,
            stale_days=2, stale_thr=-0.02,
        )
        # current_price 149 → −0.7% loss → ABOVE -2% threshold (less negative)
        _setup_held_position(mock_market_data, mock_adapter, current_price=149.0)
        _set_hold_secs(pt, 5 * 86400)  # held long
        loop._min_hold_secs = 0
        loop._stale_pnl_threshold = -0.05

        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_when_days_zero(
        self, mock_adapter, mock_market_data, mock_registry,
    ):
        loop, pt = _build_loop(
            mock_adapter, mock_market_data, mock_registry,
            stale_days=0, stale_thr=-0.02,
        )
        _setup_held_position(mock_market_data, mock_adapter, current_price=145.0)
        _set_hold_secs(pt, 100 * 86400)  # arbitrarily long
        loop._min_hold_secs = 0
        loop._stale_pnl_threshold = -0.05  # NOT reached either (-3.3% > -5%)

        await loop._evaluate_all()
        mock_adapter.create_sell_order.assert_not_called()
