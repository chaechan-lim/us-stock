"""Tests for PositionTracker."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from exchange.base import Position
from engine.risk_manager import RiskManager, RiskParams
from engine.order_manager import OrderManager
from engine.position_tracker import PositionTracker, TrackedPosition


@pytest.fixture
def adapter():
    a = AsyncMock()
    a.fetch_positions = AsyncMock(return_value=[])
    return a


@pytest.fixture
def risk():
    return RiskManager(RiskParams(
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.20,
    ))


@pytest.fixture
def order_mgr(adapter, risk):
    return OrderManager(adapter=adapter, risk_manager=risk)


@pytest.fixture
def tracker(adapter, risk, order_mgr):
    return PositionTracker(adapter, risk, order_mgr)


def test_track_and_untrack(tracker):
    tracker.track("AAPL", 150.0, 10, strategy="trend_following")
    assert "AAPL" in tracker.tracked_symbols
    assert len(tracker.tracked_symbols) == 1

    tracker.untrack("AAPL")
    assert "AAPL" not in tracker.tracked_symbols


def test_get_status(tracker):
    tracker.track("AAPL", 150.0, 10, strategy="trend_following")
    tracker.track("TSLA", 200.0, 5, strategy="macd_histogram")

    status = tracker.get_status()
    assert len(status) == 2
    symbols = {s["symbol"] for s in status}
    assert symbols == {"AAPL", "TSLA"}


@pytest.mark.asyncio
async def test_check_all_empty(tracker):
    result = await tracker.check_all()
    assert result == []


@pytest.mark.asyncio
async def test_stop_loss_triggered(adapter, risk, order_mgr):
    """Position drops below SL threshold -> sell triggered."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=135.0),  # -10% < -8% SL
    ])
    from exchange.base import OrderResult
    adapter.create_sell_order = AsyncMock(return_value=OrderResult(
        order_id="sell1", symbol="AAPL", side="SELL",
        order_type="market", quantity=10, status="filled", filled_price=135.0,
    ))

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "stop_loss"
    assert triggered[0]["symbol"] == "AAPL"
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_take_profit_triggered(adapter, risk, order_mgr):
    """Position rises above TP threshold -> sell triggered."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="MSFT", exchange="NASD", quantity=5,
                 avg_price=300.0, current_price=365.0),  # +21.7% > 20% TP
    ])
    from exchange.base import OrderResult
    adapter.create_sell_order = AsyncMock(return_value=OrderResult(
        order_id="sell2", symbol="MSFT", side="SELL",
        order_type="market", quantity=5, status="filled", filled_price=365.0,
    ))

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("MSFT", 300.0, 5)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "take_profit"
    assert triggered[0]["pnl"] == pytest.approx(325.0)


@pytest.mark.asyncio
async def test_trailing_stop_triggered(adapter, risk, order_mgr):
    """Price ran up then dropped -> trailing stop fires."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="GOOG", exchange="NASD", quantity=8,
                 avg_price=100.0, current_price=108.0),  # 8% above entry
    ])
    from exchange.base import OrderResult
    adapter.create_sell_order = AsyncMock(return_value=OrderResult(
        order_id="sell3", symbol="GOOG", side="SELL",
        order_type="market", quantity=8, status="filled", filled_price=108.0,
    ))

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("GOOG", 100.0, 8)
    # Enable trailing stop for this test
    tracker._tracked["GOOG"].trailing_activation_pct = 0.05
    tracker._tracked["GOOG"].trailing_stop_pct = 0.03
    # Simulate that price had gone to 115 before
    tracker._tracked["GOOG"].highest_price = 115.0
    # Drop from 115 to 108 = ~6.1% > 3% trail

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "trailing_stop"
    assert triggered[0]["highest"] == 115.0


@pytest.mark.asyncio
async def test_no_trigger_when_within_range(adapter, risk, order_mgr):
    """Price is within normal range -> no trigger."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=148.0),  # -1.3%, well within SL
    ])

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 0
    assert "AAPL" in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_position_gone_removes_tracker(adapter, risk, order_mgr):
    """If position no longer exists on exchange, tracker is removed."""
    adapter.fetch_positions = AsyncMock(return_value=[])  # no positions

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert "AAPL" in tracker.tracked_symbols

    await tracker.check_all()
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_highest_price_update(adapter, risk, order_mgr):
    """Highest price tracks the peak for trailing stop."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=160.0),
    ])

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert tracker._tracked["AAPL"].highest_price == 150.0

    await tracker.check_all()
    assert tracker._tracked["AAPL"].highest_price == 160.0


@pytest.mark.asyncio
async def test_notification_on_stop_loss(adapter, risk, order_mgr):
    """Notification is sent when stop-loss fires."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=135.0),
    ])
    from exchange.base import OrderResult
    adapter.create_sell_order = AsyncMock(return_value=OrderResult(
        order_id="sell4", symbol="AAPL", side="SELL",
        order_type="market", quantity=10, status="filled", filled_price=135.0,
    ))

    notif = AsyncMock()
    tracker = PositionTracker(adapter, risk, order_mgr, notification=notif)
    tracker.track("AAPL", 150.0, 10)

    await tracker.check_all()
    notif.notify_stop_loss.assert_called_once()
    args = notif.notify_stop_loss.call_args
    assert args[0][0] == "AAPL"


@pytest.mark.asyncio
async def test_fetch_error_graceful(adapter, risk, order_mgr):
    """Adapter error doesn't crash the tracker."""
    adapter.fetch_positions = AsyncMock(side_effect=Exception("network"))

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    result = await tracker.check_all()
    assert result == []
    assert "AAPL" in tracker.tracked_symbols  # still tracked


# ── Startup Restoration Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_from_exchange_no_positions(adapter, risk, order_mgr):
    """Empty exchange -> nothing restored."""
    adapter.fetch_positions = AsyncMock(return_value=[])
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
    assert len(tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_from_exchange_with_positions(adapter, risk, order_mgr):
    """Exchange positions are restored into tracker."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=155.0),
        Position(symbol="MSFT", exchange="NASD", quantity=5,
                 avg_price=300.0, current_price=310.0),
    ])
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert len(restored) == 2
    assert set(tracker.tracked_symbols) == {"AAPL", "MSFT"}

    # Check correct entry prices from exchange avg_price
    assert tracker._tracked["AAPL"].entry_price == 150.0
    assert tracker._tracked["MSFT"].entry_price == 300.0

    # Check PnL calculation
    aapl = next(r for r in restored if r["symbol"] == "AAPL")
    assert aapl["pnl_pct"] == pytest.approx(3.33, abs=0.1)


@pytest.mark.asyncio
async def test_restore_skips_zero_quantity(adapter, risk, order_mgr):
    """Positions with 0 quantity are not restored."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=0,
                 avg_price=150.0, current_price=155.0),
    ])
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
    assert len(tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_uses_default_risk_params(adapter, risk, order_mgr):
    """Restored positions get default SL/TP from RiskManager."""
    adapter.fetch_positions = AsyncMock(return_value=[
        Position(symbol="AAPL", exchange="NASD", quantity=10,
                 avg_price=150.0, current_price=155.0),
    ])
    tracker = PositionTracker(adapter, risk, order_mgr)

    await tracker.restore_from_exchange()
    tracked = tracker._tracked["AAPL"]
    assert tracked.stop_loss_pct == 0.08
    assert tracked.take_profit_pct == 0.20


@pytest.mark.asyncio
async def test_restore_fetch_error_graceful(adapter, risk, order_mgr):
    """Adapter error during restore returns empty list."""
    adapter.fetch_positions = AsyncMock(side_effect=Exception("timeout"))
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
