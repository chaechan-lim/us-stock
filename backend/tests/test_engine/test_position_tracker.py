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
    return RiskManager(
        RiskParams(
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
    )


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
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=135.0
            ),  # -10% < -8% SL
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell1",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=135.0,
        )
    )

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
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=365.0
            ),  # +21.7% > 20% TP
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell2",
            symbol="MSFT",
            side="SELL",
            order_type="market",
            quantity=5,
            status="filled",
            filled_price=365.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("MSFT", 300.0, 5)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "take_profit"
    assert triggered[0]["pnl"] == pytest.approx(325.0)


@pytest.mark.asyncio
async def test_trailing_stop_triggered(adapter, risk, order_mgr):
    """Price ran up then dropped -> trailing stop fires."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="GOOG", exchange="NASD", quantity=8, avg_price=100.0, current_price=108.0
            ),  # 8% above entry
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell3",
            symbol="GOOG",
            side="SELL",
            order_type="market",
            quantity=8,
            status="filled",
            filled_price=108.0,
        )
    )

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
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=148.0
            ),  # -1.3%, well within SL
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 0
    assert "AAPL" in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_position_gone_removes_tracker(adapter, risk, order_mgr):
    """If position no longer exists on exchange, tracker is removed after grace period."""
    adapter.fetch_positions = AsyncMock(return_value=[])  # no positions

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert "AAPL" in tracker.tracked_symbols

    # Within grace period (5 min) — should NOT be removed
    await tracker.check_all()
    assert "AAPL" in tracker.tracked_symbols

    # Simulate grace period elapsed
    tracker._tracked["AAPL"].tracked_at -= 400  # 400s ago (> 300s grace)
    await tracker.check_all()
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_highest_price_update(adapter, risk, order_mgr):
    """Highest price tracks the peak for trailing stop."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=160.0
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    tracker.track("AAPL", 150.0, 10)
    assert tracker._tracked["AAPL"].highest_price == 150.0

    await tracker.check_all()
    assert tracker._tracked["AAPL"].highest_price == 160.0


@pytest.mark.asyncio
async def test_notification_on_stop_loss(adapter, risk, order_mgr):
    """Notification is sent when stop-loss fires."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=135.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell4",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=135.0,
        )
    )

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
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=155.0
            ),
            Position(
                symbol="MSFT", exchange="NASD", quantity=5, avg_price=300.0, current_price=310.0
            ),
        ]
    )
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
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=0, avg_price=150.0, current_price=155.0
            ),
        ]
    )
    tracker = PositionTracker(adapter, risk, order_mgr)

    restored = await tracker.restore_from_exchange()
    assert restored == []
    assert len(tracker.tracked_symbols) == 0


@pytest.mark.asyncio
async def test_restore_uses_default_risk_params(adapter, risk, order_mgr):
    """Restored positions get default SL/TP from RiskManager."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=150.0, current_price=155.0
            ),
        ]
    )
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


# ── Paper/Live order separation (STOCK-6) ────────────────────────────


@pytest.mark.asyncio
async def test_restore_excludes_paper_orders(adapter, risk, order_mgr):
    """restore_from_exchange queries only non-paper orders for entry info."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from core.models import Base, Order

    # Set up in-memory DB with both paper and live orders
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Insert a paper BUY order (should be ignored)
    async with session_factory() as session:
        paper_order = Order(
            symbol="AAPL",
            exchange="NASD",
            side="BUY",
            order_type="market",
            quantity=38,
            price=145.0,
            filled_quantity=38,
            filled_price=145.0,
            status="filled",
            strategy_name="paper_strategy",
            is_paper=True,
            market="US",
        )
        session.add(paper_order)
        await session.commit()

    # Insert a live BUY order (should be used for entry info)
    async with session_factory() as session:
        live_order = Order(
            symbol="AAPL",
            exchange="NASD",
            side="BUY",
            order_type="limit",
            quantity=42,
            price=150.0,
            filled_quantity=42,
            filled_price=150.0,
            status="filled",
            strategy_name="trend_following",
            kis_order_id="KIS001",
            is_paper=False,
            market="US",
        )
        session.add(live_order)
        await session.commit()

    # Exchange shows 42 shares (live only)
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=42,
                avg_price=150.0,
                current_price=155.0,
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    restored = await tracker.restore_from_exchange(session_factory=session_factory)

    assert len(restored) == 1
    assert restored[0]["symbol"] == "AAPL"
    # Strategy should come from the live order, not the paper one
    assert restored[0]["strategy"] == "trend_following"

    await engine.dispose()


# ── Exchange field propagation (STOCK-5) ──────────────────────────────


@pytest.mark.asyncio
async def test_execute_sell_passes_exchange_kr(adapter, risk, order_mgr):
    """KR position tracker passes exchange='KRX' to place_sell on SL/TP."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="005930",
                exchange="KRX",
                quantity=10,
                avg_price=70000.0,
                current_price=63000.0,  # -10% < -8% SL
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_kr1",
            symbol="005930",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=63000.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr, market="KR")
    tracker.track("005930", 70000.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "stop_loss"

    # Verify place_sell was called with exchange="KRX"
    sell_call = adapter.create_sell_order.call_args
    assert sell_call is not None
    assert sell_call.kwargs.get("exchange") == "KRX" or (
        len(sell_call.args) > 4 and sell_call.args[4] == "KRX"
    )


@pytest.mark.asyncio
async def test_execute_sell_passes_exchange_us(adapter, risk, order_mgr):
    """US position tracker passes exchange='NASD' to place_sell on SL/TP."""
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL",
                exchange="NASD",
                quantity=10,
                avg_price=150.0,
                current_price=135.0,  # -10% < -8% SL
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_us1",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=135.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr, market="US")
    tracker.track("AAPL", 150.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "stop_loss"

    # Verify place_sell was called with exchange="NASD"
    sell_call = adapter.create_sell_order.call_args
    assert sell_call is not None
    assert sell_call.kwargs.get("exchange") == "NASD" or (
        len(sell_call.args) > 4 and sell_call.args[4] == "NASD"
    )


@pytest.mark.asyncio
async def test_resolve_exchange_kr(adapter, risk, order_mgr):
    """_resolve_exchange returns 'KRX' for KR market."""
    tracker = PositionTracker(adapter, risk, order_mgr, market="KR")
    assert tracker._resolve_exchange("005930") == "KRX"
    assert tracker._resolve_exchange("263750") == "KRX"


@pytest.mark.asyncio
async def test_resolve_exchange_us(adapter, risk, order_mgr):
    """_resolve_exchange returns 'NASD' for US market."""
    tracker = PositionTracker(adapter, risk, order_mgr, market="US")
    assert tracker._resolve_exchange("AAPL") == "NASD"
    assert tracker._resolve_exchange("NVDA") == "NASD"


# ── Paper/Live order separation (STOCK-6) ────────────────────────────


@pytest.mark.asyncio
async def test_restore_paper_order_only_uses_unknown(adapter, risk, order_mgr):
    """When only paper orders exist, strategy defaults to 'unknown'."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from core.models import Base, Order

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Only paper order exists
    async with session_factory() as session:
        paper_order = Order(
            symbol="MSFT",
            exchange="NASD",
            side="BUY",
            order_type="market",
            quantity=10,
            price=400.0,
            filled_quantity=10,
            filled_price=400.0,
            status="filled",
            strategy_name="paper_strat",
            is_paper=True,
            market="US",
        )
        session.add(paper_order)
        await session.commit()

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="MSFT",
                exchange="NASD",
                quantity=5,
                avg_price=400.0,
                current_price=410.0,
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr)
    restored = await tracker.restore_from_exchange(session_factory=session_factory)

    assert len(restored) == 1
    # No live order found → defaults to "unknown"
    assert restored[0]["strategy"] == "unknown"

    await engine.dispose()


# ── Profit-taking tests (STOCK-19) ──────────────────────────────────


@pytest.fixture
def risk_with_profit_taking():
    """RiskManager with profit-taking enabled."""
    return RiskManager(
        RiskParams(
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
            profit_taking_enabled=True,
            profit_taking_threshold_pct=0.10,
            profit_taking_sell_ratio=0.50,
            default_trailing_activation_pct=0.06,
            default_trailing_stop_pct=0.03,
        )
    )


@pytest.mark.asyncio
async def test_profit_taking_triggered(adapter, risk_with_profit_taking, order_mgr):
    """Position at +12% gain triggers partial profit-taking (50% sell)."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="NVDA", exchange="NASD", quantity=20, avg_price=100.0, current_price=112.0
            ),  # +12% gain (above 10% threshold, below 20% TP)
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_pt1",
            symbol="NVDA",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=112.0,
            filled_quantity=10,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("NVDA", 100.0, 20)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "profit_taking"
    assert triggered[0]["partial_qty"] == 10  # 50% of 20
    assert triggered[0]["gain_pct"] == pytest.approx(12.0)

    # Position should still be tracked with reduced quantity
    assert "NVDA" in tracker.tracked_symbols
    assert tracker._tracked["NVDA"].quantity == 10  # 20 - 10 sold
    assert tracker._tracked["NVDA"].partial_profit_taken is True
    # SL should be tightened to at most 3%
    assert tracker._tracked["NVDA"].stop_loss_pct <= 0.03


@pytest.mark.asyncio
async def test_profit_taking_not_triggered_below_threshold(adapter, risk_with_profit_taking):
    """Position at +8% gain should NOT trigger profit-taking (threshold is 10%)."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=100.0, current_price=108.0
            ),  # +8% gain, below 10% threshold
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("AAPL", 100.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 0
    assert tracker._tracked["AAPL"].partial_profit_taken is False


@pytest.mark.asyncio
async def test_profit_taking_skipped_when_above_tp(adapter, risk_with_profit_taking):
    """Position at +22% (above 20% TP) should do full TP, not partial profit-taking."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="MSFT", exchange="NASD", quantity=10, avg_price=100.0, current_price=122.0
            ),  # +22% gain, above 20% TP
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_tp2",
            symbol="MSFT",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=122.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("MSFT", 100.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "take_profit"  # full TP, not partial
    assert "MSFT" not in tracker.tracked_symbols  # fully untracked


@pytest.mark.asyncio
async def test_profit_taking_only_once(adapter, risk_with_profit_taking):
    """Profit-taking should only fire once per position (partial_profit_taken flag)."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="TSLA", exchange="NASD", quantity=20, avg_price=100.0, current_price=115.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_pt2",
            symbol="TSLA",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=115.0,
            filled_quantity=10,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("TSLA", 100.0, 20)

    # First check: profit-taking fires
    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "profit_taking"
    assert tracker._tracked["TSLA"].partial_profit_taken is True

    # Second check with same price: should NOT fire again
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="TSLA", exchange="NASD", quantity=10, avg_price=100.0, current_price=115.0
            ),
        ]
    )
    triggered2 = await tracker.check_all()
    assert len(triggered2) == 0


@pytest.mark.asyncio
async def test_profit_taking_disabled(adapter):
    """Profit-taking disabled in RiskParams → no partial sell."""
    risk_no_pt = RiskManager(
        RiskParams(
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
            profit_taking_enabled=False,
        )
    )
    order_mgr_no = OrderManager(adapter=adapter, risk_manager=risk_no_pt)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="GOOG", exchange="NASD", quantity=10, avg_price=100.0, current_price=115.0
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk_no_pt, order_mgr_no)
    tracker.track("GOOG", 100.0, 10)

    triggered = await tracker.check_all()
    assert len(triggered) == 0  # +15% but profit-taking disabled, below TP


@pytest.mark.asyncio
async def test_profit_taking_min_quantity(adapter, risk_with_profit_taking):
    """Position with only 1 share should not trigger profit-taking."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="BRK.A", exchange="NASD", quantity=1, avg_price=100.0, current_price=115.0
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("BRK.A", 100.0, 1)

    triggered = await tracker.check_all()
    assert len(triggered) == 0  # can't split 1 share


# ── Trailing stop propagation tests (STOCK-19) ──────────────────────


def test_track_with_trailing_stop_params(adapter, risk_with_profit_taking):
    """track() accepts and stores trailing stop parameters."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)
    tracker = PositionTracker(adapter, risk, order_mgr_pt)

    tracker.track(
        "AAPL", 150.0, 10,
        strategy="trend_following",
        stop_loss_pct=0.08,
        take_profit_pct=0.20,
        trailing_activation_pct=0.08,
        trailing_stop_pct=0.05,
    )

    tracked = tracker._tracked["AAPL"]
    assert tracked.trailing_activation_pct == 0.08
    assert tracked.trailing_stop_pct == 0.05


def test_track_trailing_defaults_from_risk_params(adapter, risk_with_profit_taking):
    """track() falls back to RiskParams defaults when trailing not specified."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)
    tracker = PositionTracker(adapter, risk, order_mgr_pt)

    tracker.track("AAPL", 150.0, 10, strategy="trend_following")

    tracked = tracker._tracked["AAPL"]
    assert tracked.trailing_activation_pct == 0.06  # from RiskParams default
    assert tracked.trailing_stop_pct == 0.03  # from RiskParams default


@pytest.mark.asyncio
async def test_trailing_stop_with_defaults(adapter, risk_with_profit_taking):
    """Trailing stop fires using default activation/trail from RiskParams."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=100.0, current_price=103.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_trail1",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=103.0,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("AAPL", 100.0, 10)  # defaults: activation=6%, trail=3%

    # Simulate price went to 110 (10% gain, above 6% activation)
    tracker._tracked["AAPL"].highest_price = 110.0
    # Now at 103 → drop from peak = (110-103)/110 = 6.4% > 3% trail

    triggered = await tracker.check_all()
    assert len(triggered) == 1
    assert triggered[0]["reason"] == "trailing_stop"
    assert "AAPL" not in tracker.tracked_symbols


@pytest.mark.asyncio
async def test_trailing_stop_not_activated_yet(adapter, risk_with_profit_taking):
    """Trailing stop should not fire if price hasn't risen enough to activate."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=10, avg_price=100.0, current_price=104.0
            ),
        ]
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("AAPL", 100.0, 10)  # defaults: activation=6%, trail=3%
    # highest_price = 104 (4% gain, below 6% activation threshold)

    triggered = await tracker.check_all()
    assert len(triggered) == 0


# ── TP cap tests (STOCK-19) ─────────────────────────────────────────


def test_tp_cap_us_lowered():
    """US TP cap should be 20% (was 30%)."""
    risk = RiskManager()
    # High volatility stock: ATR = 10% of price → TP would be 35% uncapped
    sl, tp = risk.calculate_dynamic_sl_tp(100.0, 10.0, market="US")
    assert tp <= 0.20  # capped at 20%


def test_tp_cap_kr_lowered():
    """KR TP cap should be 25% (was 30%)."""
    risk = RiskManager()
    # High volatility stock: ATR = 10% of price → TP would be 50% uncapped
    sl, tp = risk.calculate_dynamic_sl_tp(100.0, 10.0, market="KR")
    assert tp <= 0.25  # capped at 25%


def test_tp_cap_low_volatility_us():
    """Low-vol US stock gets reasonable TP, not capped."""
    risk = RiskManager()
    # Low volatility: ATR = 2% of price → TP = 2% * 3.5 = 7%
    sl, tp = risk.calculate_dynamic_sl_tp(100.0, 2.0, market="US")
    assert tp == pytest.approx(0.07)
    assert sl == pytest.approx(0.04)


# ── Notification tests for profit-taking (STOCK-19) ─────────────────


@pytest.mark.asyncio
async def test_profit_taking_notification(adapter, risk_with_profit_taking):
    """Notification sent when profit-taking fires."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="AAPL", exchange="NASD", quantity=20, avg_price=100.0, current_price=112.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="sell_pt_notif",
            symbol="AAPL",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=112.0,
            filled_quantity=10,
        )
    )

    notif = AsyncMock()
    # Simulate notify_profit_taking not available (fallback to notify_take_profit)
    notif.notify_profit_taking = AsyncMock(side_effect=AttributeError)
    tracker = PositionTracker(adapter, risk, order_mgr_pt, notification=notif)
    tracker.track("AAPL", 100.0, 20)

    await tracker.check_all()
    # Should try notify_profit_taking first, then fallback
    notif.notify_take_profit.assert_called_once()


# ── Integration: profit-taking then trailing stop (STOCK-19) ────────


@pytest.mark.asyncio
async def test_profit_taking_then_trailing_stop(adapter, risk_with_profit_taking):
    """After profit-taking at 12%, remaining position hits trailing stop."""
    risk = risk_with_profit_taking
    order_mgr_pt = OrderManager(adapter=adapter, risk_manager=risk)

    # Phase 1: Price at +12% → profit-taking
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="META", exchange="NASD", quantity=20, avg_price=100.0, current_price=112.0
            ),
        ]
    )
    from exchange.base import OrderResult

    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="pt1",
            symbol="META",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=112.0,
            filled_quantity=10,
        )
    )

    tracker = PositionTracker(adapter, risk, order_mgr_pt)
    tracker.track("META", 100.0, 20)

    triggered1 = await tracker.check_all()
    assert len(triggered1) == 1
    assert triggered1[0]["reason"] == "profit_taking"
    assert tracker._tracked["META"].quantity == 10

    # Phase 2: Price peaks at 115, then drops to 109
    # Trail from peak: (115-109)/115 = 5.2% > 3% trail
    adapter.fetch_positions = AsyncMock(
        return_value=[
            Position(
                symbol="META", exchange="NASD", quantity=10, avg_price=100.0, current_price=109.0
            ),
        ]
    )
    adapter.create_sell_order = AsyncMock(
        return_value=OrderResult(
            order_id="trail1",
            symbol="META",
            side="SELL",
            order_type="market",
            quantity=10,
            status="filled",
            filled_price=109.0,
        )
    )
    tracker._tracked["META"].highest_price = 115.0

    triggered2 = await tracker.check_all()
    assert len(triggered2) == 1
    assert triggered2[0]["reason"] == "trailing_stop"
    assert "META" not in tracker.tracked_symbols


# ── Profit-taking sell quantity calculation (STOCK-19) ───────────────


def test_calculate_profit_take_qty():
    """Verify profit-take quantity calculation."""
    risk = RiskManager(
        RiskParams(profit_taking_sell_ratio=0.50)
    )
    order_mgr_mock = MagicMock()
    adapter_mock = MagicMock()
    tracker = PositionTracker(adapter_mock, risk, order_mgr_mock)

    # 20 shares → sell 10
    tracker.track("A", 100.0, 20)
    qty = tracker._calculate_profit_take_qty(tracker._tracked["A"])
    assert qty == 10

    # 3 shares → sell 1 (min 1, but can't sell all)
    tracker.track("B", 100.0, 3)
    qty = tracker._calculate_profit_take_qty(tracker._tracked["B"])
    assert qty == 1

    # 2 shares → sell 1 (max = quantity - 1 = 1)
    tracker.track("C", 100.0, 2)
    qty = tracker._calculate_profit_take_qty(tracker._tracked["C"])
    assert qty == 1
