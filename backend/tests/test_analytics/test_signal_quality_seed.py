"""P2: seed_tracker_from_db pulls SELLs from DB and populates tracker."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from analytics.signal_quality import SignalQualityTracker
from analytics.signal_quality_seed import seed_tracker_from_db


def _order(
    symbol="AAPL",
    side="SELL",
    status="filled",
    strategy_name="supertrend:profit_taking",
    buy_strategy="supertrend",
    pnl_pct=5.2,
    market="US",
):
    o = AsyncMock()
    o.symbol = symbol
    o.side = side
    o.status = status
    o.strategy_name = strategy_name
    o.buy_strategy = buy_strategy
    o.pnl_pct = pnl_pct
    o.market = market
    o.created_at = datetime(2026, 4, 1, 10, 0, 0)
    return o


def _factory(orders):
    """Build an async-context-manager session_factory whose repo returns `orders`."""

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    sess = _Session()

    async def fake_get_history(*, limit=5000, exclude_paper=True):
        return orders

    # Patch TradeRepository so its get_trade_history returns our list.
    import analytics.signal_quality_seed as mod

    class _Repo:
        def __init__(self, *_a, **_k):
            pass

        async def get_trade_history(self, *, limit=5000, exclude_paper=True):
            return orders

    mod.TradeRepository = _Repo  # type: ignore[assignment]

    def factory():
        return sess

    return factory


class TestSeedTrackerFromDb:
    @pytest.mark.asyncio
    async def test_seeds_filled_sell_with_buy_strategy(self):
        tracker = SignalQualityTracker()
        orders = [
            _order(buy_strategy="supertrend", strategy_name="supertrend:trailing_stop", pnl_pct=8.0),
            _order(buy_strategy="dual_momentum", strategy_name="dual_momentum", pnl_pct=-3.0),
        ]
        n = await seed_tracker_from_db(tracker, _factory(orders))
        assert n == 2
        assert tracker.get_metrics("supertrend").total_trades == 1
        assert tracker.get_metrics("dual_momentum").total_trades == 1

    @pytest.mark.asyncio
    async def test_strips_role_suffix_from_sell_strategy(self):
        """When buy_strategy is None, fall back to SELL strategy_name and strip ':role'."""
        tracker = SignalQualityTracker()
        o = _order(buy_strategy=None, strategy_name="supertrend:stop_loss")
        n = await seed_tracker_from_db(tracker, _factory([o]))
        assert n == 1
        # Bucketed under bare 'supertrend', not 'supertrend:stop_loss'
        assert tracker.get_metrics("supertrend").total_trades == 1

    @pytest.mark.asyncio
    async def test_skips_buy_orders(self):
        tracker = SignalQualityTracker()
        orders = [_order(side="BUY")]
        n = await seed_tracker_from_db(tracker, _factory(orders))
        assert n == 0

    @pytest.mark.asyncio
    async def test_skips_unfilled(self):
        tracker = SignalQualityTracker()
        orders = [_order(status="cancelled"), _order(status="error")]
        n = await seed_tracker_from_db(tracker, _factory(orders))
        assert n == 0

    @pytest.mark.asyncio
    async def test_skips_null_pnl(self):
        tracker = SignalQualityTracker()
        orders = [_order(pnl_pct=None)]
        n = await seed_tracker_from_db(tracker, _factory(orders))
        assert n == 0

    @pytest.mark.asyncio
    async def test_market_filter_us_only(self):
        tracker = SignalQualityTracker()
        orders = [
            _order(market="US", buy_strategy="supertrend", pnl_pct=5.0),
            _order(market="KR", buy_strategy="dual_momentum", pnl_pct=2.0),
        ]
        n = await seed_tracker_from_db(tracker, _factory(orders), market="US")
        assert n == 1
        assert tracker.get_metrics("supertrend").total_trades == 1
        assert tracker.get_metrics("dual_momentum").total_trades == 0

    @pytest.mark.asyncio
    async def test_pct_division_to_fraction(self):
        """DB pnl_pct=5.2 (percent) → tracker stores 0.052 (fraction)."""
        tracker = SignalQualityTracker()
        orders = [_order(buy_strategy="supertrend", pnl_pct=5.2)]
        await seed_tracker_from_db(tracker, _factory(orders))
        rec = tracker._trades["supertrend"][0]
        assert rec.return_pct == pytest.approx(0.052)
