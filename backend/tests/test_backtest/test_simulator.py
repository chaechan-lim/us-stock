"""Tests for backtest simulator."""

import numpy as np
import pandas as pd
import pytest

from backtest.simulator import BacktestSimulator, SimConfig, SimPosition
from strategies.base import Signal
from core.enums import SignalType


def _make_df(n: int = 50, start_price: float = 100.0) -> pd.DataFrame:
    """Create a simple uptrending DataFrame."""
    np.random.seed(42)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.01)))

    close = np.array(prices)
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    }, index=dates)


def _buy_signal(name: str = "test") -> Signal:
    return Signal(
        signal_type=SignalType.BUY,
        confidence=0.8,
        strategy_name=name,
        reason="test buy",
    )


def _sell_signal(name: str = "test") -> Signal:
    return Signal(
        signal_type=SignalType.SELL,
        confidence=0.8,
        strategy_name=name,
        reason="test sell",
    )


class TestSimulator:
    def test_no_signals_equity_unchanged(self):
        sim = BacktestSimulator(SimConfig(initial_equity=100_000))
        df = _make_df()
        sim.run(df, {}, "AAPL")
        curve = sim.equity_curve
        assert len(curve) == len(df)
        # No trades, equity should stay at initial
        assert curve.iloc[0] == pytest.approx(100_000, rel=0.01)
        assert len(sim.trades) == 0

    def test_buy_and_sell(self):
        config = SimConfig(initial_equity=100_000, slippage_pct=0.0)
        sim = BacktestSimulator(config)
        df = _make_df(50, start_price=100.0)

        signals = {
            5: _buy_signal(),
            25: _sell_signal(),
        }
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        trade = sim.trades[0]
        assert trade.symbol == "AAPL"
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.quantity > 0

    def test_slippage_applied(self):
        config = SimConfig(initial_equity=100_000, slippage_pct=1.0)
        sim = BacktestSimulator(config)
        df = _make_df(30)
        price_at_5 = float(df.iloc[5]["close"])

        signals = {5: _buy_signal()}
        sim.run(df, signals, "AAPL")

        pos = sim.positions.get("AAPL")
        assert pos is not None
        # Buy price should be higher than close (slippage)
        assert pos.avg_price > price_at_5

    def test_position_sizing_respects_max(self):
        config = SimConfig(initial_equity=100_000, max_position_pct=0.05)
        sim = BacktestSimulator(config)
        df = _make_df(20, start_price=100.0)

        signals = {5: _buy_signal()}
        sim.run(df, signals, "AAPL")

        pos = sim.positions.get("AAPL")
        assert pos is not None
        position_value = pos.quantity * pos.avg_price
        assert position_value <= 100_000 * 0.05 + 200  # allow some tolerance

    def test_max_positions_limit(self):
        config = SimConfig(initial_equity=1_000_000, max_total_positions=1)
        sim = BacktestSimulator(config)
        df = _make_df(20)

        # Buy at bar 5 for AAPL
        signals = {5: _buy_signal()}
        sim.run(df, signals, "AAPL")

        # Try to buy TSLA — should be blocked
        sim.run(df, {5: _buy_signal()}, "TSLA")
        assert "TSLA" not in sim.positions

    def test_no_duplicate_position(self):
        config = SimConfig(initial_equity=100_000, slippage_pct=0.0)
        sim = BacktestSimulator(config)
        df = _make_df(20)

        signals = {3: _buy_signal(), 5: _buy_signal()}
        sim.run(df, signals, "AAPL")

        # Should only have one position
        assert len(sim.positions) == 1

    def test_sell_without_position_ignored(self):
        sim = BacktestSimulator()
        df = _make_df(10)
        sim.run(df, {5: _sell_signal()}, "AAPL")
        assert len(sim.trades) == 0

    def test_pnl_calculation(self):
        config = SimConfig(initial_equity=100_000, slippage_pct=0.0)
        sim = BacktestSimulator(config)

        # Create a dataframe where price goes up
        prices = [100.0] * 5 + [110.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal(), 7: _sell_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        assert sim.trades[0].pnl > 0
        assert sim.trades[0].pnl_pct == pytest.approx(10.0)

    def test_equity_curve_length(self):
        sim = BacktestSimulator()
        df = _make_df(30)
        sim.run(df, {}, "AAPL")
        assert len(sim.equity_curve) == 30

    def test_reset(self):
        sim = BacktestSimulator(SimConfig(initial_equity=50_000))
        df = _make_df(10)
        sim.run(df, {2: _buy_signal()}, "AAPL")
        assert len(sim.positions) > 0

        sim.reset()
        assert len(sim.positions) == 0
        assert len(sim.trades) == 0
        assert len(sim.equity_curve) == 0

    def test_hold_signal_ignored(self):
        sim = BacktestSimulator()
        df = _make_df(10)
        hold = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strategy_name="test",
            reason="hold",
        )
        sim.run(df, {5: hold}, "AAPL")
        assert len(sim.trades) == 0
        assert len(sim.positions) == 0


class TestStopLossTakeProfit:
    def test_stop_loss_triggers(self):
        """Position should be closed when price drops below SL."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.05, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Price drops 10%: 100 -> 90
        prices = [100.0] * 5 + [90.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        # Position should have been stopped out
        assert len(sim.trades) == 1
        assert sim.trades[0].pnl < 0
        assert "AAPL" not in sim.positions

    def test_stop_loss_exact_level(self):
        """SL triggers when low touches SL price."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.10, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Entry at 100, SL at 90. Low goes to 89 on bar 5
        prices = [100.0] * 5 + [95.0] * 5
        lows = [99.0] * 5 + [89.0] + [94.0] * 4
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices, "high": [p + 1 for p in prices],
            "low": lows, "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")
        assert len(sim.trades) == 1  # SL triggered

    def test_take_profit_triggers(self):
        """Position closed when price hits TP level."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            take_profit_pct=0.10, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Price goes up 15%: 100 -> 115
        prices = [100.0] * 5 + [115.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        assert sim.trades[0].pnl > 0

    def test_trailing_stop_triggers(self):
        """Trailing stop activates after gain, then triggers on pullback."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            trailing_stop_activation_pct=0.05,
            trailing_stop_trail_pct=0.03,
            max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Price: entry 100, rises to 110 (+10%), then drops to 105 (~4.5% from peak)
        prices = [100.0, 100.0, 105.0, 108.0, 110.0, 108.0, 105.0, 103.0, 100.0, 98.0]
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        # Trailing stop should trigger when price drops 3% from 111 peak
        assert len(sim.trades) == 1
        assert sim.trades[0].pnl > 0  # Still profitable (locked in gains)

    def test_no_sl_tp_when_disabled(self):
        """With SL/TP at 0, positions are not auto-closed."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.0, take_profit_pct=0.0,
            max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Big drop but no SL
        prices = [100.0] * 5 + [50.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 0  # No auto-close, still holding
        assert "AAPL" in sim.positions

    def test_sl_tp_combined(self):
        """SL should fire before TP when price drops."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.05, take_profit_pct=0.20,
            max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Price drops 8%
        prices = [100.0, 100.0, 100.0, 92.0, 90.0, 88.0, 85.0, 83.0, 80.0, 78.0]
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        assert sim.trades[0].pnl < 0  # Stopped out at a loss


class TestGapThrough:
    """Gap-through fill modeling: when bar opens beyond SL/TP, fill at open."""

    def test_sl_gap_through_fills_at_open(self):
        """Bar opens below SL → fill at open price (worse than SL)."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.08, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Entry at 100, SL at 92. Bar 5 gaps down: open=85 (below SL)
        prices = [100.0] * 5 + [85.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        trade = sim.trades[0]
        # Should fill at open (85), not at SL price (92)
        assert trade.exit_price == 85.0

    def test_sl_intraday_fills_at_sl_price(self):
        """Bar opens above SL but low breaches it → fill at SL price."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            stop_loss_pct=0.10, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Entry at 100, SL at 90. Bar 5: open=95, low=88 (breaches SL intraday)
        opens = [100.0] * 5 + [95.0] * 5
        closes = [100.0] * 5 + [93.0] * 5
        lows = [99.0] * 5 + [88.0] + [93.0] * 4
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": opens, "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "low": lows, "close": closes,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        trade = sim.trades[0]
        # Open=95 > SL=90, so fill at SL price (90)
        assert trade.exit_price == 90.0

    def test_tp_gap_through_fills_at_open(self):
        """Bar opens above TP → fill at open price (bonus)."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.0,
            take_profit_pct=0.10, max_position_pct=0.95,
        )
        sim = BacktestSimulator(config)

        # Entry at 100, TP at 110. Bar 5 gaps up: open=115
        opens = [100.0] * 5 + [115.0] * 5
        closes = [100.0] * 5 + [117.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=10)
        df = pd.DataFrame({
            "open": opens,
            "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1_000_000.0] * 10,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert len(sim.trades) == 1
        trade = sim.trades[0]
        # Should fill at open (115), not at TP price (110) — bonus
        assert trade.exit_price == 115.0


class TestVolumeAdjustedSlippage:
    """Volume-adjusted slippage scales by participation rate."""

    def test_high_volume_base_slippage(self):
        """High volume → base slippage (participation < 1%)."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.05,
            volume_adjusted_slippage=True, max_position_pct=0.10,
        )
        sim = BacktestSimulator(config)

        # Volume = 10M, buying ~$10K = ~100 shares = 0.001% participation
        prices = [100.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=5)
        df = pd.DataFrame({
            "open": prices, "high": [101] * 5, "low": [99] * 5,
            "close": prices, "volume": [10_000_000] * 5,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert "AAPL" in sim.positions
        pos = sim.positions["AAPL"]
        # Should be base slippage: 100 * 1.0005 = 100.05
        assert abs(pos.avg_price - 100.05) < 0.01

    def test_low_volume_higher_slippage(self):
        """Low volume → 3x slippage (participation > 10%)."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.05,
            volume_adjusted_slippage=True, max_position_pct=0.10,
        )
        sim = BacktestSimulator(config)

        # Volume = 50, buying ~$10K = ~100 shares = 200% participation → 3x
        prices = [100.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=5)
        df = pd.DataFrame({
            "open": prices, "high": [101] * 5, "low": [99] * 5,
            "close": prices, "volume": [50] * 5,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert "AAPL" in sim.positions
        pos = sim.positions["AAPL"]
        # Should be 3x slippage: 100 * 1.0015 = 100.15
        assert abs(pos.avg_price - 100.15) < 0.01

    def test_disabled_uses_base_slippage(self):
        """volume_adjusted_slippage=False → always base slippage."""
        config = SimConfig(
            initial_equity=100_000, slippage_pct=0.05,
            volume_adjusted_slippage=False, max_position_pct=0.10,
        )
        sim = BacktestSimulator(config)

        # Very low volume but feature disabled
        prices = [100.0] * 5
        dates = pd.bdate_range("2021-01-01", periods=5)
        df = pd.DataFrame({
            "open": prices, "high": [101] * 5, "low": [99] * 5,
            "close": prices, "volume": [10] * 5,
        }, index=dates)

        signals = {1: _buy_signal()}
        sim.run(df, signals, "AAPL")

        assert "AAPL" in sim.positions
        pos = sim.positions["AAPL"]
        # Base slippage only
        assert abs(pos.avg_price - 100.05) < 0.01


class TestSimConfig:
    def test_defaults(self):
        c = SimConfig()
        assert c.initial_equity == 100_000.0
        assert c.slippage_pct == 0.05
        assert c.commission_per_order == 0.0
        assert c.max_position_pct == 0.10
        assert c.max_total_positions == 20
        assert c.stop_loss_pct == 0.0
        assert c.take_profit_pct == 0.0
        assert c.trailing_stop_activation_pct == 0.0

    def test_custom_values(self):
        c = SimConfig(initial_equity=50_000, slippage_pct=0.1)
        assert c.initial_equity == 50_000
        assert c.slippage_pct == 0.1
