"""Tests for extended hours backtest simulation."""

import pytest
import random
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd

from backtest.metrics import Trade, MetricsCalculator, BacktestMetrics
from backtest.full_pipeline import PipelineConfig, FullPipelineBacktest, _Position


class TestTradeSession:
    """Trade dataclass session field."""

    def test_trade_default_session_is_regular(self):
        t = Trade(symbol="AAPL", side="SELL", entry_date="2024-01-01", entry_price=100)
        assert t.session == "regular"

    def test_trade_extended_session(self):
        t = Trade(symbol="AAPL", side="SELL", entry_date="2024-01-01", entry_price=100, session="extended")
        assert t.session == "extended"


class TestExtendedHoursMetrics:
    """MetricsCalculator extended hours stats."""

    def test_metrics_with_extended_trades(self):
        trades = [
            Trade(symbol="AAPL", side="SELL", entry_date="2024-01-01",
                  entry_price=100, exit_price=110, pnl=100, pnl_pct=10,
                  session="regular"),
            Trade(symbol="MSFT", side="SELL", entry_date="2024-01-01",
                  entry_price=200, exit_price=210, pnl=50, pnl_pct=5,
                  session="extended"),
            Trade(symbol="GOOGL", side="SELL", entry_date="2024-01-01",
                  entry_price=150, exit_price=140, pnl=-100, pnl_pct=-6.7,
                  session="extended"),
        ]
        equity = pd.Series([100000, 100100, 100050, 100150],
                           index=pd.date_range("2024-01-01", periods=4))

        metrics = MetricsCalculator.calculate(equity, trades, 100000)

        assert metrics.extended_trades == 2
        assert metrics.extended_wins == 1
        assert metrics.extended_win_rate == 50.0
        assert metrics.extended_pnl == -50  # 50 + (-100)

    def test_metrics_no_extended_trades(self):
        trades = [
            Trade(symbol="AAPL", side="SELL", entry_date="2024-01-01",
                  entry_price=100, exit_price=110, pnl=100, pnl_pct=10),
        ]
        equity = pd.Series([100000, 100100],
                           index=pd.date_range("2024-01-01", periods=2))

        metrics = MetricsCalculator.calculate(equity, trades, 100000)

        assert metrics.extended_trades == 0
        assert metrics.extended_pnl == 0.0


class TestPipelineConfigExtendedHours:
    """PipelineConfig extended hours fields."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.extended_hours_enabled is False
        assert cfg.extended_hours_max_position_pct == 0.05
        assert cfg.extended_hours_slippage_multiplier == 2.0
        assert cfg.extended_hours_fill_probability == 0.90
        assert cfg.extended_hours_min_confidence == 0.55

    def test_custom_values(self):
        cfg = PipelineConfig(
            extended_hours_enabled=True,
            extended_hours_max_position_pct=0.05,
            extended_hours_fill_probability=0.80,
        )
        assert cfg.extended_hours_enabled is True
        assert cfg.extended_hours_max_position_pct == 0.05
        assert cfg.extended_hours_fill_probability == 0.80


class TestPositionSession:
    """_Position session field."""

    def test_default_session(self):
        pos = _Position(
            symbol="AAPL", quantity=10, avg_price=150,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=150, stop_loss_pct=0.08, take_profit_pct=0.20,
        )
        assert pos.session == "regular"

    def test_extended_session(self):
        pos = _Position(
            symbol="AAPL", quantity=10, avg_price=150,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=150, stop_loss_pct=0.08, take_profit_pct=0.20,
            session="extended",
        )
        assert pos.session == "extended"


class TestExtendedHoursSLCheck:
    """Test _check_extended_hours_sl (gap-down + gap-up defense)."""

    def _make_pipeline(self, **kwargs):
        cfg = PipelineConfig(
            extended_hours_enabled=True,
            slippage_pct=0.05,
            extended_hours_slippage_multiplier=3.0,
            **kwargs,
        )
        return FullPipelineBacktest(cfg)

    def _make_stock_data(self, opens, closes, highs=None, lows=None):
        """Create mock stock data with OHLCV."""
        n = len(opens)
        if highs is None:
            highs = [max(o, c) * 1.01 for o, c in zip(opens, closes)]
        if lows is None:
            lows = [min(o, c) * 0.99 for o, c in zip(opens, closes)]

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000000] * n,
        }, index=pd.date_range("2024-01-01", periods=n))

        data = MagicMock()
        data.df = df
        return data

    def test_gap_down_triggers_sl_at_open(self):
        """Stock gaps below SL at open → exit at open with 3x slippage."""
        bt = self._make_pipeline()
        bt._cash = 100000
        bt._day_count = 1

        # Position bought at $100, SL at 8% = $92
        bt._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=105, stop_loss_pct=0.08, take_profit_pct=0.20,
        )

        # Day 2: opens at $90 (gap down below SL of $92)
        stock_data = {
            "AAPL": self._make_stock_data(
                opens=[100, 90], closes=[100, 88],
            ),
        }

        bt._check_extended_hours_sl(stock_data, 1, "2024-01-02")

        # Position should be closed
        assert "AAPL" not in bt._positions
        assert len(bt._trades) == 1

        trade = bt._trades[0]
        assert trade.session == "extended"
        assert trade.pnl < 0  # Loss
        # Exit price should be open * (1 - 0.15%) = 90 * 0.9985
        expected_exec = 90 * (1 - 0.15 / 100)
        assert abs(trade.exit_price - expected_exec) < 0.01

    def test_no_trigger_when_open_above_sl(self):
        """Stock opens above SL → no extended hours exit."""
        bt = self._make_pipeline()
        bt._cash = 100000

        bt._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=105, stop_loss_pct=0.08, take_profit_pct=0.20,
        )

        # Opens at $95 (above SL of $92)
        stock_data = {
            "AAPL": self._make_stock_data(
                opens=[100, 95], closes=[100, 93],
            ),
        }

        bt._check_extended_hours_sl(stock_data, 1, "2024-01-02")

        # Position should still be held
        assert "AAPL" in bt._positions
        assert len(bt._trades) == 0

    def test_gap_up_triggers_tp_at_open(self):
        """Stock gaps above TP at open → exit at open (better than TP price)."""
        bt = self._make_pipeline()
        bt._cash = 100000
        bt._day_count = 1

        # Position bought at $100, TP at 20% = $120
        bt._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=115, stop_loss_pct=0.08, take_profit_pct=0.20,
        )

        # Day 2: opens at $125 (gap up above TP of $120)
        stock_data = {
            "AAPL": self._make_stock_data(
                opens=[100, 125], closes=[100, 128],
            ),
        }

        bt._check_extended_hours_sl(stock_data, 1, "2024-01-02")

        # Position should be closed at open (better than TP)
        assert "AAPL" not in bt._positions
        assert len(bt._trades) == 1

        trade = bt._trades[0]
        assert trade.session == "extended"
        assert trade.pnl > 0  # Profit
        # Exit at 125 * (1 - 0.15%) which is still > 120 (TP price)
        assert trade.exit_price > 120

    def test_skips_system_positions(self):
        """Cash parking / ETF positions not checked for extended SL."""
        bt = self._make_pipeline()
        bt._cash = 100000

        bt._positions["SPY"] = _Position(
            symbol="SPY", quantity=100, avg_price=400,
            entry_date="2024-01-01", strategy_name="cash_parking",
            highest_price=400, stop_loss_pct=9.99, take_profit_pct=9.99,
        )

        stock_data = {
            "SPY": self._make_stock_data(
                opens=[400, 300], closes=[400, 290],
            ),
        }

        bt._check_extended_hours_sl(stock_data, 1, "2024-01-02")

        # Should NOT be closed (system position)
        assert "SPY" in bt._positions


class TestExtendedHoursBuys:
    """Test _execute_extended_hours_buys."""

    def _make_pipeline(self, max_positions=10, **kwargs):
        cfg = PipelineConfig(
            extended_hours_enabled=True,
            initial_equity=100000,
            slippage_pct=0.05,
            extended_hours_max_position_pct=0.03,
            extended_hours_slippage_multiplier=3.0,
            extended_hours_fill_probability=1.0,  # Always fill for deterministic tests
            extended_hours_min_confidence=0.70,
            max_positions=max_positions,
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
            **kwargs,
        )
        bt = FullPipelineBacktest(cfg)
        bt._cash = 100000
        return bt

    def _make_stock_data(self, symbol, n=3, base_price=100, next_open_discount=0.02):
        """Create mock stock data where day 1 open has a gap down vs day 0 close."""
        closes = [base_price] * n
        # Day 1 opens lower than day 0 close (gap-down = discount)
        opens = [base_price] * n
        opens[1] = base_price * (1 - next_open_discount)  # 2% discount on day 1

        df = pd.DataFrame({
            "open": opens,
            "high": [base_price * 1.02] * n,
            "low": [base_price * 0.98] * n,
            "close": closes,
            "volume": [1000000] * n,
        }, index=pd.date_range("2024-01-01", periods=n))
        data = MagicMock()
        data.df = df
        return data

    def test_extended_buy_on_dip(self):
        """High-confidence signal + gap-down discount → extended hours buy."""
        bt = self._make_pipeline()

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80

        # Day 1 opens at 98 vs day 0 close 100 (2% discount > 0.5% threshold)
        stock_data = {"AAPL": self._make_stock_data("AAPL", next_open_discount=0.02)}
        candidates = [(0.80, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        assert "AAPL" in bt._positions
        pos = bt._positions["AAPL"]
        assert pos.session == "extended"
        # Position should be ~3% of equity ($3000)
        assert pos.quantity * pos.avg_price < 100000 * 0.04

    def test_extended_buy_on_spillover(self):
        """Max positions reached → extended buy for high-confidence signals."""
        bt = self._make_pipeline(max_positions=2)

        # Fill up regular positions
        for sym in ["MSFT", "GOOGL"]:
            bt._positions[sym] = _Position(
                symbol=sym, quantity=10, avg_price=200,
                entry_date="2024-01-01", strategy_name="momentum",
                highest_price=200, stop_loss_pct=0.08, take_profit_pct=0.20,
            )

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80

        # No discount needed for spillover — regular session was full
        stock_data = {"AAPL": self._make_stock_data("AAPL", next_open_discount=0.0)}
        candidates = [(0.80, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        assert "AAPL" in bt._positions
        assert bt._positions["AAPL"].session == "extended"

    def test_extended_buy_skips_low_confidence(self):
        """Low-confidence signal → no extended hours buy."""
        bt = self._make_pipeline()

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.50  # Below 0.70 threshold

        stock_data = {"AAPL": self._make_stock_data("AAPL")}
        candidates = [(0.50, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        assert "AAPL" not in bt._positions

    def test_extended_buy_skips_already_held(self):
        """Already holding symbol → no extended buy."""
        bt = self._make_pipeline()

        bt._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=10, avg_price=100,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=100, stop_loss_pct=0.08, take_profit_pct=0.20,
        )

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.90

        stock_data = {"AAPL": self._make_stock_data("AAPL")}
        candidates = [(0.90, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        # Position should still be original, not replaced
        assert bt._positions["AAPL"].quantity == 10

    def test_extended_buy_skips_no_advantage(self):
        """No discount and not at max positions → skip (no advantage)."""
        bt = self._make_pipeline(max_positions=20)  # Lots of capacity

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80

        # No discount (open == close)
        stock_data = {"AAPL": self._make_stock_data("AAPL", next_open_discount=0.0)}
        candidates = [(0.80, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        # Should NOT buy — no discount AND not at max positions
        assert "AAPL" not in bt._positions

    def test_fill_probability_miss(self):
        """Fill probability causes some misses."""
        cfg = PipelineConfig(
            extended_hours_enabled=True,
            initial_equity=100000,
            slippage_pct=0.05,
            extended_hours_max_position_pct=0.03,
            extended_hours_slippage_multiplier=3.0,
            extended_hours_fill_probability=0.0,  # 0% fill
            extended_hours_min_confidence=0.70,
            max_positions=2,
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
        bt = FullPipelineBacktest(cfg)
        bt._cash = 100000

        # Fill up positions to trigger spillover
        bt._positions["MSFT"] = _Position(
            symbol="MSFT", quantity=10, avg_price=200,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=200, stop_loss_pct=0.08, take_profit_pct=0.20,
        )
        bt._positions["GOOGL"] = _Position(
            symbol="GOOGL", quantity=10, avg_price=200,
            entry_date="2024-01-01", strategy_name="momentum",
            highest_price=200, stop_loss_pct=0.08, take_profit_pct=0.20,
        )

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.90

        n = 3
        base_price = 100
        df = pd.DataFrame({
            "open": [base_price] * n,
            "high": [base_price * 1.02] * n,
            "low": [base_price * 0.98] * n,
            "close": [base_price] * n,
            "volume": [1000000] * n,
        }, index=pd.date_range("2024-01-01", periods=n))
        data = MagicMock()
        data.df = df
        stock_data = {"AAPL": data}
        candidates = [(0.90, "AAPL", signal)]

        bt._execute_extended_hours_buys(
            candidates, stock_data, 1, "2024-01-02", "uptrend",
        )

        assert "AAPL" not in bt._positions  # fill probability = 0, always miss
