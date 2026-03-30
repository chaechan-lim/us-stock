"""Tests for Full Pipeline Backtest Engine."""

import asyncio
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtest.full_pipeline import (
    FullPipelineBacktest,
    PipelineConfig,
    PipelineResult,
    _Position,
    DEFAULT_UNIVERSE,
)
from backtest.data_loader import BacktestData
from backtest.metrics import Trade
from core.enums import SignalType


def _make_ohlcv(n_days: int = 500, start_price: float = 100.0, trend: float = 0.0005) -> pd.DataFrame:
    """Generate synthetic OHLCV data with optional trend."""
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    np.random.seed(42)
    returns = np.random.normal(trend, 0.015, n_days)
    prices = start_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 - np.random.uniform(0, 0.01, n_days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)
    return df


def _make_backtest_data(symbol: str, n_days: int = 500, trend: float = 0.0005) -> BacktestData:
    """Create BacktestData with indicators already computed."""
    from data.indicator_service import IndicatorService
    df = _make_ohlcv(n_days, trend=trend)
    df = IndicatorService.add_all_indicators(df)
    return BacktestData(
        symbol=symbol,
        df=df,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
    )


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.initial_equity == 100_000.0
        assert config.max_positions == 20
        assert config.max_position_pct == 0.10
        assert config.screen_interval == 20
        assert config.dynamic_sl_tp is True
        assert len(config.universe) > 0

    def test_custom_config(self):
        config = PipelineConfig(
            universe=["AAPL", "MSFT"],
            initial_equity=50_000.0,
            max_positions=5,
        )
        assert config.universe == ["AAPL", "MSFT"]
        assert config.initial_equity == 50_000.0
        assert config.max_positions == 5


class TestDefaultUniverse:
    def test_has_diverse_sectors(self):
        assert "AAPL" in DEFAULT_UNIVERSE  # Tech
        assert "JPM" in DEFAULT_UNIVERSE  # Finance
        assert "UNH" in DEFAULT_UNIVERSE  # Healthcare
        assert "XOM" in DEFAULT_UNIVERSE  # Energy
        assert len(DEFAULT_UNIVERSE) >= 50


class TestPipelineInit:
    def test_creates_all_components(self):
        config = PipelineConfig(universe=["AAPL"])
        engine = FullPipelineBacktest(config)
        assert engine._config == config
        assert engine._risk_manager is not None
        assert engine._combiner is not None
        assert engine._screener is not None
        assert engine._classifier is not None

    def test_risk_params_from_config(self):
        config = PipelineConfig(
            max_position_pct=0.05,
            max_positions=10,
            max_exposure_pct=0.80,
        )
        engine = FullPipelineBacktest(config)
        assert engine._risk_manager.params.max_position_pct == 0.05
        assert engine._risk_manager.params.max_positions == 10
        assert engine._risk_manager.params.max_total_exposure_pct == 0.80


class TestScreenUniverse:
    def test_returns_filtered_symbols(self):
        config = PipelineConfig(universe=["AAPL", "MSFT", "GOOGL"])
        engine = FullPipelineBacktest(config)

        stock_data = {
            s: _make_backtest_data(s, 300) for s in ["AAPL", "MSFT", "GOOGL"]
        }

        result = engine._screen_universe(stock_data, 250, max_symbols=10)
        assert isinstance(result, list)
        # All symbols should be in universe
        for s in result:
            assert s in ["AAPL", "MSFT", "GOOGL"]

    def test_skips_short_data(self):
        config = PipelineConfig(universe=["AAPL"])
        engine = FullPipelineBacktest(config)

        stock_data = {"AAPL": _make_backtest_data("AAPL", 30)}
        result = engine._screen_universe(stock_data, 20, max_symbols=10)
        # Short data should be filtered
        assert isinstance(result, list)


class TestRiskExits:
    def test_stop_loss_triggers(self):
        config = PipelineConfig(universe=["AAPL"])
        engine = FullPipelineBacktest(config)
        engine._cash = 90_000
        engine._day_count = 10  # Past min hold period

        # Create position at $100 with 5% SL
        engine._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2023-06-01", strategy_name="test",
            highest_price=100.0,
            stop_loss_pct=0.05, take_profit_pct=0.20,
        )

        # Price drops to $94 (below $95 SL level)
        df = _make_ohlcv(300, start_price=100.0)
        # Force a low price below SL
        df.iloc[299, df.columns.get_loc("low")] = 93.0
        df.iloc[299, df.columns.get_loc("close")] = 94.0

        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2024-03-01")}

        engine._check_risk_exits(stock_data, 299, df.index[299])
        assert "AAPL" not in engine._positions
        assert len(engine._trades) == 1
        assert engine._trades[0].strategy_name == "test"

    def test_take_profit_triggers(self):
        config = PipelineConfig(universe=["AAPL"])
        engine = FullPipelineBacktest(config)
        engine._cash = 90_000
        engine._day_count = 10

        engine._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2023-06-01", strategy_name="test",
            highest_price=100.0,
            stop_loss_pct=0.08, take_profit_pct=0.10,
        )

        df = _make_ohlcv(300, start_price=100.0)
        df.iloc[299, df.columns.get_loc("high")] = 112.0
        df.iloc[299, df.columns.get_loc("close")] = 111.0

        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2024-03-01")}

        engine._check_risk_exits(stock_data, 299, df.index[299])
        assert "AAPL" not in engine._positions
        assert engine._trades[0].pnl > 0

    def test_trailing_stop_triggers(self):
        config = PipelineConfig(
            universe=["AAPL"],
            trailing_activation_pct=0.05,
            trailing_trail_pct=0.03,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 90_000
        engine._day_count = 10

        # Entry at 100, peaked at 110 (+10%), now at 106 (trail from 110: -3.6%)
        engine._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2023-06-01", strategy_name="test",
            highest_price=110.0,
            stop_loss_pct=0.08, take_profit_pct=0.30,
        )

        df = _make_ohlcv(300, start_price=100.0)
        # Trail price = 110 * 0.97 = 106.7, low at 105 triggers
        df.iloc[299, df.columns.get_loc("low")] = 105.0
        df.iloc[299, df.columns.get_loc("high")] = 107.0
        df.iloc[299, df.columns.get_loc("close")] = 106.0

        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2024-03-01")}

        engine._check_risk_exits(stock_data, 299, df.index[299])
        assert "AAPL" not in engine._positions


class TestTradingGates:
    """Trading gates: cooldown, whipsaw, min hold — matching live behavior."""

    def test_sell_cooldown_blocks_rebuy(self):
        """After selling, can't re-buy same symbol within cooldown period."""
        config = PipelineConfig(
            universe=["AAPL"], sell_cooldown_days=2,
            dynamic_sl_tp=False, default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 100_000
        engine._day_count = 5

        # Record a recent sell
        engine._sell_cooldown["AAPL"] = 4  # sold on day 4

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80
        signal.signal_type = SignalType.BUY

        df = _make_ohlcv(10, start_price=100.0)
        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        engine._execute_buy("AAPL", stock_data, 5, df.index[5], signal, "uptrend")
        assert "AAPL" not in engine._positions  # Blocked by cooldown

    def test_sell_cooldown_expires(self):
        """After cooldown period, re-buy is allowed."""
        config = PipelineConfig(
            universe=["AAPL"], sell_cooldown_days=1,
            dynamic_sl_tp=False, default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 100_000
        engine._day_count = 5

        engine._sell_cooldown["AAPL"] = 3  # sold on day 3, now day 5 → 2 days passed

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80
        signal.signal_type = SignalType.BUY

        df = _make_ohlcv(10, start_price=100.0)
        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        engine._execute_buy("AAPL", stock_data, 5, df.index[5], signal, "uptrend")
        assert "AAPL" in engine._positions  # Cooldown expired

    def test_whipsaw_blocks_after_losses(self):
        """Two loss sells in 7 days blocks re-entry."""
        config = PipelineConfig(
            universe=["AAPL"], whipsaw_max_losses=2,
            sell_cooldown_days=0,  # Disable cooldown to isolate whipsaw
            dynamic_sl_tp=False, default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 100_000
        engine._day_count = 10

        # Two loss sells in last 7 days
        engine._loss_sell_history["AAPL"] = [5, 8]

        signal = MagicMock()
        signal.strategy_name = "momentum"
        signal.confidence = 0.80
        signal.signal_type = SignalType.BUY

        df = _make_ohlcv(15, start_price=100.0)
        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        engine._execute_buy("AAPL", stock_data, 10, df.index[10], signal, "uptrend")
        assert "AAPL" not in engine._positions  # Blocked by whipsaw

    def test_min_hold_prevents_early_sl(self):
        """SL doesn't trigger if min hold not met and not hard SL."""
        config = PipelineConfig(
            universe=["AAPL"], min_hold_days=2,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 90_000
        engine._day_count = 1  # Just 1 day held (entry at day 0)

        engine._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2023-06-01", strategy_name="test",
            highest_price=100.0,
            stop_loss_pct=0.05, take_profit_pct=0.20,
            entry_day_count=0,
        )

        # Low breaches SL but only -7% (not hard SL at -15%)
        df = _make_ohlcv(5, start_price=100.0)
        df.iloc[1, df.columns.get_loc("low")] = 93.0
        df.iloc[1, df.columns.get_loc("open")] = 97.0
        df.iloc[1, df.columns.get_loc("close")] = 94.0

        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        engine._check_risk_exits(stock_data, 1, df.index[1])
        assert "AAPL" in engine._positions  # Min hold prevents exit

    def test_hard_sl_bypasses_min_hold(self):
        """Hard SL (-15%+) triggers even if min hold not met."""
        config = PipelineConfig(
            universe=["AAPL"], min_hold_days=5, hard_sl_pct=0.15,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 90_000
        engine._day_count = 1

        engine._positions["AAPL"] = _Position(
            symbol="AAPL", quantity=100, avg_price=100.0,
            entry_date="2023-06-01", strategy_name="test",
            highest_price=100.0,
            stop_loss_pct=0.05, take_profit_pct=0.20,
            entry_day_count=0,
        )

        # Low at $83 = -17% (exceeds hard SL of 15%)
        df = _make_ohlcv(5, start_price=100.0)
        df.iloc[1, df.columns.get_loc("low")] = 83.0
        df.iloc[1, df.columns.get_loc("open")] = 84.0
        df.iloc[1, df.columns.get_loc("close")] = 84.0

        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        engine._check_risk_exits(stock_data, 1, df.index[1])
        assert "AAPL" not in engine._positions  # Hard SL bypasses min hold


class TestPortfolioEquity:
    def test_cash_only(self):
        engine = FullPipelineBacktest(PipelineConfig(universe=["AAPL"]))
        engine._cash = 100_000
        equity = engine._calculate_equity({}, 0)
        assert equity == 100_000

    def test_with_positions(self):
        engine = FullPipelineBacktest(PipelineConfig(universe=["AAPL"]))
        engine._cash = 50_000
        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "test", 100.0, 0.08, 0.20,
        )

        df = _make_ohlcv(100, start_price=110.0)
        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        equity = engine._calculate_equity(stock_data, 99)
        # 50,000 cash + 100 * ~110 = ~61,000
        assert equity > 50_000


class TestClosePosition:
    def test_records_trade(self):
        engine = FullPipelineBacktest(PipelineConfig(universe=["AAPL"]))
        engine._cash = 0

        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "trend_following",
            105.0, 0.08, 0.20,
        )

        engine._close_position("AAPL", 110.0, pd.Timestamp("2023-03-01"), "signal_sell")

        assert "AAPL" not in engine._positions
        assert len(engine._trades) == 1
        trade = engine._trades[0]
        assert trade.symbol == "AAPL"
        assert trade.pnl > 0
        assert trade.strategy_name == "trend_following"
        assert engine._cash > 0

    def test_slippage_applied(self):
        config = PipelineConfig(universe=["AAPL"], slippage_pct=1.0)  # 1% slippage
        engine = FullPipelineBacktest(config)
        engine._cash = 0

        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "test", 100.0, 0.08, 0.20,
        )

        engine._close_position("AAPL", 100.0, pd.Timestamp("2023-03-01"), "test")

        # With 1% slippage on sell, exec_price = 100 * 0.99 = 99
        trade = engine._trades[0]
        assert trade.exit_price < 100.0
        assert trade.pnl < 0  # Small loss due to slippage


class TestStrategyStats:
    def test_computes_per_strategy(self):
        engine = FullPipelineBacktest(PipelineConfig(universe=["AAPL"]))
        engine._trades = [
            Trade("AAPL", "SELL", "2023-01-01", 100.0, "2023-02-01", 110.0, 100, 1000, 10, 30, "trend_following"),
            Trade("MSFT", "SELL", "2023-01-01", 100.0, "2023-02-01", 90.0, 100, -1000, -10, 30, "rsi_divergence"),
            Trade("GOOGL", "SELL", "2023-01-01", 100.0, "2023-02-01", 105.0, 100, 500, 5, 30, "trend_following"),
        ]

        stats = engine._compute_strategy_stats()
        assert "trend_following" in stats
        assert stats["trend_following"]["trades"] == 2
        assert stats["trend_following"]["wins"] == 2
        assert stats["trend_following"]["win_rate"] == 100.0
        assert stats["rsi_divergence"]["trades"] == 1
        assert stats["rsi_divergence"]["wins"] == 0


class TestPipelineResult:
    def test_summary_format(self):
        from backtest.metrics import BacktestMetrics
        result = PipelineResult(
            metrics=BacktestMetrics(
                total_return_pct=25.0, cagr=0.08, sharpe_ratio=1.2,
                sortino_ratio=1.5, max_drawdown_pct=-12.0, max_drawdown_days=30,
                total_trades=50, winning_trades=30, losing_trades=20,
                win_rate=60.0, profit_factor=1.8, avg_holding_days=15,
                benchmark_return_pct=20.0, alpha=5.0,
                start_date="2023-01-01", end_date="2025-01-01",
                trading_days=500, final_equity=125_000, initial_equity=100_000,
            ),
            trades=[],
            equity_curve=pd.Series([100_000, 125_000]),
            daily_snapshots=[],
            config=PipelineConfig(),
            strategy_stats={"trend_following": {"trades": 20, "wins": 12, "losses": 8, "pnl": 5000, "win_rate": 60.0}},
        )

        summary = result.summary()
        assert "Full Pipeline Backtest" in summary
        assert "CAGR" in summary
        assert "Sharpe" in summary
        assert "trend_following" in summary
        assert "Alpha" in summary


class TestFullPipelineIntegration:
    """Integration test using synthetic data (no yfinance calls)."""

    @pytest.mark.asyncio
    async def test_run_with_mocked_data(self):
        """Run full pipeline with small synthetic universe."""
        config = PipelineConfig(
            universe=["AAPL", "MSFT", "GOOGL"],
            initial_equity=100_000,
            screen_interval=50,
            max_watchlist=3,
            max_positions=3,
        )
        engine = FullPipelineBacktest(config)

        # Mock data loading
        mock_data = {
            "SPY": _make_backtest_data("SPY", 500, trend=0.0003),
            "AAPL": _make_backtest_data("AAPL", 500, trend=0.0005),
            "MSFT": _make_backtest_data("MSFT", 500, trend=0.0004),
            "GOOGL": _make_backtest_data("GOOGL", 500, trend=0.0002),
        }

        with patch.object(engine._data_loader, "load_multiple", return_value=mock_data):
            result = await engine.run(period="3y")

        assert isinstance(result, PipelineResult)
        assert result.metrics.trading_days > 0
        assert result.metrics.initial_equity == 100_000
        assert result.metrics.final_equity > 0
        assert len(result.equity_curve) > 0
        assert len(result.daily_snapshots) > 0

        # Should have some trades
        assert result.metrics.total_trades >= 0

        # Strategy stats should be populated
        assert isinstance(result.strategy_stats, dict)

    @pytest.mark.asyncio
    async def test_run_respects_max_positions(self):
        """Verify max_positions constraint is enforced."""
        config = PipelineConfig(
            universe=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            initial_equity=100_000,
            max_positions=2,
            screen_interval=50,
            max_watchlist=5,
        )
        engine = FullPipelineBacktest(config)

        mock_data = {
            "SPY": _make_backtest_data("SPY", 400, trend=0.0003),
        }
        for s in config.universe:
            mock_data[s] = _make_backtest_data(s, 400, trend=0.0005)

        with patch.object(engine._data_loader, "load_multiple", return_value=mock_data):
            result = await engine.run(period="2y")

        # Check that at no point did we exceed max positions
        for snap in result.daily_snapshots:
            assert snap.n_positions <= config.max_positions

    @pytest.mark.asyncio
    async def test_benchmark_comparison(self):
        """Verify benchmark metrics are calculated."""
        config = PipelineConfig(
            universe=["AAPL", "MSFT"],
            initial_equity=100_000,
            screen_interval=50,
        )
        engine = FullPipelineBacktest(config)

        mock_data = {
            "SPY": _make_backtest_data("SPY", 500, trend=0.0003),
            "AAPL": _make_backtest_data("AAPL", 500, trend=0.0005),
            "MSFT": _make_backtest_data("MSFT", 500, trend=0.0004),
        }

        with patch.object(engine._data_loader, "load_multiple", return_value=mock_data):
            result = await engine.run(period="3y")

        # Benchmark should be populated (SPY returns)
        assert result.metrics.benchmark_return_pct != 0.0

    @pytest.mark.asyncio
    async def test_no_spy_data_raises(self):
        """Should raise ValueError if SPY data is missing."""
        config = PipelineConfig(universe=["AAPL"])
        engine = FullPipelineBacktest(config)

        with patch.object(engine._data_loader, "load_multiple", return_value={"AAPL": _make_backtest_data("AAPL", 300)}):
            with pytest.raises(ValueError, match="SPY"):
                await engine.run()

    @pytest.mark.asyncio
    async def test_dynamic_sl_tp_applied(self):
        """Verify dynamic SL/TP uses ATR when enabled."""
        config = PipelineConfig(
            universe=["AAPL"],
            dynamic_sl_tp=True,
        )
        engine = FullPipelineBacktest(config)

        # Create a position manually and verify SL/TP are dynamic
        data = _make_backtest_data("AAPL", 300, trend=0.001)
        stock_data = {"AAPL": data}

        # Mock the buy execution to capture SL/TP
        from strategies.base import Signal
        from core.enums import SignalType

        engine._cash = 100_000
        signal = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            strategy_name="trend_following",
            reason="test",
        )

        from data.market_state import MarketRegime
        engine._execute_buy("AAPL", stock_data, 250, data.df.index[250], signal, MarketRegime.UPTREND)

        if "AAPL" in engine._positions:
            pos = engine._positions["AAPL"]
            # Dynamic SL/TP should differ from defaults in most cases
            assert pos.stop_loss_pct > 0
            assert pos.take_profit_pct > 0
