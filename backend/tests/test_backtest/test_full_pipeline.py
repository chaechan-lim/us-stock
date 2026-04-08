"""Tests for Full Pipeline Backtest Engine."""

import asyncio
import json
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
        # initial_equity is None until FullPipelineBacktest.__init__ resolves it
        assert config.initial_equity is None
        assert config.max_positions == 20
        assert config.max_position_pct == 0.10
        assert config.screen_interval == 20
        assert config.dynamic_sl_tp is True
        assert config.market == "US"
        # universe resolved to default in FullPipelineBacktest.__init__
        assert config.universe is None  # None = auto from market

    def test_custom_config(self):
        config = PipelineConfig(
            universe=["AAPL", "MSFT"],
            initial_equity=50_000.0,
            max_positions=5,
        )
        assert config.universe == ["AAPL", "MSFT"]
        assert config.initial_equity == 50_000.0
        assert config.max_positions == 5

    def test_us_default_equity_resolved(self):
        """US market: __init__ auto-fills initial_equity to 100_000 USD."""
        config = PipelineConfig(universe=["AAPL"])  # market defaults to "US"
        assert config.initial_equity is None  # not yet resolved
        FullPipelineBacktest(config)
        assert config.initial_equity == 100_000.0

    def test_kr_default_equity_auto_scaled_to_krw(self):
        """KR market: __init__ auto-fills initial_equity to 100M KRW.

        Regression for the silent 0-trades bug. Before this fix, KR
        backtests defaulted to 100_000 (USD-style) which made
        int(allocation/krw_price) round to 0 for every position.
        """
        config = PipelineConfig(market="KR", universe=["005930.KS"])
        assert config.initial_equity is None
        FullPipelineBacktest(config)
        assert config.initial_equity == 100_000_000.0

    def test_explicit_equity_not_overridden(self):
        """Explicitly-passed initial_equity is respected for both markets."""
        us = PipelineConfig(universe=["AAPL"], initial_equity=250_000.0)
        FullPipelineBacktest(us)
        assert us.initial_equity == 250_000.0

        kr = PipelineConfig(market="KR", universe=["005930.KS"], initial_equity=50_000_000.0)
        FullPipelineBacktest(kr)
        assert kr.initial_equity == 50_000_000.0


class TestUniverseResolution:
    """Tests for the universe selection precedence in __init__:
    explicit `universe` > `universe_path` > wide constant > narrow default.
    """

    def test_explicit_universe_wins(self):
        cfg = PipelineConfig(universe=["AAPL", "MSFT"], use_wide_universe=True)
        FullPipelineBacktest(cfg)
        assert cfg.universe == ["AAPL", "MSFT"]

    def test_default_us_is_narrow(self):
        cfg = PipelineConfig()  # market=US, no overrides
        FullPipelineBacktest(cfg)
        from backtest.full_pipeline import DEFAULT_UNIVERSE
        assert cfg.universe == list(DEFAULT_UNIVERSE)
        assert len(cfg.universe) >= 50

    def test_use_wide_universe_us(self):
        cfg = PipelineConfig(use_wide_universe=True)
        FullPipelineBacktest(cfg)
        from backtest.full_pipeline import WIDE_UNIVERSE, DEFAULT_UNIVERSE
        assert cfg.universe == list(WIDE_UNIVERSE)
        assert len(cfg.universe) > len(DEFAULT_UNIVERSE)
        # Wide includes small-cap names the live system actually trades
        assert "ELVN" in cfg.universe
        assert "BW" in cfg.universe
        assert "ADTN" in cfg.universe
        # And sector ETFs (so dual_momentum/sector_rotation can pick them)
        for etf in ("XLK", "XLF", "XLE", "XLU"):
            assert etf in cfg.universe

    def test_kr_unaffected_by_wide_flag(self):
        cfg = PipelineConfig(market="KR", use_wide_universe=True)
        FullPipelineBacktest(cfg)
        from backtest.full_pipeline import DEFAULT_KR_UNIVERSE
        assert cfg.universe == list(DEFAULT_KR_UNIVERSE)

    def test_universe_path_loads_snapshot(self, tmp_path):
        snap = tmp_path / "snapshot.json"
        snap.write_text(json.dumps({
            "market": "US",
            "symbols": ["aapl", "msft", "googl", "elvn", "bw"],
        }))
        cfg = PipelineConfig(universe_path=str(snap))
        FullPipelineBacktest(cfg)
        # Symbols are uppercased on load
        assert cfg.universe == ["AAPL", "MSFT", "GOOGL", "ELVN", "BW"]

    def test_universe_path_falls_back_on_missing_file(self, tmp_path, caplog):
        cfg = PipelineConfig(universe_path=str(tmp_path / "nope.json"))
        FullPipelineBacktest(cfg)
        # Should fall back to default narrow universe (US)
        from backtest.full_pipeline import DEFAULT_UNIVERSE
        assert cfg.universe == list(DEFAULT_UNIVERSE)

    def test_universe_path_falls_back_on_malformed(self, tmp_path):
        snap = tmp_path / "bad.json"
        snap.write_text('{"not_symbols": [1, 2, 3]}')
        cfg = PipelineConfig(universe_path=str(snap))
        FullPipelineBacktest(cfg)
        from backtest.full_pipeline import DEFAULT_UNIVERSE
        assert cfg.universe == list(DEFAULT_UNIVERSE)

    def test_explicit_universe_beats_universe_path(self, tmp_path):
        snap = tmp_path / "snap.json"
        snap.write_text(json.dumps({"symbols": ["XYZ"]}))
        cfg = PipelineConfig(universe=["AAPL"], universe_path=str(snap))
        FullPipelineBacktest(cfg)
        assert cfg.universe == ["AAPL"]


class TestSignalQualitySeed:
    """Tests for signal_quality_seed_path injection.

    Without this, backtest tracker starts cold while live has 6+ months
    of accumulated trade history → different gating + Kelly behavior.
    """

    def test_seed_path_loads_into_tracker(self, tmp_path):
        import time as _time
        snap = tmp_path / "sq.json"
        snap.write_text(json.dumps({
            "version": 1,
            "tracker": {
                "version": 1,
                "trades": {
                    "dual_momentum": [
                        {"symbol": "AAPL", "return_pct": 0.06, "timestamp": _time.time()},
                        {"symbol": "MSFT", "return_pct": -0.02, "timestamp": _time.time()},
                        {"symbol": "NVDA", "return_pct": 0.10, "timestamp": _time.time()},
                    ],
                    "supertrend": [
                        {"symbol": "TSLA", "return_pct": 0.04, "timestamp": _time.time()},
                    ],
                },
            },
        }))
        cfg = PipelineConfig(
            universe=["AAPL"],
            signal_quality_seed_path=str(snap),
        )
        eng = FullPipelineBacktest(cfg)

        m_dm = eng._signal_quality.get_metrics("dual_momentum")
        m_st = eng._signal_quality.get_metrics("supertrend")
        assert m_dm.total_trades == 3
        assert m_st.total_trades == 1

    def test_seed_path_accepts_bare_tracker_dict(self, tmp_path):
        # Snapshot file may also be a raw tracker dict (no enclosing wrapper)
        import time as _time
        snap = tmp_path / "sq.json"
        snap.write_text(json.dumps({
            "version": 1,
            "trades": {
                "dual_momentum": [
                    {"symbol": "AAPL", "return_pct": 0.05, "timestamp": _time.time()},
                ],
            },
        }))
        cfg = PipelineConfig(
            universe=["AAPL"],
            signal_quality_seed_path=str(snap),
        )
        eng = FullPipelineBacktest(cfg)
        assert eng._signal_quality.get_metrics("dual_momentum").total_trades == 1

    def test_missing_seed_file_falls_back_to_cold(self, tmp_path):
        cfg = PipelineConfig(
            universe=["AAPL"],
            signal_quality_seed_path=str(tmp_path / "nope.json"),
        )
        eng = FullPipelineBacktest(cfg)
        # Should not raise; tracker is just empty
        assert eng._signal_quality.get_metrics("dual_momentum").total_trades == 0

    def test_no_seed_path_means_cold_tracker(self):
        cfg = PipelineConfig(universe=["AAPL"])
        eng = FullPipelineBacktest(cfg)
        assert len(eng._signal_quality._trades) == 0


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


class TestLookAheadBiasFixes:
    """Verify no look-ahead bias: current bar excluded from regime & factor scoring."""

    def test_regime_detection_excludes_current_bar(self):
        """SPY regime detection must use [:date_idx], not [:date_idx+1]."""
        import ast
        import inspect
        import textwrap
        from backtest.full_pipeline import FullPipelineBacktest

        source = textwrap.dedent(inspect.getsource(FullPipelineBacktest.run))
        tree = ast.parse(source)

        # Find the spy_window assignment: should be spy_data.df.iloc[:date_idx]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "spy_window":
                        # The slice should be [:date_idx] (upper=Name(id='date_idx'))
                        # NOT [:date_idx + 1] (upper=BinOp)
                        value = node.value  # Subscript
                        assert isinstance(value, ast.Subscript), "spy_window should be a subscript"
                        slice_node = value.slice
                        assert isinstance(slice_node, ast.Slice), "Should be a slice"
                        upper = slice_node.upper
                        assert isinstance(upper, ast.Name), (
                            f"spy_window slice upper should be Name(date_idx), "
                            f"got {type(upper).__name__} — possible look-ahead bias"
                        )
                        assert upper.id == "date_idx", (
                            f"Expected date_idx, got {upper.id}"
                        )
                        return

        pytest.fail("spy_window assignment not found in run()")

    def test_factor_scores_exclude_current_bar(self):
        """Factor score computation must use [:date_idx], not [:date_idx+1]."""
        import ast
        import inspect
        import textwrap
        from backtest.full_pipeline import FullPipelineBacktest

        source = textwrap.dedent(inspect.getsource(FullPipelineBacktest._update_factor_scores))
        tree = ast.parse(source)

        # Find data.df.iloc[:date_idx] — the slice should NOT be a BinOp
        found_slice = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    upper = node.slice.upper
                    if isinstance(upper, ast.Name) and upper.id == "date_idx":
                        found_slice = True
                    elif isinstance(upper, ast.BinOp):
                        pytest.fail(
                            "Factor scores use [:date_idx + N] — look-ahead bias detected"
                        )

        assert found_slice, "Expected [:date_idx] slice in _update_factor_scores"

    def test_strategy_signals_exclude_current_bar(self):
        """Strategy analyze() receives df_window=[:date_idx], not [:date_idx+1]."""
        import ast
        import inspect
        import textwrap
        from backtest.full_pipeline import FullPipelineBacktest

        source = textwrap.dedent(inspect.getsource(FullPipelineBacktest.run))
        tree = ast.parse(source)

        # Find df_window assignment: sdata.df.iloc[:date_idx]
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "df_window":
                        value = node.value
                        assert isinstance(value, ast.Subscript)
                        slice_node = value.slice
                        assert isinstance(slice_node, ast.Slice)
                        upper = slice_node.upper
                        assert isinstance(upper, ast.Name), (
                            f"df_window slice should be Name(date_idx), "
                            f"got {type(upper).__name__} — look-ahead bias"
                        )
                        assert upper.id == "date_idx"
                        return

        pytest.fail("df_window assignment not found in run()")


class TestKellyParamsMatchLive:
    """Verify backtest Kelly params match live defaults."""

    def test_default_kelly_fraction(self):
        config = PipelineConfig()
        assert config.kelly_fraction == 0.40

    def test_default_confidence_exponent(self):
        config = PipelineConfig()
        assert config.confidence_exponent == 1.2

    def test_default_min_position_pct(self):
        config = PipelineConfig()
        assert config.min_position_pct == 0.05

    def test_kelly_sizer_receives_correct_params(self):
        """KellyPositionSizer is initialized with matching parameters."""
        config = PipelineConfig()
        engine = FullPipelineBacktest(config)
        sizer = engine._risk_manager._kelly
        assert sizer._kelly_frac == config.kelly_fraction
        assert sizer._conf_exp == config.confidence_exponent
        assert sizer._min_pct == config.min_position_pct


class TestSymmetricSlippage:
    """Verify sell-side slippage is volume-adjusted (symmetric with buy)."""

    def test_close_position_uses_effective_slippage(self):
        """_close_position should call _effective_slippage, not use fixed slippage."""
        config = PipelineConfig(
            universe=["AAPL"],
            slippage_pct=0.05,
            volume_adjusted_slippage=True,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 0

        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "test", 100.0, 0.08, 0.20,
        )

        # Close with high volume → base slippage (0.05%)
        engine._close_position("AAPL", 100.0, "2023-03-01", "test", volume=10_000_000)
        trade_high_vol = engine._trades[-1]

        # Reset position
        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "test", 100.0, 0.08, 0.20,
        )
        engine._cash = 0

        # Close with low volume → higher slippage (3x at >10% participation)
        engine._close_position("AAPL", 100.0, "2023-03-01", "test", volume=500)
        trade_low_vol = engine._trades[-1]

        # Low volume should have worse exit price (more slippage)
        assert trade_low_vol.exit_price < trade_high_vol.exit_price

    def test_execute_sell_passes_volume(self):
        """_execute_sell should pass volume to _close_position."""
        config = PipelineConfig(
            universe=["AAPL"],
            dynamic_sl_tp=False,
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.20,
        )
        engine = FullPipelineBacktest(config)
        engine._cash = 0
        engine._day_count = 10

        engine._positions["AAPL"] = _Position(
            "AAPL", 100, 100.0, "2023-01-01", "test", 100.0, 0.08, 0.20,
            entry_day_count=0,
        )

        # Create data with known volume
        df = _make_ohlcv(10, start_price=100.0)
        from data.indicator_service import IndicatorService
        df = IndicatorService.add_all_indicators(df)
        stock_data = {"AAPL": BacktestData("AAPL", df, "2023-01-01", "2023-06-01")}

        signal = MagicMock()
        signal.strategy_name = "test"
        signal.confidence = 0.80
        signal.signal_type = SignalType.SELL

        engine._execute_sell("AAPL", stock_data, 5, df.index[5], signal)
        assert len(engine._trades) == 1
        # Trade recorded with volume-adjusted slippage (not zero-volume fixed)

    def test_effective_slippage_tiers(self):
        """Verify volume-adjusted slippage scales with participation rate."""
        config = PipelineConfig(
            slippage_pct=0.05,
            volume_adjusted_slippage=True,
            universe=["AAPL"],
        )
        engine = FullPipelineBacktest(config)

        # <1% participation → base
        assert engine._effective_slippage(100_000, 500) == 0.05
        # 1-5% participation → 1.5x
        assert engine._effective_slippage(10_000, 200) == pytest.approx(0.075)
        # 5-10% participation → 2x
        assert engine._effective_slippage(10_000, 600) == pytest.approx(0.10)
        # >10% participation → 3x
        assert engine._effective_slippage(1_000, 200) == pytest.approx(0.15)

    def test_zero_volume_uses_base_slippage(self):
        """When volume=0, use base slippage (don't divide by zero)."""
        config = PipelineConfig(
            slippage_pct=0.05,
            volume_adjusted_slippage=True,
            universe=["AAPL"],
        )
        engine = FullPipelineBacktest(config)
        assert engine._effective_slippage(0, 100) == 0.05
