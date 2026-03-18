"""Tests for BaseStrategy.evaluate_exit() and PositionContext.

Covers the three default scenarios:
1. SELL + profitable → confidence boost
2. HOLD + high profit + technical weakness → profit-taking SELL
3. All other cases → pass through unchanged
Plus technical weakness detection and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from core.enums import SignalType
from strategies.base import BaseStrategy, PositionContext, Signal

# --- Concrete strategy for testing (BaseStrategy is abstract) ---


class _TestStrategy(BaseStrategy):
    """Minimal concrete strategy for testing evaluate_exit()."""

    @property
    def name(self) -> str:
        return "test_strategy"

    @property
    def display_name(self) -> str:
        return "Test Strategy"

    @property
    def applicable_market_types(self) -> list[str]:
        return ["all"]

    @property
    def required_timeframe(self) -> str:
        return "1D"

    @property
    def min_candles_required(self) -> int:
        return 20

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strategy_name=self.name,
            reason="test",
        )

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass


# --- Fixtures ---


@pytest.fixture(autouse=True)
def reset_profit_exit_params():
    """Reset class-level params before each test."""
    original = BaseStrategy._profit_exit_params.copy()
    yield
    BaseStrategy._profit_exit_params = original


@pytest.fixture
def strategy():
    return _TestStrategy()


def _make_context(
    pnl_pct: float = 0.10,
    entry_price: float = 100.0,
    current_price: float | None = None,
    hold_seconds: float = 86400.0,
) -> PositionContext:
    if current_price is None:
        current_price = entry_price * (1 + pnl_pct)
    return PositionContext(
        symbol="AAPL",
        entry_price=entry_price,
        current_price=current_price,
        highest_price=current_price * 1.02,
        quantity=100,
        pnl_pct=pnl_pct,
        hold_seconds=hold_seconds,
        strategy="trend_following",
    )


def _make_df_with_indicators(
    n: int = 30,
    rsi: float = 75.0,
    macd_declining: bool = True,
    volume_weak: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame with indicator columns for testing."""
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(100000, 500000, n).astype(float),
        }
    )

    # RSI column
    df["rsi"] = 50.0
    df.iloc[-1, df.columns.get_loc("rsi")] = rsi

    # MACD histogram
    df["macd_histogram"] = 1.0
    if macd_declining:
        df.iloc[-2, df.columns.get_loc("macd_histogram")] = 2.0
        df.iloc[-1, df.columns.get_loc("macd_histogram")] = 1.0
    else:
        df.iloc[-2, df.columns.get_loc("macd_histogram")] = 1.0
        df.iloc[-1, df.columns.get_loc("macd_histogram")] = 2.0

    # Volume weakness
    if volume_weak:
        avg_vol = df["volume"].tail(20).mean()
        df.iloc[-1, df.columns.get_loc("volume")] = avg_vol * 0.5
    else:
        avg_vol = df["volume"].tail(20).mean()
        df.iloc[-1, df.columns.get_loc("volume")] = avg_vol * 1.5

    return df


# --- PositionContext Tests ---


class TestPositionContext:
    def test_create_position_context(self):
        ctx = _make_context(pnl_pct=0.15)
        assert ctx.symbol == "AAPL"
        assert ctx.pnl_pct == 0.15
        assert ctx.quantity == 100
        assert ctx.strategy == "trend_following"

    def test_context_negative_pnl(self):
        ctx = _make_context(pnl_pct=-0.05)
        assert ctx.pnl_pct == -0.05

    def test_context_zero_pnl(self):
        ctx = _make_context(pnl_pct=0.0)
        assert ctx.pnl_pct == 0.0


# --- Scenario 1: SELL + profitable → boost confidence ---


class TestSellConfidenceBoost:
    def test_sell_profitable_boosts_confidence(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="bearish crossover",
        )
        ctx = _make_context(pnl_pct=0.10)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.SELL
        assert result.confidence > signal.confidence
        assert "profit_boost" in result.reason

    def test_sell_profitable_boost_proportional_to_pnl(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx_small = _make_context(pnl_pct=0.04)
        ctx_large = _make_context(pnl_pct=0.20)
        df = _make_df_with_indicators()

        r_small = strategy.evaluate_exit(signal, ctx_small, df)
        r_large = strategy.evaluate_exit(signal, ctx_large, df)

        assert r_large.confidence > r_small.confidence

    def test_sell_profitable_boost_capped_at_max(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.90,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx = _make_context(pnl_pct=0.50)  # Very high PnL
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.confidence <= 1.0

    def test_sell_below_min_pnl_not_boosted(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx = _make_context(pnl_pct=0.01)  # Below default 2% min
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.confidence == signal.confidence
        assert result is signal  # Unchanged, same object

    def test_sell_negative_pnl_not_boosted(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx = _make_context(pnl_pct=-0.05)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.confidence == signal.confidence
        assert result is signal

    def test_sell_at_exact_min_pnl_not_boosted(self, strategy):
        """PnL must be strictly greater than min_pnl for boost."""
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx = _make_context(pnl_pct=0.02)  # Exactly at default threshold
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        # 0.02 is NOT > 0.02, so no boost
        assert result is signal


# --- Scenario 2: HOLD + high profit + weakness → profit-take SELL ---


class TestProfitTakeSell:
    def test_hold_high_profit_with_weakness_generates_sell(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.12)
        df = _make_df_with_indicators(rsi=75, macd_declining=True, volume_weak=True)

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.SELL
        assert "profit_take" in result.reason
        assert result.confidence >= 0.65

    def test_hold_high_profit_no_weakness_stays_hold(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.12)
        # No technical weakness
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=False)

        # weakness_required = 1 by default, 0 weakness = no trigger
        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.HOLD
        assert result is signal

    def test_hold_low_profit_stays_hold(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.03)  # Below 8% default
        df = _make_df_with_indicators(rsi=75, macd_declining=True, volume_weak=True)

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.HOLD
        assert result is signal

    def test_profit_take_confidence_increases_with_pnl(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        df = _make_df_with_indicators(rsi=75, macd_declining=True, volume_weak=True)

        ctx_low = _make_context(pnl_pct=0.08)
        ctx_high = _make_context(pnl_pct=0.25)

        r_low = strategy.evaluate_exit(signal, ctx_low, df)
        r_high = strategy.evaluate_exit(signal, ctx_high, df)

        assert r_low.signal_type == SignalType.SELL
        assert r_high.signal_type == SignalType.SELL
        assert r_high.confidence > r_low.confidence

    def test_profit_take_preserves_strategy_name(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.12)
        df = _make_df_with_indicators(rsi=75, macd_declining=True, volume_weak=True)

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.strategy_name == "test_strategy"

    def test_hold_at_exact_min_pnl_with_weakness(self, strategy):
        """PnL at exactly profit_take_min_pnl should trigger."""
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.08)  # Exactly at default threshold
        df = _make_df_with_indicators(rsi=75, macd_declining=True, volume_weak=True)

        result = strategy.evaluate_exit(signal, ctx, df)

        # >= 0.08, so should trigger
        assert result.signal_type == SignalType.SELL


# --- Scenario 3: Pass-through (no modification) ---


class TestPassThrough:
    def test_buy_signal_not_modified(self, strategy):
        signal = Signal(
            signal_type=SignalType.BUY,
            confidence=0.70,
            strategy_name="test_strategy",
            reason="buy signal",
        )
        ctx = _make_context(pnl_pct=0.10)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result is signal
        assert result.signal_type == SignalType.BUY

    def test_hold_low_profit_not_modified(self, strategy):
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.02)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result is signal

    def test_sell_not_profitable_not_modified(self, strategy):
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.60,
            strategy_name="test_strategy",
            reason="stop loss",
        )
        ctx = _make_context(pnl_pct=-0.03)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result is signal


# --- Technical Weakness Detection ---


class TestTechnicalWeakness:
    def test_all_three_weakness_signals(self, strategy):
        df = _make_df_with_indicators(rsi=80, macd_declining=True, volume_weak=True)
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 3

    def test_no_weakness_signals(self, strategy):
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=False)
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 0

    def test_only_rsi_overbought(self, strategy):
        df = _make_df_with_indicators(rsi=80, macd_declining=False, volume_weak=False)
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 1

    def test_only_macd_declining(self, strategy):
        df = _make_df_with_indicators(rsi=50, macd_declining=True, volume_weak=False)
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 1

    def test_only_volume_weak(self, strategy):
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=True)
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 1

    def test_empty_df_returns_zero(self, strategy):
        df = pd.DataFrame()
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 0

    def test_single_row_df_returns_zero(self, strategy):
        df = pd.DataFrame({"close": [100], "volume": [1000], "rsi": [80]})
        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 0

    def test_alternative_rsi_column_name(self, strategy):
        """Test with RSI_14 column name (pandas-ta format)."""
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=False)
        df = df.drop(columns=["rsi"])
        df["RSI_14"] = 50.0
        df.iloc[-1, df.columns.get_loc("RSI_14")] = 80.0

        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 1  # RSI overbought

    def test_alternative_macd_column_name(self, strategy):
        """Test with MACDh_12_26_9 column name."""
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=False)
        df = df.drop(columns=["macd_histogram"])
        df["MACDh_12_26_9"] = 1.0
        df.iloc[-2, df.columns.get_loc("MACDh_12_26_9")] = 2.0
        df.iloc[-1, df.columns.get_loc("MACDh_12_26_9")] = 1.0

        params = BaseStrategy._profit_exit_params
        count = BaseStrategy._detect_technical_weakness(df, params)
        assert count == 1  # MACD declining


# --- Config Parameter Tests ---


class TestProfitExitParams:
    def test_set_profit_exit_params(self, strategy):
        custom = {"sell_confidence_boost_factor": 0.8, "rsi_overbought": 80.0}
        BaseStrategy.set_profit_exit_params(custom)

        assert BaseStrategy._profit_exit_params["sell_confidence_boost_factor"] == 0.8
        assert BaseStrategy._profit_exit_params["rsi_overbought"] == 80.0
        # Other defaults should still be present
        assert "profit_take_min_pnl" in BaseStrategy._profit_exit_params

    def test_custom_params_affect_behavior(self, strategy):
        # Set very high weakness requirement (3 of 3)
        BaseStrategy.set_profit_exit_params({"profit_take_weakness_required": 3})

        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.12)
        # Only 1 weakness signal (RSI overbought)
        df = _make_df_with_indicators(rsi=80, macd_declining=False, volume_weak=False)

        result = strategy.evaluate_exit(signal, ctx, df)

        # 1 weakness < 3 required → no profit-take SELL
        assert result.signal_type == SignalType.HOLD
        assert result is signal

    def test_custom_min_pnl_threshold(self, strategy):
        BaseStrategy.set_profit_exit_params({"profit_take_min_pnl": 0.20})

        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.15)  # Below new 20% threshold
        df = _make_df_with_indicators(rsi=80, macd_declining=True, volume_weak=True)

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.HOLD
        assert result is signal

    def test_zero_weakness_required_always_triggers(self, strategy):
        """If weakness_required = 0, any high-profit HOLD should trigger."""
        BaseStrategy.set_profit_exit_params({"profit_take_weakness_required": 0})

        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.50,
            strategy_name="test_strategy",
            reason="neutral",
        )
        ctx = _make_context(pnl_pct=0.10)
        # No weakness at all
        df = _make_df_with_indicators(rsi=50, macd_declining=False, volume_weak=False)

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.signal_type == SignalType.SELL
        assert "profit_take" in result.reason

    def test_confidence_capped_at_one(self, strategy):
        """Even with extreme boost, confidence should never exceed 1.0."""
        BaseStrategy.set_profit_exit_params(
            {
                "sell_confidence_boost_factor": 10.0,
                "sell_confidence_boost_max": 5.0,
            }
        )

        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.90,
            strategy_name="test_strategy",
            reason="sell",
        )
        ctx = _make_context(pnl_pct=0.50)
        df = _make_df_with_indicators()

        result = strategy.evaluate_exit(signal, ctx, df)

        assert result.confidence <= 1.0
