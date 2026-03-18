from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from core.enums import SignalType


@dataclass
class Signal:
    """Trading signal produced by a strategy."""

    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    reason: str
    suggested_price: float | None = None
    indicators: dict = field(default_factory=dict)


@dataclass
class PositionContext:
    """Context about a held position, passed to evaluate_exit().

    Provides strategies with position-level data (entry price, PnL, hold
    duration, etc.) so they can make profit-taking exit decisions without
    modifying the analyze() contract.
    """

    symbol: str
    entry_price: float
    current_price: float
    highest_price: float
    quantity: int
    pnl_pct: float  # (current - entry) / entry
    hold_seconds: float  # seconds since position was opened
    strategy: str  # original buy strategy name


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    All tunable parameters are loaded from config/strategies.yaml
    and can be updated at runtime via set_params().
    """

    # Class-level profit-exit parameters, shared by all strategy instances.
    # Set by StrategyRegistry on load / reload from config/strategies.yaml.
    _profit_exit_params: dict = {
        "sell_confidence_boost_min_pnl": 0.02,
        "sell_confidence_boost_factor": 0.5,
        "sell_confidence_boost_max": 0.15,
        "profit_take_min_pnl": 0.08,
        "profit_take_base_confidence": 0.65,
        "profit_take_weakness_required": 1,
        "rsi_overbought": 70.0,
        "volume_weakness_ratio": 0.8,
    }

    @classmethod
    def set_profit_exit_params(cls, params: dict) -> None:
        """Update class-level profit-exit parameters from config.

        Called by StrategyRegistry when loading/reloading strategies.yaml.
        """
        cls._profit_exit_params = {**cls._profit_exit_params, **params}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier matching config key."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    @abstractmethod
    def applicable_market_types(self) -> list[str]:
        """Market types: 'trending', 'sideways', 'all'."""
        ...

    @property
    @abstractmethod
    def required_timeframe(self) -> str:
        """OHLCV timeframe: '1D', '4h', '1h', etc."""
        ...

    @property
    @abstractmethod
    def min_candles_required(self) -> int:
        ...

    @abstractmethod
    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyze OHLCV data and produce a trading signal.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                and pre-computed indicators.
            symbol: Stock ticker symbol (e.g. 'AAPL').

        Returns:
            Signal with type, confidence, reason, and indicator snapshot.
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return current tunable parameters."""
        ...

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Update tunable parameters at runtime (from YAML reload)."""
        ...

    def evaluate_exit(
        self,
        signal: Signal,
        context: PositionContext,
        df: pd.DataFrame,
    ) -> Signal:
        """Evaluate whether a held position should be exited for profit.

        Called by EvaluationLoop for held positions after analyze().
        The default implementation handles three scenarios:

        1. **SELL + profitable**: Boost confidence proportional to PnL.
        2. **HOLD + high profit + technical weakness**: Generate a
           profit-taking SELL signal even when analyze() returned HOLD.
        3. **Other cases**: Return the original signal unchanged.

        Strategies can override this method for custom exit logic.

        Args:
            signal: The Signal from analyze().
            context: Position context with entry price, PnL, etc.
            df: OHLCV DataFrame with indicators (same as passed to analyze).

        Returns:
            Possibly modified Signal with adjusted type or confidence.
        """
        p = self._profit_exit_params

        # Scenario 1: SELL + profitable → boost confidence
        if (
            signal.signal_type == SignalType.SELL
            and context.pnl_pct > p["sell_confidence_boost_min_pnl"]
        ):
            boost = min(
                context.pnl_pct * p["sell_confidence_boost_factor"],
                p["sell_confidence_boost_max"],
            )
            new_confidence = min(1.0, signal.confidence + boost)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=new_confidence,
                strategy_name=signal.strategy_name,
                reason=f"{signal.reason} [profit_boost pnl={context.pnl_pct:.1%}]",
                suggested_price=signal.suggested_price,
                indicators=signal.indicators,
            )

        # Scenario 2: HOLD + high profit + technical weakness → profit-take SELL
        if (
            signal.signal_type == SignalType.HOLD
            and context.pnl_pct >= p["profit_take_min_pnl"]
        ):
            weakness_count = self._detect_technical_weakness(df, p)
            if weakness_count >= p["profit_take_weakness_required"]:
                confidence = min(
                    1.0,
                    p["profit_take_base_confidence"]
                    + context.pnl_pct * p["sell_confidence_boost_factor"],
                )
                return Signal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    strategy_name=signal.strategy_name,
                    reason=(
                        f"profit_take(pnl={context.pnl_pct:.1%}, "
                        f"weakness={weakness_count}/3)"
                    ),
                    suggested_price=signal.suggested_price,
                    indicators=signal.indicators,
                )

        # Scenario 3: No modification
        return signal

    @staticmethod
    def _detect_technical_weakness(df: pd.DataFrame, params: dict) -> int:
        """Count technical weakness indicators from OHLCV data.

        Checks three conditions:
        1. RSI overbought (above threshold)
        2. MACD histogram declining (current < previous)
        3. Volume ratio below threshold (weak participation)

        Returns:
            Number of weakness signals detected (0-3).
        """
        if df.empty or len(df) < 2:
            return 0

        count = 0
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. RSI overbought
        rsi_col = None
        for col in ("rsi", "RSI_14", "RSI_10"):
            if col in df.columns:
                rsi_col = col
                break
        if rsi_col is not None:
            try:
                rsi_val = float(last[rsi_col])
                if rsi_val > params.get("rsi_overbought", 70.0):
                    count += 1
            except (ValueError, TypeError):
                pass

        # 2. MACD histogram declining
        macd_col = None
        for col in ("macd_histogram", "MACDh_12_26_9", "MACDh_8_20_7"):
            if col in df.columns:
                macd_col = col
                break
        if macd_col is not None:
            try:
                curr_macd = float(last[macd_col])
                prev_macd = float(prev[macd_col])
                if curr_macd < prev_macd:
                    count += 1
            except (ValueError, TypeError):
                pass

        # 3. Volume weakness (current volume < ratio * moving average)
        if "volume" in df.columns:
            try:
                vol = float(last["volume"])
                vol_ma = float(df["volume"].tail(20).mean())
                if vol_ma > 0 and vol / vol_ma < params.get(
                    "volume_weakness_ratio", 0.8
                ):
                    count += 1
            except (ValueError, TypeError):
                pass

        return count
