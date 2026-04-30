"""Dual Momentum Strategy.

Monthly rebalance using absolute + relative momentum.
BUY: Top-N performers with positive absolute momentum.
SELL: When absolute momentum turns negative or momentum gradient declines.
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class DualMomentumStrategy(BaseStrategy):
    name = "dual_momentum"
    display_name = "Dual Momentum"
    applicable_market_types = ["trending"]
    required_timeframe = "1M"
    min_candles_required = 30

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._lookback_months = p.get("lookback_months", 12)
        self._min_absolute_return = p.get("min_absolute_return", 0.0)
        self._volatility_filter = p.get("volatility_filter", True)
        self._max_volatility_pct = p.get("max_volatility_pct", 3.0)
        self._gradient_sell = p.get("gradient_sell", True)
        self._gradient_threshold = p.get("gradient_threshold", -0.30)
        # F2 (2026-04-30): anti-overbought filter — skip BUY when the
        # symbol has rallied more than `max_5d_gain` in the last 5 trading
        # days (already pulled into a vertical move, high pullback risk).
        # Live whipsaw cases (247540, 028260) showed +20-30% over 5d
        # right before the system bought, then -5% in 1-3 days. None disables.
        self._max_5d_gain: float | None = p.get("max_5d_gain")

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        price = float(row["close"])

        ema_50 = row.get("ema_50")
        ema_20 = row.get("ema_20")

        # Calculate lookback return manually
        lookback_days = self._lookback_months * 21
        lookback_idx = max(0, len(df) - lookback_days)
        if lookback_idx < len(df) - 1:
            past_price = float(df.iloc[lookback_idx]["close"])
            absolute_return = (price - past_price) / past_price
        else:
            absolute_return = 0.0

        # 3-month momentum for recency
        short_idx = max(0, len(df) - 63)
        short_price = float(df.iloc[short_idx]["close"])
        short_return = (
            (price - short_price) / short_price if short_price > 0 else 0.0
        )

        # Momentum gradient: compare recent vs earlier momentum
        gradient = self._calc_momentum_gradient(df)

        # Volatility (20-day daily returns std, annualized)
        volatility_pct = self._calc_volatility(df)

        # EMA confirmation
        ema_bullish = False
        if (
            ema_20 is not None
            and ema_50 is not None
            and not pd.isna(ema_20)
            and not pd.isna(ema_50)
        ):
            ema_bullish = float(ema_20) > float(ema_50)

        indicators = {
            "absolute_return": absolute_return,
            "short_return": short_return,
            "gradient": round(gradient, 4) if gradient is not None else None,
            "volatility_pct": round(volatility_pct, 2),
            "ema_bullish": ema_bullish,
        }

        # Gradient sell: momentum decelerating sharply
        if (
            self._gradient_sell
            and gradient is not None
            and gradient < self._gradient_threshold
            and absolute_return > 0
        ):
            confidence = 0.55 if gradient < -0.50 else 0.45
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=(
                    f"Momentum fading: gradient={gradient:.2f}, "
                    f"abs={absolute_return:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # BUY: positive absolute momentum + strong short-term trend
        if absolute_return > self._min_absolute_return and short_return > 0:
            # Volatility filter: suppress buy in high-vol
            if self._volatility_filter and volatility_pct > self._max_volatility_pct:
                return self._hold(
                    f"High volatility ({volatility_pct:.1f}%) — buy suppressed"
                )

            # F2 anti-overbought: refuse to chase symbols that already
            # rallied hard over the last 5 days. The exhausted-momentum
            # entry is the source of most whipsaw losses.
            if self._max_5d_gain is not None and len(df) >= 6:
                p_5d = float(df.iloc[-6]["close"])
                if p_5d > 0:
                    gain_5d = (price - p_5d) / p_5d
                    if gain_5d > self._max_5d_gain:
                        return self._hold(
                            f"Overbought 5d ({gain_5d:.1%} > {self._max_5d_gain:.1%})"
                        )

            confidence = 0.5
            if absolute_return > 0.15:
                confidence += 0.15
            elif absolute_return > 0.08:
                confidence += 0.10
            if short_return > 0.05:
                confidence += 0.10
            if ema_bullish:
                confidence += 0.10
            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 0.95),
                strategy_name=self.name,
                reason=(
                    f"Abs momentum {absolute_return:.1%}, "
                    f"Short {short_return:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # SELL: negative absolute momentum
        if absolute_return < 0:
            confidence = 0.6 if absolute_return < -0.05 else 0.4
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason=f"Negative momentum {absolute_return:.1%}",
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold("Momentum inconclusive")

    def _calc_momentum_gradient(self, df: pd.DataFrame) -> float | None:
        """Compare 1-month return vs 3-month return to detect deceleration."""
        if len(df) < 63:
            return None
        price = float(df["close"].iloc[-1])
        p_1m = float(df["close"].iloc[-21])
        p_3m = float(df["close"].iloc[-63])
        if p_1m <= 0 or p_3m <= 0:
            return None
        ret_1m = (price - p_1m) / p_1m
        ret_3m = (price - p_3m) / p_3m
        # Gradient: if recent momentum is much weaker than 3m, negative
        return ret_1m - ret_3m / 3

    def _calc_volatility(self, df: pd.DataFrame) -> float:
        """Calculate 20-day daily return std as a percentage."""
        if len(df) < 21:
            return 0.0
        returns = df["close"].iloc[-21:].pct_change().dropna()
        if len(returns) < 5:
            return 0.0
        return float(returns.std() * 100)

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "lookback_months": self._lookback_months,
            "min_absolute_return": self._min_absolute_return,
            "volatility_filter": self._volatility_filter,
            "max_volatility_pct": self._max_volatility_pct,
            "gradient_sell": self._gradient_sell,
            "gradient_threshold": self._gradient_threshold,
        }

    def set_params(self, params: dict) -> None:
        for key in [
            "lookback_months", "min_absolute_return",
            "volatility_filter", "max_volatility_pct",
            "gradient_sell", "gradient_threshold",
        ]:
            if key in params:
                setattr(self, f"_{key}", params[key])
