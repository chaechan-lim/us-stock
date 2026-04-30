"""Supertrend Strategy.

Uses the Supertrend indicator for trend direction with confirmation.

Buy: Supertrend direction flips bullish (direction changes from -1 to 1)
     with confirmation
Sell: Supertrend direction flips bearish, with dynamic confidence
      based on distance below supertrend + ADX gradient

Additional signals:
- ADX gradient: adjusts confidence based on ADX trend direction
- RSI overbought exit: reduces buy confidence when RSI > 75 as
  an early warning of potential reversal
"""

import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class SupertrendStrategy(BaseStrategy):
    name = "supertrend"
    display_name = "Supertrend"
    applicable_market_types = ["trending"]
    required_timeframe = "1D"
    min_candles_required = 20

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._confirmation_bars = p.get("confirmation_bars", 2)
        self._rsi_overbought = p.get("rsi_overbought", 75)
        self._adx_lookback = p.get("adx_lookback", 3)
        # E2 (2026-04-24): optional 20d daily-return std filter to suppress
        # BUY on high-vol whipsaw-prone symbols like ALM (vol 0.97 annualised
        # → daily std ~6%, routinely -5% in 4h → position_cleanup churn).
        self._volatility_filter = p.get("volatility_filter", False)
        self._max_volatility_pct = p.get("max_volatility_pct", 3.0)
        # F2 (2026-04-30): pullback filter — only BUY when close is within
        # `pullback_max_pct` above the supertrend line. The line acts as
        # support; entries far above it (fresh breakout extension) are the
        # ones that whipsaw down 5-7% in 4h. None disables.
        # Live observation: ALM/CRML/247540/028260 all bought at 6%+ above
        # the supertrend line, then immediately pulled back.
        self._pullback_max_pct: float | None = p.get("pullback_max_pct")
        # G1 (2026-04-30): when set, signal.suggested_price = line × (1+offset)
        # instead of current close → live order_manager places LIMIT order at
        # that price. Order sits in the book; if price never pulls back, no
        # fill happens (signal naturally expires when supertrend turns
        # bearish). Daily-bar backtest can't validate this (intraday spike
        # invisible) so this is a live-only experimental knob.
        self._entry_offset_pct: float | None = p.get("entry_offset_pct")

    async def analyze(
        self, df: pd.DataFrame, symbol: str
    ) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        row = df.iloc[-1]
        price = float(row["close"])

        st_dir = row.get("supertrend_direction")
        supertrend = row.get("supertrend")
        adx = row.get("adx")
        rsi = row.get("rsi")

        if st_dir is None or pd.isna(st_dir):
            return self._hold("Supertrend not ready")

        adx_grad = self._calc_adx_gradient(df)
        rsi_ob = self._is_rsi_overbought(rsi)

        indicators = {
            "supertrend": (
                float(supertrend)
                if supertrend and not pd.isna(supertrend)
                else 0
            ),
            "supertrend_direction": float(st_dir),
            "adx": (
                float(adx)
                if adx and not pd.isna(adx)
                else 0
            ),
            "rsi": (
                float(rsi)
                if rsi and not pd.isna(rsi)
                else 50
            ),
            "adx_gradient": adx_grad,
            "rsi_overbought": rsi_ob,
        }

        confirmed_bull = self._check_confirmation(
            df, bullish=True
        )
        confirmed_bear = self._check_confirmation(
            df, bullish=False
        )

        # BUY: bullish supertrend confirmed
        if (
            st_dir > 0
            and confirmed_bull
            and price > supertrend
        ):
            # E2 volatility filter: suppress BUY on high-vol whipsaw symbols
            if self._volatility_filter:
                vol_pct = self._calc_volatility(df)
                if vol_pct > self._max_volatility_pct:
                    return self._hold(
                        f"High volatility ({vol_pct:.1f}% > {self._max_volatility_pct}%)"
                    )

            # F2 pullback filter: only BUY when close is reasonably close to
            # the supertrend line. Entries 6-10%+ above the line have shown
            # high whipsaw rate (live ALM/CRML/247540 all bought extended).
            if (
                self._pullback_max_pct is not None
                and supertrend
                and not pd.isna(supertrend)
                and float(supertrend) > 0
            ):
                extension = (price - float(supertrend)) / float(supertrend)
                if extension > self._pullback_max_pct:
                    return self._hold(
                        f"Too extended above supertrend line "
                        f"({extension:.1%} > {self._pullback_max_pct:.1%})"
                    )

            confidence = self._calc_confidence(
                adx, rsi, price, supertrend
            )
            confidence = self._apply_adx_gradient(
                confidence, adx_grad
            )
            reason = (
                f"Supertrend bullish "
                f"(confirmed {self._confirmation_bars} bars)"
            )

            # RSI overbought warning: reduce confidence
            if rsi_ob:
                confidence *= 0.80
                reason += " [RSI overbought warning]"

            confidence = max(0.1, min(confidence, 0.95))

            # G1: limit-at-line entry. When entry_offset_pct is set and
            # current close is above (line × (1+offset)), suggest the
            # lower limit price so the order_manager places a limit order
            # there instead of a market order at the spike. If close is
            # already at/below the limit, market entry is fine.
            entry_price = price
            if (
                self._entry_offset_pct is not None
                and supertrend
                and not pd.isna(supertrend)
                and float(supertrend) > 0
            ):
                line = float(supertrend)
                limit_price = line * (1 + self._entry_offset_pct)
                if limit_price < price:
                    entry_price = limit_price
                    reason += f" (limit @ line+{self._entry_offset_pct:.0%})"

            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name=self.name,
                reason=reason,
                suggested_price=entry_price,
                indicators=indicators,
            )

        # SELL: bearish supertrend confirmed
        if (
            st_dir < 0
            and confirmed_bear
            and price < supertrend
        ):
            confidence = self._calc_sell_confidence(
                price, supertrend, adx_grad
            )
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name=self.name,
                reason="Supertrend bearish",
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold("No supertrend signal")

    def _check_confirmation(
        self, df: pd.DataFrame, bullish: bool = True
    ) -> bool:
        """Check supertrend direction consistent for N bars."""
        if len(df) < self._confirmation_bars + 1:
            return False

        recent = df.iloc[-self._confirmation_bars :]
        directions = recent.get("supertrend_direction")
        if directions is None:
            return False

        if bullish:
            return all(
                not pd.isna(d) and d > 0 for d in directions
            )
        return all(
            not pd.isna(d) and d < 0 for d in directions
        )

    def _calc_confidence(
        self, adx, rsi, price, supertrend
    ) -> float:
        conf = 0.55
        if adx and not pd.isna(adx) and adx > 25:
            conf += 0.15
        if rsi and not pd.isna(rsi) and 40 < rsi < 70:
            conf += 0.10
        if (
            supertrend
            and not pd.isna(supertrend)
            and supertrend > 0
        ):
            gap_pct = (
                (price - supertrend) / supertrend * 100
            )
            if 1 < gap_pct < 5:
                conf += 0.10
        return min(conf, 0.95)

    def _calc_sell_confidence(
        self,
        price: float,
        supertrend: float,
        adx_gradient: float,
    ) -> float:
        """Dynamic sell confidence based on distance below
        supertrend and ADX gradient.

        - Base confidence: 0.55
        - Distance bonus: up to +0.25 based on how far price
          is below supertrend (0-8% range mapped linearly)
        - ADX gradient adjustment: rising ADX in bearish trend
          increases sell confidence (trend strengthening)
        """
        base = 0.55

        # Distance-based scaling
        if supertrend and supertrend > 0:
            gap_pct = (
                (supertrend - price) / supertrend * 100
            )
            # Map 0-8% gap to 0-0.25 bonus (clamped)
            distance_bonus = min(
                max(gap_pct / 8.0 * 0.25, 0.0), 0.25
            )
            base += distance_bonus

        # ADX gradient: rising ADX = stronger trend = more
        # confident sell
        base = self._apply_adx_gradient(
            base, adx_gradient, for_sell=True
        )

        return max(0.3, min(base, 0.95))

    def _calc_volatility(self, df: pd.DataFrame) -> float:
        """20d daily-return std as a percentage (mirrors dual_momentum).

        Used by the optional volatility_filter to suppress BUY signals on
        high-vol, whipsaw-prone symbols (ALM, AMPX, CIFR — vol 0.6+).
        """
        if len(df) < 21:
            return 0.0
        returns = df["close"].iloc[-21:].pct_change().dropna()
        if len(returns) < 5:
            return 0.0
        return float(returns.std() * 100)

    def _calc_adx_gradient(
        self, df: pd.DataFrame
    ) -> float:
        """Calculate ADX gradient (rate of change) over
        recent bars.

        Returns positive value if ADX is increasing (trend
        strengthening), negative if decreasing (trend
        weakening). Returns 0.0 if insufficient data.
        """
        lookback = self._adx_lookback
        if len(df) < lookback + 1:
            return 0.0

        recent = df.iloc[-(lookback + 1) :]
        adx_vals = recent.get("adx")
        if adx_vals is None:
            return 0.0

        valid = [
            float(v)
            for v in adx_vals
            if v is not None and not pd.isna(v)
        ]
        if len(valid) < 2:
            return 0.0

        # Average per-bar change over lookback
        gradient = (valid[-1] - valid[0]) / len(valid)
        return round(gradient, 4)

    def _apply_adx_gradient(
        self,
        confidence: float,
        adx_gradient: float,
        for_sell: bool = False,
    ) -> float:
        """Adjust confidence based on ADX gradient.

        For buy: rising ADX boosts, falling ADX reduces
        For sell: rising ADX boosts (trend strengthening =
        more confident bearish), falling ADX reduces
        """
        if abs(adx_gradient) < 0.5:
            return confidence

        # Normalize gradient effect: cap at ±0.10
        adjustment = min(
            max(adx_gradient / 5.0, -0.10), 0.10
        )

        if for_sell:
            # For sell: rising ADX = stronger bearish trend
            confidence += adjustment
        else:
            # For buy: rising ADX = stronger bullish trend
            confidence += adjustment

        return confidence

    def _is_rsi_overbought(self, rsi) -> bool:
        """Check if RSI indicates overbought condition."""
        if rsi is None or pd.isna(rsi):
            return False
        return float(rsi) > self._rsi_overbought

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "confirmation_bars": self._confirmation_bars,
            "rsi_overbought": self._rsi_overbought,
            "adx_lookback": self._adx_lookback,
        }

    def set_params(self, params: dict) -> None:
        self._confirmation_bars = params.get(
            "confirmation_bars", self._confirmation_bars
        )
        self._rsi_overbought = params.get(
            "rsi_overbought", self._rsi_overbought
        )
        self._adx_lookback = params.get(
            "adx_lookback", self._adx_lookback
        )
