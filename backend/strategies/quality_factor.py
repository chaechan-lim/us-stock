"""Quality Factor Strategy (Price-Derived Quality Metrics).

Inspired by Novy-Marx (2013) profitability factor. Since the analyze()
interface only provides OHLCV data, we use price-derived quality proxies:

1. **Earnings stability**: Low volatility of returns (stable businesses)
2. **Trend consistency**: Percentage of up-days above threshold (persistent growth)
3. **Risk-adjusted momentum**: Sharpe-like ratio of returns (quality alpha)
4. **Drawdown resilience**: Max drawdown over lookback (quality = less drawdown)

BUY: High quality score (stable, consistent, risk-adjusted outperformance)
SELL: Deteriorating quality (rising volatility, broken trend, deep drawdown)
"""

import numpy as np
import pandas as pd

from core.enums import SignalType
from strategies.base import BaseStrategy, Signal


class QualityFactorStrategy(BaseStrategy):
    name = "quality_factor"
    display_name = "Quality Factor"
    applicable_market_types = ["all"]
    required_timeframe = "1D"
    min_candles_required = 126  # ~6 months of daily data

    def __init__(self, params: dict | None = None):
        p = params or {}
        self._lookback_days = p.get("lookback_days", 126)
        self._vol_lookback = p.get("vol_lookback", 60)
        self._max_volatility = p.get("max_volatility", 0.40)
        self._min_up_ratio = p.get("min_up_ratio", 0.52)
        self._min_sharpe = p.get("min_sharpe", 0.5)
        self._max_drawdown = p.get("max_drawdown", -0.15)
        self._sell_quality_threshold = p.get("sell_quality_threshold", 0.25)

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.min_candles_required:
            return self._hold("Insufficient data")

        close = df["close"]
        price = float(close.iloc[-1])

        # 1. Earnings stability proxy: annualized volatility of daily returns
        returns = close.iloc[-self._vol_lookback:].pct_change().dropna()
        if len(returns) < 20:
            return self._hold("Insufficient return data")
        daily_vol = float(returns.std())
        ann_vol = daily_vol * np.sqrt(252)

        # 2. Trend consistency: fraction of positive return days
        up_ratio = float((returns > 0).sum() / len(returns))

        # 3. Risk-adjusted momentum: return / vol (Sharpe-like, no risk-free)
        total_return = float(
            (close.iloc[-1] - close.iloc[-self._lookback_days])
            / close.iloc[-self._lookback_days]
        )
        ann_return = total_return * (252 / self._lookback_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # 4. Drawdown resilience: max drawdown over lookback
        rolling_max = close.iloc[-self._lookback_days:].cummax()
        drawdown = (close.iloc[-self._lookback_days:] - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Composite quality score (0-1)
        quality_score = self._compute_quality_score(
            ann_vol, up_ratio, sharpe, max_dd,
        )

        # EMA alignment check
        ema_20 = df.iloc[-1].get("ema_20")
        ema_50 = df.iloc[-1].get("ema_50")
        ema_aligned = (
            ema_20 is not None
            and ema_50 is not None
            and not pd.isna(ema_20)
            and not pd.isna(ema_50)
            and float(ema_20) > float(ema_50)
        )

        indicators = {
            "ann_vol": round(ann_vol, 4),
            "up_ratio": round(up_ratio, 4),
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
            "quality_score": round(quality_score, 4),
            "ema_aligned": ema_aligned,
        }

        # SELL: very low quality + broken trend
        if quality_score < self._sell_quality_threshold and not ema_aligned:
            confidence = 0.45
            if quality_score < 0.15:
                confidence += 0.10
            if max_dd < -0.25:
                confidence += 0.10
            return Signal(
                signal_type=SignalType.SELL,
                confidence=min(confidence, 0.80),
                strategy_name=self.name,
                reason=(
                    f"Low quality: score={quality_score:.2f}, "
                    f"vol={ann_vol:.1%}, dd={max_dd:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        # BUY: high quality score with trend support
        if quality_score >= 0.60:
            # Volatility filter
            if ann_vol > self._max_volatility:
                return self._hold(
                    f"High volatility ({ann_vol:.1%}) despite quality"
                )

            confidence = 0.50
            if quality_score >= 0.80:
                confidence += 0.15
            elif quality_score >= 0.70:
                confidence += 0.10
            if ema_aligned:
                confidence += 0.10
            if sharpe > 1.5:
                confidence += 0.10

            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 0.95),
                strategy_name=self.name,
                reason=(
                    f"High quality: score={quality_score:.2f}, "
                    f"Sharpe={sharpe:.2f}, dd={max_dd:.1%}"
                ),
                suggested_price=price,
                indicators=indicators,
            )

        return self._hold(
            f"Quality neutral ({quality_score:.2f})"
        )

    def _compute_quality_score(
        self,
        ann_vol: float,
        up_ratio: float,
        sharpe: float,
        max_dd: float,
    ) -> float:
        """Compute composite quality score from 0 to 1.

        Each component contributes 0.25 (equal weight):
        - Low volatility → high score
        - High up-day ratio → high score
        - High Sharpe → high score
        - Shallow drawdown → high score
        """
        # Volatility score: lower is better (0-1)
        # ann_vol of 0.10 (10%) → 1.0, ann_vol of 0.50 (50%) → 0.0
        vol_score = max(0, min(1, 1.0 - (ann_vol - 0.10) / 0.40))

        # Up-ratio score: higher is better (0-1)
        # up_ratio of 0.55+ → good, 0.45- → bad
        ratio_score = max(0, min(1, (up_ratio - 0.40) / 0.20))

        # Sharpe score: higher is better (0-1)
        # Sharpe 2.0+ → 1.0, Sharpe 0.0 → 0.0
        sharpe_score = max(0, min(1, sharpe / 2.0))

        # Drawdown score: shallower is better (0-1)
        # max_dd of 0% → 1.0, max_dd of -30% → 0.0
        dd_score = max(0, min(1, 1.0 + max_dd / 0.30))

        return (vol_score + ratio_score + sharpe_score + dd_score) / 4.0

    def _hold(self, reason: str) -> Signal:
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self.name,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {
            "lookback_days": self._lookback_days,
            "vol_lookback": self._vol_lookback,
            "max_volatility": self._max_volatility,
            "min_up_ratio": self._min_up_ratio,
            "min_sharpe": self._min_sharpe,
            "max_drawdown": self._max_drawdown,
            "sell_quality_threshold": self._sell_quality_threshold,
        }

    def set_params(self, params: dict) -> None:
        for key in [
            "lookback_days", "vol_lookback", "max_volatility",
            "min_up_ratio", "min_sharpe", "max_drawdown",
            "sell_quality_threshold",
        ]:
            if key in params:
                setattr(self, f"_{key}", params[key])
