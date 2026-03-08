"""Signal Quality & Confidence Calibration.

Tracks per-strategy trade outcomes and calculates calibrated metrics:
- Win rate per strategy
- Average win / loss magnitudes
- Profit factor (gross profit / gross loss)
- Sharpe-like signal quality score

Use these metrics to:
1. Gate weak strategies (disable strategies with profit factor < 1.0)
2. Feed Kelly Criterion (win_rate, avg_win, avg_loss)
3. Weight signal contributions (higher quality → more influence)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade outcome."""
    strategy: str
    symbol: str
    return_pct: float  # +0.08 = 8% gain, -0.03 = 3% loss
    timestamp: float = 0.0


@dataclass
class StrategyMetrics:
    """Calibrated performance metrics for a strategy."""
    strategy: str
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    quality_score: float = 0.0  # Composite quality metric

    @property
    def has_edge(self) -> bool:
        """Strategy has positive expected value."""
        return self.profit_factor > 1.0 and self.total_trades >= 5

    @property
    def kelly_inputs(self) -> tuple[float, float, float]:
        """Return (win_rate, avg_win, avg_loss) for Kelly calculation."""
        return self.win_rate, self.avg_win, self.avg_loss


class SignalQualityTracker:
    """Track and calibrate strategy signal quality.

    Records trade outcomes per strategy and computes rolling
    performance metrics. Used to:
    - Gate strategies that don't have edge (profit_factor < 1.0)
    - Feed Kelly Criterion for optimal position sizing
    - Weight strategy signals proportional to quality
    """

    def __init__(
        self,
        max_trades_per_strategy: int = 200,
        min_trades_for_gating: int = 10,
        lookback_days: int = 180,
    ):
        self._max_trades = max_trades_per_strategy
        self._min_trades = min_trades_for_gating
        self._lookback_days = lookback_days
        self._trades: dict[str, list[TradeRecord]] = defaultdict(list)

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        return_pct: float,
    ) -> None:
        """Record a completed trade outcome."""
        record = TradeRecord(
            strategy=strategy,
            symbol=symbol,
            return_pct=return_pct,
            timestamp=time.time(),
        )
        trades = self._trades[strategy]
        trades.append(record)

        # Keep only recent trades
        if len(trades) > self._max_trades:
            self._trades[strategy] = trades[-self._max_trades:]

    def get_metrics(self, strategy: str) -> StrategyMetrics:
        """Calculate calibrated metrics for a strategy."""
        trades = self._get_recent_trades(strategy)

        if not trades:
            return StrategyMetrics(strategy=strategy)

        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total = len(returns)
        win_rate = len(wins) / total if total > 0 else 0

        avg_win = float(sum(wins) / len(wins)) if wins else 0
        avg_loss = float(abs(sum(losses) / len(losses))) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = min(
            gross_profit / gross_loss if gross_loss > 0 else (
                10.0 if gross_profit > 0 else 0
            ),
            10.0,
        )

        # Quality score: combines win rate, profit factor, and consistency
        # Range: 0-100
        quality = self._compute_quality(win_rate, profit_factor, returns, total)

        return StrategyMetrics(
            strategy=strategy,
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(profit_factor, 4),
            total_trades=total,
            quality_score=round(quality, 1),
        )

    def get_all_metrics(self) -> dict[str, StrategyMetrics]:
        """Get metrics for all tracked strategies."""
        return {name: self.get_metrics(name) for name in self._trades}

    def get_active_strategies(self) -> list[str]:
        """Return strategies with positive edge (profit_factor > 1.0)."""
        return [
            name for name in self._trades
            if self.get_metrics(name).has_edge
        ]

    def get_gated_strategies(self) -> list[str]:
        """Return strategies that should be disabled (no edge)."""
        gated = []
        for name in self._trades:
            metrics = self.get_metrics(name)
            if (metrics.total_trades >= self._min_trades
                    and not metrics.has_edge):
                gated.append(name)
        return gated

    def get_strategy_weights(self) -> dict[str, float]:
        """Calculate relative weights based on quality scores.

        Higher quality → higher weight. Strategies without edge get 0.
        """
        all_metrics = self.get_all_metrics()
        weights = {}
        for name, m in all_metrics.items():
            if m.has_edge:
                weights[name] = m.quality_score
            else:
                weights[name] = 0.0

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def _get_recent_trades(self, strategy: str) -> list[TradeRecord]:
        """Get trades within lookback period."""
        cutoff = time.time() - self._lookback_days * 86400
        return [
            t for t in self._trades.get(strategy, [])
            if t.timestamp >= cutoff or t.timestamp == 0  # 0 = legacy
        ]

    @staticmethod
    def _compute_quality(
        win_rate: float,
        profit_factor: float,
        returns: list[float],
        total: int,
    ) -> float:
        """Compute composite quality score (0-100)."""
        if total < 3:
            return 0.0

        # Win rate component (0-30): linear 40%-70% → 0-30
        wr_score = max(0, min(30, (win_rate - 0.4) / 0.3 * 30))

        # Profit factor component (0-30): 1.0-3.0 → 0-30
        pf = min(profit_factor, 3.0)
        pf_score = max(0, (pf - 1.0) / 2.0 * 30)

        # Consistency (0-20): lower stdev of returns → higher
        import numpy as np
        std = np.std(returns) if len(returns) > 1 else 0.5
        mean_ret = np.mean(returns)
        # Risk-adjusted return (mini Sharpe)
        mini_sharpe = mean_ret / std if std > 0 else 0
        cons_score = max(0, min(20, (mini_sharpe + 0.5) / 1.5 * 20))

        # Sample size confidence (0-20): more trades → more reliable
        size_score = min(20, total / 50 * 20)

        return wr_score + pf_score + cons_score + size_score
