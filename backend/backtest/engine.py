"""Backtest engine orchestrator.

Coordinates data loading, strategy execution, simulation, and metrics.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtest.data_loader import BacktestDataLoader, BacktestData
from backtest.simulator import BacktestSimulator, SimConfig
from backtest.metrics import MetricsCalculator, BacktestMetrics, Trade
from strategies.base import BaseStrategy, Signal
from core.enums import SignalType
from data.indicator_service import IndicatorService

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: pd.Series
    config: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.metrics.passes_minimum()

    def summary(self) -> str:
        m = self.metrics
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.strategy_name} on {self.symbol}\n"
            f"  Period: {m.start_date} ~ {m.end_date} ({m.trading_days} days)\n"
            f"  Return: {m.total_return_pct:.1f}% | CAGR: {m.cagr:.1%}\n"
            f"  Sharpe: {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f}\n"
            f"  MDD: {m.max_drawdown_pct:.1f}% ({m.max_drawdown_days} days)\n"
            f"  Trades: {m.total_trades} | Win Rate: {m.win_rate:.1f}%\n"
            f"  Profit Factor: {m.profit_factor:.2f}\n"
            f"  Final Equity: ${m.final_equity:,.0f}\n"
        )


class BacktestEngine:
    """Main backtest orchestrator."""

    def __init__(
        self,
        data_loader: BacktestDataLoader | None = None,
        sim_config: SimConfig | None = None,
    ):
        self._data_loader = data_loader or BacktestDataLoader()
        self._sim_config = sim_config or SimConfig()

    async def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        period: str = "3y",
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        """Run backtest for a single strategy on a single symbol.

        Args:
            strategy: Strategy instance to test
            symbol: Stock ticker
            period: Data period (if start not specified)
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD
        """
        # Load data
        data = self._data_loader.load(
            symbol, period=period, start=start, end=end,
        )

        # Generate signals
        signals = await self._generate_signals(strategy, data.df, symbol)

        # Simulate
        simulator = BacktestSimulator(config=self._sim_config)
        simulator.run(data.df, signals, symbol)

        # Calculate metrics
        metrics = MetricsCalculator.calculate(
            equity_curve=simulator.equity_curve,
            trades=simulator.trades,
            initial_equity=self._sim_config.initial_equity,
        )

        result = BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            metrics=metrics,
            trades=simulator.trades,
            equity_curve=simulator.equity_curve,
            config=strategy.get_params(),
        )

        logger.info(result.summary())
        return result

    async def run_multiple(
        self,
        strategy: BaseStrategy,
        symbols: list[str],
        period: str = "3y",
    ) -> list[BacktestResult]:
        """Run backtest across multiple symbols."""
        results = []
        for symbol in symbols:
            try:
                result = await self.run(strategy, symbol, period=period)
                results.append(result)
            except Exception as e:
                logger.warning("Backtest failed for %s: %s", symbol, e)
        return results

    async def _generate_signals(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        symbol: str,
    ) -> dict[int, Signal]:
        """Generate signals by running strategy on each bar.

        Uses data up to (but not including) bar i to generate the signal,
        which is then executed at bar i's close. This matches live trading
        where strategies analyze yesterday's completed bar and trade today.
        """
        signals = {}
        min_bars = strategy.min_candles_required

        for i in range(min_bars + 1, len(df)):
            window = df.iloc[:i]  # Exclude current bar (no look-ahead)
            try:
                signal = await strategy.analyze(window, symbol)
                if signal.signal_type != SignalType.HOLD:
                    signals[i] = signal
            except Exception as e:
                logger.debug("Signal generation error at bar %d: %s", i, e)

        logger.info(
            "Generated %d signals for %s using %s",
            len(signals), symbol, strategy.name,
        )
        return signals
