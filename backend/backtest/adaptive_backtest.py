"""Adaptive weight backtest - compare weighting strategies.

Runs a portfolio-level backtest comparing:
1. Equal weights (baseline)
2. Market-state weights (from config profiles)
3. Category-based weights (from stock classification)
4. Full adaptive weights (category + rolling performance)

Each mode runs the same strategies on the same data, only differing
in how strategy signals are combined.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestDataLoader
from backtest.metrics import MetricsCalculator, BacktestMetrics, Trade
from backtest.simulator import SimConfig
from data.indicator_service import IndicatorService
from engine.stock_classifier import StockClassifier, StockCategory
from engine.adaptive_weights import AdaptiveWeightManager, DEFAULT_STOCK_PROFILES
from strategies.registry import StrategyRegistry, STRATEGY_CLASSES
from strategies.combiner import SignalCombiner
from core.enums import SignalType

logger = logging.getLogger(__name__)


@dataclass
class WeightModeResult:
    """Result for a single weighting mode across all symbols."""
    mode: str
    metrics: BacktestMetrics
    per_symbol: dict[str, BacktestMetrics] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    total_trades: int = 0


@dataclass
class AdaptiveBacktestResult:
    """Comparison result across all weight modes."""
    symbols: list[str]
    period: str
    strategies_used: list[str]
    modes: dict[str, WeightModeResult] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Adaptive Weight Backtest: {len(self.symbols)} symbols, period={self.period}",
            f"Strategies: {', '.join(self.strategies_used)}",
            "",
            f"{'Mode':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Trades':>8}",
            "-" * 60,
        ]
        for name, mr in self.modes.items():
            m = mr.metrics
            lines.append(
                f"{name:<20} {m.cagr:>7.1%} {m.sharpe_ratio:>8.2f} "
                f"{m.max_drawdown_pct:>7.1f}% {m.win_rate:>7.1f}% {mr.total_trades:>8d}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def safe(v: float) -> float:
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                return 0.0
            return v

        result = {
            "symbols": self.symbols,
            "period": self.period,
            "strategies_used": self.strategies_used,
            "modes": {},
        }
        for mode_name, mr in self.modes.items():
            m = mr.metrics
            result["modes"][mode_name] = {
                "cagr": round(safe(m.cagr * 100), 2),
                "sharpe_ratio": round(safe(m.sharpe_ratio), 2),
                "sortino_ratio": round(safe(m.sortino_ratio), 2),
                "max_drawdown_pct": round(safe(m.max_drawdown_pct), 2),
                "total_return_pct": round(safe(m.total_return_pct), 2),
                "win_rate": round(safe(m.win_rate), 1),
                "profit_factor": round(safe(m.profit_factor), 2),
                "total_trades": mr.total_trades,
                "final_equity": round(safe(m.final_equity), 2),
                "categories": mr.categories,
                "per_symbol": {
                    sym: {
                        "cagr": round(safe(sm.cagr * 100), 2),
                        "sharpe_ratio": round(safe(sm.sharpe_ratio), 2),
                        "max_drawdown_pct": round(safe(sm.max_drawdown_pct), 2),
                        "total_trades": sm.total_trades,
                        "win_rate": round(safe(sm.win_rate), 1),
                    }
                    for sym, sm in mr.per_symbol.items()
                },
            }
        return result


class AdaptiveBacktestEngine:
    """Run portfolio-level backtests comparing weight modes."""

    WEIGHT_MODES = ["equal", "market_state", "category", "adaptive"]

    def __init__(
        self,
        sim_config: SimConfig | None = None,
        market_state: str = "uptrend",
    ):
        self._sim_config = sim_config or SimConfig(
            initial_equity=100_000,
            max_position_pct=0.10,
            max_total_positions=20,
        )
        self._data_loader = BacktestDataLoader()
        self._classifier = StockClassifier()
        self._combiner = SignalCombiner()
        self._market_state = market_state

    async def run(
        self,
        symbols: list[str],
        period: str = "3y",
        modes: list[str] | None = None,
        strategy_names: list[str] | None = None,
    ) -> AdaptiveBacktestResult:
        """Run comparative backtest across weight modes.

        Args:
            symbols: Stock tickers to test
            period: Data period
            modes: Weight modes to compare (default: all 4)
            strategy_names: Specific strategies (default: all enabled)
        """
        modes = modes or self.WEIGHT_MODES

        # Load strategies
        registry = StrategyRegistry()
        if strategy_names:
            strategies = [
                STRATEGY_CLASSES[n]()
                for n in strategy_names
                if n in STRATEGY_CLASSES
            ]
        else:
            strategies = registry.get_enabled()

        strat_names = [s.name for s in strategies]
        logger.info(
            "Adaptive backtest: %d symbols, %d strategies, modes=%s",
            len(symbols), len(strategies), modes,
        )

        # Load data for all symbols
        data_cache = self._data_loader.load_multiple(symbols, period=period)
        loaded_symbols = list(data_cache.keys())
        if not loaded_symbols:
            raise ValueError("No data loaded for any symbol")

        # Classify all stocks
        classifications = {}
        for symbol in loaded_symbols:
            df = data_cache[symbol].df
            profile = self._classifier.classify(df, symbol)
            classifications[symbol] = profile
            logger.info(
                "Classified %s as %s (vol=%.2f, mom=%.2f)",
                symbol, profile.category.value, profile.volatility,
                profile.momentum_score,
            )

        # Generate signals for all strategies × symbols
        all_signals = {}
        for strategy in strategies:
            for symbol in loaded_symbols:
                df = data_cache[symbol].df
                signals = await self._generate_signals(strategy, df, symbol)
                all_signals[(strategy.name, symbol)] = signals

        # Get market state weights from registry
        market_weights = registry.get_profile_weights(self._market_state)

        # Run each mode
        result = AdaptiveBacktestResult(
            symbols=loaded_symbols,
            period=period,
            strategies_used=strat_names,
        )

        for mode in modes:
            logger.info("Running mode: %s", mode)
            mode_result = await self._run_mode(
                mode=mode,
                loaded_symbols=loaded_symbols,
                data_cache=data_cache,
                strategies=strategies,
                all_signals=all_signals,
                classifications=classifications,
                market_weights=market_weights,
            )
            result.modes[mode] = mode_result

        logger.info("\n%s", result.summary())
        return result

    async def _run_mode(
        self,
        mode: str,
        loaded_symbols: list[str],
        data_cache: dict,
        strategies: list,
        all_signals: dict,
        classifications: dict,
        market_weights: dict[str, float],
    ) -> WeightModeResult:
        """Run backtest for a single weight mode across all symbols."""
        strat_names = [s.name for s in strategies]

        # Build weight function for this mode
        adaptive_mgr = AdaptiveWeightManager(alpha=0.6, min_signals_for_adaptation=5)

        # Pre-set categories for adaptive/category modes
        if mode in ("category", "adaptive"):
            for symbol, profile in classifications.items():
                adaptive_mgr.set_category(symbol, profile.category)

        all_trades = []
        equity_curves = {}
        per_symbol_metrics = {}
        categories = {}

        for symbol in loaded_symbols:
            df = data_cache[symbol].df
            profile = classifications[symbol]
            categories[symbol] = profile.category.value

            # Get weights for this mode
            weights = self._get_mode_weights(
                mode, strat_names, market_weights, adaptive_mgr, symbol,
            )

            # Combine signals across strategies for each bar
            sim_signals = self._combine_bar_signals(
                df, strategies, all_signals, symbol, weights,
            )

            # Simulate
            from backtest.simulator import BacktestSimulator
            simulator = BacktestSimulator(config=self._sim_config)
            simulator.run(df, sim_signals, symbol)

            # Track results
            all_trades.extend(simulator.trades)
            if not simulator.equity_curve.empty:
                equity_curves[symbol] = simulator.equity_curve

            sym_metrics = MetricsCalculator.calculate(
                equity_curve=simulator.equity_curve,
                trades=simulator.trades,
                initial_equity=self._sim_config.initial_equity,
            )
            per_symbol_metrics[symbol] = sym_metrics

            # For adaptive mode: record signal results to build performance data
            if mode == "adaptive":
                for trade in simulator.trades:
                    was_correct = trade.pnl > 0
                    if trade.strategy_name:
                        adaptive_mgr.record_signal_result(
                            symbol, trade.strategy_name, was_correct,
                        )

        # Aggregate portfolio equity curve
        portfolio_equity = self._aggregate_equity(equity_curves)

        # Calculate portfolio-level metrics
        portfolio_metrics = MetricsCalculator.calculate(
            equity_curve=portfolio_equity,
            trades=all_trades,
            initial_equity=self._sim_config.initial_equity * len(loaded_symbols),
        )

        return WeightModeResult(
            mode=mode,
            metrics=portfolio_metrics,
            per_symbol=per_symbol_metrics,
            categories=categories,
            total_trades=len(all_trades),
        )

    def _get_mode_weights(
        self,
        mode: str,
        strat_names: list[str],
        market_weights: dict[str, float],
        adaptive_mgr: AdaptiveWeightManager,
        symbol: str,
    ) -> dict[str, float]:
        """Get strategy weights for a given mode."""
        if mode == "equal":
            n = len(strat_names)
            return {name: 1.0 / n for name in strat_names}

        if mode == "market_state":
            # Use market state weights, fill missing with small default
            weights = {}
            for name in strat_names:
                weights[name] = market_weights.get(name, 0.01)
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()} if total > 0 else weights

        if mode in ("category", "adaptive"):
            return adaptive_mgr.get_weights(symbol, market_weights)

        return {name: 1.0 / len(strat_names) for name in strat_names}

    def _combine_bar_signals(
        self,
        df: pd.DataFrame,
        strategies: list,
        all_signals: dict,
        symbol: str,
        weights: dict[str, float],
    ) -> dict[int, "Signal"]:
        """Combine signals from multiple strategies per bar using weights."""
        combined = {}

        for i in range(len(df)):
            bar_signals = []
            for strategy in strategies:
                sig = all_signals.get((strategy.name, symbol), {}).get(i)
                if sig:
                    bar_signals.append(sig)

            if not bar_signals:
                continue

            # Use combiner with specified weights
            result = self._combiner.combine(bar_signals, weights)
            if result.signal_type != SignalType.HOLD:
                combined[i] = result

        return combined

    def _aggregate_equity(
        self, equity_curves: dict[str, pd.Series]
    ) -> pd.Series:
        """Sum equity curves across symbols into a portfolio curve."""
        if not equity_curves:
            return pd.Series(dtype=float)

        # Align all curves to a common date index
        combined = pd.DataFrame(equity_curves)
        combined = combined.ffill().bfill()
        portfolio = combined.sum(axis=1)
        return portfolio

    async def _generate_signals(
        self,
        strategy,
        df: pd.DataFrame,
        symbol: str,
    ) -> dict[int, "Signal"]:
        """Generate signals by running strategy on each bar."""
        signals = {}
        min_bars = strategy.min_candles_required

        for i in range(min_bars, len(df)):
            window = df.iloc[:i + 1]
            try:
                signal = await strategy.analyze(window, symbol)
                if signal.signal_type != SignalType.HOLD:
                    signals[i] = signal
            except Exception as e:
                logger.debug("Strategy analysis failed at bar %d for %s: %s", i, symbol, e)

        return signals
