"""Full Pipeline Backtest Engine.

Simulates the complete live trading pipeline on historical data:
  1. SPY-based market state detection (regime-aware strategy weights)
  2. Rolling universe screening via IndicatorScreener
  3. Per-stock classification + adaptive weight blending
  4. 14 strategies → SignalCombiner (group consensus)
  5. Kelly position sizing with factor scores
  6. ATR-based dynamic SL/TP per stock
  7. Portfolio-level constraints (max positions, exposure, daily loss)

Usage:
    config = PipelineConfig(universe=["AAPL", "MSFT", ...])
    engine = FullPipelineBacktest(config)
    result = await engine.run(period="3y")
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestDataLoader
from backtest.metrics import MetricsCalculator, BacktestMetrics, Trade
from data.indicator_service import IndicatorService
from data.market_state import MarketStateDetector, MarketRegime
from scanner.indicator_screener import IndicatorScreener
from strategies.registry import StrategyRegistry
from strategies.combiner import SignalCombiner
from strategies.base import Signal
from engine.stock_classifier import StockClassifier
from engine.adaptive_weights import AdaptiveWeightManager
from engine.risk_manager import RiskManager, RiskParams
from analytics.factor_model import MultiFactorModel
from analytics.position_sizing import KellyPositionSizer
from analytics.signal_quality import SignalQualityTracker
from strategies.config_loader import StrategyConfigLoader
from core.enums import SignalType

logger = logging.getLogger(__name__)

# Default broad universe (S&P 100 subset + growth/value mix)
DEFAULT_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    # Tech / Semis
    "AMD", "INTC", "QCOM", "AMAT", "MU", "CRM", "ADBE", "ORCL",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "C",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT",
    # Consumer
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "PG", "KO",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "DE",
    # Other
    "DIS", "NFLX", "PYPL", "V", "MA", "BRK-B",
]


@dataclass
class PipelineConfig:
    """Configuration for full pipeline backtest."""
    # Universe
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    spy_symbol: str = "SPY"

    # Simulation
    initial_equity: float = 100_000.0
    slippage_pct: float = 0.05  # 0.05%
    commission_per_order: float = 0.0

    # Screening
    screen_interval: int = 20  # Re-screen every N trading days
    max_watchlist: int = 30  # Max symbols in active watchlist
    min_screen_grade: str = "B"

    # Risk (mirrors RiskParams)
    max_position_pct: float = 0.10
    max_positions: int = 20
    max_exposure_pct: float = 0.90
    daily_loss_limit_pct: float = 0.03

    # SL/TP
    dynamic_sl_tp: bool = True
    default_stop_loss_pct: float = 0.08
    default_take_profit_pct: float = 0.20
    trailing_activation_pct: float = 0.05
    trailing_trail_pct: float = 0.03

    # Protective sells
    enable_regime_sells: bool = True  # Sell losing positions on regime deterioration

    # Signal combiner
    min_active_ratio: float = 0.15  # Min fraction of strategies that must be active
    min_confidence: float = 0.50  # Min combined confidence to generate BUY/SELL

    # Kelly sizing
    kelly_fraction: float = 0.25  # Fractional Kelly (0.25 = quarter Kelly)
    confidence_exponent: float = 2.0  # Confidence scaling power (lower = bigger positions)
    min_position_pct: float = 0.02  # Minimum position size

    # Momentum factor tilt
    enable_momentum_tilt: bool = False  # Pass momentum z-scores to Kelly sizer
    momentum_update_interval: int = 20  # Recompute momentum every N days

    # Strategy quality amplification
    enable_quality_amplification: bool = False  # Boost weights of winning strategies
    quality_blend_alpha: float = 0.3  # 30% quality-based, 70% original weights
    min_trades_for_quality: int = 30  # Min trades before quality weights activate

    # Strategy gating
    enable_strategy_gating: bool = False  # Disable strategies with no edge

    # Re-entry after stop loss
    recovery_watch_days: int = 20  # Keep stopped-out symbols in eval set for N days

    # Strategy config path
    strategy_config_path: str | None = None


@dataclass
class _Position:
    """Internal position tracking for backtest."""
    symbol: str
    quantity: int
    avg_price: float
    entry_date: str
    strategy_name: str
    highest_price: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class DailySnapshot:
    """Daily portfolio state for logging."""
    date: str
    equity: float
    cash: float
    n_positions: int
    regime: str
    n_watchlist: int


@dataclass
class PipelineResult:
    """Result of full pipeline backtest."""
    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: pd.Series
    daily_snapshots: list[DailySnapshot]
    config: PipelineConfig
    strategy_stats: dict[str, dict]  # per-strategy trade counts

    def summary(self) -> str:
        m = self.metrics
        status = "PASS" if m.passes_minimum() else "FAIL"
        lines = [
            f"[{status}] Full Pipeline Backtest",
            f"  Period: {m.start_date} ~ {m.end_date} ({m.trading_days} days)",
            f"  Return: {m.total_return_pct:.1f}% | CAGR: {m.cagr:.1%}",
            f"  Sharpe: {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f}",
            f"  MDD: {m.max_drawdown_pct:.1f}% ({m.max_drawdown_days} days)",
            f"  Trades: {m.total_trades} | Win Rate: {m.win_rate:.1f}%",
            f"  Profit Factor: {m.profit_factor:.2f}",
            f"  Avg Hold: {m.avg_holding_days:.0f} days",
            f"  Final Equity: ${m.final_equity:,.0f}",
            f"  Benchmark (SPY): {m.benchmark_return_pct:.1f}% | Alpha: {m.alpha:.1f}%",
        ]
        if self.strategy_stats:
            lines.append("  Strategy breakdown:")
            for name, stats in sorted(
                self.strategy_stats.items(),
                key=lambda x: x[1]["trades"], reverse=True,
            ):
                if stats["trades"] > 0:
                    lines.append(
                        f"    {name}: {stats['trades']} trades, "
                        f"WR={stats['win_rate']:.0f}%, "
                        f"PnL=${stats['pnl']:+,.0f}"
                    )
        return "\n".join(lines)


class FullPipelineBacktest:
    """Simulates the complete live trading pipeline on historical data."""

    def __init__(self, config: PipelineConfig | None = None):
        self._config = config or PipelineConfig()
        self._data_loader = BacktestDataLoader()
        self._indicator_svc = IndicatorService()
        self._screener = IndicatorScreener(min_grade=self._config.min_screen_grade)
        self._market_state_detector = MarketStateDetector()
        self._classifier = StockClassifier()
        self._adaptive = AdaptiveWeightManager()
        self._factor_model = MultiFactorModel()
        self._signal_quality = SignalQualityTracker()
        self._risk_manager = RiskManager(RiskParams(
            max_position_pct=self._config.max_position_pct,
            max_total_exposure_pct=self._config.max_exposure_pct,
            max_positions=self._config.max_positions,
            daily_loss_limit_pct=self._config.daily_loss_limit_pct,
            default_stop_loss_pct=self._config.default_stop_loss_pct,
            default_take_profit_pct=self._config.default_take_profit_pct,
        ))
        # Override Kelly sizer params for backtest tuning
        self._risk_manager._kelly = KellyPositionSizer(
            kelly_fraction=self._config.kelly_fraction,
            max_position_pct=self._config.max_position_pct,
            min_position_pct=self._config.min_position_pct,
            confidence_exponent=self._config.confidence_exponent,
        )

        # Initialize strategy registry and combiner
        config_loader = StrategyConfigLoader(self._config.strategy_config_path)
        self._registry = StrategyRegistry(config_loader=config_loader)
        consensus_cfg = config_loader.get_consensus_config()
        self._combiner = SignalCombiner(
            consensus_config=consensus_cfg,
            min_active_ratio=self._config.min_active_ratio,
        )

        # Portfolio state
        self._cash: float = 0.0
        self._positions: dict[str, _Position] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []
        self._equity_dates: list = []
        self._daily_snapshots: list[DailySnapshot] = []
        self._watchlist: list[str] = []
        self._prev_regime: str = "uptrend"

        # Momentum factor scores (updated periodically)
        self._factor_scores: dict[str, float] = {}  # symbol → composite z-score
        self._last_factor_update: int = -9999
        self._gated_strategies: set[str] = set()

        # Recovery watch: recently sold symbols stay in eval set
        # {symbol: day_count_when_sold}
        self._recovery_watch: dict[str, int] = {}
        self._day_count: int = 0

    async def run(
        self,
        period: str = "3y",
        start: str | None = None,
        end: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline backtest.

        Args:
            period: Data period (e.g. '3y', '5y')
            start: Start date YYYY-MM-DD (overrides period)
            end: End date YYYY-MM-DD

        Returns:
            PipelineResult with metrics, trades, equity curve
        """
        cfg = self._config

        # 1. Load all data upfront
        logger.info(
            "Loading data for %d universe symbols + SPY...",
            len(cfg.universe),
        )
        all_symbols = list(dict.fromkeys([cfg.spy_symbol] + cfg.universe))
        all_data = self._data_loader.load_multiple(
            all_symbols, period=period,
        )

        if cfg.spy_symbol not in all_data:
            raise ValueError(f"Failed to load SPY data (required for market state)")

        spy_data = all_data[cfg.spy_symbol]
        stock_data = {s: d for s, d in all_data.items() if s != cfg.spy_symbol}

        if not stock_data:
            raise ValueError("No stock data loaded")

        logger.info(
            "Data loaded: %d stocks, %d ~ %d bars",
            len(stock_data),
            min(len(d.df) for d in stock_data.values()),
            max(len(d.df) for d in stock_data.values()),
        )

        # 2. Find common date range
        spy_dates = spy_data.df.index
        common_start = spy_dates[250]  # Need 200+ bars for SMA200 + warmup
        common_end = spy_dates[-1]

        # 3. Initialize state
        self._cash = cfg.initial_equity
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._equity_dates.clear()
        self._daily_snapshots.clear()
        self._watchlist.clear()
        self._risk_manager.reset_daily()
        self._factor_scores.clear()
        self._last_factor_update = -9999
        self._gated_strategies.clear()
        self._recovery_watch.clear()

        # Pre-classify all stocks once
        for symbol, data in stock_data.items():
            if len(data.df) >= 60:
                profile = self._classifier.classify(data.df, symbol)
                self._adaptive.set_category(symbol, profile.category)

        # 4. Day-by-day simulation
        day_count = 0
        self._day_count = 0
        last_screen_day = -cfg.screen_interval  # Force screen on first day

        for date_idx in range(len(spy_dates)):
            date = spy_dates[date_idx]
            if date < common_start:
                continue

            day_count += 1
            self._day_count = day_count

            # Daily reset of PnL tracking
            if day_count % 1 == 0:
                # Actually reset only on real new days, but for daily bars this is every bar
                pass  # We reset in _end_of_day

            # 4a. Detect market state from SPY
            spy_window = spy_data.df.iloc[:date_idx + 1]
            market_state = self._market_state_detector.detect(spy_window)
            regime_str = market_state.regime.value
            self._risk_manager.set_eval_regime(regime_str)

            # Map regime to profile name
            profile_name = regime_str
            if profile_name == "strong_uptrend":
                profile_name = "strong_uptrend"

            # 4b. Screening: refresh watchlist periodically
            if day_count - last_screen_day >= cfg.screen_interval:
                self._watchlist = self._screen_universe(
                    stock_data, date_idx, cfg.max_watchlist,
                )
                last_screen_day = day_count
                logger.debug(
                    "Day %d: Screened %d → watchlist %d symbols",
                    day_count, len(stock_data), len(self._watchlist),
                )

            # 4b2. Update momentum factor scores periodically
            if cfg.enable_momentum_tilt:
                if day_count - self._last_factor_update >= cfg.momentum_update_interval:
                    self._update_factor_scores(stock_data, date_idx)
                    self._last_factor_update = day_count

            # 4b3. Update strategy gating
            if cfg.enable_strategy_gating:
                self._gated_strategies = set(
                    self._signal_quality.get_gated_strategies()
                )

            # Expire old recovery watch entries
            expired = [
                s for s, sold_day in self._recovery_watch.items()
                if day_count - sold_day > cfg.recovery_watch_days
            ]
            for s in expired:
                del self._recovery_watch[s]

            # Merge watchlist + held positions + recovery watch
            held = set(self._positions.keys())
            recovery = set(self._recovery_watch.keys()) - held
            eval_symbols = list(dict.fromkeys(
                self._watchlist + sorted(held) + sorted(recovery)
            ))

            # 4c. Regime-change protective sells
            if cfg.enable_regime_sells and self._positions:
                self._check_regime_sells(
                    stock_data, date_idx, date, regime_str,
                )
            self._prev_regime = regime_str

            # 4d. Check SL/TP/trailing stop on existing positions
            self._check_risk_exits(stock_data, date_idx, date)

            # 4e. Evaluate signals and execute
            buy_candidates: list[tuple[float, str, Signal]] = []

            for symbol in eval_symbols:
                if symbol not in stock_data:
                    continue
                sdata = stock_data[symbol]
                if date_idx >= len(sdata.df):
                    continue

                df_window = sdata.df.iloc[:date_idx + 1]
                if len(df_window) < 50:
                    continue

                # Run all strategies
                strategies = self._registry.get_enabled()
                signals = []
                for strategy in strategies:
                    try:
                        signal = await strategy.analyze(df_window, symbol)
                        signals.append(signal)
                    except Exception:
                        pass

                # Get weights: market-state profile + stock category blending
                market_weights = self._registry.get_profile_weights(profile_name)
                weights = self._adaptive.get_weights(symbol, market_weights)

                # Amplify weights of winning strategies
                if cfg.enable_quality_amplification:
                    weights = self._get_quality_adjusted_weights(weights)

                # Soft-gate losing strategies (halve their weight)
                if cfg.enable_strategy_gating and self._gated_strategies:
                    for gated in self._gated_strategies:
                        if gated in weights:
                            weights[gated] *= 0.5

                # Combine signals
                combined = self._combiner.combine(
                    signals, weights, min_confidence=cfg.min_confidence,
                )

                # Execute SELLs immediately
                if combined.signal_type == SignalType.SELL:
                    self._execute_sell(symbol, stock_data, date_idx, date, combined)
                elif combined.signal_type == SignalType.BUY:
                    buy_candidates.append((combined.confidence, symbol, combined))

            # Execute BUYs ranked by confidence (highest first)
            if buy_candidates:
                buy_candidates.sort(key=lambda x: x[0], reverse=True)
                for _conf, symbol, combined in buy_candidates:
                    self._execute_buy(
                        symbol, stock_data, date_idx, date, combined,
                        market_state.regime,
                    )

            # 4f. Update equity and snapshot
            equity = self._calculate_equity(stock_data, date_idx)
            self._equity_curve.append(equity)
            self._equity_dates.append(date)
            self._daily_snapshots.append(DailySnapshot(
                date=str(date),
                equity=round(equity, 2),
                cash=round(self._cash, 2),
                n_positions=len(self._positions),
                regime=regime_str,
                n_watchlist=len(self._watchlist),
            ))

            # End of day: reset daily PnL
            self._risk_manager.reset_daily()

        # 5. Close all remaining positions at last price
        self._close_all_positions(stock_data, len(spy_dates) - 1, spy_dates[-1])

        # 6. Calculate metrics
        equity_series = pd.Series(self._equity_curve, index=self._equity_dates)
        spy_returns = spy_data.df["close"].pct_change().dropna()
        # Align benchmark returns to our simulation period
        spy_returns = spy_returns.loc[
            (spy_returns.index >= common_start) & (spy_returns.index <= common_end)
        ]

        metrics = MetricsCalculator.calculate(
            equity_curve=equity_series,
            trades=self._trades,
            initial_equity=cfg.initial_equity,
            benchmark_returns=spy_returns,
        )

        # 7. Strategy breakdown
        strategy_stats = self._compute_strategy_stats()

        result = PipelineResult(
            metrics=metrics,
            trades=self._trades,
            equity_curve=equity_series,
            daily_snapshots=self._daily_snapshots,
            config=self._config,
            strategy_stats=strategy_stats,
        )

        logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Momentum factor scoring
    # ------------------------------------------------------------------

    def _update_factor_scores(
        self, stock_data: dict, date_idx: int,
    ) -> None:
        """Compute cross-sectional momentum z-scores for all stocks."""
        price_data = {}
        for symbol, data in stock_data.items():
            if date_idx < len(data.df):
                price_data[symbol] = data.df.iloc[:date_idx + 1]

        if len(price_data) < 3:
            return

        # Use factor model which computes momentum + z-scores
        scores = self._factor_model.score_universe(price_data)
        self._factor_scores = {s.symbol: s.composite for s in scores}

    def _get_quality_adjusted_weights(
        self, base_weights: dict[str, float],
    ) -> dict[str, float]:
        """Blend base weights with signal quality performance weights."""
        cfg = self._config
        quality_weights = self._signal_quality.get_strategy_weights()
        if not quality_weights:
            return base_weights

        # Check if enough trades have been recorded
        total_trades = sum(
            self._signal_quality.get_metrics(name).total_trades
            for name in quality_weights
        )
        if total_trades < cfg.min_trades_for_quality:
            return base_weights

        # Blend: (1-alpha)*base + alpha*quality
        alpha = cfg.quality_blend_alpha
        blended = {}
        all_keys = set(base_weights) | set(quality_weights)
        for key in all_keys:
            base = base_weights.get(key, 0.0)
            qual = quality_weights.get(key, 0.0)
            blended[key] = (1 - alpha) * base + alpha * qual

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def _screen_universe(
        self,
        stock_data: dict[str, object],
        date_idx: int,
        max_symbols: int,
    ) -> list[str]:
        """Screen universe and return top symbols by indicator score."""
        scores = []
        for symbol, data in stock_data.items():
            if date_idx >= len(data.df):
                continue
            df_window = data.df.iloc[:date_idx + 1]
            if len(df_window) < 50:
                continue
            try:
                score = self._screener.score(df_window, symbol)
                scores.append(score)
            except Exception:
                pass

        filtered = self._screener.filter_candidates(
            scores, max_candidates=max_symbols,
        )
        return [s.symbol for s in filtered]

    # ------------------------------------------------------------------
    # Risk exits (SL/TP/trailing stop)
    # ------------------------------------------------------------------

    def _check_risk_exits(
        self, stock_data: dict[str, object], date_idx: int, date,
    ) -> None:
        """Check SL/TP/trailing stop on all held positions."""
        for symbol in list(self._positions.keys()):
            if symbol not in stock_data:
                continue
            data = stock_data[symbol]
            if date_idx >= len(data.df):
                continue

            row = data.df.iloc[date_idx]
            pos = self._positions[symbol]
            price = float(row["close"])
            high = float(row["high"]) if "high" in row.index else price
            low = float(row["low"]) if "low" in row.index else price

            # Update highest price for trailing stop
            if high > pos.highest_price:
                pos.highest_price = high

            # Stop-loss
            sl_price = pos.avg_price * (1 - pos.stop_loss_pct)
            if low <= sl_price:
                self._close_position(symbol, sl_price, date, "stop_loss")
                continue

            # Take-profit
            tp_price = pos.avg_price * (1 + pos.take_profit_pct)
            if high >= tp_price:
                self._close_position(symbol, tp_price, date, "take_profit")
                continue

            # Trailing stop
            cfg = self._config
            if cfg.trailing_activation_pct > 0 and cfg.trailing_trail_pct > 0:
                gain = (pos.highest_price - pos.avg_price) / pos.avg_price
                if gain >= cfg.trailing_activation_pct:
                    trail_price = pos.highest_price * (1 - cfg.trailing_trail_pct)
                    if low <= trail_price:
                        self._close_position(
                            symbol, trail_price, date, "trailing_stop",
                        )

    # ------------------------------------------------------------------
    # Regime-change protective sells
    # ------------------------------------------------------------------

    def _check_regime_sells(
        self,
        stock_data: dict,
        date_idx: int,
        date,
        current_regime: str,
    ) -> None:
        """Sell losing positions when regime transitions to downtrend.

        Mirrors live evaluation_loop._check_protective_sells():
        when market moves from uptrend/strong_uptrend to downtrend,
        positions with negative PnL are closed to protect capital.
        """
        _BEARISH = {"downtrend"}

        regime_worsened = (
            current_regime in _BEARISH
            and self._prev_regime not in _BEARISH
        )
        if not regime_worsened:
            return

        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            data = stock_data.get(symbol)
            if not data or date_idx >= len(data.df):
                continue

            price = float(data.df.iloc[date_idx]["close"])
            pnl_pct = (price - pos.avg_price) / pos.avg_price

            if pnl_pct < 0:
                logger.debug(
                    "Regime sell %s: %s→%s, PnL=%.1f%%",
                    symbol, self._prev_regime, current_regime, pnl_pct * 100,
                )
                self._close_position(symbol, price, date, "regime_protect")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_buy(
        self,
        symbol: str,
        stock_data: dict,
        date_idx: int,
        date,
        signal: Signal,
        regime: MarketRegime,
    ) -> None:
        """Execute a buy order with portfolio-level risk checks."""
        if symbol in self._positions:
            return  # Already holding

        cfg = self._config
        data = stock_data[symbol]
        if date_idx >= len(data.df):
            return

        price = float(data.df.iloc[date_idx]["close"])
        if price <= 0:
            return

        equity = self._calculate_equity(stock_data, date_idx)

        # Use Kelly-enhanced sizing (falls back to fixed if no trade history)
        strategy_name = signal.strategy_name
        metrics = self._signal_quality.get_metrics(strategy_name)
        win_rate, avg_win, avg_loss = metrics.kelly_inputs

        # Get momentum factor score for this stock
        factor_score = self._factor_scores.get(symbol, 0.0)

        sizing = self._risk_manager.calculate_kelly_position_size(
            symbol=symbol,
            price=price,
            portfolio_value=equity,
            cash_available=self._cash,
            current_positions=len(self._positions),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            signal_confidence=signal.confidence,
            factor_score=factor_score,
        )

        if not sizing.allowed:
            return

        # Apply slippage
        exec_price = price * (1 + cfg.slippage_pct / 100)
        quantity = int(sizing.allocation_usd / exec_price)
        if quantity <= 0:
            return

        cost = quantity * exec_price + cfg.commission_per_order
        if cost > self._cash:
            return

        # Determine dynamic SL/TP
        if cfg.dynamic_sl_tp:
            atr_col = None
            row = data.df.iloc[date_idx]
            for col in ("atr", "ATRr_14"):
                if col in row.index and not pd.isna(row[col]):
                    atr_col = col
                    break
            if atr_col:
                atr_val = float(row[atr_col])
                sl_pct, tp_pct = self._risk_manager.calculate_dynamic_sl_tp(
                    price, atr_val,
                )
            else:
                sl_pct = cfg.default_stop_loss_pct
                tp_pct = cfg.default_take_profit_pct
        else:
            sl_pct = cfg.default_stop_loss_pct
            tp_pct = cfg.default_take_profit_pct

        # Execute
        self._cash -= cost
        self._positions[symbol] = _Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=exec_price,
            entry_date=str(date),
            strategy_name=strategy_name,
            highest_price=exec_price,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )

    def _execute_sell(
        self,
        symbol: str,
        stock_data: dict,
        date_idx: int,
        date,
        signal: Signal,
    ) -> None:
        """Execute a sell signal."""
        if symbol not in self._positions:
            return

        data = stock_data.get(symbol)
        if not data or date_idx >= len(data.df):
            return

        price = float(data.df.iloc[date_idx]["close"])
        self._close_position(symbol, price, date, "signal_sell")

    def _close_position(
        self, symbol: str, price: float, date, reason: str = "",
    ) -> None:
        """Close a position and record the trade."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        cfg = self._config
        exec_price = price * (1 - cfg.slippage_pct / 100)
        proceeds = pos.quantity * exec_price - cfg.commission_per_order
        self._cash += proceeds

        pnl = (exec_price - pos.avg_price) * pos.quantity
        pnl_pct = (exec_price - pos.avg_price) / pos.avg_price * 100

        try:
            entry = pd.Timestamp(pos.entry_date)
            exit_ = pd.Timestamp(str(date))
            holding_days = (exit_ - entry).days
        except Exception:
            holding_days = 0

        self._trades.append(Trade(
            symbol=symbol,
            side="SELL",
            entry_date=pos.entry_date,
            entry_price=pos.avg_price,
            exit_date=str(date),
            exit_price=exec_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            strategy_name=pos.strategy_name,
        ))

        # Update daily PnL for risk manager
        self._risk_manager.update_daily_pnl(pnl)

        # Record for signal quality tracking
        self._signal_quality.record_trade(
            pos.strategy_name, symbol, pnl_pct / 100,
        )

        del self._positions[symbol]

        # Add to recovery watch for re-entry evaluation
        if reason not in ("end_of_backtest",):
            self._recovery_watch[symbol] = self._day_count

    def _close_all_positions(
        self, stock_data: dict, date_idx: int, date,
    ) -> None:
        """Close all positions at end of backtest."""
        for symbol in list(self._positions.keys()):
            data = stock_data.get(symbol)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
            else:
                price = self._positions[symbol].avg_price
            self._close_position(symbol, price, date, "backtest_end")

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def _calculate_equity(
        self, stock_data: dict, date_idx: int,
    ) -> float:
        """Calculate total portfolio value (cash + positions)."""
        position_value = 0.0
        for symbol, pos in self._positions.items():
            data = stock_data.get(symbol)
            if data and date_idx < len(data.df):
                price = float(data.df.iloc[date_idx]["close"])
            else:
                price = pos.avg_price
            position_value += pos.quantity * price
        return self._cash + position_value

    # ------------------------------------------------------------------
    # Strategy stats
    # ------------------------------------------------------------------

    def _compute_strategy_stats(self) -> dict[str, dict]:
        """Compute per-strategy trade statistics."""
        stats: dict[str, dict] = {}
        for trade in self._trades:
            name = trade.strategy_name or "unknown"
            if name not in stats:
                stats[name] = {
                    "trades": 0, "wins": 0, "losses": 0,
                    "pnl": 0.0, "win_rate": 0.0,
                }
            s = stats[name]
            s["trades"] += 1
            s["pnl"] += trade.pnl
            if trade.pnl > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

        for s in stats.values():
            if s["trades"] > 0:
                s["win_rate"] = s["wins"] / s["trades"] * 100

        return stats
