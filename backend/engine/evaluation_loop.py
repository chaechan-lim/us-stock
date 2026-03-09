"""Trading engine evaluation loop.

Periodically evaluates strategies against market data,
generates combined signals, and executes trades via OrderManager.

Uses per-stock adaptive weights:
1. StockClassifier categorizes each stock (growth/stable/cyclical/volatile)
2. AdaptiveWeightManager blends category weights + rolling performance
3. SignalCombiner uses per-stock weights instead of global market weights
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

from exchange.base import ExchangeAdapter
from data.market_data_service import MarketDataService
from data.indicator_service import IndicatorService
from strategies.registry import StrategyRegistry
from strategies.combiner import SignalCombiner
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from engine.stock_classifier import StockClassifier
from engine.adaptive_weights import AdaptiveWeightManager
from analytics.factor_model import MultiFactorModel, FactorScores
from analytics.signal_quality import SignalQualityTracker
from services.exchange_resolver import ExchangeResolver
from core.enums import SignalType

if TYPE_CHECKING:
    from agents.risk_assessment import RiskAssessmentAgent

logger = logging.getLogger(__name__)


class EvaluationLoop:
    """Main trading loop: data -> strategy -> signal -> order."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        market_data: MarketDataService,
        indicator_svc: IndicatorService,
        registry: StrategyRegistry,
        combiner: SignalCombiner,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        watchlist: list[str] | None = None,
        market_state: str = "uptrend",
        interval_sec: int = 300,
        adaptive_weights: AdaptiveWeightManager | None = None,
        factor_model: MultiFactorModel | None = None,
        signal_quality: SignalQualityTracker | None = None,
        risk_agent: RiskAssessmentAgent | None = None,
        exchange_resolver: ExchangeResolver | None = None,
    ):
        self._adapter = adapter
        self._market_data = market_data
        self._indicator_svc = indicator_svc
        self._registry = registry
        self._combiner = combiner
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._watchlist = watchlist or []
        self._market_state = market_state
        self._interval_sec = interval_sec
        self._running = False
        self._classifier = StockClassifier()
        self._adaptive = adaptive_weights or AdaptiveWeightManager()
        self._factor_model = factor_model or MultiFactorModel()
        self._signal_quality = signal_quality or SignalQualityTracker()
        self._risk_agent = risk_agent
        self._exchange_resolver = exchange_resolver or ExchangeResolver()
        self._factor_scores: dict[str, FactorScores] = {}
        self._last_classify_time: dict[str, float] = {}
        self._reclassify_interval = 86400  # re-classify every 24h
        self._last_factor_update: float = 0.0
        self._factor_update_interval = 3600  # update factor scores hourly

    @property
    def running(self) -> bool:
        return self._running

    def set_watchlist(self, symbols: list[str]) -> None:
        self._watchlist = symbols

    def set_market_state(self, state: str) -> None:
        self._market_state = state

    async def start(self) -> None:
        """Start the evaluation loop."""
        self._running = True
        logger.info("Evaluation loop started (interval=%ds)", self._interval_sec)

        while self._running:
            try:
                await self._evaluate_all()
            except Exception as e:
                logger.error("Evaluation loop error: %s", e)

            await asyncio.sleep(self._interval_sec)

    async def stop(self) -> None:
        """Stop the evaluation loop."""
        self._running = False
        logger.info("Evaluation loop stopped")

    async def evaluate_symbol(self, symbol: str) -> None:
        """Evaluate a single symbol through all strategies with per-stock weights."""
        try:
            # Fetch OHLCV data
            df = await self._market_data.get_ohlcv(symbol, limit=250)
            if df.empty:
                return

            # Classify stock if needed (every reclassify_interval)
            self._maybe_classify(symbol, df)

            # Add indicators
            df = self._indicator_svc.add_all_indicators(df)

            # Run all enabled strategies
            strategies = self._registry.get_enabled()
            signals = []
            for strategy in strategies:
                try:
                    signal = await strategy.analyze(df, symbol)
                    signals.append(signal)
                except Exception as e:
                    logger.debug("Strategy %s failed on %s: %s", strategy.name, symbol, e)

            # Get per-stock blended weights
            market_weights = self._registry.get_profile_weights(self._market_state)
            weights = self._adaptive.get_weights(symbol, market_weights)

            # Combine signals with per-stock weights
            combined = self._combiner.combine(signals, weights)

            # Log weight selection
            category = self._adaptive.get_category(symbol)
            if category:
                logger.debug(
                    "%s [%s] signal=%s conf=%.2f",
                    symbol, category.value, combined.signal_type.value,
                    combined.confidence,
                )

            # Execute
            await self._execute_signal(combined, symbol, df)

        except Exception as e:
            logger.error("Failed to evaluate %s: %s", symbol, e)

    def _maybe_classify(self, symbol: str, df: pd.DataFrame) -> None:
        """Classify stock if not yet classified or stale."""
        now = time.time()
        last = self._last_classify_time.get(symbol, 0)
        if now - last < self._reclassify_interval:
            return

        profile = self._classifier.classify(df, symbol)
        self._adaptive.set_category(symbol, profile.category)
        self._last_classify_time[symbol] = now
        logger.info(
            "Classified %s as %s (vol=%.2f, mom=%.2f)",
            symbol, profile.category.value, profile.volatility, profile.momentum_score,
        )

    async def _evaluate_all(self) -> None:
        """Evaluate all symbols in watchlist."""
        # Update factor scores periodically
        now = time.time()
        if now - self._last_factor_update > self._factor_update_interval:
            await self._update_factor_scores()
            self._last_factor_update = now

        for symbol in self._watchlist:
            await self.evaluate_symbol(symbol)

    async def _update_factor_scores(self) -> None:
        """Compute cross-sectional factor scores for the watchlist."""
        price_data = {}
        for symbol in self._watchlist:
            try:
                df = await self._market_data.get_ohlcv(symbol, limit=260)
                if not df.empty:
                    price_data[symbol] = df
            except Exception:
                pass

        if len(price_data) < 3:
            return

        try:
            scores = self._factor_model.score_universe(price_data)
            self._factor_scores = {s.symbol: s for s in scores}
            logger.info(
                "Factor scores updated: %d stocks, top=%s",
                len(scores),
                [f"{s.symbol}({s.composite:+.2f})" for s in scores[:3]],
            )
        except Exception as e:
            logger.warning("Factor score update failed: %s", e)

    async def _execute_signal(
        self, signal, symbol: str, df: pd.DataFrame
    ) -> None:
        """Execute a combined signal with Kelly-enhanced position sizing."""
        if signal.signal_type == SignalType.HOLD:
            return

        price = float(df.iloc[-1]["close"])

        if signal.signal_type == SignalType.BUY:
            # Skip if there's already a pending buy order for this symbol
            if self._order_manager.has_pending_order(symbol, "BUY"):
                logger.debug("Skipping BUY for %s: pending order exists", symbol)
                return

            balance = await self._market_data.get_balance()
            positions = await self._market_data.get_positions()

            # Get factor score for this stock
            factor = self._factor_scores.get(symbol)
            factor_score = factor.composite if factor else 0.0

            # Get strategy quality metrics for Kelly inputs
            strategy_name = signal.strategy_name
            metrics = self._signal_quality.get_metrics(strategy_name)
            win_rate, avg_win, avg_loss = metrics.kelly_inputs

            # Use Kelly-enhanced sizing
            sizing = self._risk_manager.calculate_kelly_position_size(
                symbol=symbol,
                price=price,
                portfolio_value=balance.total,
                cash_available=balance.available,
                current_positions=len(positions),
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                signal_confidence=signal.confidence,
                factor_score=factor_score,
            )

            if not sizing.allowed:
                logger.info(
                    "Buy rejected for %s: %s", symbol, sizing.reason,
                )
                return

            # AI pre-trade risk check (non-blocking: failures default to approved)
            if self._risk_agent:
                try:
                    risk_check = await self._risk_agent.assess_pre_trade(
                        symbol=symbol,
                        proposed_size=sizing.allocation_usd,
                        current_positions=[
                            {"symbol": p.symbol, "value": p.current_price * p.quantity}
                            for p in positions
                        ],
                        portfolio_summary={
                            "total_value": balance.total,
                            "cash": balance.available,
                            "positions": len(positions),
                        },
                    )
                    if not risk_check.get("approved", True):
                        logger.info(
                            "Buy blocked by risk agent for %s: %s",
                            symbol, risk_check.get("reason", ""),
                        )
                        return
                except Exception as e:
                    logger.debug("Risk agent pre-trade check error: %s", e)

            exchange = self._exchange_resolver.resolve(symbol)
            await self._order_manager.place_buy(
                symbol=symbol,
                price=price,
                portfolio_value=balance.total,
                cash_available=balance.available,
                current_positions=len(positions),
                strategy_name=strategy_name,
                sizing_override=sizing,
                exchange=exchange,
            )

        elif signal.signal_type == SignalType.SELL:
            positions = await self._market_data.get_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos and pos.quantity > 0:
                exchange = self._exchange_resolver.resolve(symbol)
                await self._order_manager.place_sell(
                    symbol=symbol,
                    quantity=int(pos.quantity),
                    price=price,
                    strategy_name=signal.strategy_name,
                    exchange=exchange,
                )

    def record_trade_result(
        self, strategy: str, symbol: str, return_pct: float,
    ) -> None:
        """Record a trade result for signal quality tracking."""
        self._signal_quality.record_trade(strategy, symbol, return_pct)

    @property
    def factor_scores(self) -> dict[str, FactorScores]:
        return dict(self._factor_scores)

    @property
    def signal_quality(self) -> SignalQualityTracker:
        return self._signal_quality
