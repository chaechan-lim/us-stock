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
from strategies.base import Signal
from strategies.combiner import SignalCombiner
from strategies.registry import StrategyRegistry
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
        position_tracker=None,
        market: str = "US",
        event_calendar=None,
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
        self._position_tracker = position_tracker
        self._market = market
        self._event_calendar = event_calendar
        self._other_market_data: MarketDataService | None = None
        self._exchange_rate: float = 1450.0  # USD/KRW default
        self._factor_scores: dict[str, FactorScores] = {}
        self._last_classify_time: dict[str, float] = {}
        self._reclassify_interval = 86400  # re-classify every 24h
        self._last_factor_update: float = 0.0
        self._factor_update_interval = 3600  # update factor scores hourly
        self._prev_market_state: str = market_state
        self._news_sentiment: dict[str, float] = {}  # symbol -> sentiment (-1 to +1)
        # Signal dedup: prevent re-executing the same daily signal within 24h
        self._last_signal: dict[str, tuple[str, float]] = {}  # symbol -> (signal_type, timestamp)
        # Recovery watch: recently sold symbols get re-evaluated for re-entry
        self._recovery_watch: dict[str, float] = {}  # {symbol: timestamp_when_sold}
        self._recovery_watch_secs = 7 * 86400  # 7 days
        # Daily buy counter (resets at midnight)
        self._daily_buy_count: int = 0
        self._daily_buy_date: str = ""
        # Recent signals buffer for frontend display (last N signals)
        from collections import deque

        self._recent_signals: deque[dict] = deque(maxlen=200)

    @property
    def running(self) -> bool:
        return self._running

    # ETF-only symbols that should NOT be traded by the stock combiner.
    # These are managed exclusively by the ETF Engine.
    _ETF_ONLY = frozenset(
        {
            # Leveraged / inverse
            "TQQQ",
            "SQQQ",
            "UPRO",
            "SPXU",
            "SOXL",
            "SOXS",
            "TECL",
            "TECS",
            "FAS",
            "ERX",
            "LABU",
            "SARK",
            # Volatility products (structural decay on long hold)
            "VXX",
            "UVXY",
            "SVXY",
            # Safe-haven / non-equity
            "SHY",
            "TLT",
            "GLD",
            "UUP",
            # Index ETFs (benchmark, not for active trading)
            "SPY",
            "QQQ",
            "SOXX",
            "ARKK",
        }
    )

    def set_other_market_data(self, other_md: MarketDataService) -> None:
        """Set the other market's data service for combined portfolio calculation."""
        self._other_market_data = other_md

    def set_exchange_rate(self, rate: float) -> None:
        """Update USD/KRW exchange rate for cross-market value conversion."""
        if rate > 0:
            self._exchange_rate = rate

    def set_watchlist(self, symbols: list[str]) -> None:
        self._watchlist = [s for s in symbols if s not in self._ETF_ONLY]

    def set_market_state(self, state: str) -> None:
        self._prev_market_state = self._market_state
        self._market_state = state
        # Propagate to risk manager for regime-adaptive sizing
        self._risk_manager.set_eval_regime(state)

    def update_news_sentiment(self, sentiments: dict[str, float]) -> None:
        """Update per-symbol news sentiment scores.

        Args:
            sentiments: {symbol: sentiment_score} where score is -1.0 to +1.0
        """
        self._news_sentiment.update(sentiments)

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

    async def evaluate_symbol(self, symbol: str, is_held: bool = False) -> None:
        """Evaluate a single symbol through all strategies with per-stock weights.

        Args:
            symbol: Stock ticker symbol.
            is_held: If True, remap BUY signals to HOLD before combining
                so that the combiner evaluates SELL vs HOLD (exit decision).
        """
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
            signals: list[Signal] = []
            for strategy in strategies:
                try:
                    signal = await strategy.analyze(df, symbol)
                    signals.append(signal)
                except Exception as e:
                    logger.debug("Strategy %s failed on %s: %s", strategy.name, symbol, e)

            # Get per-stock blended weights
            market_weights = self._registry.get_profile_weights(self._market_state)
            weights = self._adaptive.get_weights(symbol, market_weights)

            # For held positions, remap BUY→HOLD for exit evaluation
            if is_held:
                signals = [
                    Signal(
                        signal_type=SignalType.HOLD,
                        confidence=s.confidence,
                        strategy_name=s.strategy_name,
                        reason=s.reason,
                        indicators=s.indicators,
                    )
                    if s.signal_type == SignalType.BUY
                    else s
                    for s in signals
                ]

            # Combine signals with per-stock weights
            combined = self._combiner.combine(
                signals,
                weights,
                min_active_ratio=0.15 if is_held else None,
            )

            # Log weight selection
            category = self._adaptive.get_category(symbol)
            if category:
                logger.debug(
                    "%s [%s] signal=%s conf=%.2f (held=%s)",
                    symbol,
                    category.value,
                    combined.signal_type.value,
                    combined.confidence,
                    is_held,
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
            symbol,
            profile.category.value,
            profile.volatility,
            profile.momentum_score,
        )

    async def _evaluate_all(self) -> None:
        """Evaluate all symbols: collect signals, then execute ranked by confidence.

        SELLs execute immediately. BUYs are ranked by confidence so
        the highest-conviction signals get filled first when cash is limited.
        """
        # Update factor scores periodically
        now = time.time()
        if now - self._last_factor_update > self._factor_update_interval:
            await self._update_factor_scores()
            self._last_factor_update = now

        # Expire old recovery watch entries
        expired = [
            s for s, ts in self._recovery_watch.items() if now - ts > self._recovery_watch_secs
        ]
        for s in expired:
            del self._recovery_watch[s]

        # Merge watchlist + held positions + recovery watch
        held = set(self._position_tracker.tracked_symbols) if self._position_tracker else set()

        # Defense-in-depth: also include exchange positions so held stocks
        # get SELL evaluations even when position_tracker is empty (e.g.
        # after restart before restore_from_exchange completes).
        try:
            exchange_positions = await self._market_data.get_positions()
            exchange_held = {p.symbol for p in exchange_positions if p.quantity > 0}
            held = held | exchange_held
        except Exception:
            pass

        recovery = set(self._recovery_watch.keys()) - held
        eval_symbols = list(dict.fromkeys(self._watchlist + sorted(held) + sorted(recovery)))

        # Phase 0: Regime-change and sentiment-based protective sells
        if held and self._position_tracker:
            await self._check_protective_sells(held)

        # Phase 1: Collect all signals (no execution yet)
        buy_candidates: list[tuple[float, str, object, pd.DataFrame]] = []

        for symbol in eval_symbols:
            try:
                df = await self._market_data.get_ohlcv(symbol, limit=250)
                if df.empty:
                    continue

                self._maybe_classify(symbol, df)
                df = self._indicator_svc.add_all_indicators(df)

                strategies = self._registry.get_enabled()
                signals = []
                for strategy in strategies:
                    try:
                        signal = await asyncio.wait_for(
                            strategy.analyze(df, symbol),
                            timeout=10.0,
                        )
                        signals.append(signal)
                    except asyncio.TimeoutError:
                        logger.warning("Strategy %s timed out on %s", strategy.name, symbol)
                    except Exception as e:
                        logger.debug("Strategy %s failed on %s: %s", strategy.name, symbol, e)

                market_weights = self._registry.get_profile_weights(self._market_state)
                weights = self._adaptive.get_weights(symbol, market_weights)

                # For held positions, remap BUY signals to HOLD before combining.
                # Rationale: we already own the stock, so BUY signals (meaning
                # "this stock looks good") are not actionable — the relevant
                # question is SELL vs HOLD (exit or keep).  Without remapping,
                # BUY votes from trend-following strategies drown out SELL votes,
                # making strategy-based exits nearly impossible.
                is_held = symbol in held
                if is_held:
                    n_remapped = sum(
                        1 for s in signals if s.signal_type == SignalType.BUY
                    )
                    if n_remapped:
                        signals = [
                            Signal(
                                signal_type=SignalType.HOLD,
                                confidence=s.confidence,
                                strategy_name=s.strategy_name,
                                reason=s.reason,
                                indicators=s.indicators,
                            )
                            if s.signal_type == SignalType.BUY
                            else s
                            for s in signals
                        ]
                        logger.debug(
                            "Held %s: remapped %d BUY→HOLD for exit evaluation",
                            symbol,
                            n_remapped,
                        )

                combined = self._combiner.combine(
                    signals,
                    weights,
                    min_active_ratio=0.15 if is_held else None,
                )

                category = self._adaptive.get_category(symbol)
                if category:
                    logger.debug(
                        "%s [%s] signal=%s conf=%.2f (held=%s)",
                        symbol,
                        category.value,
                        combined.signal_type.value,
                        combined.confidence,
                        is_held,
                    )

                # Log signal for frontend visibility
                if combined.signal_type in (SignalType.BUY, SignalType.SELL):
                    from datetime import datetime, timezone

                    self._recent_signals.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                            "signal": combined.signal_type.value,
                            "confidence": round(combined.confidence, 3),
                            "strategy": combined.strategy_name,
                            "market_state": self._market_state,
                            "market": self._market,
                        }
                    )

                # SELLs execute immediately (no competition for cash)
                if combined.signal_type == SignalType.SELL:
                    await self._execute_signal(combined, symbol, df)
                elif combined.signal_type == SignalType.BUY:
                    buy_candidates.append((combined.confidence, symbol, combined, df))

            except Exception as e:
                logger.error("Failed to evaluate %s: %s", symbol, e)

        # Phase 2: Execute BUYs ranked by confidence (highest first)
        if buy_candidates:
            buy_candidates.sort(key=lambda x: x[0], reverse=True)
            logger.info(
                "Buy candidates ranked: %s",
                [(s, f"{c:.2f}") for c, s, _, _ in buy_candidates[:10]],
            )
            for _conf, symbol, combined, df in buy_candidates:
                await self._execute_signal(combined, symbol, df)

    async def _check_protective_sells(self, held: set[str]) -> None:
        """Sell positions on regime deterioration or negative news sentiment.

        Regime sell: when market transitions to downtrend, sell losing positions
        to protect capital. Winning positions are kept (trailing stop will handle).

        Sentiment sell: when a held stock has strongly negative news sentiment
        (score <= -0.70), sell if held for at least 4 hours (avoid churn).
        """
        _BEARISH_REGIMES = {"downtrend"}
        regime_worsened = (
            self._market_state in _BEARISH_REGIMES
            and self._prev_market_state not in _BEARISH_REGIMES
        )

        if not regime_worsened and not self._news_sentiment:
            return

        positions = await self._market_data.get_positions()
        position_map = {p.symbol: p for p in positions}

        for symbol in list(held):
            pos = position_map.get(symbol)
            if not pos or pos.quantity <= 0:
                continue

            sell_reason = None

            # 1. Regime deterioration: sell losing positions
            if regime_worsened and pos.avg_price > 0:
                pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
                if pnl_pct < 0:
                    sell_reason = f"regime_protect(pnl={pnl_pct:.1%})"
                    logger.warning(
                        "Regime sell %s: %s→%s, PnL=%.1f%%",
                        symbol,
                        self._prev_market_state,
                        self._market_state,
                        pnl_pct * 100,
                    )

            # 2. Strongly negative news sentiment (threshold -0.70, min hold 4h)
            sentiment = self._news_sentiment.get(symbol, 0.0)
            if sentiment <= -0.70:
                # Don't sell if position was opened recently (avoid buy-sell churn)
                hold_secs = float("inf")
                if self._position_tracker:
                    try:
                        tracked = self._position_tracker._tracked.get(symbol)
                        if tracked:
                            hold_secs = time.time() - tracked.tracked_at
                    except AttributeError:
                        pass  # mock or missing _tracked
                min_hold = 4 * 3600  # 4 hours minimum
                if hold_secs >= min_hold:
                    sell_reason = f"negative_sentiment({sentiment:.2f})"
                    logger.warning(
                        "Sentiment sell %s: score=%.2f (held %.1fh)",
                        symbol,
                        sentiment,
                        hold_secs / 3600,
                    )
                else:
                    logger.info(
                        "Sentiment sell skipped %s: score=%.2f but held only %.1fh (min 4h)",
                        symbol,
                        sentiment,
                        hold_secs / 3600,
                    )

            if sell_reason:
                exchange = (
                    "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
                )
                # Look up original buy strategy from position tracker
                orig_strategy = ""
                if self._position_tracker:
                    orig_strategy = self._position_tracker.get_buy_strategy(symbol)
                sell_order = await self._order_manager.place_sell(
                    symbol=symbol,
                    quantity=int(pos.quantity),
                    price=pos.current_price,
                    strategy_name=sell_reason,
                    exchange=exchange,
                    entry_price=pos.avg_price,
                    buy_strategy=orig_strategy,
                )
                if sell_order and self._position_tracker:
                    self._position_tracker.untrack(symbol)
                    self._recovery_watch[symbol] = time.time()

        # Clear processed sentiments to avoid re-selling
        for symbol in held:
            self._news_sentiment.pop(symbol, None)

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

    async def _get_combined_portfolio_value(self, own_balance_total: float) -> float | None:
        """Compute combined portfolio value across both markets.

        For integrated margin (통합증거금) accounts, returns the total value
        of both markets converted to this market's currency. Returns None
        if other market data is not available.

        Avoids double-counting the shared deposit by adding only the other
        market's position value (total - available = invested portion).
        """
        if not self._other_market_data:
            return None
        try:
            other_balance = await self._other_market_data.get_balance()
            # Only add position value from other market (not deposit, to avoid double-count)
            other_positions = max(0, other_balance.total - other_balance.available)
            if self._market == "US":
                # Own is USD, other KR positions are in KRW → convert to USD
                other_in_own = other_positions / self._exchange_rate
            else:
                # Own is KRW, other US positions are in USD → convert to KRW
                other_in_own = other_positions * self._exchange_rate
            combined = own_balance_total + other_in_own
            return combined
        except Exception as e:
            logger.debug("Failed to fetch other market balance for combined total: %s", e)
            return None

    async def _execute_signal(self, signal, symbol: str, df: pd.DataFrame) -> None:
        """Execute a combined signal with Kelly-enhanced position sizing."""
        if signal.signal_type == SignalType.HOLD:
            return

        # Use real-time price from KIS API for order placement;
        # fall back to OHLCV close if unavailable (e.g. paper mode)
        try:
            exchange = "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
            price = await self._market_data.get_price(symbol, exchange)
        except Exception:
            price = float(df.iloc[-1]["close"])

        if signal.signal_type == SignalType.BUY:
            # Daily buy budget with dynamic confidence escalation
            # As more slots are used, require higher confidence to preserve
            # remaining slots for stronger opportunities later in the day.
            from datetime import date as _date

            today = _date.today().isoformat()
            if self._daily_buy_date != today:
                self._daily_buy_count = 0
                self._daily_buy_date = today
            daily_limit = getattr(self, "_daily_buy_limit", 5)
            if daily_limit > 0 and self._daily_buy_count >= daily_limit:
                # Hard cap reached — only ultra-high confidence (0.90+) can override
                if signal.confidence < 0.90:
                    logger.info(
                        "Skipping BUY for %s: daily limit reached (%d/%d, conf=%.2f < 0.90)",
                        symbol,
                        self._daily_buy_count,
                        daily_limit,
                        signal.confidence,
                    )
                    return
                logger.info(
                    "High-confidence override for %s (conf=%.2f, %d/%d used)",
                    symbol,
                    signal.confidence,
                    self._daily_buy_count,
                    daily_limit,
                )
            elif daily_limit > 0:
                # Dynamic confidence bar: higher as budget depletes
                usage_ratio = self._daily_buy_count / daily_limit
                if usage_ratio >= 0.8:
                    min_conf = 0.75
                elif usage_ratio >= 0.6:
                    min_conf = 0.65
                else:
                    min_conf = 0.0  # Use normal min_confidence from combiner
                if min_conf > 0 and signal.confidence < min_conf:
                    logger.info(
                        "Skipping BUY for %s: budget %d/%d, need conf>=%.2f (got %.2f)",
                        symbol,
                        self._daily_buy_count,
                        daily_limit,
                        min_conf,
                        signal.confidence,
                    )
                    return

            # Skip if there's already a pending buy order for this symbol
            if self._order_manager.has_pending_order(symbol, "BUY"):
                logger.debug("Skipping BUY for %s: pending order exists", symbol)
                return

            # Skip if we already hold this symbol (prevent duplicate buys)
            if self._position_tracker and symbol in self._position_tracker.tracked_symbols:
                logger.debug("Skipping BUY for %s: already held", symbol)
                return

            # Skip if signal hasn't changed since last evaluation (daily strategies
            # produce the same signal all day — no point re-buying)
            last = self._last_signal.get(symbol)
            if last and last[0] == "BUY" and time.time() - last[1] < 86400:
                logger.debug("Skipping BUY for %s: same signal within 24h", symbol)
                return
            self._last_signal[symbol] = ("BUY", time.time())

            # Event calendar checks (earnings proximity, FOMC day)
            if self._event_calendar:
                skip, reason = self._event_calendar.should_skip_buy(symbol)
                if skip:
                    logger.info("Skipping BUY for %s: %s", symbol, reason)
                    return

            balance = await self._market_data.get_balance()
            positions = await self._market_data.get_positions()

            # Defense-in-depth: block buy if already holding via exchange
            # positions.  This catches duplicates even when position_tracker
            # is empty (e.g. after restart before restore_from_exchange).
            if any(p.symbol == symbol and p.quantity > 0 for p in positions):
                logger.info(
                    "Skipping BUY for %s: already held (exchange positions)",
                    symbol,
                )
                return

            # Combined portfolio value for integrated margin allocation
            combined_pv = await self._get_combined_portfolio_value(balance.total)

            # Get factor score for this stock
            factor = self._factor_scores.get(symbol)
            factor_score = factor.composite if factor else 0.0

            # Insider confidence adjustment
            confidence = signal.confidence
            if self._event_calendar:
                confidence += self._event_calendar.get_confidence_adjustment(symbol)
                confidence = max(0.0, min(1.0, confidence))

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
                signal_confidence=confidence,
                factor_score=factor_score,
                market=self._market,
                combined_portfolio_value=combined_pv,
            )

            # Macro event sizing reduction (CPI/JOBS day = half size)
            if sizing.allowed and self._event_calendar:
                macro_mult = self._event_calendar.get_sizing_multiplier()
                if macro_mult < 1.0:
                    sizing.quantity = max(1, int(sizing.quantity * macro_mult))
                    sizing.allocation_usd *= macro_mult
                    logger.info(
                        "Macro event sizing: %s reduced to %.0f%%", symbol, macro_mult * 100
                    )

            if not sizing.allowed:
                logger.info(
                    "Buy rejected for %s: %s",
                    symbol,
                    sizing.reason,
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
                            symbol,
                            risk_check.get("reason", ""),
                        )
                        return
                except Exception as e:
                    logger.debug("Risk agent pre-trade check error: %s", e)

            exchange = "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
            order = await self._order_manager.place_buy(
                symbol=symbol,
                price=price,
                portfolio_value=balance.total,
                cash_available=balance.available,
                current_positions=len(positions),
                strategy_name=strategy_name,
                sizing_override=sizing,
                exchange=exchange,
            )

            # Increment daily buy counter
            if order:
                self._daily_buy_count += 1

            # Register position for SL/TP/trailing stop monitoring
            if order and self._position_tracker:
                # Dynamic ATR-based SL/TP per stock volatility
                atr_val = None
                if "atr" in df.columns:
                    atr_val = float(df["atr"].iloc[-1])
                elif "ATRr_14" in df.columns:
                    atr_val = float(df["ATRr_14"].iloc[-1])

                if atr_val and atr_val > 0:
                    sl_pct, tp_pct = self._risk_manager.calculate_dynamic_sl_tp(
                        price,
                        atr_val,
                        market=self._market,
                    )
                else:
                    sl_pct = self._risk_manager.params.default_stop_loss_pct
                    tp_pct = self._risk_manager.params.default_take_profit_pct

                self._position_tracker.track(
                    symbol=symbol,
                    entry_price=price,
                    quantity=order.quantity,
                    strategy=strategy_name,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )

        elif signal.signal_type == SignalType.SELL:
            positions = await self._market_data.get_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos and pos.quantity > 0:
                exchange = (
                    "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
                )
                orig_strategy = ""
                if self._position_tracker:
                    orig_strategy = self._position_tracker.get_buy_strategy(symbol)
                sell_order = await self._order_manager.place_sell(
                    symbol=symbol,
                    quantity=int(pos.quantity),
                    price=price,
                    strategy_name=signal.strategy_name,
                    exchange=exchange,
                    entry_price=pos.avg_price,
                    buy_strategy=orig_strategy,
                )
                if sell_order and self._position_tracker:
                    self._position_tracker.untrack(symbol)
                    # Add to recovery watch for re-entry evaluation
                    self._recovery_watch[symbol] = time.time()

    def record_trade_result(
        self,
        strategy: str,
        symbol: str,
        return_pct: float,
    ) -> None:
        """Record a trade result for signal quality tracking."""
        self._signal_quality.record_trade(strategy, symbol, return_pct)

    @property
    def factor_scores(self) -> dict[str, FactorScores]:
        return dict(self._factor_scores)

    @property
    def signal_quality(self) -> SignalQualityTracker:
        return self._signal_quality
