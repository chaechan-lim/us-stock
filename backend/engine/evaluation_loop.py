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
from strategies.base import PositionContext, Signal
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
    from services.cache import CacheService

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
        # Sell cooldown: block BUY for recently-sold symbols (STOCK-20)
        self._sell_cooldown_secs: int = 24 * 3600  # 24 hours
        # Redis cache for persisting _recovery_watch across restarts (STOCK-43)
        self._cache: CacheService | None = None
        self._cache_key: str = f"sell_cooldown:{market}"
        # Per-symbol maximum position weight (% of portfolio) (STOCK-20)
        self._max_per_symbol_pct: float = 0.10  # 10% max per symbol
        # STOCK-47: Minimum hold period before non-emergency sells (4 hours)
        self._min_hold_secs: int = 4 * 3600
        # STOCK-47: Hard stop-loss threshold that bypasses min hold (-7%)
        self._hard_sl_pct: float = -0.07
        # STOCK-47: Whipsaw counter — track loss sell timestamps per symbol
        self._loss_sell_history: dict[str, list[float]] = {}  # {symbol: [timestamps]}
        self._max_loss_sells: int = 2  # block re-entry after N loss sells in 7 days
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

    def set_cache(self, cache: CacheService) -> None:
        """Set Redis cache for persisting sell cooldown data (STOCK-43)."""
        self._cache = cache

    def register_sell_cooldown(
        self,
        symbol: str,
        sell_ts: float,
        *,
        is_loss: bool = False,
    ) -> None:
        """Record a sell event for cooldown enforcement (STOCK-43).

        Called as a callback from PositionTracker after stop-loss/take-profit/
        trailing stop sells, ensuring the evaluation loop blocks immediate
        re-buy of the same symbol.

        Also persists to Redis so cooldowns survive service restarts.

        Args:
            is_loss: If True, also records in whipsaw counter (STOCK-47).
        """
        self._recovery_watch[symbol] = sell_ts
        logger.info(
            "Sell cooldown registered for %s (%s market, cooldown=%dh)",
            symbol,
            self._market,
            self._sell_cooldown_secs // 3600,
        )
        # STOCK-47: Track loss sells for whipsaw detection
        if is_loss:
            cutoff = sell_ts - 7 * 86400
            history = self._loss_sell_history.get(symbol, [])
            history = [ts for ts in history if ts > cutoff]
            history.append(sell_ts)
            self._loss_sell_history[symbol] = history
            logger.info(
                "Whipsaw counter for %s: %d loss sells in 7d",
                symbol,
                len(history),
            )
        # Fire-and-forget Redis persistence (non-blocking from sync callback)
        if self._cache:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_sell_cooldown(symbol, sell_ts))
            except RuntimeError:
                pass  # No running loop (e.g. in tests)

    async def _persist_sell_cooldown(self, symbol: str, sell_ts: float) -> None:
        """Persist a single sell cooldown entry to Redis."""
        if not self._cache:
            return
        try:
            # Use recovery_watch_secs as TTL (entries expire after 7 days)
            await self._cache.set(
                f"{self._cache_key}:{symbol}",
                str(sell_ts),
                ex=self._recovery_watch_secs,
            )
        except Exception as e:
            logger.warning("Failed to persist sell cooldown for %s: %s", symbol, e)

    async def load_sell_cooldowns(self) -> int:
        """Load persisted sell cooldowns from Redis on startup (STOCK-43).

        Returns the number of cooldowns loaded.
        """
        if not self._cache or not self._cache.available:
            return 0
        try:
            import redis.asyncio as aioredis

            r: aioredis.Redis | None = self._cache._redis
            if r is None:
                return 0
            pattern = f"{self._cache_key}:*"
            loaded = 0
            now = time.time()
            async for key in r.scan_iter(match=pattern, count=100):
                raw = await r.get(key)
                if raw is None:
                    continue
                try:
                    sell_ts = float(raw)
                except (ValueError, TypeError):
                    continue
                # Only load entries still within recovery window
                if now - sell_ts > self._recovery_watch_secs:
                    continue
                # Extract symbol from key "sell_cooldown:{market}:{symbol}"
                symbol = str(key).split(":", 2)[-1] if ":" in str(key) else ""
                if symbol:
                    self._recovery_watch[symbol] = sell_ts
                    loaded += 1
            if loaded:
                logger.info("Loaded %d sell cooldowns from Redis (%s market)", loaded, self._market)
            return loaded
        except Exception as e:
            logger.warning("Failed to load sell cooldowns from Redis: %s", e)
            return 0

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

                # STOCK-21: evaluate_exit() for held positions
                current_price = float(df.iloc[-1]["close"])
                pos_ctx = self._build_position_context(symbol, current_price)
                if pos_ctx is not None:
                    strategy_map = {s.name: s for s in strategies}
                    evaluated = []
                    for sig in signals:
                        strat = strategy_map.get(sig.strategy_name)
                        if strat is not None:
                            try:
                                evaluated.append(strat.evaluate_exit(sig, pos_ctx, df))
                            except Exception as e:
                                logger.warning(
                                    "evaluate_exit failed for %s/%s, using original signal: %s",
                                    symbol,
                                    sig.strategy_name,
                                    e,
                                )
                                evaluated.append(sig)
                        else:
                            evaluated.append(sig)
                    signals = evaluated

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
        # Also build position_map for P&L-based exit decisions (STOCK-7).
        position_map: dict[str, object] = {}
        try:
            exchange_positions = await self._market_data.get_positions()
            exchange_held = {p.symbol for p in exchange_positions if p.quantity > 0}
            held = held | exchange_held
            position_map = {p.symbol: p for p in exchange_positions if p.quantity > 0}
        except Exception as e:
            logger.warning(
                "Exchange position fetch failed, using tracker only: %s", e, exc_info=True
            )

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
                    n_remapped = sum(1 for s in signals if s.signal_type == SignalType.BUY)
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

                    # STOCK-21: evaluate_exit() — strategy-level profit-taking
                    # Build position context and let each strategy evaluate
                    # whether the held position should be exited for profit.
                    current_price = float(df.iloc[-1]["close"])
                    pos_ctx = self._build_position_context(symbol, current_price)
                    if pos_ctx is not None:
                        strategy_map = {s.name: s for s in strategies}
                        evaluated = []
                        for sig in signals:
                            strat = strategy_map.get(sig.strategy_name)
                            if strat is not None:
                                try:
                                    evaluated.append(strat.evaluate_exit(sig, pos_ctx, df))
                                except Exception as e:
                                    logger.debug(
                                        "evaluate_exit failed for %s/%s: %s",
                                        sig.strategy_name,
                                        symbol,
                                        e,
                                    )
                                    evaluated.append(sig)
                            else:
                                evaluated.append(sig)
                        signals = evaluated

                # Held positions: lower SELL threshold + bias for easier exits (STOCK-7)
                # STOCK-47: Raised thresholds to reduce whipsaw
                if is_held:
                    _hsb = getattr(self, "_held_sell_bias", 0.05)
                    _hmc = getattr(self, "_held_min_confidence", 0.40)
                    combined = self._combiner.combine(
                        signals,
                        weights,
                        min_confidence=_hmc,
                        min_active_ratio=0.15,
                        held_sell_bias=_hsb,
                    )
                else:
                    combined = self._combiner.combine(signals, weights)

                # Sell on indifference: if no strategy has a strong opinion
                # (HOLD) and the held position is losing beyond threshold,
                # free up capital by selling (STOCK-7).
                # STOCK-47: threshold -3% → -5%, add min hold check
                if is_held and combined.signal_type == SignalType.HOLD and symbol in position_map:
                    pos = position_map[symbol]
                    if hasattr(pos, "avg_price") and pos.avg_price > 0:
                        pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
                        _spt = getattr(self, "_stale_pnl_threshold", -0.05)
                        if pnl_pct < _spt:
                            # STOCK-47: Check min hold (skip for hard SL)
                            hold_ok = True
                            if pnl_pct >= self._hard_sl_pct:
                                hold_ok = self._check_min_hold(symbol)
                            if hold_ok:
                                combined = Signal(
                                    signal_type=SignalType.SELL,
                                    confidence=0.50,
                                    strategy_name="position_cleanup",
                                    reason=(
                                        f"Sell on indifference: P&L={pnl_pct:.1%},"
                                        f" no strategy recommends"
                                    ),
                                )
                                logger.info(
                                    "Position cleanup SELL for %s: P&L=%.1f%%",
                                    symbol,
                                    pnl_pct * 100,
                                )

                # STOCK-34: Profit protection — sell on high profit when
                # all strategies say HOLD. This is a defense-in-depth
                # safety net: if position_tracker TP doesn't fire and
                # evaluate_exit() doesn't convert to SELL, this ensures
                # highly profitable positions are eventually exited.
                _ppp = getattr(self, "_profit_protection_pct", 0.15)
                if (
                    is_held
                    and combined.signal_type == SignalType.HOLD
                    and symbol in position_map
                    and _ppp > 0
                ):
                    pos = position_map[symbol]
                    if hasattr(pos, "avg_price") and pos.avg_price > 0:
                        pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
                        if pnl_pct >= _ppp:
                            combined = Signal(
                                signal_type=SignalType.SELL,
                                confidence=0.60,
                                strategy_name="profit_protection",
                                reason=(f"Profit protection: P&L={pnl_pct:.1%}, securing gains"),
                            )
                            logger.info(
                                "Profit protection SELL for %s: P&L=%.1f%%",
                                symbol,
                                pnl_pct * 100,
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
                # STOCK-47: Min hold check for strategy sells (not position_cleanup
                # or profit_protection which have their own checks above).
                if combined.signal_type == SignalType.SELL:
                    if is_held and combined.strategy_name not in (
                        "position_cleanup",
                        "profit_protection",
                    ):
                        # Check if hard SL bypass applies
                        pos = position_map.get(symbol)
                        is_hard_sl = False
                        if pos and hasattr(pos, "avg_price") and pos.avg_price > 0:
                            pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
                            is_hard_sl = pnl_pct < self._hard_sl_pct
                        if not is_hard_sl and not self._check_min_hold(symbol):
                            combined = Signal(
                                signal_type=SignalType.HOLD,
                                confidence=combined.confidence,
                                strategy_name="combiner",
                                reason=f"Min hold not met for {symbol}",
                            )
                    if combined.signal_type == SignalType.SELL:
                        await self._execute_signal(combined, symbol, df)
                elif combined.signal_type == SignalType.BUY:
                    buy_candidates.append((combined.confidence, symbol, combined, df))

            except Exception as e:
                logger.error("Failed to evaluate %s: %s", symbol, e)

        # Phase 2: Execute BUYs ranked by confidence (highest first)
        # STOCK-26: Deduplicate by symbol — keep only highest confidence
        # per symbol. This prevents the same stock from appearing multiple
        # times in buy_candidates (e.g. from different evaluation paths).
        if buy_candidates:
            buy_candidates = self._dedup_buy_candidates(buy_candidates)
            logger.info(
                "Buy candidates ranked: %s",
                [(s, f"{c:.2f}") for c, s, _, _ in buy_candidates[:10]],
            )
            for _conf, symbol, combined, df in buy_candidates:
                await self._execute_signal(combined, symbol, df)

    @staticmethod
    def _dedup_buy_candidates(
        candidates: list[tuple[float, str, object, pd.DataFrame]],
    ) -> list[tuple[float, str, object, pd.DataFrame]]:
        """Sort by confidence descending and keep only the first entry per symbol.

        STOCK-26: Prevents the same stock from appearing multiple times in
        buy_candidates (e.g. from different evaluation paths or strategies).
        """
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen_symbols: set[str] = set()
        deduped: list[tuple[float, str, object, pd.DataFrame]] = []
        for entry in candidates:
            sym = entry[1]
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                deduped.append(entry)
        if len(deduped) < len(candidates):
            logger.info(
                "Deduplicated buy candidates: %d -> %d (removed %d duplicates)",
                len(candidates),
                len(deduped),
                len(candidates) - len(deduped),
            )
        return deduped

    def _check_min_hold(self, symbol: str) -> bool:
        """Return True if the position has been held for at least _min_hold_secs.

        STOCK-47: Prevents whipsaw by enforcing a minimum holding period
        before non-emergency sells. Hard stop-loss bypasses this check.
        """
        if self._min_hold_secs <= 0:
            return True
        if not self._position_tracker:
            return True
        tracked_dict = getattr(self._position_tracker, "_tracked", None)
        if not tracked_dict:
            return True
        tracked = tracked_dict.get(symbol)
        if not tracked:
            return True
        try:
            hold_secs = time.monotonic() - float(tracked.tracked_at)
        except (TypeError, ValueError):
            return True  # Can't determine hold time — allow sell
        if hold_secs < self._min_hold_secs:
            logger.info(
                "Min hold not met for %s: held %.1fh < %.1fh required",
                symbol,
                hold_secs / 3600,
                self._min_hold_secs / 3600,
            )
            return False
        return True

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
                            hold_secs = time.monotonic() - tracked.tracked_at
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
                    # Protective sells are always loss sells (regime/sentiment)
                    self.register_sell_cooldown(
                        symbol,
                        time.time(),
                        is_loss=True,
                    )

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
            except Exception as e:
                logger.warning("Failed to fetch OHLCV for factor scoring (%s): %s", symbol, e)

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

        STOCK-53: Uses other market's full total (not just positions) to get
        a more accurate combined value. The deposit overlap is acceptable
        because _apply_market_cap limits actual cash usage via
        min(capped_cash, cash_available).
        """
        if not self._other_market_data:
            return None
        try:
            other_balance = await self._other_market_data.get_balance()
            # Add full other market total for accurate allocation sizing.
            # Deposit double-counting is safe: actual orders are bounded by
            # real cash_available via min() in _apply_market_cap.
            other_total = other_balance.total
            if other_total <= 0:
                other_total = max(0, other_balance.total - other_balance.available)
            if self._market == "US":
                # Own is USD, other KR total is in KRW → convert to USD
                other_in_own = other_total / self._exchange_rate
            else:
                # Own is KRW, other US total is in USD → convert to KRW
                other_in_own = other_total * self._exchange_rate
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
        except Exception as e:
            logger.warning("Real-time price fetch failed for %s, using OHLCV close: %s", symbol, e)
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

            # STOCK-20: Sell cooldown — block BUY for recently-sold symbols.
            # After a stop-loss or strategy sell, wait at least _sell_cooldown_secs
            # before re-buying to prevent sell-then-immediately-rebuy churn.
            sell_ts = self._recovery_watch.get(symbol)
            if sell_ts is not None and self._sell_cooldown_secs > 0:
                elapsed = time.time() - sell_ts
                if elapsed < self._sell_cooldown_secs:
                    hours_ago = elapsed / 3600
                    cooldown_h = self._sell_cooldown_secs / 3600
                    logger.info(
                        "Skipping BUY for %s: sell cooldown (sold %.1fh ago, cooldown=%.0fh)",
                        symbol,
                        hours_ago,
                        cooldown_h,
                    )
                    return

            # STOCK-47: Whipsaw counter — block re-entry after repeated loss sells
            loss_history = self._loss_sell_history.get(symbol, [])
            if loss_history:
                cutoff = time.time() - 7 * 86400
                recent = [ts for ts in loss_history if ts > cutoff]
                if len(recent) >= self._max_loss_sells:
                    logger.info(
                        "Skipping BUY for %s: whipsaw block (%d loss sells in 7d, max=%d)",
                        symbol,
                        len(recent),
                        self._max_loss_sells,
                    )
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

            # STOCK-20: Per-symbol position concentration limit.
            # Block additional buys if existing position value exceeds
            # max_per_symbol_pct of portfolio. Defense-in-depth for cases
            # where position_tracker is unavailable (e.g. after restart).
            existing_pos = next(
                (p for p in positions if p.symbol == symbol and p.quantity > 0),
                None,
            )
            # Compute existing position value once — reused below for
            # Kelly sizing's existing_position_value parameter (STOCK-26).
            existing_value = 0.0
            if existing_pos and hasattr(existing_pos, "current_price"):
                existing_value = existing_pos.current_price * existing_pos.quantity
            if existing_value > 0 and balance.total > 0:
                max_value = balance.total * self._max_per_symbol_pct
                if existing_value >= max_value:
                    logger.info(
                        "Skipping BUY for %s: position concentration %.1f%% >= limit %.1f%%",
                        symbol,
                        (existing_value / balance.total) * 100,
                        self._max_per_symbol_pct * 100,
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

            # Use Kelly-enhanced sizing (existing_value computed above)
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
                existing_position_value=existing_value,
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

                # Look up trailing stop config from strategy YAML
                trail_act: float | None = None
                trail_pct: float | None = None
                trail_cfg = self._registry.get_trailing_stop_config(strategy_name)
                if trail_cfg and trail_cfg.get("enabled", False):
                    trail_act = trail_cfg.get("activation_pct")
                    trail_pct = trail_cfg.get("trail_pct")

                self._position_tracker.track(
                    symbol=symbol,
                    entry_price=price,
                    quantity=order.quantity,
                    strategy=strategy_name,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                    trailing_activation_pct=trail_act,
                    trailing_stop_pct=trail_pct,
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
                    # STOCK-47: Determine if this was a loss sell
                    is_loss = False
                    if pos.avg_price > 0:
                        is_loss = price < pos.avg_price
                    # Add to recovery watch for re-entry evaluation
                    self.register_sell_cooldown(
                        symbol,
                        time.time(),
                        is_loss=is_loss,
                    )

    def _build_position_context(self, symbol: str, current_price: float) -> PositionContext | None:
        """Build a PositionContext for a held symbol.

        Returns None if the position tracker is unavailable or the symbol
        is not tracked. Uses TrackedPosition data to avoid extra API calls.
        """
        if not self._position_tracker:
            return None
        try:
            tracked_dict = getattr(self._position_tracker, "_tracked", None)
            if tracked_dict is None:
                return None
            tracked = tracked_dict.get(symbol)
            if tracked is None:
                return None
            pnl_pct = (
                (current_price - tracked.entry_price) / tracked.entry_price
                if tracked.entry_price > 0
                else 0.0
            )
            hold_secs = time.monotonic() - tracked.tracked_at
            return PositionContext(
                symbol=symbol,
                entry_price=tracked.entry_price,
                current_price=current_price,
                highest_price=tracked.highest_price,
                quantity=tracked.quantity,
                pnl_pct=pnl_pct,
                hold_seconds=hold_secs,
                strategy=tracked.strategy,
            )
        except Exception as e:
            logger.debug("Failed to build position context for %s: %s", symbol, e)
            return None

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
