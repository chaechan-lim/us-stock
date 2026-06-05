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
from typing import Any, TYPE_CHECKING

import pandas as pd

from core.constants import USD_KRW_FALLBACK
from analytics.factor_model import FactorScores, MultiFactorModel
from analytics.signal_quality import SignalQualityTracker
from core.enums import SignalType
from data.indicator_service import IndicatorService
from data.market_data_service import MarketDataService
from engine.adaptive_weights import AdaptiveWeightManager
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager
from engine.stock_classifier import StockClassifier
from exchange.base import ExchangeAdapter
from services.exchange_resolver import ExchangeResolver
from strategies.base import BaseStrategy, PositionContext, Signal
from strategies.combiner import SignalCombiner
from strategies.registry import StrategyRegistry

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
        account_id: str = "ACC001",
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
        # Phase 4 will use account_id for multi-account order routing and logging
        self._account_id = account_id
        self._event_calendar = event_calendar
        self._other_market_data: MarketDataService | None = None
        self._exchange_rate: float = USD_KRW_FALLBACK
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
        # Hard stop-loss threshold that bypasses min hold (aligned with YAML -15%)
        self._hard_sl_pct: float = -0.15
        # P1 (#55, 2026-05-13): time-based stale exit. Backtest V3 (2d, -2%)
        # showed +6.8pp Ret, +0.55 Sharpe with MDD unchanged. Disabled by
        # default (0); enabled per-market via _apply_*_eval_overrides.
        self._stale_time_days: int = 0
        self._stale_time_pnl_threshold: float = 0.0
        # STOCK-47: Whipsaw counter — track loss sell timestamps per symbol
        self._loss_sell_history: dict[str, list[float]] = {}  # {symbol: [timestamps]}
        self._max_loss_sells: int = 2  # block re-entry after N loss sells in 7 days
        # Daily buy counter (resets at midnight)
        self._daily_buy_count: int = 0
        self._daily_buy_date: str = ""
        # Daily buy budget + dynamic confidence escalation. Defaults match
        # the prior hardcoded behavior (KR backtest VP0). US relaxes to
        # 10/0.50/0.60 via yaml — compare_daily_buy_limit.py V1 was 4/4 OK.
        self._daily_buy_limit: int = 5
        self._daily_buy_escalation_low: float = 0.65   # ≥60% usage
        self._daily_buy_escalation_high: float = 0.75  # ≥80% usage
        self._daily_buy_override: float = 0.90         # over-cap override
        # 2026-05-22 (#59-C): sizing-up — allow add-on BUY for symbols
        # currently held below `sizing_up_threshold × min_position_pct`
        # of equity. Closes the placeholder gap created by the 1-share
        # round-up path (KR blue chips) once that path is disabled, by
        # bringing stuck 1-share positions up to the meaningful size
        # Kelly would have chosen given enough cash. Disabled by default;
        # opt in per market via set_sizing_up_enabled(True).
        self._sizing_up_enabled: bool = False
        self._sizing_up_threshold: float = 0.5
        self._sizing_up_min_confidence: float = 0.50
        self._sizing_up_cooldown_secs: int = 86400  # 24h
        self._sizing_up_history: dict[str, float] = {}  # symbol -> last add-on ts
        # F1 (2026-05-09) attribution funnel: count BUY-flow rejections by
        # category so the dashboard can answer "why is cash sitting?"
        # Reset daily by _reset_daily_counters_if_needed.
        self._reject_counters: dict[str, int] = {}
        self._buy_flow_counters: dict[str, int] = {
            "buy_signals_total": 0,
            "buys_placed": 0,
        }
        # Recent signals buffer for frontend display (last N signals)
        from collections import deque

        self._recent_signals: deque[dict] = deque(maxlen=200)
        # STOCK-65/66: Market-specific disabled strategies
        self._disabled_strategies: frozenset[str] = frozenset()
        # STOCK-65/66: Per-market min_confidence override (None = use combiner default)
        self._min_confidence: float | None = None
        # STOCK-65/66: Per-market min_active_ratio override (None = use per-call defaults)
        self._min_active_ratio: float | None = None
        # Signal quality dynamic weighting: boost/suppress strategies by live profit factor
        self._quality_weight_enabled: bool = True
        self._quality_min_trades: int = 25
        # Sector concentration limit (blocks BUY if sector > max_sector_pct of portfolio)
        self._max_sector_pct: float = 0.40
        self._sector_cache: dict[str, str] = {}  # symbol -> sector name
        # D1 sector strength boost on BUY: multiplier applied to signal.confidence
        # before sizing. 0.0 = off. Backtest sweep (2026-04-24): KR 0.3, US 0.2
        # gave the cleanest 4-axis improvement vs baseline.
        self._sector_boost_weight: float = 0.0
        self._sector_scores: dict[str, float] = {}  # sector_name -> strength 0-100
        # 2026-04-24: skip BUY during the first N minutes after market open.
        # 0 = off. SELL signals still pass (position_tracker SL/TP also runs
        # on its own schedule, so risk exits aren't blocked).
        self._opening_avoidance_minutes: int = 0
        # 2026-05-28: minimum BUY price (defense-in-depth for #158).
        # ScannerPipeline.min_price filters Layer 1, but universe_expander
        # / KIS ranking can push symbols straight into evaluation without
        # going through the pipeline. CDXS ($2.69) + BWEN ($3.85) slipped
        # through that gap on 5-27. Setting a market-level floor in the
        # eval loop catches every path. 0 = off; live US sets 5.0 via yaml.
        self._min_buy_price: float = 0.0

        # Commission rate per order (one-way). Used to filter out trades
        # where expected PnL < round-trip commission. KIS US = 0.25%,
        # KR = ~0.015% brokerage + 0.18% sell tax ≈ 0.20% effective.
        # 2026-04-13: Added after cash_parking churn showed the system
        # was completely commission-blind (860k KRW in 2 days).
        self._commission_rate: float = 0.0025 if market == "US" else 0.0020
        self._min_profit_after_commission: float = self._commission_rate * 2  # round-trip

        # Cash parking (port from backtest/full_pipeline.py — recovers ~13pp
        # alpha by parking idle cash in SPY/KODEX 200 instead of letting it
        # sit during low-conviction periods). Default disabled; opt-in via
        # markets.<MARKET>.cash_parking.enabled in strategies.yaml.
        self._cash_parking_enabled: bool = False
        self._cash_parking_enable_unpark: bool = False
        self._cash_parking_min_hold_days: int = 10  # 2 weeks default
        self._cash_parking_parked_at: float = 0.0   # timestamp when parked
        self._cash_parking_symbol: str = "SPY" if market == "US" else "069500"
        self._cash_parking_threshold: float = 0.30  # park if cash > 30% of equity
        self._cash_parking_buffer: float = 0.10     # keep 10% cash buffer for opportunities
        # P3 (2026-05-15) re-enable safety: hard cap on total parking
        # exposure as a fraction of total equity. Prevents parking from
        # eating the US BUY exposure cap (was the 2026-04-28 disable
        # cause — parking grew to 57% of US positions, blocking strategy
        # BUYs). When existing parking value ≥ this cap, skip new park.
        self._cash_parking_max_pct: float = 0.25     # 25% of equity max
        # P3-A (2026-05-15) per-cycle split: cap a single buy to this
        # fraction of equity so a 40% target isn't dumped in one order
        # (averaging-in vs one-shot single-day exposure to SPY).
        # Default 10% → 4 cycles minimum to reach 40% cap; with 1h
        # cooldown that's 4 hours of averaging.
        self._cash_parking_per_cycle_pct: float = 0.10
        # Per-cycle position cache — set at start of _evaluate_all, cleared after.
        # Downstream methods use _get_positions() which returns this cache when fresh.
        self._cycle_positions: list | None = None
        # CON-B1 (2026-06-05): serialise concurrent _evaluate_all calls.
        # The scheduler awaits one tick at a time, but POST /engine/run-
        # evaluation lets MCP / operator re-enter the same coroutine,
        # which races on _cycle_positions, _last_signal, _daily_buy_count,
        # _recent_signals etc. Same-symbol double BUYs are possible
        # without this guard.
        self._evaluate_lock = asyncio.Lock()

    async def _get_positions(self) -> list:
        """Return cached positions from current eval cycle, or fetch fresh."""
        if self._cycle_positions is not None:
            return self._cycle_positions
        return await self._market_data.get_positions()

    def _bump_reject(self, reason: str) -> None:
        self._reject_counters[reason] = self._reject_counters.get(reason, 0) + 1
        # Hermes Phase 3: persist funnel event for counterfactual replay.
        # _current_* are set at the top of _execute_signal so every reject
        # call site automatically gets symbol/strategy/price context.
        sym = getattr(self, "_current_symbol", None)
        if sym:
            asyncio.create_task(self._persist_funnel_event(
                sym, decision="rejected", reason=reason,
            ))

    async def _persist_funnel_event(
        self, symbol: str, decision: str, reason: str | None = None,
    ) -> None:
        """Fire-and-forget DB insert. Failures must not affect trading —
        wrap broadly and swallow. 30d retention is enforced by a daily
        scheduler task (see main.py)."""
        try:
            from core.models import FunnelEvent
            from db.session import get_session_factory
            sig = getattr(self, "_current_signal", None)
            price = getattr(self, "_current_price", None)
            factory = get_session_factory()
            async with factory() as s:
                s.add(FunnelEvent(
                    market=self._market,
                    symbol=symbol,
                    strategy_name=getattr(sig, "strategy_name", None) if sig else None,
                    signal_confidence=getattr(sig, "confidence", None) if sig else None,
                    decision=decision,
                    reject_reason=reason,
                    price=price,
                ))
                await s.commit()
        except Exception as e:
            logger.debug("FunnelEvent persist failed (non-fatal): %s", e)

    def _reset_daily_counters_if_needed(self) -> None:
        """Roll over _daily_buy_count, _reject_counters, _buy_flow_counters
        when the date changes. Must run before any per-signal counter touch
        so the first BUY of a new day is attributed to the new day.
        """
        from datetime import date as _date

        today = _date.today().isoformat()
        if self._daily_buy_date != today:
            self._daily_buy_count = 0
            self._daily_buy_date = today
            self._reject_counters.clear()
            self._buy_flow_counters = {
                "buy_signals_total": 0,
                "buys_placed": 0,
            }

    def set_disabled_strategies(self, names: list[str]) -> None:
        """Set market-specific disabled strategy names."""
        self._disabled_strategies = frozenset(names)
        logger.info(
            "Market %s: disabled strategies = %s",
            self._market,
            sorted(self._disabled_strategies),
        )

    def set_cash_parking_config(
        self,
        enabled: bool,
        symbol: str | None = None,
        threshold: float | None = None,
        buffer: float | None = None,
        min_hold_days: int | None = None,
        enable_unpark: bool | None = None,
        max_pct: float | None = None,
        per_cycle_pct: float | None = None,
    ) -> None:
        """Configure cash parking (idle cash → SPY/KODEX 200).

        Args:
            enabled: Master switch.
            symbol: Parking symbol (default SPY for US, 069500 for KR).
            threshold: Park when cash/equity > threshold (default 0.30).
            buffer: Keep this fraction of equity in cash (default 0.10).
            min_hold_days: Minimum days before unpark allowed (default 10).
            enable_unpark: Allow selling parking when BUY needs cash.
            max_pct: Hard cap on parking value as fraction of equity
                (default 0.25). Skip new parks when at/over cap.
            per_cycle_pct: Max fraction of equity to buy in a single
                cycle (default 0.10). Splits the build-up to max_pct
                across multiple cycles for averaging-in.
        """
        self._cash_parking_enabled = bool(enabled)
        if symbol:
            self._cash_parking_symbol = symbol
        if threshold is not None:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"cash_parking threshold must be in [0,1], got {threshold}")
            self._cash_parking_threshold = float(threshold)
        if buffer is not None:
            if not (0.0 <= buffer <= 1.0):
                raise ValueError(f"cash_parking buffer must be in [0,1], got {buffer}")
            self._cash_parking_buffer = float(buffer)
        if min_hold_days is not None:
            self._cash_parking_min_hold_days = int(min_hold_days)
        if enable_unpark is not None:
            self._cash_parking_enable_unpark = bool(enable_unpark)
        if max_pct is not None:
            if not (0.0 <= max_pct <= 1.0):
                raise ValueError(f"cash_parking max_pct must be in [0,1], got {max_pct}")
            self._cash_parking_max_pct = float(max_pct)
        if per_cycle_pct is not None:
            if not (0.0 < per_cycle_pct <= 1.0):
                raise ValueError(
                    f"cash_parking per_cycle_pct must be in (0,1], got {per_cycle_pct}"
                )
            self._cash_parking_per_cycle_pct = float(per_cycle_pct)
        logger.info(
            "Market %s: cash_parking enabled=%s symbol=%s threshold=%.2f buffer=%.2f "
            "min_hold=%dd unpark=%s",
            self._market, self._cash_parking_enabled, self._cash_parking_symbol,
            self._cash_parking_threshold, self._cash_parking_buffer,
            self._cash_parking_min_hold_days, self._cash_parking_enable_unpark,
        )

    def set_min_confidence(self, value: float | None) -> None:
        """Set per-market minimum confidence threshold for signal combining."""
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {value}")
        self._min_confidence = value
        logger.info("Market %s: min_confidence = %s", self._market, value)

    def set_min_active_ratio(self, value: float | None) -> None:
        """Set per-market minimum active signal ratio for signal combining."""
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"min_active_ratio must be in [0, 1], got {value}")
        self._min_active_ratio = value
        logger.info("Market %s: min_active_ratio = %s", self._market, value)

    def set_sell_cooldown_secs(self, value: int) -> None:
        """Set the sell-cooldown duration for this market instance."""
        if value < 0:
            raise ValueError(f"sell_cooldown_secs must be >= 0, got {value}")
        self._sell_cooldown_secs = value
        logger.info("Market %s: sell_cooldown_secs=%d", self._market, value)

    def set_max_loss_sells(self, value: int) -> None:
        """Set the whipsaw loss-sell limit for this market instance."""
        if value < 0:
            raise ValueError(f"max_loss_sells must be >= 0, got {value}")
        self._max_loss_sells = value
        logger.info("Market %s: max_loss_sells=%d", self._market, value)

    def set_min_hold_secs(self, value: int) -> None:
        """Set the minimum hold period for this market instance."""
        if value < 0:
            raise ValueError(f"min_hold_secs must be >= 0, got {value}")
        self._min_hold_secs = value
        logger.info(
            "Market %s: min_hold_secs=%d (%.2fh)", self._market, value, value / 3600
        )

    def set_stale_time_exit(
        self,
        days: int | None,
        pnl_threshold: float | None = None,
    ) -> None:
        """P1 (#55) time-based stale exit. Set days=0 to disable.

        When days > 0, cleanup also fires for held positions where
        elapsed_days >= days AND pnl_pct < pnl_threshold (default 0%),
        even if loss hasn't reached the standard stale_pnl_threshold.

        Targets the live "bounce_die" cleanup bucket (positions that
        touched only +1% high then drifted to -5%, missing profit_taking
        / trailing / breakeven thresholds).
        """
        days = int(days or 0)
        if days < 0:
            raise ValueError(f"stale_time_days must be >= 0, got {days}")
        thr = float(pnl_threshold) if pnl_threshold is not None else 0.0
        if not -0.20 <= thr <= 0.20:
            raise ValueError(f"stale_time_pnl_threshold out of [-0.20, 0.20]: {thr}")
        self._stale_time_days = days
        self._stale_time_pnl_threshold = thr
        if days > 0:
            logger.info(
                "Market %s: stale_time_exit enabled (≥%dd, pnl<%.0f%%)",
                self._market, days, thr * 100,
            )
        else:
            logger.info("Market %s: stale_time_exit disabled", self._market)

    def set_max_sector_pct(self, value: float) -> None:
        """Set maximum portfolio concentration per sector."""
        self._max_sector_pct = value
        logger.info("Market %s: max_sector_pct=%.0f%%", self._market, value * 100)

    def set_sector_cache(self, cache: dict[str, str]) -> None:
        """Set symbol → sector mapping for sector concentration checks."""
        self._sector_cache = cache

    def set_sector_boost_weight(self, weight: float) -> None:
        """Enable sector strength boost on BUY confidence. 0.0 = off."""
        if weight < 0:
            raise ValueError(f"sector_boost_weight must be >= 0, got {weight}")
        self._sector_boost_weight = float(weight)
        logger.info(
            "Market %s: sector_boost_weight=%.2f", self._market, self._sector_boost_weight,
        )

    def set_sector_scores(self, scores: dict[str, float]) -> None:
        """Update sector strength scores (0-100), typically called from
        the ETF evaluation task that already fetches sector performance."""
        self._sector_scores = dict(scores)

    def set_daily_buy_budget(
        self,
        *,
        limit: int | None = None,
        escalation_low: float | None = None,
        escalation_high: float | None = None,
        override: float | None = None,
    ) -> None:
        """Configure daily buy budget + confidence escalation thresholds.

        compare_daily_buy_limit.py 2026-05-07 found different optima per market:
          KR: 5/0.65/0.75 (default, untouched). Relaxing hurt -2pp Ret.
          US: 10/0.50/0.60. 4/4 floor improvement, Cash 34→31%.
        """
        if limit is not None:
            if limit < 0:
                raise ValueError(f"daily_buy_limit must be >= 0, got {limit}")
            self._daily_buy_limit = int(limit)
        if escalation_low is not None:
            if not 0.0 <= escalation_low <= 1.0:
                raise ValueError(f"escalation_low out of [0,1]: {escalation_low}")
            self._daily_buy_escalation_low = float(escalation_low)
        if escalation_high is not None:
            if not 0.0 <= escalation_high <= 1.0:
                raise ValueError(f"escalation_high out of [0,1]: {escalation_high}")
            self._daily_buy_escalation_high = float(escalation_high)
        if override is not None:
            if not 0.0 <= override <= 1.0:
                raise ValueError(f"override out of [0,1]: {override}")
            self._daily_buy_override = float(override)
        logger.info(
            "Market %s: daily_buy_budget limit=%d esc=%.2f/%.2f override=%.2f",
            self._market,
            self._daily_buy_limit,
            self._daily_buy_escalation_low,
            self._daily_buy_escalation_high,
            self._daily_buy_override,
        )

    def set_opening_avoidance_minutes(self, minutes: int) -> None:
        """Skip BUY eval during the first N minutes after regular open.
        0 disables. SELL still passes; position_tracker SL/TP unaffected."""
        if minutes < 0:
            raise ValueError(f"opening_avoidance_minutes must be >= 0, got {minutes}")
        self._opening_avoidance_minutes = int(minutes)
        logger.info(
            "Market %s: opening_avoidance_minutes=%d",
            self._market, self._opening_avoidance_minutes,
        )

    def set_min_buy_price(self, min_price: float) -> None:
        """Defense-in-depth minimum BUY price. Rejects BUY signals when
        the real-time price is below `min_price`. 0 disables. Matches
        the watchlist's ScannerPipeline.min_price filter (#158); catches
        symbols that enter evaluation via universe_expander / KIS
        ranking without going through the scanner."""
        if min_price < 0:
            raise ValueError(f"min_buy_price must be >= 0, got {min_price}")
        self._min_buy_price = float(min_price)
        logger.info(
            "Market %s: min_buy_price=$%.2f",
            self._market, self._min_buy_price,
        )

    def set_sizing_up_config(
        self,
        *,
        enabled: bool,
        threshold: float | None = None,
        min_confidence: float | None = None,
        cooldown_secs: int | None = None,
    ) -> None:
        """Configure the sizing-up path (#59-C).

        When enabled, an `already_held` BUY signal becomes an add-on order
        instead of a reject, provided the current position value sits below
        `threshold × min_position_pct × equity` and the signal confidence
        clears `min_confidence`. Cooldown prevents repeated add-ons for the
        same symbol within `cooldown_secs`.
        """
        self._sizing_up_enabled = bool(enabled)
        if threshold is not None:
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"sizing_up_threshold out of (0,1]: {threshold}")
            self._sizing_up_threshold = float(threshold)
        if min_confidence is not None:
            if not 0.0 <= min_confidence <= 1.0:
                raise ValueError(f"sizing_up_min_confidence out of [0,1]: {min_confidence}")
            self._sizing_up_min_confidence = float(min_confidence)
        if cooldown_secs is not None:
            if cooldown_secs < 0:
                raise ValueError(f"sizing_up_cooldown_secs must be >= 0, got {cooldown_secs}")
            self._sizing_up_cooldown_secs = int(cooldown_secs)
        logger.info(
            "Market %s: sizing_up enabled=%s thr=%.2f min_conf=%.2f cd=%ds",
            self._market, self._sizing_up_enabled, self._sizing_up_threshold,
            self._sizing_up_min_confidence, self._sizing_up_cooldown_secs,
        )

    def _is_sizing_up_eligible(
        self, symbol: str, existing_value: float, equity: float, signal_confidence: float,
    ) -> tuple[bool, str]:
        """Return (eligible, reason) for converting an already_held reject
        into an add-on BUY. Pure function — easy to unit-test."""
        if not self._sizing_up_enabled:
            return False, "disabled"
        if equity <= 0:
            return False, "no_equity"
        if existing_value <= 0:
            return False, "no_existing_position"
        min_pct = self._risk_manager._params.min_position_pct
        target_floor = self._sizing_up_threshold * min_pct * equity
        if existing_value >= target_floor:
            cur_pct = existing_value / equity * 100
            return False, f"already_sized ({cur_pct:.1f}% ≥ floor {min_pct*100*self._sizing_up_threshold:.1f}%)"
        if signal_confidence < self._sizing_up_min_confidence:
            return False, f"conf {signal_confidence:.2f} < {self._sizing_up_min_confidence:.2f}"
        last_ts = self._sizing_up_history.get(symbol)
        if last_ts is not None and self._sizing_up_cooldown_secs > 0:
            elapsed = time.time() - last_ts
            if elapsed < self._sizing_up_cooldown_secs:
                return False, f"cooldown {elapsed/3600:.1f}h < {self._sizing_up_cooldown_secs/3600:.1f}h"
        return True, "undersized"

    def _get_sector(self, symbol: str) -> str:
        """Look up sector for a symbol (cached)."""
        return self._sector_cache.get(symbol, "Unknown")

    def _check_sector_limit(
        self,
        symbol: str,
        cost: float,
        positions: list,
        portfolio_value: float,
    ) -> bool:
        """Check if buying symbol would breach sector concentration limit.

        Returns True if buy is allowed, False if blocked.
        """
        if self._max_sector_pct >= 1.0 or portfolio_value <= 0:
            return True

        sector = self._get_sector(symbol)
        if sector == "Unknown":
            return True  # Unknown sector → allow (don't block on missing data)

        # Sum current exposure in this sector
        sector_value = 0.0
        for pos in positions:
            if pos.quantity > 0 and self._get_sector(pos.symbol) == sector:
                pos_price = getattr(pos, "current_price", 0) or getattr(pos, "avg_price", 0)
                sector_value += pos_price * pos.quantity

        new_sector_pct = (sector_value + cost) / portfolio_value
        if new_sector_pct > self._max_sector_pct:
            logger.info(
                "Skipping BUY for %s: sector %s at %.1f%% (limit %.1f%%)",
                symbol, sector, new_sector_pct * 100, self._max_sector_pct * 100,
            )
            return False
        return True

    def set_quality_weight_enabled(self, enabled: bool) -> None:
        """Enable/disable signal quality dynamic weighting."""
        self._quality_weight_enabled = enabled
        logger.info("Market %s: quality_weight_enabled=%s", self._market, enabled)

    def _apply_quality_weights(
        self, weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply signal quality multiplier to strategy weights.

        Strategies with proven edge (high profit factor, sufficient trades)
        get boosted. Poor performers get suppressed. Normalizes after.

        Multiplier logic:
        - PF >= 1.5 → boost up to 1.5x (0.25 per PF point above 1.0)
        - PF 1.0-1.5 → neutral 1.0x
        - PF < 1.0 → suppress to max(0.3, PF)
        - < min_trades → no adjustment
        """
        if not self._quality_weight_enabled:
            return weights

        adjusted = dict(weights)
        any_adjusted = False

        for name in adjusted:
            metrics = self._signal_quality.get_metrics(name)
            if metrics.total_trades < self._quality_min_trades:
                continue

            pf = metrics.profit_factor
            if pf >= 1.5:
                mult = 1.0 + min(0.5, (pf - 1.0) * 0.25)
            elif pf >= 1.0:
                mult = 1.0
            else:
                mult = max(0.3, pf)

            adjusted[name] = adjusted[name] * mult
            any_adjusted = True

        if not any_adjusted:
            return weights

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def _get_active_strategies(self) -> list[BaseStrategy]:
        """Return enabled strategies minus the market-specific disabled list."""
        all_strategies = self._registry.get_enabled()
        if not self._disabled_strategies:
            return all_strategies
        return [s for s in all_strategies if s.name not in self._disabled_strategies]

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

    def reload_hard_sl_pct(self, hard_sl_pct: float) -> None:
        """Update hard_sl_pct from config during hot-reload (STOCK-61).

        Called by StrategyRegistry.reload_config() to propagate config changes
        to the evaluation loop without restarting the service.

        Args:
            hard_sl_pct: New hard stop-loss threshold from config (e.g., -0.15).
        """
        old_value = self._hard_sl_pct
        self._hard_sl_pct = hard_sl_pct
        logger.info(
            "Updated hard_sl_pct for %s market: %.2f → %.2f",
            self._market,
            old_value * 100,
            hard_sl_pct * 100,
        )

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

    def set_etf_exclusions(self, symbols: set[str]) -> None:
        """Extend ETF exclusion set (e.g. with KR ETF symbols from config)."""
        self._etf_exclude = self._ETF_ONLY | frozenset(symbols)

    def set_watchlist(self, symbols: list[str]) -> None:
        exclude = getattr(self, "_etf_exclude", self._ETF_ONLY)
        self._watchlist = [s for s in symbols if s not in exclude]

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

            # Run all enabled strategies (filtered by market-specific disabled list)
            strategies = self._get_active_strategies()
            signals: list[Signal] = []
            for strategy in strategies:
                try:
                    signal = await strategy.analyze(df, symbol)
                    signals.append(signal)
                except Exception as e:
                    logger.warning("Strategy %s failed on %s: %s", strategy.name, symbol, e)

            # Get per-stock blended weights
            market_weights = self._registry.get_profile_weights(self._market_state)
            weights = self._adaptive.get_weights(symbol, market_weights)
            weights = self._apply_quality_weights(weights)

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
            # STOCK-65/66: per-market overrides take precedence;
            # fall back to global defaults (0.15 for held, None otherwise).
            combine_kwargs: dict[str, Any] = {
                "min_active_ratio": (
                    self._min_active_ratio
                    if self._min_active_ratio is not None
                    else (0.15 if is_held else None)
                ),
            }
            if self._min_confidence is not None:
                combine_kwargs["min_confidence"] = self._min_confidence
            combined = self._combiner.combine(signals, weights, **combine_kwargs)

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

    def _resolve_strategy_sl_tp(
        self,
        strategy_name: str,
        price: float,
        atr_val: float | None,
        df: "pd.DataFrame",
    ) -> tuple[float, float]:
        """Resolve SL/TP percentages for a position based on strategy YAML.

        Reads the strategy's `stop_loss.type` and `take_profit.type` config
        and converts them into the (sl_pct, tp_pct) tuple that
        position_tracker uses. Falls back to ATR-based dynamic SL/TP if
        the strategy has no SL config or if a required input is missing.

        Supported `stop_loss.type` values:
            fixed_pct  — uses `max_pct` directly (e.g. 0.05 → 5% SL)
            atr        — uses `atr_multiplier` × ATR / price
            supertrend — uses (entry - supertrend_line) / entry from df,
                         falling back to atr × 2.0 if the supertrend column
                         is missing or the line is above price

        2026-04-09: Before this fix, ``evaluation_loop`` ignored every
        strategy YAML stop_loss block and used a single ATR/default SL
        for all positions. This caused 펄어비스 (-7.19%) and 삼성전기
        (-6.72%) where supertrend's tight line-based SL never fired.
        See ``docs/IMPROVEMENT_PLAN.md`` §1 for the full diagnosis.
        """
        from engine.sl_tp_resolver import (
            resolve_strategy_sl_pct,
            resolve_strategy_tp_pct,
        )

        # Try strategy-specific SL config first
        sl_cfg = self._registry.get_stop_loss_config(strategy_name) if hasattr(
            self._registry, "get_stop_loss_config"
        ) else {}
        tp_cfg = self._registry.get_take_profit_config(strategy_name) if hasattr(
            self._registry, "get_take_profit_config"
        ) else {}

        sl_pct = resolve_strategy_sl_pct(sl_cfg, price, atr_val, df)

        # If strategy SL config didn't yield a value, fall back to dynamic ATR
        if sl_pct is None:
            if atr_val and atr_val > 0:
                sl_pct, tp_pct = self._risk_manager.calculate_dynamic_sl_tp(
                    price, atr_val, market=self._market,
                )
                return sl_pct, tp_pct
            sl_pct = self._risk_manager.params.default_stop_loss_pct
            tp_pct = self._risk_manager.params.default_take_profit_pct
            return sl_pct, tp_pct

        # SL came from strategy config — pair it with a TP
        tp_pct = resolve_strategy_tp_pct(tp_cfg, sl_pct)

        logger.info(
            "SL/TP from strategy YAML for %s: type=%s sl=%.1f%% tp=%.1f%%",
            strategy_name, (sl_cfg or {}).get("type"), sl_pct * 100, tp_pct * 100,
        )
        return sl_pct, tp_pct

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

        CON-B1 (2026-06-05): wrapped in `_evaluate_lock` so a manual
        /engine/run-evaluation cannot race the scheduler tick — two
        concurrent passes shared `_last_signal`, `_daily_buy_count`,
        `_cycle_positions`, and could double-fire a BUY for the same
        symbol if `has_pending_order` hadn't yet recorded the first.
        """
        if self._evaluate_lock.locked():
            logger.warning(
                "Skipping concurrent _evaluate_all on %s — another caller "
                "is mid-cycle (CON-B1 guard)", self._market,
            )
            return
        async with self._evaluate_lock:
            await self._evaluate_all_locked()

    async def _evaluate_all_locked(self) -> None:
        """Locked body — same code as before. Split out so callers that
        already hold the lock (none today) can bypass."""
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
        # Cache positions for this cycle so downstream methods don't refetch.
        position_map: dict[str, object] = {}
        try:
            exchange_positions = await self._market_data.get_positions()
            self._cycle_positions = exchange_positions
            exchange_held = {p.symbol for p in exchange_positions if p.quantity > 0}
            held = held | exchange_held
            position_map = {p.symbol: p for p in exchange_positions if p.quantity > 0}
            # 2026-06-05: gap-fill tracker for any held symbols that
            # dropped out. Without this, `_check_min_hold` returns
            # fail-closed for the missing symbol and stuck SELLs (incl.
            # position_cleanup at -3% to -15% PnL) silently block,
            # leaving the position to mature unchecked.
            if self._position_tracker:
                try:
                    held_positions = [
                        p for p in exchange_positions if p.quantity > 0
                    ]
                    await self._position_tracker.ensure_tracked(held_positions)
                except Exception as e:
                    logger.warning(
                        "ensure_tracked failed: %s — min_hold may fail-closed "
                        "on untracked symbols this cycle", e,
                    )
        except Exception as e:
            self._cycle_positions = None
            logger.warning(
                "Exchange position fetch failed, using tracker only: %s", e, exc_info=True
            )

        recovery = set(self._recovery_watch.keys()) - held
        # Filter ETF symbols from held + recovery so ETFs managed by
        # etf_engine don't get evaluated by individual-stock strategies.
        exclude = getattr(self, "_etf_exclude", self._ETF_ONLY)
        filtered_held = sorted(s for s in held if s not in exclude)
        filtered_recovery = sorted(s for s in recovery if s not in exclude)
        eval_symbols = list(dict.fromkeys(self._watchlist + filtered_held + filtered_recovery))

        # Phase 0: Regime-change and sentiment-based protective sells
        sold_in_phase0: set[str] = set()
        if held and self._position_tracker:
            # STOCK-54: Remove symbols sold in Phase 0 from held set so
            # Phase 1 does not treat them as is_held=True (which would
            # cause BUY→HOLD remapping, held_sell_bias, and potential
            # double-sell attempts).
            sold_in_phase0 = await self._check_protective_sells(held)
            if sold_in_phase0:
                held -= sold_in_phase0
                logger.info(
                    "Phase 0 sold %d symbols, removed from held: %s",
                    len(sold_in_phase0),
                    sorted(sold_in_phase0),
                )

        # Phase 1: Collect all signals (no execution yet)
        buy_candidates: list[tuple[float, str, object, pd.DataFrame]] = []

        for symbol in eval_symbols:
            # STOCK-54: Skip symbols already sold in Phase 0 to prevent
            # double-sell attempts.
            if symbol in sold_in_phase0:
                logger.debug("Skipping %s in Phase 1: already sold in Phase 0", symbol)
                continue
            # Cash parking position is system-managed — never let strategies
            # generate BUY/SELL signals on it (would cause double-sell or
            # phantom buy alongside the unpark/park logic).
            if (
                self._cash_parking_enabled
                and symbol == self._cash_parking_symbol
            ):
                continue
            try:
                df = await self._market_data.get_ohlcv(symbol, limit=250)
                if df.empty:
                    continue

                self._maybe_classify(symbol, df)
                df = self._indicator_svc.add_all_indicators(df)

                strategies = self._get_active_strategies()
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
                        logger.warning("Strategy %s failed on %s: %s", strategy.name, symbol, e)

                market_weights = self._registry.get_profile_weights(self._market_state)
                weights = self._adaptive.get_weights(symbol, market_weights)
                weights = self._apply_quality_weights(weights)

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
                    _ckw: dict[str, Any] = {}
                    if self._min_confidence is not None:
                        _ckw["min_confidence"] = self._min_confidence
                    if self._min_active_ratio is not None:
                        _ckw["min_active_ratio"] = self._min_active_ratio
                    combined = self._combiner.combine(signals, weights, **_ckw)

                # Sell on indifference: if no strategy has a strong opinion
                # (HOLD) and the held position is losing beyond threshold,
                # free up capital by selling (STOCK-7).
                # STOCK-47: threshold -3% → -5%, add min hold check
                # P1 (#55, 2026-05-13): add second trigger based on hold
                # time — targets the "bounce_die" cleanup bucket (positions
                # that touched only +1% high then drifted to -5%, missing
                # all existing profit mechanisms). Backtest V3 (2d, -2%):
                # +6.8pp Ret, +0.55 Sharpe, MDD even improved.
                if is_held and combined.signal_type == SignalType.HOLD and symbol in position_map:
                    pos = position_map[symbol]
                    if hasattr(pos, "avg_price") and pos.avg_price > 0:
                        pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
                        _spt = getattr(self, "_stale_pnl_threshold", -0.05)
                        loss_trigger = pnl_pct < _spt
                        # Time trigger: held >= N days + below pnl floor.
                        time_trigger = False
                        st_days = getattr(self, "_stale_time_days", 0)
                        st_thr = getattr(self, "_stale_time_pnl_threshold", 0.0)
                        if (
                            st_days > 0
                            and self._position_tracker is not None
                            and symbol in self._position_tracker._tracked
                            and pnl_pct < st_thr
                        ):
                            tracked = self._position_tracker._tracked[symbol]
                            # Wall-clock preferred — survives process restart.
                            _eu = getattr(tracked, "entry_unix", None)
                            if isinstance(_eu, (int, float)) and _eu > 0:
                                hold_secs = time.time() - float(_eu)
                            else:
                                hold_secs = time.monotonic() - tracked.tracked_at
                            time_trigger = hold_secs >= st_days * 86400

                        if loss_trigger or time_trigger:
                            # STOCK-47: Check min hold (skip for hard SL)
                            hold_ok = True
                            if pnl_pct >= self._hard_sl_pct:
                                hold_ok = self._check_min_hold(symbol)
                            if hold_ok:
                                reason = (
                                    f"Sell on indifference: P&L={pnl_pct:.1%},"
                                    f" no strategy recommends"
                                    if loss_trigger
                                    else f"Stale time exit: held ≥{st_days}d,"
                                         f" P&L={pnl_pct:.1%} below {st_thr:.0%}"
                                )
                                combined = Signal(
                                    signal_type=SignalType.SELL,
                                    confidence=0.50,
                                    strategy_name="position_cleanup",
                                    reason=reason,
                                )
                                logger.info(
                                    "Position cleanup SELL for %s: %s",
                                    symbol, reason,
                                )

                # STOCK-34: Profit protection — sell on high profit when
                # all strategies say HOLD. This is a defense-in-depth
                # safety net: if position_tracker TP doesn't fire and
                # evaluate_exit() doesn't convert to SELL, this ensures
                # highly profitable positions are eventually exited.
                _ppp = getattr(self, "_profit_protection_pct", 0.25)
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

        # Phase 3: Park leftover cash in SPY/KODEX 200 if above threshold.
        # 2026-04-11: Removed the preemptive unpark-before-buys that caused
        # 93 SPY round-trips in 2 days (~860k KRW commissions). Now the
        # parking position is placed ONCE and only sold when the position
        # tracker detects a strategy SELL or when _unpark_if_needed() is
        # called from _execute_signal on actual cash shortfall.
        if self._cash_parking_enabled:
            await self._park_excess_cash()

        # Clear per-cycle position cache
        self._cycle_positions = None

    async def _park_excess_cash(self) -> None:
        """Park idle cash in SPY/KODEX 200 — ONCE, then hold.

        2026-04-11 rewrite: the previous version placed a BUY every
        5-minute evaluation cycle even when SPY was already held, because
        it failed to detect the existing position reliably (pending order
        race + position check timing). This caused 93 SPY round-trips
        in 2 days and ~860k KRW in commissions.

        New invariants:
        - Only BUY when (a) parking enabled AND (b) cash > threshold AND
          (c) NOT already holding parking symbol AND (d) no pending
          parking BUY order exists AND (e) at least 1 hour since last
          park/unpark action (cooldown).
        - Place exactly ONE order per trigger, then stop.
        """
        if not self._cash_parking_enabled:
            return
        sym = self._cash_parking_symbol

        # Cooldown: at most 1 park action per hour
        now = time.time()
        last = getattr(self, "_last_park_action", 0.0)
        if now - last < 3600:
            return

        # Already holding? skip (parking is one-shot).
        # P3-A (2026-05-15): existing parking value vs cap.
        # Skip new buy if already at/over max_pct cap. Otherwise add a
        # per-cycle chunk towards the cap (averaging-in).
        try:
            positions = await self._get_positions()
            existing_value = float(sum(
                p.quantity * (getattr(p, "current_price", 0) or 0)
                for p in positions
                if p.symbol == sym and p.quantity > 0
            ))
        except Exception as e:
            logger.debug("park: position check failed: %s", e)
            return

        # Pending BUY order for parking symbol? skip (don't stack orders)
        if self._order_manager.has_pending_order(sym, "BUY"):
            return

        # Cash check
        try:
            balance = await self._adapter.fetch_balance()
        except Exception as e:
            logger.debug("park: balance fetch failed: %s", e)
            return
        equity = float(getattr(balance, "total", 0.0) or 0.0)
        cash = float(getattr(balance, "available", 0.0) or 0.0)
        if equity <= 0 or cash <= 0:
            return
        cash_pct = cash / equity
        if cash_pct < self._cash_parking_threshold:
            return

        # Caps:
        #   1. headroom = max_pct × equity − existing_value  (don't exceed cap)
        #   2. per-cycle = per_cycle_pct × equity            (averaging-in)
        #   3. cash-buffer = cash − buffer × equity         (keep some cash)
        max_park_value = equity * self._cash_parking_max_pct
        headroom = max_park_value - existing_value
        if headroom <= 0:
            return  # Already at/over cap
        cash_after_buffer = cash - equity * self._cash_parking_buffer
        if cash_after_buffer <= 0:
            return
        per_cycle = equity * self._cash_parking_per_cycle_pct
        non_chunk_cap = min(headroom, cash_after_buffer)
        park_amount = min(non_chunk_cap, per_cycle)
        if park_amount <= 0:
            return
        if park_amount < non_chunk_cap:
            # per_cycle was the binding constraint — log for visibility.
            logger.debug(
                "Cash parking: per-cycle chunk %.0f (headroom=%.0f, cap=%.0f%%, "
                "existing=%.0f)",
                park_amount, headroom,
                self._cash_parking_max_pct * 100, existing_value,
            )

        try:
            df = await self._market_data.get_ohlcv(sym, limit=2)
            if df.empty:
                return
            price = float(df["close"].iloc[-1])
        except Exception as e:
            logger.debug("park: price fetch failed for %s: %s", sym, e)
            return
        if price <= 0:
            return

        quantity = int(park_amount / price)
        if quantity <= 0:
            return

        from engine.risk_manager import PositionSizeResult
        sizing = PositionSizeResult(
            quantity=quantity,
            allocation_usd=quantity * price,
            risk_per_share=0.0,
            reason="cash_parking",
            allowed=True,
        )
        logger.info(
            "Cash parking: buying %d %s @ %.2f (cash=%.0f, equity=%.0f, cash_pct=%.0f%%)",
            quantity, sym, price, cash, equity, cash_pct * 100,
        )
        try:
            exchange = self._exchange_resolver.resolve(sym) if self._exchange_resolver else "NASD"
        except Exception:
            exchange = "NASD" if self._market == "US" else "KRX"
        try:
            order = await self._order_manager.place_buy(
                symbol=sym,
                price=price,
                portfolio_value=equity,
                cash_available=cash,
                current_positions=0,
                strategy_name="cash_parking",
                order_type="limit",
                exchange=exchange,
                sizing_override=sizing,
                skip_position_limit=True,
            )
            if order:
                self._last_park_action = now
                self._cash_parking_parked_at = now
        except Exception as e:
            logger.warning("Cash parking BUY for %s failed: %s", sym, e)

    async def _unpark_for_buy(self, needed: float, available: float) -> None:
        """Sell parking position to free cash for a BUY signal.

        Only sells if parking has been held >= min_hold_days (2 weeks default)
        to ensure round-trip commission is covered by parking gains.
        """
        sym = self._cash_parking_symbol
        min_hold_secs = self._cash_parking_min_hold_days * 86400
        held_secs = time.time() - self._cash_parking_parked_at

        if held_secs < min_hold_secs:
            logger.debug(
                "Unpark skipped for %s: held %.1f days < min %d days",
                sym, held_secs / 86400, self._cash_parking_min_hold_days,
            )
            return

        # Check if parking symbol is actually held
        try:
            positions = await self._get_positions()
        except Exception:
            return
        parked = next((p for p in positions if p.symbol == sym and p.quantity > 0), None)
        if not parked:
            return

        try:
            df = await self._market_data.get_ohlcv(sym, limit=2)
            if df.empty:
                return
            price = float(df["close"].iloc[-1])
        except Exception:
            return

        try:
            exchange = self._exchange_resolver.resolve(sym) if self._exchange_resolver else "NASD"
        except Exception:
            exchange = "NASD" if self._market == "US" else "KRX"

        logger.info(
            "Unpark: selling %d %s @ %.2f (held %.0f days, need %.0f, have %.0f)",
            int(parked.quantity), sym, price, held_secs / 86400, needed, available,
        )
        try:
            await self._order_manager.place_sell(
                symbol=sym,
                price=price,
                quantity=int(parked.quantity),
                strategy_name="cash_parking:unpark",
                order_type="market" if self._market == "US" else "limit",
                exchange=exchange,
            )
            self._cash_parking_parked_at = 0.0  # reset
        except Exception as e:
            logger.warning("Unpark SELL for %s failed: %s", sym, e)

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

        2026-06-04: Fail-closed on missing tracker entry. Previously
        returned True (allow sell) for untracked symbols, which silently
        bypassed the anti-churn gate whenever a position lived in KIS
        balance but not in `_tracked` (race, restart edge case, etc.).
        Live KR session showed 9 round-trips that day in <55 minutes
        despite yaml min_hold_days=3.

        Source of truth is wall-clock `entry_unix` so the gate survives
        process restarts (restored from PositionRecord.opened_at by
        restore_from_db). Falls back to monotonic tracked_at when
        entry_unix is missing (e.g. mocked test fixtures).
        """
        if self._min_hold_secs <= 0:
            return True
        if not self._position_tracker:
            return True
        tracked_dict = getattr(self._position_tracker, "_tracked", None)
        if not tracked_dict or symbol not in tracked_dict:
            logger.warning(
                "Min hold check: %s not in %s tracker — blocking SELL "
                "(fail-closed; investigate why position is untracked).",
                symbol, self._market,
            )
            return False
        tracked = tracked_dict[symbol]

        # Prefer wall-clock — survives process restart via DB opened_at.
        entry_unix = getattr(tracked, "entry_unix", None)
        if isinstance(entry_unix, (int, float)) and entry_unix > 0:
            # max(0, ...) guards against NTP stepping the clock backwards
            # (or a tiny scheduling skew); without it the gate could see
            # a negative hold and treat it as zero-then-pass on the next
            # comparison.
            hold_secs = max(0.0, time.time() - float(entry_unix))
        else:
            # Legacy / mocked: fall back to monotonic tracked_at.
            try:
                hold_secs = max(0.0, time.monotonic() - float(tracked.tracked_at))
            except (TypeError, ValueError):
                logger.warning(
                    "Min hold check: %s has no usable timestamp — blocking SELL.",
                    symbol,
                )
                return False

        if hold_secs < self._min_hold_secs:
            logger.info(
                "Min hold not met for %s: held %.1fh < %.1fh required",
                symbol,
                hold_secs / 3600,
                self._min_hold_secs / 3600,
            )
            return False
        return True

    async def _check_protective_sells(self, held: set[str]) -> set[str]:
        """Sell positions on regime deterioration or negative news sentiment.

        Regime sell: when market transitions to downtrend, sell losing positions
        to protect capital. Winning positions are kept (trailing stop will handle).

        Sentiment sell: when a held stock has strongly negative news sentiment
        (score <= -0.70), sell if held for at least 4 hours (avoid churn).

        Returns:
            Set of symbols that were sold (filled) so the caller can remove
            them from the held set and avoid double-sell in Phase 1 (STOCK-54).
        """
        _BEARISH_REGIMES = {"downtrend"}
        regime_worsened = (
            self._market_state in _BEARISH_REGIMES
            and self._prev_market_state not in _BEARISH_REGIMES
        )

        if not regime_worsened and not self._news_sentiment:
            return set()

        sold_symbols: set[str] = set()
        positions = await self._get_positions()
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
                # STOCK-52: Only untrack when order is confirmed filled.
                # Pending limit orders (status != "filled") are handled by
                # reconciliation via position_tracker.handle_sell_fill().
                # "submitted"/"open" are also non-filled — untrack deferred.
                # "failed"/"cancelled" leave the position tracked so the next
                # check_all cycle can retry the sell automatically.
                if sell_order and self._position_tracker:
                    if sell_order.status == "filled":
                        self._position_tracker.untrack(symbol)
                        # STOCK-54: Track sold symbols so caller can update held set
                        sold_symbols.add(symbol)
                        # Protective sells are always loss sells (regime/sentiment)
                        self.register_sell_cooldown(
                            symbol,
                            time.time(),
                            is_loss=True,
                        )
                    elif sell_order.status in ("submitted", "open", "pending"):
                        # STOCK-54: Sell order in flight — also exclude from Phase 1
                        # to prevent BUY→HOLD remapping while the order is pending.
                        # Untrack is deferred to reconciliation. "failed"/"cancelled"
                        # are intentionally excluded so the next cycle can retry.
                        sold_symbols.add(symbol)

        # Clear processed sentiments to avoid re-selling
        for symbol in held:
            self._news_sentiment.pop(symbol, None)

        return sold_symbols

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

        own_adapter = getattr(self._market_data, "_adapter", None)
        other_adapter = getattr(self._other_market_data, "_adapter", None)
        if self._market == "US":
            us_adapter = own_adapter
            kr_adapter = other_adapter
        else:
            kr_adapter = own_adapter
            us_adapter = other_adapter

        us_position_value_krw = float(getattr(us_adapter, "_us_position_value_krw", 0) or 0)
        kr_tot_evlu_krw = float(getattr(kr_adapter, "_tot_evlu_amt", 0) or 0)
        try:
            other_balance = await self._other_market_data.get_balance()
            kr_total_krw = own_balance_total if self._market == "KR" else other_balance.total
            if kr_tot_evlu_krw > 0 and us_position_value_krw > 0:
                combined_krw = kr_tot_evlu_krw + us_position_value_krw
                return combined_krw / self._exchange_rate if self._market == "US" else combined_krw
            if kr_total_krw > 0 and us_position_value_krw > 0:
                combined_krw = kr_total_krw + us_position_value_krw
                return combined_krw / self._exchange_rate if self._market == "US" else combined_krw

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

        # Hermes Phase 3: stash context for _bump_reject / _persist_funnel_event
        # so every reject + placed-buy automatically records symbol/conf/price.
        self._current_symbol = symbol
        self._current_signal = signal
        self._current_price = price

        if signal.signal_type == SignalType.BUY:
            # F1 funnel: roll counters BEFORE any increments so the first
            # BUY of a new day is attributed to the new day even if it gets
            # rejected at the first gate.
            self._reset_daily_counters_if_needed()
            self._buy_flow_counters["buy_signals_total"] += 1
            # Defense-in-depth min_price gate (#158 follow-up). Catches
            # symbols that bypass ScannerPipeline (universe_expander +
            # KIS ranking can push directly to evaluation). 0 = off.
            if self._min_buy_price > 0 and price < self._min_buy_price:
                logger.info(
                    "Skipping BUY for %s: price $%.2f < min_buy_price $%.2f",
                    symbol, price, self._min_buy_price,
                )
                self._bump_reject("min_price")
                return
            # 2026-04-24: skip BUY during the first N minutes after market open.
            # Live BUY pattern showed ~60% of fills land in the opening 30 min
            # and lose ~5% within 4h (ALM/AMPX whipsaw). SELL signals pass
            # through so held positions can still exit — position_tracker also
            # runs SL/TP via task_position_check independently of this loop.
            if self._opening_avoidance_minutes > 0:
                from engine.scheduler import is_opening_minutes
                if is_opening_minutes(self._market, self._opening_avoidance_minutes):
                    logger.debug(
                        "Skipping BUY for %s: within %dmin post-open avoidance window",
                        symbol, self._opening_avoidance_minutes,
                    )
                    self._bump_reject("opening_avoidance")
                    return

            # Daily buy budget with dynamic confidence escalation
            # As more slots are used, require higher confidence to preserve
            # remaining slots for stronger opportunities later in the day.
            # (Date rollover handled by _reset_daily_counters_if_needed above.)
            daily_limit = self._daily_buy_limit
            override = self._daily_buy_override
            if daily_limit > 0 and self._daily_buy_count >= daily_limit:
                # Hard cap reached — only ultra-high confidence override
                if signal.confidence < override:
                    logger.info(
                        "Skipping BUY for %s: daily limit reached (%d/%d, conf=%.2f < %.2f)",
                        symbol,
                        self._daily_buy_count,
                        daily_limit,
                        signal.confidence,
                        override,
                    )
                    self._bump_reject("daily_limit")
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
                    min_conf = self._daily_buy_escalation_high
                elif usage_ratio >= 0.6:
                    min_conf = self._daily_buy_escalation_low
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
                    self._bump_reject("daily_budget_conf_bar")
                    return

            # Skip if there's already a pending buy order for this symbol
            if self._order_manager.has_pending_order(symbol, "BUY"):
                logger.debug("Skipping BUY for %s: pending order exists", symbol)
                self._bump_reject("pending_order")
                return

            # Skip if we already hold this symbol (prevent duplicate buys).
            # #59-C: when sizing_up is enabled, defer the decision to the
            # exchange-position check below where balance/positions are
            # available — that check computes full eligibility (undersized
            # + confidence + cooldown). Sizing-up converts the reject into
            # an add-on BUY.
            if self._position_tracker and symbol in self._position_tracker.tracked_symbols:
                if not self._sizing_up_enabled:
                    logger.debug("Skipping BUY for %s: already held", symbol)
                    self._bump_reject("already_held")
                    return

            # STOCK-20: Sell cooldown — block BUY for recently-sold symbols.
            # After a stop-loss or strategy sell, wait at least _sell_cooldown_secs
            # before re-buying to prevent sell-then-immediately-rebuy churn.
            #
            # Bypass: when the symbol is still held (partial sell left a
            # placeholder) AND sizing_up is enabled, the BUY signal is an
            # add-on to an existing position, not a re-entry. The
            # whipsaw_block gate (2+ loss sells in 7d) below still catches
            # genuine loser-symbol cases. Live evidence 2026-05-29 EOD:
            # 173/239 KR rejections were sell_cooldown, of which 8 held
            # placeholders kept hitting the gate even though sizing-up
            # eligibility passed — they never reached line ~2004.
            held_for_sizing_up = (
                self._sizing_up_enabled
                and self._position_tracker is not None
                and symbol in self._position_tracker.tracked_symbols
            )
            sell_ts = self._recovery_watch.get(symbol)
            if (
                not held_for_sizing_up
                and sell_ts is not None
                and self._sell_cooldown_secs > 0
            ):
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
                    self._bump_reject("sell_cooldown")
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
                    self._bump_reject("whipsaw_block")
                    return

            # Skip if signal hasn't changed since last evaluation (daily strategies
            # produce the same signal all day — no point re-buying).
            #
            # Exception: when sizing_up_enabled and the symbol is already held,
            # we WANT the repeated daily BUY signal to reach the sizing-up
            # branch (line ~2000) so 1-share placeholders can be grown.
            # Without this bypass, sizing-up is silently no-op: daily strategies
            # re-fire the same signal every 5 min eval cycle and get rejected
            # here long before the sizing-up eligibility check at line ~2004.
            sizing_up_eligible_held = (
                self._sizing_up_enabled
                and self._position_tracker is not None
                and symbol in self._position_tracker.tracked_symbols
            )
            last = self._last_signal.get(symbol)
            if (
                not sizing_up_eligible_held
                and last
                and last[0] == "BUY"
                and time.time() - last[1] < 86400
            ):
                logger.debug("Skipping BUY for %s: same signal within 24h", symbol)
                self._bump_reject("same_signal_24h")
                return
            # CON-H2 (2026-06-05): _last_signal used to be set HERE,
            # before place_buy ran. If the order then failed (network,
            # KIS reject, sizing return None), the 24h dedup gate was
            # still armed and blocked every subsequent attempt for the
            # rest of the day. Now we set it only after the order
            # actually returns from place_buy (search for the matching
            # comment below).

            # Event calendar checks (earnings proximity, FOMC day)
            if self._event_calendar:
                skip, reason = self._event_calendar.should_skip_buy(symbol)
                if skip:
                    logger.info("Skipping BUY for %s: %s", symbol, reason)
                    self._bump_reject("event_calendar")
                    return

            balance = await self._market_data.get_balance()
            positions = await self._get_positions()

            # Sector concentration check.
            # est_cost approximates the largest plausible BUY for this symbol
            # so the sector pre-filter can reject before Kelly sizing is run.
            # 2026-04-25: changed from a fixed `price × 10` to the per-symbol
            # cap (max_position_pct × portfolio_value). The old fixed quantity
            # rejected expensive stocks (EQIX $800 × 10 = $8000 → inflated
            # sector concentration to 126% of a $6.5k US portfolio) even when
            # Kelly would have sized the actual order at 1 share.
            est_cost = (
                self._max_per_symbol_pct * balance.total
                if balance.total > 0
                else price
            )
            if not self._check_sector_limit(symbol, est_cost, positions, balance.total):
                self._bump_reject("sector_limit")
                return

            # Defense-in-depth: block buy if already holding via exchange
            # positions.  This catches duplicates even when position_tracker
            # is empty (e.g. after restart before restore_from_exchange).
            # #59-C: sizing_up_enabled converts already_held into an add-on
            # BUY when the existing position is well below min_position_pct
            # of equity (placeholder symbol from 1-share round-up path).
            existing_match = next(
                (p for p in positions if p.symbol == symbol and p.quantity > 0),
                None,
            )
            if existing_match is not None:
                sizing_up_taken = False
                if self._sizing_up_enabled:
                    cur_value = (
                        getattr(existing_match, "current_price", 0) or price
                    ) * existing_match.quantity
                    eligible, reason = self._is_sizing_up_eligible(
                        symbol, cur_value, balance.total, signal.confidence,
                    )
                    if eligible:
                        logger.info(
                            "Sizing-up BUY for %s: existing %.0f → target ≥%.0f (reason=%s)",
                            symbol,
                            cur_value,
                            self._sizing_up_threshold
                            * self._risk_manager._params.min_position_pct
                            * balance.total,
                            reason,
                        )
                        self._sizing_up_history[symbol] = time.time()
                        sizing_up_taken = True
                    else:
                        logger.debug(
                            "Sizing-up declined for %s: %s", symbol, reason,
                        )
                if not sizing_up_taken:
                    logger.info(
                        "Skipping BUY for %s: already held (exchange positions)",
                        symbol,
                    )
                    self._bump_reject("already_held_exchange")
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
                    self._bump_reject("position_concentration")
                    return

            # Combined portfolio value for integrated margin allocation
            combined_pv = await self._get_combined_portfolio_value(balance.total)

            # STOCK-57: Real per-market invested value. In 통합증거금 accounts
            # balance.total reflects the whole-account total (not just this
            # market), so the risk manager cannot infer market-specific invested
            # from `balance.total - balance.available`. Compute it directly
            # from this market's position list.
            market_invested = sum(
                (getattr(p, "current_price", 0) or 0) * p.quantity
                for p in positions
                if p.quantity > 0
            )

            # Get factor score for this stock
            factor = self._factor_scores.get(symbol)
            factor_score = factor.composite if factor else 0.0

            # Insider confidence adjustment
            confidence = signal.confidence
            if self._event_calendar:
                confidence += self._event_calendar.get_confidence_adjustment(symbol)
                confidence = max(0.0, min(1.0, confidence))

            # D1 sector strength boost: multiply confidence by a factor derived
            # from the symbol's sector's 1w/1m/3m strength score. Neutral when
            # weight=0 or sector is Unknown / not scored yet. See
            # scripts/compare_entry_and_sector_boost.py for 2y sweep evidence.
            if self._sector_boost_weight > 0 and self._sector_scores:
                sector = self._sector_cache.get(symbol, "Unknown")
                strength = self._sector_scores.get(sector)
                if strength is not None:
                    # multiplier = 1 + w * (strength - 50) / 50
                    # strength=100, w=0.3 → 1.30 ; strength=0 → 0.70
                    mult = max(0.1, 1.0 + self._sector_boost_weight * (strength - 50) / 50)
                    boosted = confidence * mult
                    confidence = max(0.0, min(1.0, boosted))

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
                max_drawdown=metrics.max_drawdown,
                market_invested=market_invested,
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

            # Risk-parity: scale position inversely proportional to volatility
            if sizing.allowed and price > 0:
                atr_val = None
                if "atr" in df.columns:
                    atr_val = float(df["atr"].iloc[-1])
                elif "ATRr_14" in df.columns:
                    atr_val = float(df["ATRr_14"].iloc[-1])
                if atr_val and atr_val > 0:
                    atr_pct = atr_val / price
                    # Vol scaling now reads target/min/max from RiskParams
                    # (per-market tuning). KR overrides via yaml block
                    # markets.KR.risk.volatility_scaling.
                    sizing = self._risk_manager.apply_volatility_scaling(
                        sizing, atr_pct, price,
                    )

            if not sizing.allowed:
                logger.info(
                    "Buy rejected for %s: %s",
                    symbol,
                    sizing.reason,
                )
                # Sizing rejection reason is fine-grained (insufficient_cash,
                # max_positions, max_exposure, kelly_zero, etc) — surface the
                # raw token from RiskManager so the funnel can break down why.
                reason_token = (sizing.reason or "sizing_rejected").split(":", 1)[0].strip()
                self._bump_reject(f"sizing_{reason_token}")
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
                        self._bump_reject("risk_agent_block")
                        return
                except Exception as e:
                    logger.debug("Risk agent pre-trade check error: %s", e)

            # Unpark: if cash is insufficient for this buy but parking is held
            # long enough (min_hold_days), sell parking to free cash.
            if (
                self._cash_parking_enabled
                and sizing.allocation_usd > balance.available
                and getattr(self, "_cash_parking_enable_unpark", False)
            ):
                await self._unpark_for_buy(sizing.allocation_usd, balance.available)
                # Refresh balance after unpark
                balance = await self._market_data.get_balance()

            exchange = "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
            # G1: honour signal.suggested_price as a LIMIT below the market.
            # Strategies (currently supertrend with entry_offset_pct) use this
            # to anchor entry near support, refusing to chase intraday spikes.
            # If the strategy didn't set a lower limit, falls through to the
            # market price as before.
            order_price = price
            sp = getattr(signal, "suggested_price", None)
            if sp and sp > 0 and sp < price:
                order_price = float(sp)
                logger.info(
                    "Limit-at-line BUY %s @ $%.2f (vs market $%.2f, gap %.1f%%)",
                    symbol, order_price, price, (price - order_price) / price * 100,
                )

            order = await self._order_manager.place_buy(
                symbol=symbol,
                price=order_price,
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
                self._buy_flow_counters["buys_placed"] += 1
                # CON-H2 (2026-06-05): arm 24h dedup ONLY on successful
                # order placement. Pre-fix the gate was armed at the
                # signal stage so a network error / KIS reject silently
                # blocked the next 24h of attempts.
                self._last_signal[symbol] = ("BUY", time.time())
                # STATE-H3 (2026-06-05): invalidate the per-cycle
                # position cache so downstream sector-cap, sizing-up
                # and max-positions checks within this same cycle
                # see the BUY that just executed. Without this, two
                # rapid BUYs in the same cycle could both pass the
                # sector-cap check against the pre-BUY snapshot.
                self._cycle_positions = None
                # Hermes Phase 3: record placed buy for counterfactual
                # baseline (so replay knows which signals already passed
                # all gates, not just which were rejected).
                asyncio.create_task(self._persist_funnel_event(
                    symbol, decision="placed",
                ))

            # Register position for SL/TP/trailing stop monitoring
            if order and self._position_tracker:
                # Dynamic ATR-based SL/TP per stock volatility (default fallback)
                atr_val = None
                if "atr" in df.columns:
                    atr_val = float(df["atr"].iloc[-1])
                elif "ATRr_14" in df.columns:
                    atr_val = float(df["ATRr_14"].iloc[-1])

                # Strategy-specific SL config (yaml `stop_loss.type`).
                # 2026-04-09: this used to be silently ignored — every position
                # got the same ATR/default SL regardless of what the strategy
                # YAML said. That caused 펄어비스 (-7.19%) and 삼성전기 (-6.72%)
                # losses where supertrend's intended tight line-based SL never
                # fired. See docs/IMPROVEMENT_PLAN.md §1 for the diagnosis.
                sl_pct, tp_pct = self._resolve_strategy_sl_tp(
                    strategy_name, price, atr_val, df,
                )

                # Look up trailing stop config from strategy YAML
                trail_act: float | None = None
                trail_pct: float | None = None
                trail_cfg = self._registry.get_trailing_stop_config(strategy_name)
                if trail_cfg and trail_cfg.get("enabled", False):
                    trail_act = trail_cfg.get("activation_pct")
                    trail_pct = trail_cfg.get("trail_pct")

                # 2026-06-05 (FIN-B1): sizing-up add-on BUYs must blend
                # the cost basis with the existing lot. Calling track()
                # for an already-tracked symbol used to overwrite
                # entry_price / quantity / highest_price — every KR
                # sizing-up event from 2026-05-22 onward corrupted PnL
                # math and made SL fire on the wrong anchor. Now we
                # route to add_on(), which preserves the original
                # entry_unix (min_hold clock) and partial_profit_taken
                # while blending entry_price.
                if symbol in self._position_tracker.tracked_symbols:
                    self._position_tracker.add_on(
                        symbol=symbol,
                        add_price=price,
                        add_quantity=order.quantity,
                        strategy=strategy_name,
                        stop_loss_pct=sl_pct,
                        take_profit_pct=tp_pct,
                        trailing_activation_pct=trail_act,
                        trailing_stop_pct=trail_pct,
                    )
                else:
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
            # 2026-04-18: Smart sell escalation. If a pending limit SELL
            # exists from a previous cycle, cancel it and escalate to
            # market order (US only). This prevents the XLC scenario where
            # limit SELL at $119.48 was cancelled because price dropped to
            # $118.52. First attempt is always limit (better price for
            # liquid stocks), escalation to market only on retry.
            has_pending = self._order_manager.has_pending_order(symbol, "SELL")
            if has_pending and self._market != "US":
                logger.debug("Skipping SELL for %s: pending sell order exists (KR)", symbol)
                return
            # US: if pending, we'll cancel + escalate below

            positions = await self._get_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos and pos.quantity > 0:
                exchange = (
                    "KRX" if self._market == "KR" else self._exchange_resolver.resolve(symbol)
                )
                orig_strategy = ""
                if self._position_tracker:
                    orig_strategy = self._position_tracker.get_buy_strategy(symbol)
                sell_order_type = "limit"
                if has_pending and self._market == "US":
                    logger.info("SELL escalation for %s: pending limit unfilled → cancel + market", symbol)
                    await self._order_manager.cancel_pending_orders(symbol, "SELL")
                    sell_order_type = "market"

                sell_order = await self._order_manager.place_sell(
                    symbol=symbol,
                    quantity=int(pos.quantity),
                    price=price,
                    strategy_name=signal.strategy_name,
                    order_type=sell_order_type,
                    exchange=exchange,
                    entry_price=pos.avg_price,
                    buy_strategy=orig_strategy,
                )
                # STOCK-52: Only untrack when order is confirmed filled.
                # Pending limit orders (status != "filled") are handled by
                # reconciliation via position_tracker.handle_sell_fill().
                # "failed"/"cancelled" leave the position tracked so the next
                # evaluation cycle can retry the sell automatically.
                if sell_order and self._position_tracker:
                    if sell_order.status == "filled":
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
            # Wall-clock preferred — survives process restart.
            _eu = getattr(tracked, "entry_unix", None)
            if isinstance(_eu, (int, float)) and _eu > 0:
                hold_secs = time.time() - float(_eu)
            else:
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
            logger.warning("Failed to build position context for %s: %s", symbol, e)
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
