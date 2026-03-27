"""ETF Trading Engine.

Manages leveraged/inverse ETF trading independently from the stock engine:
1. Regime-based leveraged pair switching (TQQQ <-> SQQQ based on market state)
2. Sector ETF rotation (XLK, XLF, etc. based on sector strength)
3. ETF-specific risk rules (max hold days, position limits)

Uses MarketStateDetector for regime signals and SectorAnalyzer for sector rotation.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select, desc

from data.market_data_service import MarketDataService
from data.market_state import MarketStateDetector, MarketRegime, MarketState
from scanner.etf_universe import ETFUniverse
from scanner.sector_analyzer import SectorAnalyzer, SectorScore
from engine.order_manager import OrderManager
from engine.risk_manager import RiskManager, RiskParams
from services.notification import NotificationService

logger = logging.getLogger(__name__)


@dataclass
class ETFPosition:
    """Track an ETF position with entry metadata."""
    symbol: str
    entry_date: float  # timestamp
    reason: str  # "regime_bull", "regime_bear", "sector_rotation"
    sector: str = ""


@dataclass
class ETFRiskParams:
    """ETF-specific risk parameters (from etf_universe.yaml)."""
    max_hold_days: int = 10
    max_portfolio_pct: float = 0.30
    max_single_etf_pct: float = 0.15
    require_stop_loss: bool = True
    default_stop_loss_pct: float = 0.08
    min_hold_leveraged_hours: int = 4  # Min hold for leveraged ETFs before selling
    min_hold_sector_hours: int = 2     # Min hold for sector ETFs before selling
    sell_cooldown_hours: int = 4       # Prevent rebuy of same ETF for N hours after sell


class ETFEngine:
    """Dedicated engine for leveraged/inverse ETF and sector ETF trading."""

    def __init__(
        self,
        market_data: MarketDataService,
        order_manager: OrderManager,
        etf_universe: ETFUniverse | None = None,
        sector_analyzer: SectorAnalyzer | None = None,
        notification: NotificationService | None = None,
        max_regime_etfs: int = 2,
        max_sector_etfs: int = 3,
        bear_min_distance_pct: float = -5.0,
        bear_min_confidence: float = 0.7,
        bear_size_ratio: float = 0.4,
        market: str = "US",
        risk_manager: RiskManager | None = None,
    ):
        self._market_data = market_data
        self._order_manager = order_manager
        self._market = market
        self._risk_manager = risk_manager
        self._etf = etf_universe or ETFUniverse()
        self._sector_analyzer = sector_analyzer or SectorAnalyzer()
        self._notification = notification
        self._max_regime_etfs = max_regime_etfs
        self._max_sector_etfs = max_sector_etfs
        # Bear (inverse) ETF entry conditions — stricter than bull.
        # Backtest: naive bear entry loses money on regime whipsaws.
        # Require SPY to be clearly below SMA200 and high confidence.
        self._bear_min_distance = bear_min_distance_pct  # SPY must be this % below SMA200
        self._bear_min_confidence = bear_min_confidence  # MarketState confidence threshold
        self._bear_size_ratio = bear_size_ratio  # Position size = bull size × this ratio

        # ETF-specific risk params from config
        rules = self._etf.risk_rules
        self._risk = ETFRiskParams(
            max_hold_days=rules.max_hold_days,
            max_portfolio_pct=rules.max_portfolio_pct,
            max_single_etf_pct=rules.max_single_etf_pct,
            require_stop_loss=rules.require_stop_loss,
            default_stop_loss_pct=rules.default_stop_loss_pct,
            min_hold_leveraged_hours=rules.min_hold_leveraged_hours,
            min_hold_sector_hours=rules.min_hold_sector_hours,
            sell_cooldown_hours=rules.sell_cooldown_hours,
        )

        # Regime-adaptive allocation: stronger trend → bigger ETF positions
        # Keep modest to avoid excessive leveraged exposure on top of stock positions
        self._regime_alloc_pct: dict[str, float] = {
            "strong_uptrend": 0.10,  # 10% per ETF → max 20% nominal, 60% effective
            "uptrend": 0.07,         # 7% per ETF → max 14% nominal, 42% effective
            "sideways": 0.00,        # Exit all leveraged
            "downtrend": 0.05,       # 5% per ETF (× bear_size_ratio → 2%)
        }

        # Track ETF positions managed by this engine
        self._managed_positions: dict[str, ETFPosition] = {}
        self._last_regime: MarketRegime | None = None
        self._last_top_sectors: list[str] = []

        # Track when each ETF was last sold (for sell_cooldown).
        # NOTE: This state is in-memory only. On process restart (crash, deploy,
        # systemd restart) all cooldown timestamps are lost, which may allow
        # immediate re-buys that should still be cooling down. A future improvement
        # would persist these to Redis with a TTL equal to sell_cooldown_hours, or
        # seed from same-day sell records in the orders DB on startup.
        self._last_sell_times: dict[str, float] = {}  # symbol -> unix timestamp

    async def evaluate(
        self,
        market_state: MarketState,
        sector_data: dict[str, dict] | None = None,
    ) -> dict:
        """Main evaluation: regime ETFs + sector ETFs.

        Args:
            market_state: Current market state from MarketStateDetector.
            sector_data: Sector performance data for sector rotation.

        Returns:
            Summary dict of actions taken.
        """
        actions = {"regime": [], "sector": [], "risk": []}

        # Fetch positions and balance once for all steps
        positions = await self._market_data.get_positions()
        balance = await self._market_data.get_balance()

        # Apply market-level allocation cap
        if self._risk_manager:
            capped_total, capped_avail = self._risk_manager._apply_market_cap(
                balance.total, balance.available, self._market,
            )
            balance = type(balance)(
                total=capped_total, available=capped_avail,
                locked=balance.locked, currency=balance.currency,
            )

        # Step 1: Check hold-day limits on existing positions
        expired = await self._check_hold_limits(positions)
        actions["risk"].extend(expired)

        # Step 2: Regime-based leveraged ETF management
        regime_actions = await self._manage_regime_etfs(market_state, positions, balance)
        actions["regime"].extend(regime_actions)

        # Re-fetch after regime trades so sector rotation sees updated balance
        if regime_actions:
            positions = await self._market_data.get_positions()
            balance = await self._market_data.get_balance()

        # Step 3: Sector ETF rotation
        if sector_data:
            sector_actions = await self._manage_sector_etfs(sector_data, positions, balance)
            actions["sector"].extend(sector_actions)

        # Step 4: Check total ETF exposure
        exposure_actions = self._check_exposure_limits(positions, balance)
        actions["risk"].extend(exposure_actions)

        return actions

    def _can_sell_etf(self, symbol: str) -> tuple[bool, str]:
        """Check if an ETF can be sold based on min_hold requirement.

        Args:
            symbol: ETF symbol to check.

        Returns:
            (can_sell, reason) tuple. reason is empty if can_sell=True.
        """
        pos = self._managed_positions.get(symbol)
        if not pos:
            # Position exists on broker but not tracked in-memory (e.g. after restart
            # before restore_managed_positions runs). Allow the sell but warn so
            # operators can detect the gap and investigate.
            logger.warning(
                "ETF Engine: _can_sell_etf: %s not in _managed_positions — "
                "allowing sell without min_hold check (may have restarted before "
                "restore_managed_positions ran)",
                symbol,
            )
            return True, ""

        now = time.time()
        hold_seconds = now - pos.entry_date

        # Determine min hold based on ETF type
        if self._etf.is_leveraged(symbol):
            min_hold_seconds = self._risk.min_hold_leveraged_hours * 3600
        else:
            min_hold_seconds = self._risk.min_hold_sector_hours * 3600

        if hold_seconds < min_hold_seconds:
            hold_hours = hold_seconds / 3600
            min_hold_hours = min_hold_seconds / 3600
            reason = f"held only {hold_hours:.1f}h (min={min_hold_hours:.0f}h)"
            return False, reason

        return True, ""

    def _can_buy_etf(self, symbol: str) -> tuple[bool, str]:
        """Check if an ETF can be bought based on sell_cooldown.

        Args:
            symbol: ETF symbol to check.

        Returns:
            (can_buy, reason) tuple. reason is empty if can_buy=True.
        """
        last_sell_time = self._last_sell_times.get(symbol)
        if last_sell_time is None:
            return True, ""

        now = time.time()
        cooldown_seconds = now - last_sell_time
        min_cooldown_seconds = self._risk.sell_cooldown_hours * 3600

        if cooldown_seconds < min_cooldown_seconds:
            remaining_hours = (min_cooldown_seconds - cooldown_seconds) / 3600
            reason = f"cooldown {remaining_hours:.1f}h remaining"
            return False, reason

        return True, ""

    async def _manage_regime_etfs(
        self, state: MarketState, positions=None, balance=None,
    ) -> list[str]:
        """Switch leveraged ETFs based on market regime.

        Bull entry: SPY above SMA200 (confirmed) → buy TQQQ, SOXL
        Bear entry: SPY below SMA200 by >3%, confidence >0.7, half-size → buy SQQQ, SOXS
        Sideways: exit all leveraged to cash
        """
        actions = []
        regime = state.regime

        # No change in regime -> no action
        if regime == self._last_regime:
            return actions

        # First eval after restart: record current regime without trading.
        # Prevents spurious buy attempts every time the server restarts.
        if self._last_regime is None:
            logger.info(
                "ETF Engine: initial regime detected as %s (confidence=%.2f) — no action on first eval",
                regime.value, state.confidence,
            )
            self._last_regime = regime
            return actions

        logger.info(
            "Regime change: %s -> %s (confidence=%.2f, SPY dist=%.1f%%)",
            self._last_regime, regime.value, state.confidence,
            state.spy_distance_pct,
        )

        # Use provided positions/balance or fetch
        if positions is None:
            positions = await self._market_data.get_positions()
        if balance is None:
            balance = await self._market_data.get_balance()
        held_symbols = {p.symbol for p in positions}
        current_count = len(positions)

        # Determine target direction and ETFs
        target_etfs: list[str] = []
        exit_etfs: list[str] = []
        is_bear_entry = False

        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND):
            target_etfs = self._etf.get_regime_etfs("bull")[:self._max_regime_etfs]
            exit_etfs = self._etf.get_regime_etfs("bear")
        elif regime == MarketRegime.DOWNTREND:
            # Always exit bull positions on downtrend
            exit_etfs = self._etf.get_regime_etfs("bull")
            # Bear (inverse) entry requires stricter conditions:
            #   1. SPY must be clearly below SMA200 (distance < threshold)
            #   2. High confidence from MarketStateDetector
            if (state.spy_distance_pct <= self._bear_min_distance
                    and state.confidence >= self._bear_min_confidence):
                target_etfs = self._etf.get_regime_etfs("bear")[:self._max_regime_etfs]
                is_bear_entry = True
                logger.info(
                    "Bear entry qualified: SPY dist=%.1f%% (threshold=%.1f%%), "
                    "confidence=%.2f (threshold=%.2f)",
                    state.spy_distance_pct, self._bear_min_distance,
                    state.confidence, self._bear_min_confidence,
                )
            else:
                logger.info(
                    "Bear entry skipped: SPY dist=%.1f%% (need <=%.1f%%), "
                    "confidence=%.2f (need >=%.2f) — exit to cash",
                    state.spy_distance_pct, self._bear_min_distance,
                    state.confidence, self._bear_min_confidence,
                )
        else:  # SIDEWAYS
            exit_etfs = (
                self._etf.get_regime_etfs("bull")
                + self._etf.get_regime_etfs("bear")
            )

        # Exit opposite-regime positions
        for sym in exit_etfs:
            if sym in held_symbols:
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos and pos.quantity > 0:
                    # Check min_hold before selling
                    can_sell, reason = self._can_sell_etf(sym)
                    if not can_sell:
                        logger.info(
                            "ETF Engine: SKIP SELL %s (regime=%s) — min_hold constraint: %s",
                            sym, regime.value, reason,
                        )
                        continue

                    price = float(pos.current_price) if pos.current_price else 0
                    sell_result = await self._order_manager.place_sell(
                        symbol=sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_regime",
                        exchange=self._etf.get_exchange(sym),
                        entry_price=pos.avg_price,
                        buy_strategy="etf_engine",
                    )
                    if sell_result is None:
                        logger.warning("ETF Engine: SELL %s failed", sym)
                        continue
                    self._last_sell_times[sym] = time.time()
                    self._managed_positions.pop(sym, None)
                    held_symbols.discard(sym)  # keep set current so sibling check is accurate
                    actions.append(f"SELL {sym} (regime={regime.value})")
                    logger.info("ETF Engine: SELL %s on regime %s", sym, regime.value)

        # Enter target ETFs (sell conflicting siblings first for mutual exclusivity)
        for sym in target_etfs:
            if sym not in held_symbols:
                # Mutual exclusivity: ensure no sibling (base/bull/bear) remains held.
                # Case A: sibling in exit_etfs — already handled above, but may have
                #         been skipped by min_hold. If it's still in held_symbols, the
                #         exit sell was blocked, so we must not buy sym either.
                # Case B: sibling not in exit_etfs — try to sell now, but if min_hold
                #         blocks the sell, skip buying sym to avoid holding both sides.
                siblings = self._etf.get_pair_siblings(sym)
                skip_buy_for_sibling = False
                for sib in siblings:
                    if sib not in held_symbols:
                        continue  # already cleared or never held

                    if sib in exit_etfs:
                        # Exit loop already tried to sell this sibling.
                        # Still in held_symbols → sell did not complete (min_hold blocked or order failed).
                        logger.info(
                            "ETF Engine: SKIP BUY %s — sibling %s still held "
                            "(exit sell did not complete)",
                            sym, sib,
                        )
                        skip_buy_for_sibling = True
                        break

                    # Sibling not in exit_etfs — try to sell it now
                    pos = next((p for p in positions if p.symbol == sib), None)
                    if pos:  # quantity > 0 guaranteed by held_symbols construction
                        can_sell_sib, reason_sib = self._can_sell_etf(sib)
                        if not can_sell_sib:
                            logger.info(
                                "ETF Engine: SKIP BUY %s — sibling %s min_hold not met: %s",
                                sym, sib, reason_sib,
                            )
                            skip_buy_for_sibling = True
                            break  # stop checking siblings; can't buy anyway

                        price_sib = float(pos.current_price) if pos.current_price else 0
                        sell_result = await self._order_manager.place_sell(
                            symbol=sib,
                            quantity=int(pos.quantity),
                            price=price_sib,
                            strategy_name="etf_engine_regime",
                            exchange=self._etf.get_exchange(sib),
                            entry_price=pos.avg_price,
                            buy_strategy="etf_engine",
                        )
                        if sell_result:
                            self._last_sell_times[sib] = time.time()
                            self._managed_positions.pop(sib, None)
                            held_symbols.discard(sib)
                            actions.append(f"SELL {sib} (mutual exclusivity for {sym})")
                            logger.info(
                                "ETF Engine: SELL %s (sibling conflict with target %s)",
                                sib, sym,
                            )
                        else:
                            # Order failed — sibling still held; can't guarantee exclusivity
                            logger.warning(
                                "ETF Engine: SELL %s (sibling of %s) failed — skipping buy",
                                sib, sym,
                            )
                            skip_buy_for_sibling = True
                            break

                if skip_buy_for_sibling:
                    continue

                # Check sell_cooldown before buying
                can_buy, reason = self._can_buy_etf(sym)
                if not can_buy:
                    logger.info(
                        "ETF Engine: SKIP BUY %s (regime=%s) — sell cooldown: %s",
                        sym, regime.value, reason,
                    )
                    continue

                try:
                    df = await self._market_data.get_ohlcv(sym, limit=5)
                    if df.empty:
                        continue
                    price = float(df.iloc[-1]["close"])

                    # Position sizing: regime-adaptive allocation
                    regime_pct = self._regime_alloc_pct.get(
                        regime.value, self._risk.max_single_etf_pct,
                    )
                    max_alloc = balance.total * regime_pct
                    if is_bear_entry:
                        max_alloc *= self._bear_size_ratio
                    alloc = min(max_alloc, balance.available * 0.9)
                    if alloc < price:
                        continue

                    result = await self._order_manager.place_buy(
                        symbol=sym,
                        price=price,
                        portfolio_value=balance.total,
                        cash_available=alloc,
                        current_positions=current_count,
                        strategy_name="etf_engine_regime",
                        exchange=self._etf.get_exchange(sym),
                        skip_position_limit=True,
                    )
                    if result is None:
                        logger.warning("ETF Engine: BUY %s failed — skipping", sym)
                        continue
                    self._managed_positions[sym] = ETFPosition(
                        symbol=sym,
                        entry_date=time.time(),
                        reason=f"regime_{regime.value}",
                    )
                    current_count += 1
                    size_note = " (half-size)" if is_bear_entry else ""
                    actions.append(f"BUY {sym} (regime={regime.value}){size_note}")
                    logger.info("ETF Engine: BUY %s on regime %s%s", sym, regime.value, size_note)
                except Exception as e:
                    logger.warning("ETF Engine: failed to buy %s: %s", sym, e)

        self._last_regime = regime

        if actions and self._notification:
            await self._notification.notify_system_event(
                "etf_regime_switch",
                f"Regime: {regime.value}\nActions: {', '.join(actions)}",
            )

        return actions

    async def _manage_sector_etfs(
        self, sector_data: dict, positions=None, balance=None,
    ) -> list[str]:
        """Rotate sector ETFs based on relative strength.

        Buy top N sector ETFs, sell bottom sectors.
        """
        actions = []

        scores = self._sector_analyzer.analyze(sector_data)
        if not scores:
            return actions

        top = self._sector_analyzer.get_top_sectors(
            scores, n=self._max_sector_etfs, min_score=60,
        )
        bottom = self._sector_analyzer.get_bottom_sectors(scores, n=3)

        top_names = [s.name for s in top]
        bottom_names = [s.name for s in bottom]

        # No change in top sectors -> no action
        if sorted(top_names) == sorted(self._last_top_sectors):
            return actions

        logger.info("Sector rotation: top=%s, bottom=%s", top_names, bottom_names)

        if positions is None:
            positions = await self._market_data.get_positions()
        if balance is None:
            balance = await self._market_data.get_balance()
        held_symbols = {p.symbol for p in positions}
        current_count = len(positions)

        # Sell ETFs in bottom sectors
        sectors = self._etf.get_all_sectors()
        for name in bottom_names:
            sec = sectors.get(name)
            if not sec:
                continue
            etf_sym = sec.etf
            if etf_sym in held_symbols:
                pos = next((p for p in positions if p.symbol == etf_sym), None)
                if pos and pos.quantity > 0:
                    # Check min_hold before selling
                    can_sell, reason = self._can_sell_etf(etf_sym)
                    if not can_sell:
                        logger.info(
                            "ETF Engine: SKIP SELL %s (sector %s weak) — min_hold: %s",
                            etf_sym, name, reason,
                        )
                        continue

                    price = float(pos.current_price) if pos.current_price else 0
                    sell_result = await self._order_manager.place_sell(
                        symbol=etf_sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_sector",
                        exchange=self._etf.get_exchange(etf_sym),
                        entry_price=pos.avg_price,
                        buy_strategy="etf_engine",
                    )
                    if sell_result is None:
                        logger.warning("ETF Engine: SELL %s (sector) failed", etf_sym)
                        continue
                    self._last_sell_times[etf_sym] = time.time()
                    self._managed_positions.pop(etf_sym, None)
                    actions.append(f"SELL {etf_sym} ({name} weak)")

        # Buy ETFs in top sectors
        for score in top:
            sec = sectors.get(score.name)
            if not sec:
                continue
            etf_sym = sec.etf
            if etf_sym in held_symbols:
                continue

            # Check sell_cooldown before buying
            can_buy, reason = self._can_buy_etf(etf_sym)
            if not can_buy:
                logger.info(
                    "ETF Engine: SKIP BUY %s (sector %s top) — sell cooldown: %s",
                    etf_sym, score.name, reason,
                )
                continue

            try:
                df = await self._market_data.get_ohlcv(etf_sym, limit=5)
                if df.empty:
                    continue
                price = float(df.iloc[-1]["close"])

                max_alloc = balance.total * self._risk.max_single_etf_pct
                alloc = min(max_alloc, balance.available * 0.9)
                if alloc < price:
                    continue

                result = await self._order_manager.place_buy(
                    symbol=etf_sym,
                    price=price,
                    portfolio_value=balance.total,
                    cash_available=alloc,
                    current_positions=current_count,
                    strategy_name="etf_engine_sector",
                    exchange=self._etf.get_exchange(etf_sym),
                    skip_position_limit=True,
                )
                if result is None:
                    logger.warning("ETF Engine: BUY %s (sector) failed — skipping", etf_sym)
                    continue
                self._managed_positions[etf_sym] = ETFPosition(
                    symbol=etf_sym,
                    entry_date=time.time(),
                    reason="sector_rotation",
                    sector=score.name,
                )
                current_count += 1
                actions.append(
                    f"BUY {etf_sym} ({score.name} strength={score.strength_score:.0f})"
                )
            except Exception as e:
                logger.warning("ETF Engine: failed to buy %s: %s", etf_sym, e)

        self._last_top_sectors = top_names

        if actions and self._notification:
            await self._notification.notify_system_event(
                "etf_sector_rotation",
                f"Top sectors: {top_names}\nActions: {', '.join(actions)}",
            )

        return actions

    async def _check_hold_limits(self, positions=None) -> list[str]:
        """Sell leveraged ETFs held beyond max_hold_days."""
        actions = []
        now = time.time()
        max_seconds = self._risk.max_hold_days * 86400

        to_remove = []
        for sym, etf_pos in self._managed_positions.items():
            if not self._etf.is_leveraged(sym):
                continue
            hold_seconds = now - etf_pos.entry_date
            if hold_seconds > max_seconds:
                if positions is None:
                    positions = await self._market_data.get_positions()
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos and pos.quantity > 0:
                    price = float(pos.current_price) if pos.current_price else 0
                    sell_result = await self._order_manager.place_sell(
                        symbol=sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_hold_limit",
                        exchange=self._etf.get_exchange(sym),
                        entry_price=pos.avg_price,
                        buy_strategy="etf_engine",
                    )
                    if sell_result is None:
                        logger.warning(
                            "ETF Engine: SELL failed for %s — retaining tracking for retry",
                            sym,
                        )
                        continue
                    self._last_sell_times[sym] = time.time()
                    hold_days = hold_seconds / 86400
                    actions.append(
                        f"SELL {sym} (held {hold_days:.0f}d > {self._risk.max_hold_days}d limit)"
                    )
                    logger.info(
                        "ETF Engine: SELL %s — held %.0f days (limit=%d)",
                        sym, hold_days, self._risk.max_hold_days,
                    )
                to_remove.append(sym)

        for sym in to_remove:
            self._managed_positions.pop(sym, None)

        return actions

    def _check_exposure_limits(self, positions=None, balance=None) -> list[str]:
        """Warn if total ETF exposure exceeds max_portfolio_pct."""
        actions = []

        if not balance or not positions or balance.total <= 0:
            return actions

        etf_value = 0.0
        for pos in positions:
            if self._etf.is_leveraged(pos.symbol) or pos.symbol in [
                s.etf for s in self._etf.get_all_sectors().values()
            ]:
                etf_value += pos.current_price * pos.quantity if pos.current_price else 0

        exposure_pct = etf_value / balance.total
        if exposure_pct > self._risk.max_portfolio_pct:
            msg = (
                f"ETF exposure {exposure_pct:.1%} exceeds limit "
                f"{self._risk.max_portfolio_pct:.0%}"
            )
            actions.append(msg)
            logger.warning("ETF Engine: %s", msg)

        return actions

    async def restore_managed_positions(
        self, session_factory: Any = None,
    ) -> list[dict]:
        """Restore managed ETF positions from broker + DB on startup.

        Fetches current positions from the exchange adapter, filters to known
        ETF symbols, and rebuilds _managed_positions with entry metadata from
        the orders DB table. This ensures hold-day limits, regime transitions,
        and status reporting work correctly after a server restart.

        Follows the same pattern as PositionTracker.restore_from_exchange().

        Returns:
            List of restored position summaries.
        """
        restored: list[dict] = []
        try:
            positions = await self._market_data.get_positions()
        except Exception as e:
            logger.error("ETF Engine restore: failed to fetch positions: %s", e)
            return restored

        # Build set of all ETF symbols this engine manages
        all_etf_symbols = set(self._etf.all_etf_symbols)

        # Filter to ETF positions with quantity > 0
        etf_positions = [
            p for p in positions
            if p.symbol in all_etf_symbols and p.quantity > 0
        ]
        if not etf_positions:
            logger.info("ETF Engine restore: no ETF positions found on broker")
            return restored

        # Look up entry info from DB orders table
        entry_info: dict[str, dict] = {}
        if session_factory:
            try:
                # Lazy import: core.models requires the full application stack
                # (DB, SQLAlchemy), so it is imported inside the try block to keep
                # the engine testable without the full stack and so ImportError is
                # caught like any other DB failure.
                from core.models import Order
                async with session_factory() as session:
                    for pos in etf_positions:
                        # Skip if already tracked (idempotent)
                        if pos.symbol in self._managed_positions:
                            continue
                        stmt = (
                            select(Order)
                            .where(
                                Order.symbol == pos.symbol,
                                Order.side == "BUY",
                                Order.strategy_name.like("etf_engine_%"),
                                Order.status.in_(["filled", "submitted"]),
                                Order.is_paper.is_(False),
                            )
                            .order_by(desc(Order.created_at))
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        order = result.scalar_one_or_none()
                        if order:
                            entry_info[pos.symbol] = {
                                "strategy_name": order.strategy_name,
                                "created_at": order.created_at,
                            }
            except Exception as e:
                logger.warning(
                    "ETF Engine restore: DB lookup failed, will use defaults: %s", e,
                )

        # Seed _last_sell_times from same-day SELL orders so cooldowns survive restarts.
        # Only seeds cooldowns that are still active (sold within the last
        # sell_cooldown_hours window) to avoid blocking buys for old sells.
        if session_factory:
            try:
                from core.models import Order  # already cached in sys.modules if first import succeeded
                cutoff = datetime.now(timezone.utc) - timedelta(
                    hours=self._risk.sell_cooldown_hours,
                )
                async with session_factory() as session:
                    stmt = (
                        select(Order)
                        .where(
                            Order.symbol.in_(list(all_etf_symbols)),
                            Order.side == "SELL",
                            Order.strategy_name.like("etf_engine_%"),
                            Order.status.in_(["filled", "submitted"]),
                            Order.is_paper.is_(False),
                            Order.created_at >= cutoff,
                        )
                        .order_by(desc(Order.created_at))
                    )
                    result = await session.execute(stmt)
                    sell_orders = result.scalars().all()

                seeded: list[str] = []
                for order in sell_orders:
                    sym = order.symbol
                    if sym in self._last_sell_times:
                        continue  # already have a more recent sell time
                    created_at = order.created_at
                    if isinstance(created_at, datetime):
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        self._last_sell_times[sym] = created_at.timestamp()
                        seeded.append(sym)
                    else:
                        logger.warning(
                            "ETF Engine restore: unexpected created_at type %s for %s"
                            " — cooldown not seeded",
                            type(created_at).__name__, sym,
                        )

                if seeded:
                    logger.info(
                        "ETF Engine restore: seeded sell_cooldown for %d symbol(s): %s",
                        len(seeded), seeded,
                    )
            except Exception as e:
                logger.warning(
                    "ETF Engine restore: sell-cooldown seeding failed (non-fatal): %s", e,
                )

        # Pre-compute sector mapping (invariant across positions)
        all_sectors = self._etf.get_all_sectors()

        # Rebuild _managed_positions
        for pos in etf_positions:
            sym = pos.symbol
            if sym in self._managed_positions:
                # Already tracked (idempotent)
                restored.append({
                    "symbol": sym,
                    "quantity": int(pos.quantity),
                    "reason": self._managed_positions[sym].reason,
                    "sector": self._managed_positions[sym].sector,
                    "source": "already_tracked",
                })
                continue

            # Determine reason from DB order or infer from ETF type
            info = entry_info.get(sym)
            if info:
                strategy = info["strategy_name"]
                if strategy == "etf_engine_regime":
                    reason = "regime_restored"
                elif strategy == "etf_engine_sector":
                    reason = "sector_rotation"
                else:
                    reason = f"restored_{strategy}"

                # Use actual order creation time for hold-day tracking
                created_at = info["created_at"]
                if isinstance(created_at, datetime):
                    # Ensure timezone-aware before converting to timestamp.
                    # DB stores UTC but may return naive datetime depending
                    # on driver — treat naive as UTC (matches DB convention).
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    entry_date = created_at.timestamp()
                else:
                    entry_date = time.time()
            else:
                # No DB record — infer from ETF type
                if self._etf.is_leveraged(sym):
                    reason = "regime_restored"
                else:
                    reason = "sector_rotation"
                entry_date = time.time()  # lose hold-day accuracy

            # Determine sector for sector ETFs
            sector = ""
            for sec_name, sec_info in all_sectors.items():
                if sec_info.etf == sym:
                    sector = sec_name
                    break

            self._managed_positions[sym] = ETFPosition(
                symbol=sym,
                entry_date=entry_date,
                reason=reason,
                sector=sector,
            )
            restored.append({
                "symbol": sym,
                "quantity": int(pos.quantity),
                "reason": reason,
                "sector": sector,
                "source": "exchange" if info else "inferred",
            })
            logger.info(
                "ETF Engine restore: %s — reason=%s, sector=%s, source=%s",
                sym, reason, sector, "db" if info else "inferred",
            )

        if restored:
            logger.info(
                "ETF Engine: restored %d managed positions: %s",
                len(restored), [r["symbol"] for r in restored],
            )
        return restored

    def get_status(self) -> dict:
        """Return current ETF engine status."""
        return {
            "last_regime": self._last_regime.value if self._last_regime else None,
            "top_sectors": self._last_top_sectors,
            "managed_positions": {
                sym: {
                    "reason": p.reason,
                    "sector": p.sector,
                    "hold_days": (time.time() - p.entry_date) / 86400,
                }
                for sym, p in self._managed_positions.items()
            },
            "risk_params": {
                "max_hold_days": self._risk.max_hold_days,
                "max_portfolio_pct": self._risk.max_portfolio_pct,
                "max_single_etf_pct": self._risk.max_single_etf_pct,
            },
            "bear_config": {
                "min_distance_pct": self._bear_min_distance,
                "min_confidence": self._bear_min_confidence,
                "size_ratio": self._bear_size_ratio,
            },
            "regime_allocation_pct": self._regime_alloc_pct,
        }
