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

import pandas as pd

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
        )

        # Track ETF positions managed by this engine
        self._managed_positions: dict[str, ETFPosition] = {}
        self._last_regime: MarketRegime | None = None
        self._last_top_sectors: list[str] = []

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
                    price = float(pos.current_price) if pos.current_price else 0
                    sell_result = await self._order_manager.place_sell(
                        symbol=sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_regime",
                        exchange=self._etf.get_exchange(sym),
                    )
                    if sell_result is None:
                        logger.warning("ETF Engine: SELL %s failed", sym)
                        continue
                    self._managed_positions.pop(sym, None)
                    actions.append(f"SELL {sym} (regime={regime.value})")
                    logger.info("ETF Engine: SELL %s on regime %s", sym, regime.value)

        # Enter target ETFs
        for sym in target_etfs:
            if sym not in held_symbols:
                try:
                    df = await self._market_data.get_ohlcv(sym, limit=5)
                    if df.empty:
                        continue
                    price = float(df.iloc[-1]["close"])

                    # Position sizing: bear entries use reduced size
                    max_alloc = balance.total * self._risk.max_single_etf_pct
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
                    size_note = f" (half-size)" if is_bear_entry else ""
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
                    price = float(pos.current_price) if pos.current_price else 0
                    sell_result = await self._order_manager.place_sell(
                        symbol=etf_sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_sector",
                        exchange=self._etf.get_exchange(etf_sym),
                    )
                    if sell_result is None:
                        logger.warning("ETF Engine: SELL %s (sector) failed", etf_sym)
                        continue
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
                    await self._order_manager.place_sell(
                        symbol=sym,
                        quantity=int(pos.quantity),
                        price=price,
                        strategy_name="etf_engine_hold_limit",
                        exchange=self._etf.get_exchange(sym),
                    )
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
        }
