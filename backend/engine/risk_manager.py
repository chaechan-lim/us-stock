"""Risk manager for position sizing, stop-loss, and portfolio limits.

Enforces:
- Per-position max allocation (fixed or Kelly-based)
- Total portfolio exposure limits
- Stop-loss / take-profit / trailing stop
- Daily loss limit
"""

import logging
from dataclasses import dataclass

from analytics.position_sizing import KellyPositionSizer, KellyResult

logger = logging.getLogger(__name__)


@dataclass
class RiskParams:
    max_position_pct: float = 0.10  # Max 10% per position
    max_total_exposure_pct: float = 0.90  # Max 90% invested
    max_positions: int = 20
    daily_loss_limit_pct: float = 0.03  # Stop trading at 3% daily loss
    default_stop_loss_pct: float = 0.08
    default_take_profit_pct: float = 0.20
    # Market-level fund allocation (share of total portfolio)
    market_allocations: dict[str, float] | None = None  # e.g. {"US": 0.5, "KR": 0.5}
    # Profit-taking: sell a portion of position at intermediate gain levels
    profit_taking_enabled: bool = True
    profit_taking_threshold_pct: float = 0.10  # 10% gain triggers partial sell
    profit_taking_sell_ratio: float = 0.50  # sell 50% of remaining position
    # Default trailing stop (used when strategy config doesn't specify)
    default_trailing_activation_pct: float = 0.06  # activate after 6% gain
    default_trailing_stop_pct: float = 0.03  # trail 3% from peak
    # Tiered trailing stop: tighter trail at higher gain levels (STOCK-24)
    # List of (gain_threshold, trail_pct) tuples, sorted by gain ascending.
    # At 10% gain → 5% trail, at 15% → 4%, at 20% → 3%.
    tiered_trailing_tiers: list[tuple[float, float]] | None = None
    # Breakeven stop: ratchet SL upward as gain approaches TP (STOCK-24)
    breakeven_stop_enabled: bool = True
    breakeven_stop_activation_ratio: float = 0.50  # activate at 50% of TP
    breakeven_stop_lock_ratio: float = 0.75  # at 75% of TP, lock 50% of gain
    breakeven_stop_lock_pct: float = 0.50  # lock this fraction of peak gain


@dataclass
class PositionSizeResult:
    quantity: int
    allocation_usd: float
    risk_per_share: float
    reason: str
    allowed: bool = True


# Regime-adaptive position sizing and exposure
REGIME_POSITION_PCT: dict[str, float] = {
    "strong_uptrend": 0.08,  # Max per position (diversified)
    "uptrend": 0.07,
    "sideways": 0.06,
    "downtrend": 0.04,  # Defensive
}
REGIME_EXPOSURE_PCT: dict[str, float] = {
    "strong_uptrend": 0.95,
    "uptrend": 0.90,
    "sideways": 0.70,
    "downtrend": 0.40,  # Mostly cash
}


class RiskManager:
    """Enforce risk rules before order placement."""

    def __init__(self, params: RiskParams | None = None):
        self._params = params or RiskParams()
        self._daily_pnl: float = 0.0
        self._market_regimes: dict[str, str] = {}  # {"US": "bull", "KR": "sideways"}
        self._eval_regime: str = "uptrend"  # Current regime for adaptive sizing
        self._kelly = KellyPositionSizer(
            max_position_pct=self._params.max_position_pct,
        )

    def set_eval_regime(self, regime: str) -> None:
        """Set current market regime for adaptive position/exposure sizing."""
        self._eval_regime = regime

    def _get_regime_position_pct(self) -> float:
        """Get max position % for current regime."""
        return REGIME_POSITION_PCT.get(self._eval_regime, self._params.max_position_pct)

    def _get_regime_exposure_pct(self) -> float:
        """Get max exposure % for current regime."""
        return REGIME_EXPOSURE_PCT.get(self._eval_regime, self._params.max_total_exposure_pct)

    def set_market_regime(self, market: str, regime: str) -> None:
        """Update market regime for dynamic allocation boost.

        Args:
            market: "US" or "KR"
            regime: "bull", "bear", or "sideways"
        """
        self._market_regimes[market] = regime

    def get_effective_allocation(self, market: str) -> float | None:
        """Get effective allocation for a market, including regime boost.

        Bull market gets +10% boost (from the other market's share),
        bear market gets -10% penalty. Total always sums to ≤100%.
        """
        allocs = self._params.market_allocations
        if not allocs or market not in allocs:
            return None

        base = allocs[market]
        regime = self._market_regimes.get(market, "sideways")
        boost = 0.0
        if regime == "bull":
            boost = 0.20  # Bull gets extra 20%
        elif regime == "bear":
            boost = -0.20  # Bear gives up 20%

        effective = max(0.20, min(0.70, base + boost))  # Clamp 20%-70%
        return effective

    def _apply_market_cap(
        self,
        portfolio_value: float,
        cash_available: float,
        market: str | None = None,
        combined_portfolio_value: float | None = None,
    ) -> tuple[float, float]:
        """Apply market-level allocation cap to portfolio_value and cash_available.

        If market allocations are configured (e.g. US=50%, KR=50%),
        each market can only use its share of the total portfolio for sizing.
        Regime-aware: bull markets get a boost, bear markets get reduced.

        When combined_portfolio_value is provided (integrated margin accounts),
        the cap is applied to the combined total instead of the per-market total.
        This ensures 50% means 50% of the whole account, not 50% of one market's view.
        """
        if not market:
            return portfolio_value, cash_available
        cap_pct = self.get_effective_allocation(market)
        if cap_pct is None:
            return portfolio_value, cash_available
        base = combined_portfolio_value if combined_portfolio_value else portfolio_value
        capped_portfolio = base * cap_pct
        # STOCK-53: Don't cap at portfolio_value for integrated margin (통합증거금)
        # accounts. When combined_portfolio_value is provided, the allocation
        # can exceed this market's adapter-reported total because the shared
        # deposit pool allows cross-market capacity. Safety is ensured by
        # min(capped_cash, cash_available) below — real cash is never exceeded.
        if not combined_portfolio_value:
            # Single-market mode: cap at this market's own total
            capped_portfolio = min(capped_portfolio, portfolio_value)
        # Preserve actual invested amount so exposure check works correctly.
        invested = portfolio_value - cash_available
        capped_cash = max(0.0, capped_portfolio - invested)
        capped_cash = min(capped_cash, cash_available)  # can't exceed real cash
        return capped_portfolio, capped_cash

    def _resolve_existing_value(
        self,
        portfolio_value: float,
        existing_position_value: float,
        existing_symbol_exposure: float,
    ) -> float:
        """Resolve effective existing position value from absolute or exposure inputs.

        When both are provided, returns the larger value (conservative).

        Args:
            portfolio_value: Total portfolio value.
            existing_position_value: Absolute value of existing position (USD/KRW).
            existing_symbol_exposure: Existing position as fraction of portfolio (0.0–1.0).
        """
        effective = existing_position_value
        if existing_symbol_exposure > 0 and portfolio_value > 0:
            exposure_value = existing_symbol_exposure * portfolio_value
            effective = max(effective, exposure_value)
        return effective

    def _check_concentration_limit(
        self,
        symbol: str,
        existing_position_value: float,
        portfolio_value: float,
        existing_symbol_exposure: float = 0.0,
    ) -> PositionSizeResult | None:
        """Check if existing position exceeds per-symbol concentration limit.

        Returns a rejection PositionSizeResult if the existing position is at
        or above max_position_pct, otherwise returns None (OK to proceed).
        STOCK-26: Shared by both calculate_position_size and
        calculate_kelly_position_size to avoid logic drift.
        STOCK-30: Added existing_symbol_exposure (fraction) as complementary
        input to existing_position_value (absolute).
        """
        effective_value = self._resolve_existing_value(
            portfolio_value,
            existing_position_value,
            existing_symbol_exposure,
        )
        if effective_value > 0 and portfolio_value > 0:
            existing_pct = effective_value / portfolio_value
            max_pct = self._params.max_position_pct
            if existing_pct >= max_pct:
                return PositionSizeResult(
                    quantity=0,
                    allocation_usd=0,
                    risk_per_share=0,
                    reason=(
                        f"Already holding {symbol} at {existing_pct:.1%} (>= {max_pct:.0%} limit)"
                    ),
                    allowed=False,
                )
        return None

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        cash_available: float,
        current_positions: int,
        atr: float | None = None,
        market: str | None = None,
        combined_portfolio_value: float | None = None,
        existing_position_value: float = 0.0,
        existing_symbol_exposure: float = 0.0,
    ) -> PositionSizeResult:
        """Calculate allowed position size given risk constraints.

        Args:
            existing_position_value: Current value of any existing position in
                this symbol. Used to enforce per-symbol concentration limit
                (STOCK-26). If existing value already exceeds max_position_pct,
                the buy is rejected.
            existing_symbol_exposure: Existing position as a fraction of
                portfolio (0.0–1.0). Complementary to existing_position_value;
                when both are provided, the higher implied value is used
                (STOCK-30).
        """
        # STOCK-56: Preserve original portfolio_value before market-cap is applied.
        # The daily loss limit must be computed against the full portfolio value,
        # not the market-capped slice. With 50:50 allocation the capped value is
        # halved, which would double the apparent loss percentage and trigger a
        # premature trading halt.
        uncapped_portfolio_value = portfolio_value
        portfolio_value, cash_available = self._apply_market_cap(
            portfolio_value,
            cash_available,
            market,
            combined_portfolio_value,
        )

        # STOCK-26 + STOCK-30: Per-symbol concentration check
        rejection = self._check_concentration_limit(
            symbol,
            existing_position_value,
            portfolio_value,
            existing_symbol_exposure,
        )
        if rejection:
            return rejection

        # Check position limit
        if current_positions >= self._params.max_positions:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason=f"Max positions reached ({self._params.max_positions})",
                allowed=False,
            )

        # Check total exposure limit
        reject = self._check_exposure_limit(portfolio_value, cash_available)
        if reject:
            return reject

        # Check daily loss limit using uncapped portfolio value (STOCK-56).
        # Using the market-capped value here would inflate the loss percentage
        # proportionally to the allocation split and cause premature halts.
        if self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / uncapped_portfolio_value
            if daily_loss_pct >= self._params.daily_loss_limit_pct:
                return PositionSizeResult(
                    quantity=0,
                    allocation_usd=0,
                    risk_per_share=0,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1%})",
                    allowed=False,
                )

        # Max allocation per position (regime-adaptive), minus existing position value
        max_alloc = portfolio_value * self._get_regime_position_pct()
        effective_existing = self._resolve_existing_value(
            portfolio_value,
            existing_position_value,
            existing_symbol_exposure,
        )
        if effective_existing > 0:
            max_alloc = max(0.0, max_alloc - effective_existing)

        # Respect cash available (with buffer) — also respect exposure headroom
        max_from_cash = cash_available * 0.95
        exposure_headroom = self._get_exposure_headroom(portfolio_value, cash_available)
        allocation = min(max_alloc, max_from_cash, exposure_headroom)

        if allocation <= 0 or price <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="No cash available",
                allowed=False,
            )

        quantity = int(allocation / price)
        if quantity <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="Price too high for allocation",
                allowed=False,
            )

        risk_per_share = price * self._params.default_stop_loss_pct

        return PositionSizeResult(
            quantity=quantity,
            allocation_usd=quantity * price,
            risk_per_share=risk_per_share,
            reason="OK",
            allowed=True,
        )

    def calculate_kelly_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        cash_available: float,
        current_positions: int,
        win_rate: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        signal_confidence: float = 0.5,
        factor_score: float = 0.0,
        market: str | None = None,
        combined_portfolio_value: float | None = None,
        existing_position_value: float = 0.0,
        existing_symbol_exposure: float = 0.0,
    ) -> PositionSizeResult:
        """Kelly-enhanced position sizing.

        Uses Kelly Criterion when trade history is available,
        falls back to fixed sizing otherwise. Factor score and
        signal confidence scale position size up/down.

        Args:
            existing_position_value: Current value of any existing position in
                this symbol (STOCK-26). Rejects if at/above max_position_pct.
            existing_symbol_exposure: Existing position as a fraction of
                portfolio (0.0–1.0). Complementary to existing_position_value
                (STOCK-30).
        """
        # STOCK-56: Preserve original portfolio_value before market-cap is applied.
        uncapped_portfolio_value = portfolio_value
        portfolio_value, cash_available = self._apply_market_cap(
            portfolio_value,
            cash_available,
            market,
            combined_portfolio_value,
        )

        # STOCK-26 + STOCK-30: Per-symbol concentration check (shared helper)
        rejection = self._check_concentration_limit(
            symbol,
            existing_position_value,
            portfolio_value,
            existing_symbol_exposure,
        )
        if rejection:
            return rejection

        # Standard risk checks first
        if current_positions >= self._params.max_positions:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason=f"Max positions reached ({self._params.max_positions})",
                allowed=False,
            )

        # Check total exposure limit
        reject = self._check_exposure_limit(portfolio_value, cash_available)
        if reject:
            return reject

        # Check daily loss limit using uncapped portfolio value (STOCK-56).
        if self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / uncapped_portfolio_value
            if daily_loss_pct >= self._params.daily_loss_limit_pct:
                return PositionSizeResult(
                    quantity=0,
                    allocation_usd=0,
                    risk_per_share=0,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1%})",
                    allowed=False,
                )

        # Calculate exposure headroom for capping allocations
        exposure_headroom = self._get_exposure_headroom(portfolio_value, cash_available)

        # Try Kelly sizing if we have trade history
        if win_rate > 0 and avg_win > 0 and avg_loss > 0:
            kelly_result = self._kelly.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                signal_confidence=signal_confidence,
                factor_score=factor_score,
                portfolio_value=portfolio_value,
            )

            if kelly_result.final_allocation_pct > 0:
                allocation = portfolio_value * kelly_result.final_allocation_pct
                # STOCK-26 + STOCK-30: Subtract effective existing value so
                # total concentration stays within max_position_pct.
                effective_existing = self._resolve_existing_value(
                    portfolio_value,
                    existing_position_value,
                    existing_symbol_exposure,
                )
                if effective_existing > 0:
                    allocation = max(0.0, allocation - effective_existing)
                allocation = min(allocation, cash_available * 0.95, exposure_headroom)

                if allocation > 0 and price > 0:
                    quantity = int(allocation / price)
                    if quantity > 0:
                        logger.info(
                            "Kelly sizing %s: kelly=%.3f, conf_boost=%.2f, "
                            "factor_boost=%.2f, alloc=%.1f%%",
                            symbol,
                            kelly_result.kelly_fraction,
                            kelly_result.confidence_boost,
                            kelly_result.factor_boost,
                            kelly_result.final_allocation_pct * 100,
                        )
                        return PositionSizeResult(
                            quantity=quantity,
                            allocation_usd=quantity * price,
                            risk_per_share=price * self._params.default_stop_loss_pct,
                            reason=f"Kelly (f={kelly_result.kelly_fraction:.3f})",
                            allowed=True,
                        )

            # Negative Kelly: no edge detected, fall through to minimum sizing
            # (Don't block — combiner confidence already gates entry quality)

        # Fallback: fixed sizing with factor/confidence adjustment
        base_pct = self._get_regime_position_pct()
        # Adjust by factor score: positive factor → up to 1.3x, negative → down to 0.7x
        import numpy as np

        factor_mult = 1.0 + 0.3 * np.tanh(factor_score) if factor_score != 0 else 1.0
        # Adjust by confidence: scale linearly — low confidence gets much smaller position
        # conf=0.3→0.58, conf=0.5→0.70, conf=0.7→0.82, conf=0.9→0.94, conf=1.0→1.0
        conf_mult = 0.4 + 0.6 * min(signal_confidence, 1.0) if signal_confidence > 0 else 0.4
        adjusted_pct = base_pct * factor_mult * conf_mult
        adjusted_pct = min(adjusted_pct, self._params.max_position_pct)

        allocation = portfolio_value * adjusted_pct
        # STOCK-26 + STOCK-30: Subtract effective existing value so total
        # concentration stays within max_position_pct.
        effective_existing = self._resolve_existing_value(
            portfolio_value,
            existing_position_value,
            existing_symbol_exposure,
        )
        if effective_existing > 0:
            allocation = max(0.0, allocation - effective_existing)
        allocation = min(allocation, cash_available * 0.95, exposure_headroom)

        if allocation <= 0 or price <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="No cash available",
                allowed=False,
            )

        quantity = int(allocation / price)
        if quantity <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="Price too high for allocation",
                allowed=False,
            )

        return PositionSizeResult(
            quantity=quantity,
            allocation_usd=quantity * price,
            risk_per_share=price * self._params.default_stop_loss_pct,
            reason=f"Fixed+factors (conf={conf_mult:.2f}, factor={factor_mult:.2f})",
            allowed=True,
        )

    def _check_exposure_limit(
        self,
        portfolio_value: float,
        cash_available: float,
        exposure_limit: float | None = None,
    ) -> PositionSizeResult | None:
        """Return rejection result if total exposure exceeds limit."""
        if portfolio_value <= 0:
            return None
        limit = exposure_limit or self._get_regime_exposure_pct()
        invested = portfolio_value - cash_available
        exposure = invested / portfolio_value
        if exposure >= limit:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason=f"Max exposure reached ({exposure:.0%} >= {limit:.0%})",
                allowed=False,
            )
        return None

    def _get_exposure_headroom(
        self,
        portfolio_value: float,
        cash_available: float,
    ) -> float:
        """How much more can be invested before hitting exposure limit."""
        if portfolio_value <= 0:
            return 0.0
        limit = self._get_regime_exposure_pct()
        max_invested = portfolio_value * limit
        current_invested = portfolio_value - cash_available
        return max(0.0, max_invested - current_invested)

    def calculate_extended_hours_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        cash_available: float,
        current_positions: int,
        max_position_pct: float = 0.03,
        max_positions: int = 5,
        market: str | None = None,
        combined_portfolio_value: float | None = None,
        existing_position_value: float = 0.0,
        existing_symbol_exposure: float = 0.0,
    ) -> PositionSizeResult:
        """Conservative position sizing for extended hours trading.

        Uses tighter limits than regular session:
        - 3% max per position (vs 8% regular)
        - 5 max positions (vs 20 regular)
        - Limit orders only (enforced by caller)

        Args:
            existing_position_value: Current value of any existing position in
                this symbol. Used to enforce per-symbol concentration limit
                (STOCK-32). If existing value already exceeds max_position_pct,
                the buy is rejected.
            existing_symbol_exposure: Existing position as a fraction of
                portfolio (0.0-1.0). Complementary to existing_position_value;
                when both are provided, the higher implied value is used
                (STOCK-32).
        """
        # STOCK-56: Preserve original portfolio_value before market-cap is applied.
        uncapped_portfolio_value = portfolio_value
        portfolio_value, cash_available = self._apply_market_cap(
            portfolio_value,
            cash_available,
            market,
            combined_portfolio_value,
        )

        # STOCK-32: Per-symbol concentration check (shared helper)
        rejection = self._check_concentration_limit(
            symbol,
            existing_position_value,
            portfolio_value,
            existing_symbol_exposure,
        )
        if rejection:
            return rejection

        if current_positions >= max_positions:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason=f"Extended hours max positions ({max_positions})",
                allowed=False,
            )

        reject = self._check_exposure_limit(portfolio_value, cash_available)
        if reject:
            return reject

        # Check daily loss limit using uncapped portfolio value (STOCK-56).
        if self._daily_pnl < 0 and uncapped_portfolio_value > 0:
            daily_loss_pct = abs(self._daily_pnl) / uncapped_portfolio_value
            if daily_loss_pct >= self._params.daily_loss_limit_pct:
                return PositionSizeResult(
                    quantity=0,
                    allocation_usd=0,
                    risk_per_share=0,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1%})",
                    allowed=False,
                )

        # STOCK-32: Subtract effective existing value so total
        # concentration stays within max_position_pct.
        max_alloc = portfolio_value * max_position_pct
        effective_existing = self._resolve_existing_value(
            portfolio_value,
            existing_position_value,
            existing_symbol_exposure,
        )
        if effective_existing > 0:
            max_alloc = max(0.0, max_alloc - effective_existing)
        max_from_cash = cash_available * 0.95
        exposure_headroom = self._get_exposure_headroom(portfolio_value, cash_available)
        allocation = min(max_alloc, max_from_cash, exposure_headroom)

        if allocation <= 0 or price <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="No cash available (extended hours)",
                allowed=False,
            )

        quantity = int(allocation / price)
        if quantity <= 0:
            return PositionSizeResult(
                quantity=0,
                allocation_usd=0,
                risk_per_share=0,
                reason="Price too high for extended hours allocation",
                allowed=False,
            )

        return PositionSizeResult(
            quantity=quantity,
            allocation_usd=quantity * price,
            risk_per_share=price * self._params.default_stop_loss_pct,
            reason="OK (extended hours)",
            allowed=True,
        )

    def calculate_dynamic_sl_tp(
        self,
        price: float,
        atr: float,
        market: str = "US",
    ) -> tuple[float, float]:
        """Calculate ATR-based stop-loss and take-profit percentages.

        Uses ATR relative to price to adapt SL/TP to each stock's volatility.
        Higher volatility → wider SL/TP, lower volatility → tighter SL/TP.

        Returns:
            (stop_loss_pct, take_profit_pct)
        """
        if price <= 0 or atr <= 0:
            return self._params.default_stop_loss_pct, self._params.default_take_profit_pct

        atr_pct = atr / price  # e.g. 0.02 = 2% daily ATR

        # SL = 2x ATR, clamped to [3%, 15%] for US, [5%, 20%] for KR
        # TP capped at realistic levels (was 30%, lowered to avoid unreachable targets)
        if market == "KR":
            sl = max(0.05, min(0.20, atr_pct * 2.5))
            tp = max(0.08, min(0.25, atr_pct * 4.0))
        else:
            sl = max(0.03, min(0.15, atr_pct * 2.0))
            tp = max(0.06, min(0.20, atr_pct * 3.5))

        return round(sl, 4), round(tp, 4)

    def check_stop_loss(
        self, entry_price: float, current_price: float, stop_loss_pct: float | None = None
    ) -> bool:
        """Return True if stop-loss is triggered."""
        sl = stop_loss_pct or self._params.default_stop_loss_pct
        return current_price <= entry_price * (1 - sl)

    def check_take_profit(
        self, entry_price: float, current_price: float, take_profit_pct: float | None = None
    ) -> bool:
        """Return True if take-profit is triggered."""
        tp = take_profit_pct or self._params.default_take_profit_pct
        return current_price >= entry_price * (1 + tp)

    def check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        activation_pct: float = 0.05,
        trail_pct: float = 0.03,
    ) -> bool:
        """Return True if trailing stop is triggered.

        Trailing stop activates after price rises by activation_pct
        from entry, then triggers if price drops trail_pct from peak.
        """
        if entry_price <= 0 or highest_price <= 0:
            return False
        # activation_pct=0 or trail_pct=0 means trailing stop is disabled
        if activation_pct <= 0 or trail_pct <= 0:
            return False
        gain_from_entry = (highest_price - entry_price) / entry_price
        if gain_from_entry < activation_pct:
            return False  # Not yet activated

        drop_from_peak = (highest_price - current_price) / highest_price
        return drop_from_peak >= trail_pct

    def check_tiered_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
    ) -> bool:
        """Return True if tiered trailing stop is triggered.

        Unlike the flat trailing stop, this uses gain-dependent trail
        percentages: the higher the unrealized gain, the tighter the trail.
        This protects large gains without cutting early winners.

        Tiers are checked from highest to lowest gain threshold.
        The first matching tier (highest gain) determines the trail %.
        """
        tiers = self._params.tiered_trailing_tiers
        if not tiers:
            return False
        if entry_price <= 0 or highest_price <= 0:
            return False

        gain_from_entry = (highest_price - entry_price) / entry_price

        # Sort tiers descending by gain threshold, pick highest matching
        sorted_tiers = sorted(tiers, key=lambda t: t[0], reverse=True)
        for gain_threshold, trail_pct in sorted_tiers:
            if gain_from_entry >= gain_threshold and trail_pct > 0:
                drop_from_peak = (highest_price - current_price) / highest_price
                return drop_from_peak >= trail_pct

        return False  # No tier matched (gain below all thresholds)

    def check_breakeven_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        take_profit_pct: float | None = None,
    ) -> bool:
        """Return True if breakeven stop is triggered.

        Ratchets the stop-loss upward as price approaches TP:
        - At activation_ratio * TP: SL moves to entry price (breakeven)
        - At lock_ratio * TP: SL moves to entry + lock_pct * peak_gain

        This prevents large winners from turning into losers. The stop
        only tightens (never loosens) — it uses highest_price to determine
        what gain level was reached, ensuring the ratchet is monotonic.
        """
        if not self._params.breakeven_stop_enabled:
            return False
        if entry_price <= 0 or highest_price <= 0:
            return False

        tp = take_profit_pct or self._params.default_take_profit_pct
        if tp <= 0:
            return False

        # Use highest_price to determine which tier we've reached
        peak_gain_pct = (highest_price - entry_price) / entry_price
        activation_gain = tp * self._params.breakeven_stop_activation_ratio
        lock_gain = tp * self._params.breakeven_stop_lock_ratio

        if peak_gain_pct < activation_gain:
            return False  # Haven't reached activation level yet

        # Determine the ratcheted stop price
        if peak_gain_pct >= lock_gain:
            # Lock a fraction of peak gain
            stop_price = entry_price * (1 + peak_gain_pct * self._params.breakeven_stop_lock_pct)
        else:
            # Breakeven: stop at entry price
            stop_price = entry_price

        return current_price <= stop_price

    def update_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl += pnl

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    @property
    def params(self) -> RiskParams:
        return self._params

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl
