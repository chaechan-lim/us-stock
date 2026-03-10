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


@dataclass
class PositionSizeResult:
    quantity: int
    allocation_usd: float
    risk_per_share: float
    reason: str
    allowed: bool = True


class RiskManager:
    """Enforce risk rules before order placement."""

    def __init__(self, params: RiskParams | None = None):
        self._params = params or RiskParams()
        self._daily_pnl: float = 0.0
        self._market_regimes: dict[str, str] = {}  # {"US": "bull", "KR": "sideways"}
        self._kelly = KellyPositionSizer(
            max_position_pct=self._params.max_position_pct,
        )

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
        self, portfolio_value: float, cash_available: float, market: str | None = None,
    ) -> tuple[float, float]:
        """Apply market-level allocation cap to portfolio_value and cash_available.

        If market allocations are configured (e.g. US=50%, KR=50%),
        each market can only use its share of the total portfolio for sizing.
        Regime-aware: bull markets get a boost, bear markets get reduced.
        """
        if not market:
            return portfolio_value, cash_available
        cap_pct = self.get_effective_allocation(market)
        if cap_pct is None:
            return portfolio_value, cash_available
        capped_portfolio = portfolio_value * cap_pct
        capped_cash = min(cash_available, capped_portfolio)
        return capped_portfolio, capped_cash

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        cash_available: float,
        current_positions: int,
        atr: float | None = None,
        market: str | None = None,
    ) -> PositionSizeResult:
        """Calculate allowed position size given risk constraints."""
        portfolio_value, cash_available = self._apply_market_cap(
            portfolio_value, cash_available, market,
        )

        # Check position limit
        if current_positions >= self._params.max_positions:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
                reason=f"Max positions reached ({self._params.max_positions})",
                allowed=False,
            )

        # Check total exposure limit
        reject = self._check_exposure_limit(portfolio_value, cash_available)
        if reject:
            return reject

        # Check daily loss limit
        if self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / portfolio_value
            if daily_loss_pct >= self._params.daily_loss_limit_pct:
                return PositionSizeResult(
                    quantity=0, allocation_usd=0, risk_per_share=0,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1%})",
                    allowed=False,
                )

        # Max allocation per position
        max_alloc = portfolio_value * self._params.max_position_pct

        # Respect cash available (with buffer) — also respect exposure headroom
        max_from_cash = cash_available * 0.95
        exposure_headroom = self._get_exposure_headroom(portfolio_value, cash_available)
        allocation = min(max_alloc, max_from_cash, exposure_headroom)

        if allocation <= 0 or price <= 0:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
                reason="No cash available",
                allowed=False,
            )

        quantity = int(allocation / price)
        if quantity <= 0:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
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
    ) -> PositionSizeResult:
        """Kelly-enhanced position sizing.

        Uses Kelly Criterion when trade history is available,
        falls back to fixed sizing otherwise. Factor score and
        signal confidence scale position size up/down.
        """
        portfolio_value, cash_available = self._apply_market_cap(
            portfolio_value, cash_available, market,
        )

        # Standard risk checks first
        if current_positions >= self._params.max_positions:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
                reason=f"Max positions reached ({self._params.max_positions})",
                allowed=False,
            )

        # Check total exposure limit
        reject = self._check_exposure_limit(portfolio_value, cash_available)
        if reject:
            return reject

        if self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / portfolio_value
            if daily_loss_pct >= self._params.daily_loss_limit_pct:
                return PositionSizeResult(
                    quantity=0, allocation_usd=0, risk_per_share=0,
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
                allocation = min(allocation, cash_available * 0.95, exposure_headroom)

                if allocation > 0 and price > 0:
                    quantity = int(allocation / price)
                    if quantity > 0:
                        logger.info(
                            "Kelly sizing %s: kelly=%.3f, conf_boost=%.2f, "
                            "factor_boost=%.2f, alloc=%.1f%%",
                            symbol, kelly_result.kelly_fraction,
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

            if kelly_result.kelly_fraction <= 0:
                return PositionSizeResult(
                    quantity=0, allocation_usd=0, risk_per_share=0,
                    reason=f"Kelly negative ({kelly_result.kelly_fraction:.3f}): no edge",
                    allowed=False,
                )

        # Fallback: fixed sizing with factor/confidence adjustment
        base_pct = self._params.max_position_pct
        # Adjust by factor score: positive factor → up to 1.3x, negative → down to 0.7x
        import numpy as np
        factor_mult = 1.0 + 0.3 * np.tanh(factor_score) if factor_score != 0 else 1.0
        # Adjust by confidence: scale linearly — low confidence gets much smaller position
        # conf=0.3→0.58, conf=0.5→0.70, conf=0.7→0.82, conf=0.9→0.94, conf=1.0→1.0
        conf_mult = 0.4 + 0.6 * min(signal_confidence, 1.0) if signal_confidence > 0 else 0.4
        adjusted_pct = base_pct * factor_mult * conf_mult
        adjusted_pct = min(adjusted_pct, self._params.max_position_pct)

        allocation = portfolio_value * adjusted_pct
        allocation = min(allocation, cash_available * 0.95, exposure_headroom)

        if allocation <= 0 or price <= 0:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
                reason="No cash available",
                allowed=False,
            )

        quantity = int(allocation / price)
        if quantity <= 0:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
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
        self, portfolio_value: float, cash_available: float,
    ) -> PositionSizeResult | None:
        """Return rejection result if total exposure exceeds limit."""
        if portfolio_value <= 0:
            return None
        invested = portfolio_value - cash_available
        exposure = invested / portfolio_value
        if exposure >= self._params.max_total_exposure_pct:
            return PositionSizeResult(
                quantity=0, allocation_usd=0, risk_per_share=0,
                reason=f"Max exposure reached ({exposure:.0%} >= {self._params.max_total_exposure_pct:.0%})",
                allowed=False,
            )
        return None

    def _get_exposure_headroom(
        self, portfolio_value: float, cash_available: float,
    ) -> float:
        """How much more can be invested before hitting exposure limit."""
        if portfolio_value <= 0:
            return 0.0
        max_invested = portfolio_value * self._params.max_total_exposure_pct
        current_invested = portfolio_value - cash_available
        return max(0.0, max_invested - current_invested)

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
        gain_from_entry = (highest_price - entry_price) / entry_price
        if gain_from_entry < activation_pct:
            return False  # Not yet activated

        drop_from_peak = (highest_price - current_price) / highest_price
        return drop_from_peak >= trail_pct

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
