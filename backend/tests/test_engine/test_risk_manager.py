"""Tests for Risk Manager."""

import pytest

from engine.risk_manager import RiskManager, RiskParams


class TestPositionSizing:
    def test_normal_sizing(self):
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=5,
        )
        assert result.allowed is True
        assert result.quantity > 0
        assert result.allocation_usd <= 100_000 * 0.10

    def test_max_positions_reached(self):
        rm = RiskManager(RiskParams(max_positions=5))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=5,
        )
        assert result.allowed is False
        assert "Max positions" in result.reason

    def test_no_cash(self):
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=0,
            current_positions=0,
        )
        assert result.allowed is False

    def test_price_too_high(self):
        rm = RiskManager(RiskParams(max_position_pct=0.01))
        result = rm.calculate_position_size(
            symbol="BRK.A",
            price=600_000.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
        )
        assert result.allowed is False

    def test_daily_loss_limit(self):
        rm = RiskManager(RiskParams(daily_loss_limit_pct=0.03))
        rm.update_daily_pnl(-3500)  # -3.5% of 100k
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason

    def test_respects_cash_over_position_limit(self):
        rm = RiskManager(RiskParams(max_position_pct=0.50))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=30_000,
            current_positions=0,
        )
        assert result.allowed is True
        assert result.allocation_usd <= 30_000

    def test_exposure_limit_blocks_buy(self):
        rm = RiskManager(RiskParams(max_total_exposure_pct=0.90))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=5_000,  # 95% invested
            current_positions=0,
        )
        assert result.allowed is False
        assert "exposure" in result.reason.lower()

    def test_exposure_headroom_caps_allocation(self):
        rm = RiskManager(RiskParams(max_position_pct=0.10, max_total_exposure_pct=0.90))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=15_000,  # 85% invested, 5% headroom
            current_positions=0,
        )
        assert result.allowed is True
        # Max exposure headroom = 90% - 85% = 5% = 5,000
        # Per-position max = 10% = 10,000
        # Should be capped by headroom
        assert result.allocation_usd <= 5_000


class TestStopLoss:
    def test_stop_loss_triggered(self):
        rm = RiskManager()
        assert rm.check_stop_loss(100.0, 91.0) is True  # -9% > 8% default

    def test_stop_loss_not_triggered(self):
        rm = RiskManager()
        assert rm.check_stop_loss(100.0, 95.0) is False

    def test_custom_stop_loss(self):
        rm = RiskManager()
        # 5% SL: trigger at <= 95.0
        assert rm.check_stop_loss(100.0, 94.0, stop_loss_pct=0.05) is True
        assert rm.check_stop_loss(100.0, 96.0, stop_loss_pct=0.05) is False
        # 3% SL: trigger at <= 97.0
        assert rm.check_stop_loss(100.0, 96.0, stop_loss_pct=0.03) is True
        # 10% SL: trigger at <= 90.0
        assert rm.check_stop_loss(100.0, 96.0, stop_loss_pct=0.10) is False


class TestTakeProfit:
    def test_take_profit_triggered(self):
        rm = RiskManager()
        assert rm.check_take_profit(100.0, 121.0) is True  # +21% > 20% default

    def test_take_profit_not_triggered(self):
        rm = RiskManager()
        assert rm.check_take_profit(100.0, 115.0) is False


class TestTrailingStop:
    def test_trailing_stop_triggered(self):
        rm = RiskManager()
        # Entry=100, highest=110 (+10%), current=105 → drop=4.5% > 3%
        assert rm.check_trailing_stop(100.0, 105.0, 110.0) is True

    def test_trailing_stop_not_activated(self):
        rm = RiskManager()
        # Entry=100, highest=103 (+3%), below 5% activation
        assert rm.check_trailing_stop(100.0, 101.0, 103.0) is False

    def test_trailing_stop_activated_not_triggered(self):
        rm = RiskManager()
        # Entry=100, highest=108 (+8%), current=107 → drop=0.9% < 3%
        assert rm.check_trailing_stop(100.0, 107.0, 108.0) is False

    def test_custom_trailing_params(self):
        rm = RiskManager()
        assert (
            rm.check_trailing_stop(
                100.0,
                104.0,
                110.0,
                activation_pct=0.03,
                trail_pct=0.05,
            )
            is True
        )


class TestDynamicSlTp:
    def test_low_volatility_stock(self):
        rm = RiskManager()
        # ATR=1.5 on $100 stock = 1.5% daily volatility → tighter SL/TP
        sl, tp = rm.calculate_dynamic_sl_tp(100.0, 1.5)
        assert 0.03 <= sl <= 0.05  # ~3%
        assert 0.06 <= tp <= 0.10  # ~6%

    def test_high_volatility_stock(self):
        rm = RiskManager()
        # ATR=8.0 on $100 stock = 8% daily volatility → wider SL/TP
        sl, tp = rm.calculate_dynamic_sl_tp(100.0, 8.0)
        assert sl >= 0.10
        assert tp >= 0.20

    def test_kr_market_wider_bounds(self):
        rm = RiskManager()
        # KR market should have wider minimum bounds (±30% daily limit)
        sl, tp = rm.calculate_dynamic_sl_tp(50000.0, 500.0, market="KR")
        assert sl >= 0.05  # KR min SL is 5%

    def test_zero_atr_uses_defaults(self):
        rm = RiskManager()
        sl, tp = rm.calculate_dynamic_sl_tp(100.0, 0.0)
        assert sl == rm.params.default_stop_loss_pct
        assert tp == rm.params.default_take_profit_pct

    def test_returns_different_values_per_volatility(self):
        rm = RiskManager()
        sl_low, tp_low = rm.calculate_dynamic_sl_tp(100.0, 1.0)
        sl_high, tp_high = rm.calculate_dynamic_sl_tp(100.0, 5.0)
        assert sl_high > sl_low
        assert tp_high > tp_low


class TestDailyPnL:
    def test_update_and_reset(self):
        rm = RiskManager()
        rm.update_daily_pnl(100)
        rm.update_daily_pnl(-50)
        assert rm.daily_pnl == 50.0
        rm.reset_daily()
        assert rm.daily_pnl == 0.0


class TestMarketAllocation:
    """Test market-level fund allocation caps."""

    def _make_rm(self, us=0.5, kr=0.5):
        return RiskManager(
            RiskParams(
                market_allocations={"US": us, "KR": kr},
            )
        )

    def test_caps_portfolio_value(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is True
        # Max allocation = 50,000 * 10% = 5,000 → 50 shares
        assert result.allocation_usd <= 50_000 * 0.10 + 1

    def test_no_market_param_no_cap(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
        )
        # Without market param, no cap applied → 100,000 * 10% = 10,000
        assert result.allocation_usd <= 100_000 * 0.10 + 1
        assert result.allocation_usd > 50_000 * 0.10

    def test_kr_market_capped_separately(self):
        rm = self._make_rm(us=0.7, kr=0.3)
        kr_result = rm.calculate_position_size(
            symbol="005930",
            price=50.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="KR",
        )
        us_result = rm.calculate_position_size(
            symbol="AAPL",
            price=50.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        # KR capped at 30% → 30,000 * 10% = 3,000
        # US capped at 70% → 70,000 * 10% = 7,000
        assert kr_result.allocation_usd <= 30_000 * 0.10 + 1
        assert us_result.allocation_usd <= 70_000 * 0.10 + 1

    def test_regime_boost_bull(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        rm.set_market_regime("US", "bull")
        eff = rm.get_effective_allocation("US")
        assert eff == 0.70  # 50% + 20% boost, clamped to 70%

    def test_regime_penalty_bear(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        rm.set_market_regime("KR", "bear")
        eff = rm.get_effective_allocation("KR")
        assert eff == 0.30  # 50% - 20% penalty

    def test_regime_clamp_limits(self):
        rm = self._make_rm(us=0.8, kr=0.2)
        rm.set_market_regime("US", "bull")
        # 80% + 20% = 100% → clamped to 70%
        assert rm.get_effective_allocation("US") == 0.70
        rm.set_market_regime("KR", "bear")
        # 20% - 20% = 0% → clamped to 20%
        assert rm.get_effective_allocation("KR") == 0.20

    def test_kelly_sizing_with_market(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is True
        # Capped at 50% of portfolio
        assert result.allocation_usd <= 50_000 * 0.10 + 1

    def test_no_allocations_configured(self):
        """No market_allocations → no cap applied."""
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allocation_usd <= 100_000 * 0.10 + 1
        assert result.allocation_usd > 50_000 * 0.10

    def test_combined_portfolio_value_increases_cap(self):
        """With combined_portfolio_value, 50% cap applies to combined total."""
        rm = self._make_rm(us=0.5, kr=0.5)
        # Without combined: 50% of 60,000 = 30,000
        without = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
        )
        # With combined=100,000: 50% of 100,000 = 50,000, clamped to 60,000
        with_combined = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
            combined_portfolio_value=100_000,
        )
        assert with_combined.quantity >= without.quantity

    def test_combined_clamped_to_own_portfolio(self):
        """Combined cap never exceeds own market's portfolio value."""
        rm = self._make_rm(us=0.5, kr=0.5)
        # 50% of 200,000 = 100,000 → but own portfolio is only 60,000
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
            combined_portfolio_value=200_000,
        )
        # Should be capped by portfolio_value (60,000), not combined cap (100,000)
        assert result.allocation_usd <= 60_000 * 0.10 + 1

    def test_combined_with_kelly_sizing(self):
        """Kelly sizing also uses combined_portfolio_value."""
        rm = self._make_rm(us=0.5, kr=0.5)
        without = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
        )
        with_combined = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
            combined_portfolio_value=100_000,
        )
        assert with_combined.quantity >= without.quantity

    def test_exposure_check_with_existing_positions(self):
        """When positions exist, exposure is correctly calculated after capping.

        Regression test: previously capped_cash == capped_portfolio when
        cash > cap, making invested=0 and exposure=0% — bypassing limits.
        """
        rm = self._make_rm(us=0.5, kr=0.5)
        # KR: portfolio=9M, cash=7.7M → invested=1.3M
        # combined=9.7M, cap=50% → capped_portfolio=4.85M
        # capped_cash should be 4.85M - 1.3M = 3.55M (not 4.85M!)
        result = rm.calculate_position_size(
            symbol="005930",
            price=50_000.0,
            portfolio_value=9_000_000,
            cash_available=7_700_000,
            current_positions=0,
            market="KR",
            combined_portfolio_value=9_700_000,
        )
        assert result.allowed is True
        # Max allocation: capped_portfolio(4.85M) * 7%(uptrend) = 339,500
        assert result.allocation_usd <= 4_850_000 * 0.08 + 1

    def test_exposure_blocks_when_heavily_invested(self):
        """Already invested beyond cap → no new positions allowed."""
        rm = self._make_rm(us=0.5, kr=0.5)
        # KR: portfolio=9M, cash=1M → invested=8M
        # combined=9.7M, cap=50% → capped_portfolio=4.85M
        # invested(8M) > capped_portfolio(4.85M) → capped_cash=0
        result = rm.calculate_position_size(
            symbol="005930",
            price=50_000.0,
            portfolio_value=9_000_000,
            cash_available=1_000_000,
            current_positions=0,
            market="KR",
            combined_portfolio_value=9_700_000,
        )
        assert result.allowed is False
        assert "exposure" in result.reason.lower() or "cash" in result.reason.lower()

    def test_exposure_correct_without_combined(self):
        """Without combined_portfolio, exposure still correctly tracks invested."""
        rm = self._make_rm(us=0.5, kr=0.5)
        # portfolio=100K, cash=60K → invested=40K
        # cap=50% → capped_portfolio=50K
        # capped_cash = 50K - 40K = 10K
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is True
        # allocation from 10K cash (limited)
        assert result.allocation_usd <= 10_000 + 1


class TestConfidenceBasedSizing:
    """Test that signal confidence affects position size meaningfully."""

    def test_high_confidence_gets_larger_position(self):
        rm = RiskManager()
        high = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            signal_confidence=0.9,
        )
        low = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            signal_confidence=0.3,
        )
        assert high.allowed and low.allowed
        # High confidence should get a meaningfully larger position
        assert high.allocation_usd > low.allocation_usd
        # At least 20% more allocation for high vs low confidence
        assert high.allocation_usd / low.allocation_usd > 1.2

    def test_zero_confidence_gets_minimum(self):
        rm = RiskManager()
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            signal_confidence=0.0,
        )
        assert result.allowed
        # Should get a much smaller position than max (7%)
        assert result.allocation_usd < 100_000 * 0.04

    def test_full_confidence_gets_near_max(self):
        rm = RiskManager()
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            signal_confidence=1.0,
        )
        assert result.allowed
        # Should be close to max regime position (7% for uptrend)
        assert result.allocation_usd >= 100_000 * 0.06


class TestTieredTrailingStop:
    """Tests for tiered trailing stop (STOCK-24)."""

    DEFAULT_TIERS = [(0.10, 0.05), (0.15, 0.04), (0.20, 0.03)]

    def _make_rm(self, tiers=None):
        return RiskManager(
            RiskParams(
                tiered_trailing_tiers=tiers or self.DEFAULT_TIERS,
            )
        )

    def test_no_tiers_returns_false(self):
        rm = RiskManager()  # No tiers configured
        assert rm.check_tiered_trailing_stop(100.0, 105.0, 115.0) is False

    def test_empty_tiers_returns_false(self):
        rm = RiskManager(RiskParams(tiered_trailing_tiers=[]))
        assert rm.check_tiered_trailing_stop(100.0, 105.0, 115.0) is False

    def test_below_all_tiers_not_triggered(self):
        rm = self._make_rm()
        # Peak gain = 8% < 10% (lowest tier)
        assert rm.check_tiered_trailing_stop(100.0, 105.0, 108.0) is False

    def test_tier1_triggered_10pct_gain_5pct_trail(self):
        rm = self._make_rm()
        # Peak=110 (+10%), current=104 → drop=5.45% > 5% trail
        assert rm.check_tiered_trailing_stop(100.0, 104.0, 110.0) is True

    def test_tier1_not_triggered_small_drop(self):
        rm = self._make_rm()
        # Peak=110 (+10%), current=108 → drop=1.8% < 5% trail
        assert rm.check_tiered_trailing_stop(100.0, 108.0, 110.0) is False

    def test_tier2_triggered_15pct_gain_4pct_trail(self):
        rm = self._make_rm()
        # Peak=115 (+15%), current=110 → drop=4.35% > 4% trail
        assert rm.check_tiered_trailing_stop(100.0, 110.0, 115.0) is True

    def test_tier2_not_triggered_small_drop(self):
        rm = self._make_rm()
        # Peak=115 (+15%), current=113 → drop=1.7% < 4% trail
        assert rm.check_tiered_trailing_stop(100.0, 113.0, 115.0) is False

    def test_tier3_triggered_20pct_gain_3pct_trail(self):
        rm = self._make_rm()
        # Peak=120 (+20%), current=116 → drop=3.33% > 3% trail
        assert rm.check_tiered_trailing_stop(100.0, 116.0, 120.0) is True

    def test_tier3_not_triggered_small_drop(self):
        rm = self._make_rm()
        # Peak=120 (+20%), current=118 → drop=1.67% < 3% trail
        assert rm.check_tiered_trailing_stop(100.0, 118.0, 120.0) is False

    def test_highest_tier_used_when_multiple_match(self):
        rm = self._make_rm()
        # Peak=125 (+25%) matches all tiers. Should use tier3 (3% trail)
        # current=121 → drop=3.2% > 3% → triggered
        assert rm.check_tiered_trailing_stop(100.0, 121.0, 125.0) is True
        # current=122 → drop=2.4% < 3% → not triggered (using tightest tier)
        assert rm.check_tiered_trailing_stop(100.0, 122.0, 125.0) is False

    def test_zero_entry_price(self):
        rm = self._make_rm()
        assert rm.check_tiered_trailing_stop(0.0, 100.0, 110.0) is False

    def test_zero_highest_price(self):
        rm = self._make_rm()
        assert rm.check_tiered_trailing_stop(100.0, 100.0, 0.0) is False

    def test_single_tier(self):
        rm = self._make_rm(tiers=[(0.10, 0.03)])
        # Peak=112 (+12%), current=108 → drop=3.57% > 3%
        assert rm.check_tiered_trailing_stop(100.0, 108.0, 112.0) is True

    def test_kr_market_large_gain_protection(self):
        """Simulate HPSP-like scenario: +18% gain should be protected."""
        rm = self._make_rm()
        entry = 50000.0
        peak = 59000.0  # +18%
        # Price drops to 56500 → drop from peak = 4.24% > 4% (tier2)
        assert rm.check_tiered_trailing_stop(entry, 56500.0, peak) is True

    def test_us_market_docn_scenario(self):
        """Simulate DOCN-like scenario: +23.74% gain should be protected."""
        rm = self._make_rm()
        entry = 30.0
        peak = 37.12  # +23.74%
        # Price drops to 35.80 → drop from peak = 3.55% > 3% (tier3)
        assert rm.check_tiered_trailing_stop(entry, 35.80, peak) is True
        # Price at 36.50 → drop from peak = 1.67% < 3% → not triggered
        assert rm.check_tiered_trailing_stop(entry, 36.50, peak) is False


class TestBreakevenStop:
    """Tests for breakeven stop (STOCK-24)."""

    def _make_rm(self, tp=0.20, enabled=True, activation=0.50, lock_ratio=0.75, lock_pct=0.50):
        return RiskManager(
            RiskParams(
                default_take_profit_pct=tp,
                breakeven_stop_enabled=enabled,
                breakeven_stop_activation_ratio=activation,
                breakeven_stop_lock_ratio=lock_ratio,
                breakeven_stop_lock_pct=lock_pct,
            )
        )

    def test_disabled_returns_false(self):
        rm = self._make_rm(enabled=False)
        assert rm.check_breakeven_stop(100.0, 95.0, 115.0) is False

    def test_below_activation_not_triggered(self):
        rm = self._make_rm(tp=0.20)  # activation at 10%
        # Peak gain = 8% < 10% activation
        assert rm.check_breakeven_stop(100.0, 95.0, 108.0) is False

    def test_breakeven_triggered_at_activation(self):
        rm = self._make_rm(tp=0.20)  # activation at 10%
        # Peak=110 (+10%), current=99 → below entry → triggered
        assert rm.check_breakeven_stop(100.0, 99.0, 110.0) is True

    def test_breakeven_not_triggered_above_entry(self):
        rm = self._make_rm(tp=0.20)  # activation at 10%
        # Peak=110 (+10%), current=101 → above entry → not triggered
        assert rm.check_breakeven_stop(100.0, 101.0, 110.0) is False

    def test_breakeven_exact_entry_price(self):
        rm = self._make_rm(tp=0.20)
        # Peak=110 (+10%), current=100 (exact entry) → triggered (<=)
        assert rm.check_breakeven_stop(100.0, 100.0, 110.0) is True

    def test_lock_triggered_at_lock_ratio(self):
        rm = self._make_rm(tp=0.20)  # lock at 15%
        # Peak=116 (+16%, above 15% lock), lock = entry * (1 + 0.16 * 0.50) = 108
        # current=107 → below 108 → triggered
        assert rm.check_breakeven_stop(100.0, 107.0, 116.0) is True

    def test_lock_not_triggered_above_lock_price(self):
        rm = self._make_rm(tp=0.20)  # lock at 15%
        # Peak=116 (+16%), lock price = 100 * (1 + 0.16 * 0.50) = 108
        # current=109 → above lock → not triggered
        assert rm.check_breakeven_stop(100.0, 109.0, 116.0) is False

    def test_between_activation_and_lock(self):
        rm = self._make_rm(tp=0.20)  # activation=10%, lock=15%
        # Peak=112 (+12%, between 10% and 15%), in breakeven zone
        # current=101 → above entry → not triggered
        assert rm.check_breakeven_stop(100.0, 101.0, 112.0) is False
        # current=99 → below entry → triggered
        assert rm.check_breakeven_stop(100.0, 99.0, 112.0) is True

    def test_zero_entry_price(self):
        rm = self._make_rm()
        assert rm.check_breakeven_stop(0.0, 100.0, 110.0) is False

    def test_zero_highest_price(self):
        rm = self._make_rm()
        assert rm.check_breakeven_stop(100.0, 100.0, 0.0) is False

    def test_zero_tp_returns_false(self):
        rm = self._make_rm(tp=0.0)
        assert rm.check_breakeven_stop(100.0, 95.0, 110.0, take_profit_pct=0.0) is False

    def test_custom_tp_overrides_default(self):
        rm = self._make_rm(tp=0.20)
        # With custom TP=0.30, activation = 15%, lock = 22.5%
        # Peak=116 (+16%), above activation (15%)
        # Breakeven zone: current=99 → below entry → triggered
        assert rm.check_breakeven_stop(100.0, 99.0, 116.0, take_profit_pct=0.30) is True

    def test_custom_tp_changes_lock_level(self):
        rm = self._make_rm(tp=0.20)
        # Default TP=20%: lock at 15%. Peak=116 → lock zone.
        # Custom TP=30%: lock at 22.5%. Peak=116 → breakeven zone (not lock).
        # Lock price with TP=20%: 100*(1 + 0.16*0.5) = 108
        # In breakeven zone with TP=30%: stop at entry=100
        # current=107: triggers lock with TP=20%, NOT with TP=30%
        assert rm.check_breakeven_stop(100.0, 107.0, 116.0) is True
        assert (
            rm.check_breakeven_stop(
                100.0,
                107.0,
                116.0,
                take_profit_pct=0.30,
            )
            is False
        )  # TP=30% → breakeven zone (stop=100)

    def test_docn_scenario_23pct_gain_protected(self):
        """DOCN +23.74% gain: if price drops back toward entry, breakeven fires."""
        rm = self._make_rm(tp=0.30)  # 30% TP like DOCN's original
        entry = 30.0
        peak = 37.12  # +23.74%, above lock=22.5%
        # Lock price = 30 * (1 + 0.2374 * 0.50) = 33.56
        # Price drops to 33.0 → below lock → triggered
        assert rm.check_breakeven_stop(entry, 33.0, peak, take_profit_pct=0.30) is True
        # Price at 34.0 → above lock → not triggered
        assert rm.check_breakeven_stop(entry, 34.0, peak, take_profit_pct=0.30) is False

    def test_hpsp_scenario_kr_market(self):
        """HPSP +18.31% gain: breakeven should protect entry."""
        rm = self._make_rm(tp=0.30)  # 30% TP
        entry = 50000.0
        peak = 59155.0  # +18.31%, above activation=15%, below lock=22.5%
        # In breakeven zone: stop at entry
        assert rm.check_breakeven_stop(entry, 49000.0, peak, take_profit_pct=0.30) is True
        assert rm.check_breakeven_stop(entry, 51000.0, peak, take_profit_pct=0.30) is False


class TestExistingPositionConcentration:
    """STOCK-26: Existing position value blocks new buys exceeding max_position_pct."""

    def test_existing_at_max_blocks_buy(self):
        """If already holding at max_position_pct, new buy rejected."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=5,
            existing_position_value=1_000_000,  # 10% of portfolio
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_existing_above_max_blocks_buy(self):
        """If holding above max_position_pct (e.g. 34%), buy rejected."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=5,
            existing_position_value=3_400_000,  # 34% of portfolio (like the bug)
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "263750" in result.reason

    def test_no_existing_position_allows_buy(self):
        """Without existing position, buy proceeds normally."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        assert result.allowed is True
        assert result.quantity > 0

    def test_small_existing_allows_buy_but_reduces_allocation(self):
        """Small existing position allows buy but caps remaining allocation."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # Existing position = 3% of portfolio (below 10% max)
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_position_value=3_000,  # 3% of portfolio
        )
        assert result.allowed is True
        # max_alloc should be regime_pct(7%) * portfolio - existing(3000)
        # = 7000 - 3000 = 4000. So allocation should be <= 4000
        assert result.allocation_usd <= 4_100  # small tolerance for rounding

    def test_kelly_existing_at_max_blocks_buy(self):
        """Kelly sizing also blocks when existing position at max."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_kelly_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=5,
            existing_position_value=1_000_000,  # 10% of portfolio
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_kelly_no_existing_allows_buy(self):
        """Kelly sizing proceeds normally without existing position."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        assert result.allowed is True

    def test_kelly_small_existing_reduces_allocation(self):
        """STOCK-26: Partial existing position should reduce Kelly allocation.

        A 3% existing position with 10% max_position_pct should yield a
        smaller allocation than starting from zero. This prevents the total
        concentration from exceeding max_position_pct through the Kelly path.
        """
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        without = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        with_existing = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            existing_position_value=3_000,  # 3% of portfolio
        )
        assert without.allowed is True
        assert with_existing.allowed is True
        assert with_existing.allocation_usd < without.allocation_usd

    def test_kelly_with_trade_history_existing_reduces_allocation(self):
        """STOCK-26: Kelly branch (with trade history) also reduces for existing."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        without = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.04,
            signal_confidence=0.7,
            existing_position_value=0.0,
        )
        with_existing = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.04,
            signal_confidence=0.7,
            existing_position_value=3_000,  # 3% of portfolio
        )
        assert without.allowed is True
        assert with_existing.allowed is True
        assert with_existing.allocation_usd < without.allocation_usd

    def test_zero_portfolio_value_no_division_error(self):
        """Zero portfolio value should not cause division by zero."""
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=0,
            cash_available=0,
            current_positions=0,
            existing_position_value=1000,
        )
        # Should be rejected for lack of cash, not crash
        assert result.allowed is False

    def test_kr_263750_regression_scenario(self):
        """STOCK-26 regression: 263750 at 34% of KR portfolio should be blocked.

        In the original incident, 263750 reached 34% concentration due to
        17 successive buys. With existing_position_value check, the 2nd+ buy
        would be blocked once the position exceeds max_position_pct.
        """
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        portfolio = 10_000_000  # 10M KRW

        # First buy: no existing position → allowed
        first = rm.calculate_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=portfolio,
            cash_available=5_000_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        assert first.allowed is True

        # After first buy: position value ~452K (7 shares * 64600)
        # Second buy: existing position at ~4.5% → still allowed
        second = rm.calculate_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=portfolio,
            cash_available=4_500_000,
            current_positions=1,
            existing_position_value=452_200.0,
        )
        assert second.allowed is True

        # After many buys: position at 10%+ → BLOCKED
        blocked = rm.calculate_position_size(
            symbol="263750",
            price=64600.0,
            portfolio_value=portfolio,
            cash_available=4_000_000,
            current_positions=5,
            existing_position_value=1_100_000.0,  # 11%
        )
        assert blocked.allowed is False
