"""Tests for Risk Manager."""

import pytest

from engine.risk_manager import RiskManager, RiskParams, PositionSizeResult


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


class TestP1RoundUp:
    """P1: allow buying 1 share when alloc < price but 1 share fits max_pct."""

    def test_round_up_when_alloc_below_price_but_one_share_fits(self):
        # Portfolio ₩7M, max_pct=20% → 1 share cap ₩1.4M; share ₩300K fits.
        # Allocation comes out below ₩300K (low conf scenario).
        params = RiskParams(
            max_position_pct=0.20, min_position_pct=0.04,
            allow_one_share_round_up=True,
        )
        rm_inst = RiskManager(params=params)
        result = rm_inst.calculate_kelly_position_size(
            symbol="005380", price=300_000.0,
            portfolio_value=7_000_000.0, cash_available=5_000_000.0,
            current_positions=0, signal_confidence=0.30,
            market="KR", combined_portfolio_value=7_000_000.0,
        )
        assert result.allowed is True
        assert result.quantity == 1
        assert "Round-up" in result.reason or "P1" in result.reason

    def test_round_up_blocked_when_one_share_exceeds_max_pct(self):
        # Share ₩2M, portfolio ₩7M, max_pct=20% → 1 share = 28.5% > 20%.
        # Should still reject.
        params = RiskParams(
            max_position_pct=0.20, min_position_pct=0.04,
            allow_one_share_round_up=True,
        )
        rm_inst = RiskManager(params=params)
        result = rm_inst.calculate_kelly_position_size(
            symbol="010130", price=2_000_000.0,
            portfolio_value=7_000_000.0, cash_available=5_000_000.0,
            current_positions=0, signal_confidence=0.30,
            market="KR", combined_portfolio_value=7_000_000.0,
        )
        assert result.allowed is False

    def test_disabled_keeps_legacy_behavior(self):
        params = RiskParams(
            max_position_pct=0.20, min_position_pct=0.04,
            allow_one_share_round_up=False,
        )
        rm_inst = RiskManager(params=params)
        result = rm_inst.calculate_kelly_position_size(
            symbol="005380", price=300_000.0,
            portfolio_value=7_000_000.0, cash_available=5_000_000.0,
            current_positions=0, signal_confidence=0.30,
            market="KR", combined_portfolio_value=7_000_000.0,
        )
        # Without round-up, legacy "Price too high" rejection.
        assert result.allowed is False


class TestP2MinPositionFloor:
    """P2: enforce min_position_pct floor in fixed-sizing fallback."""

    def test_floor_bumps_small_alloc(self):
        # uptrend regime → base 0.07; conf 0.30 → conf_mult 0.58 → adj 4.06%.
        # min 5% → floor bumps it. ₩7M × 5% = ₩350K → 1 share at ₩100K = 3.
        # Without floor: 4.06% × ₩7M = ₩284K → 2 shares.
        params_floor = RiskParams(
            max_position_pct=0.20, min_position_pct=0.05,
            enforce_min_position_pct_floor=True,
        )
        rm_floor = RiskManager(params=params_floor)
        rm_floor.set_eval_regime("uptrend")
        r_floor = rm_floor.calculate_kelly_position_size(
            symbol="X", price=100_000.0,
            portfolio_value=7_000_000.0, cash_available=2_000_000.0,
            current_positions=0, signal_confidence=0.30,
        )

        params_no = RiskParams(
            max_position_pct=0.20, min_position_pct=0.05,
            enforce_min_position_pct_floor=False,
        )
        rm_no = RiskManager(params=params_no)
        rm_no.set_eval_regime("uptrend")
        r_no = rm_no.calculate_kelly_position_size(
            symbol="X", price=100_000.0,
            portfolio_value=7_000_000.0, cash_available=2_000_000.0,
            current_positions=0, signal_confidence=0.30,
        )

        assert r_floor.allowed is True
        assert r_no.allowed is True
        assert r_floor.quantity > r_no.quantity

    def test_floor_capped_by_max_position_pct(self):
        # Even with floor, allocation cannot exceed max_position_pct.
        params = RiskParams(
            max_position_pct=0.10, min_position_pct=0.20,  # min > max edge
            enforce_min_position_pct_floor=True,
        )
        rm_inst = RiskManager(params=params)
        rm_inst.set_eval_regime("uptrend")
        result = rm_inst.calculate_kelly_position_size(
            symbol="X", price=100_000.0,
            portfolio_value=7_000_000.0, cash_available=2_000_000.0,
            current_positions=0, signal_confidence=0.30,
        )
        assert result.allowed is True
        # max 10% × ₩7M = ₩700K → 7 shares max
        assert result.allocation_usd <= 700_000.0


class TestRegimePositionPctOverride:
    """Per-market regime sizing override (US uses larger sizes than KR)."""

    def test_override_replaces_default_table(self):
        override = {"uptrend": 0.123, "sideways": 0.045}
        params = RiskParams(regime_position_pct=override)
        rm = RiskManager(params=params)
        rm.set_eval_regime("uptrend")
        assert rm._get_regime_position_pct() == pytest.approx(0.123)
        rm.set_eval_regime("sideways")
        assert rm._get_regime_position_pct() == pytest.approx(0.045)

    def test_no_override_uses_module_default(self):
        rm = RiskManager(params=RiskParams())
        rm.set_eval_regime("uptrend")
        # REGIME_POSITION_PCT["uptrend"] = 0.07 (KR baseline — module default)
        assert rm._get_regime_position_pct() == pytest.approx(0.07)

    def test_override_unknown_regime_falls_back_to_max_pct(self):
        params = RiskParams(
            max_position_pct=0.15,
            regime_position_pct={"uptrend": 0.10},
        )
        rm = RiskManager(params=params)
        rm.set_eval_regime("nonexistent")
        assert rm._get_regime_position_pct() == pytest.approx(0.15)


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

    def test_combined_allocation_limited_by_cash(self):
        """STOCK-53: combined cap can exceed own portfolio, but real cash limits orders."""
        rm = self._make_rm(us=0.5, kr=0.5)
        # 50% of 200,000 = 100,000 → allocation base is 100,000
        # but actual cash is only 60,000 (× 0.95 buffer)
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=60_000,
            cash_available=60_000,
            current_positions=0,
            market="US",
            combined_portfolio_value=200_000,
        )
        # Per-position = 100,000 * 7% = 7,000 (regime "uptrend")
        # Cash limit = 60,000 * 0.95 = 57,000
        # Allocation = min(7,000, 57,000, headroom) = 7,000
        assert result.allowed is True
        assert result.allocation_usd == 7_000
        # Real cash still limits total allocation
        assert result.allocation_usd <= 60_000 * 0.95

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

    def test_integrated_margin_uses_market_invested(self):
        """STOCK-57: 통합증거금 accounts — `balance.total` reflects the whole
        account (shared deposits), so `portfolio_value - cash_available` does
        NOT equal this market's real invested capital. Previously this caused
        spurious 100% exposure readings blocking every buy (2026-04-23).

        Incident numbers (US side):
          portfolio_value = $10,642 (whole-account total, inflated by 통합 deposits)
          cash_available  = $5,396
          combined_pv     = $7,888 (positions-only combined, KR+US)
          real US positions = $3,238

        Without market_invested: invested_inferred = 10642-5396 = 5246.
        capped_portfolio = 7888*0.5 = 3944. capped_cash = max(0, 3944-5246)=0.
        Exposure check sees (3944-0)/3944 = 100% → REJECTED.

        With market_invested=3238: capped_cash = max(0, 3944-3238)=706.
        Exposure check sees (3944-706)/3944 = 82% → allowed.
        """
        rm = self._make_rm(us=0.5, kr=0.5)
        # Without market_invested override (old behavior) → blocked at 100%
        blocked = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=10_642,
            cash_available=5_396,
            current_positions=0,
            market="US",
            combined_portfolio_value=7_888,
        )
        assert blocked.allowed is False
        assert "exposure" in blocked.reason.lower()

        # With explicit market_invested → cap math is correct, buy allowed
        allowed = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=10_642,
            cash_available=5_396,
            current_positions=0,
            market="US",
            combined_portfolio_value=7_888,
            market_invested=3_238,
        )
        assert allowed.allowed is True
        assert allowed.quantity > 0

    def test_integrated_margin_kelly_uses_market_invested(self):
        """Kelly sizing path also respects market_invested override."""
        rm = self._make_rm(us=0.5, kr=0.5)
        allowed = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=10_642,
            cash_available=5_396,
            current_positions=0,
            market="US",
            combined_portfolio_value=7_888,
            market_invested=3_238,
        )
        assert allowed.allowed is True
        assert allowed.quantity > 0

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


class TestSymbolConcentrationExposure:
    """STOCK-30: Per-symbol concentration limit via existing_symbol_exposure parameter."""

    def test_rejects_when_symbol_concentration_exceeds_limit(self):
        """Core test: buy rejected when existing_symbol_exposure >= max_position_pct."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=3,
            existing_symbol_exposure=0.12,  # 12% > 10% limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "AAPL" in result.reason

    def test_rejects_at_exact_limit(self):
        """Buy rejected when exposure equals max_position_pct (boundary)."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="MSFT",
            price=400.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_symbol_exposure=0.10,  # Exactly at 10% limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_allows_when_below_limit(self):
        """Buy allowed when exposure is below max_position_pct."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_symbol_exposure=0.05,  # 5% < 10% limit
        )
        assert result.allowed is True
        assert result.quantity > 0

    def test_zero_exposure_allows_buy(self):
        """Zero exposure (no existing position) proceeds normally."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_symbol_exposure=0.0,
        )
        assert result.allowed is True
        assert result.quantity > 0

    def test_exposure_reduces_allocation(self):
        """Existing exposure reduces the remaining allocation available."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # No exposure → full allocation
        without = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_symbol_exposure=0.0,
        )
        # 5% exposure → reduced allocation
        with_exposure = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_symbol_exposure=0.05,
        )
        assert without.allowed is True
        assert with_exposure.allowed is True
        assert with_exposure.allocation_usd < without.allocation_usd

    def test_both_params_uses_higher_value(self):
        """When both existing_position_value and existing_symbol_exposure are provided,
        the higher implied value is used (conservative)."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # exposure=12% → $12,000 on $100K portfolio
        # position_value=$8,000 (lower)
        # Should use exposure-implied value ($12,000) → block
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_position_value=8_000,  # 8% — below limit
            existing_symbol_exposure=0.12,  # 12% — above limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_position_value_wins_when_higher(self):
        """When position_value implies higher exposure, it takes precedence."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # exposure=5% → $5,000
        # position_value=$11,000 (higher, 11%)
        # Should use position_value → block
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=2,
            existing_position_value=11_000,  # 11% — above limit
            existing_symbol_exposure=0.05,  # 5% — below limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_kelly_rejects_when_exposure_exceeds_limit(self):
        """Kelly sizing also rejects when existing_symbol_exposure >= max_position_pct."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_kelly_position_size(
            symbol="TSLA",
            price=200.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=3,
            existing_symbol_exposure=0.15,  # 15% > 10% limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "TSLA" in result.reason

    def test_kelly_exposure_reduces_allocation(self):
        """Kelly sizing reduces allocation when exposure is below limit."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        without = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            existing_symbol_exposure=0.0,
        )
        with_exposure = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            existing_symbol_exposure=0.03,  # 3%
        )
        assert without.allowed is True
        assert with_exposure.allowed is True
        assert with_exposure.allocation_usd < without.allocation_usd

    def test_kelly_with_trade_history_exposure_reduces(self):
        """Kelly branch (with trade history) also respects existing_symbol_exposure."""
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
            existing_symbol_exposure=0.0,
        )
        with_exposure = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.04,
            signal_confidence=0.7,
            existing_symbol_exposure=0.03,  # 3%
        )
        assert without.allowed is True
        assert with_exposure.allowed is True
        assert with_exposure.allocation_usd < without.allocation_usd

    def test_kr_market_concentration_via_exposure(self):
        """KR market symbols also blocked via existing_symbol_exposure."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_position_size(
            symbol="005930",
            price=70_000.0,
            portfolio_value=10_000_000,
            cash_available=5_000_000,
            current_positions=3,
            existing_symbol_exposure=0.34,  # 34% like the 263750 incident
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "005930" in result.reason


class TestExtendedHoursConcentration:
    """STOCK-32: Extended hours position sizing must enforce per-symbol concentration."""

    def test_existing_at_max_blocks_buy(self):
        """If already holding at max_position_pct, extended-hours buy rejected."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            existing_position_value=10_000,  # 10% of portfolio = at limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "AAPL" in result.reason

    def test_existing_above_max_blocks_buy(self):
        """If holding above max_position_pct, extended-hours buy rejected."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_extended_hours_position_size(
            symbol="TSLA",
            price=200.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            existing_position_value=25_000,  # 25% of portfolio
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "TSLA" in result.reason

    def test_no_existing_position_allows_buy(self):
        """Without existing position, extended-hours buy proceeds normally."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        assert result.allowed is True
        assert result.quantity > 0
        assert "extended hours" in result.reason

    def test_small_existing_reduces_allocation(self):
        """Small existing position reduces extended-hours allocation."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # Without existing position
        without = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=10.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_position_value=0.0,
        )
        # With 1% existing position (below 10% max, so not blocked)
        with_existing = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=10.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_position_value=1_000,  # 1% of portfolio
        )
        assert without.allowed is True
        assert with_existing.allowed is True
        assert with_existing.allocation_usd < without.allocation_usd

    def test_exposure_param_blocks_concentrated_position(self):
        """existing_symbol_exposure blocks buy when at/above max_position_pct."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        result = rm.calculate_extended_hours_position_size(
            symbol="NVDA",
            price=500.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            existing_symbol_exposure=0.15,  # 15% > 10% limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason
        assert "NVDA" in result.reason

    def test_exposure_param_reduces_allocation(self):
        """existing_symbol_exposure reduces allocation when below limit."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        without = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=10.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
        )
        with_exposure = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=10.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            existing_symbol_exposure=0.01,  # 1% existing
        )
        assert without.allowed is True
        assert with_exposure.allowed is True
        assert with_exposure.allocation_usd < without.allocation_usd

    def test_both_value_and_exposure_uses_higher(self):
        """When both existing_position_value and exposure provided, use higher."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # exposure=12% implies $12,000 > value=$5,000, so exposure should win
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=1,
            existing_position_value=5_000,  # 5%
            existing_symbol_exposure=0.12,  # 12% > 10% limit
        )
        assert result.allowed is False
        assert "Already holding" in result.reason

    def test_existing_position_fills_extended_hours_alloc(self):
        """Existing position that fills extended-hours max_position_pct (3%) blocks buy."""
        rm = RiskManager(RiskParams(max_position_pct=0.10))
        # Default extended hours max is 3%. If existing = 3%, allocation = 0
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100_000,
            cash_available=50_000,
            current_positions=0,
            max_position_pct=0.03,
            existing_position_value=3_000,  # 3% = fills the 3% extended-hours alloc
        )
        # Concentration check passes (3% < 10%), but max_alloc = 3000-3000 = 0
        assert result.allowed is False


class TestDailyLossLimitUncapped:
    """STOCK-56: Daily loss limit must use uncapped (full) portfolio value.

    With a 50:50 US/KR split, _apply_market_cap halves portfolio_value for
    each market.  Using the capped value doubles the apparent loss percentage,
    which triggers the daily halt at only half the configured threshold.
    """

    def _make_rm(self, loss_limit: float = 0.03) -> RiskManager:
        return RiskManager(
            RiskParams(
                daily_loss_limit_pct=loss_limit,
                market_allocations={"US": 0.5, "KR": 0.5},
            )
        )

    # --- calculate_position_size ---

    def test_loss_below_limit_allows_trade_with_market_cap(self):
        """1.5% actual loss should NOT trigger 3% daily loss limit.

        Setup: portfolio=100k (all cash, no positions), market=US with 50:50 split.
        After cap: capped_portfolio=50k, capped_cash=50k (invested=0, so full cap).
        Bug (pre-fix): loss_pct = 1500/50000 = 3.0% → incorrectly triggers halt.
        Fix: loss_pct = 1500/100000 = 1.5% < 3% → trade allowed.
        """
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-1_500)  # -1.5% of 100k full portfolio
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,  # all cash → invested=0 → capped_cash=50k
            current_positions=0,
            market="US",  # cap = 50% → capped_portfolio = 50,000
        )
        # 1,500 / 100,000 = 1.5% < 3% limit → should be allowed
        assert result.allowed is True, (
            f"Expected allowed=True but got reason: {result.reason}"
        )

    def test_loss_at_limit_blocks_trade_with_market_cap(self):
        """3.5% actual loss SHOULD trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-3_500)  # -3.5% of 100k
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason

    def test_loss_below_limit_without_market_cap_still_allowed(self):
        """Without market= param, uncapped == portfolio_value — no regression."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-1_500)
        result = rm.calculate_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            # No market param → no cap applied
        )
        assert result.allowed is True

    # --- calculate_kelly_position_size ---

    def test_kelly_loss_below_limit_allows_trade_with_market_cap(self):
        """Kelly: 1.5% actual loss should NOT trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-1_500)
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is True, (
            f"Expected allowed=True but got reason: {result.reason}"
        )

    def test_kelly_loss_at_limit_blocks_trade_with_market_cap(self):
        """Kelly: 3.5% actual loss SHOULD trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-3_500)
        result = rm.calculate_kelly_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason

    # --- calculate_extended_hours_position_size ---

    def test_extended_hours_loss_below_limit_allows_trade_with_market_cap(self):
        """Extended hours: 1.5% actual loss should NOT trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-1_500)
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is True, (
            f"Expected allowed=True but got reason: {result.reason}"
        )

    def test_extended_hours_loss_at_limit_blocks_trade_with_market_cap(self):
        """Extended hours: 3.5% actual loss SHOULD trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-3_500)
        result = rm.calculate_extended_hours_position_size(
            symbol="AAPL",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="US",
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason

    # --- KR market path (primary failure surface for STOCK-56) ---

    def test_kr_loss_below_limit_allows_trade_with_market_cap(self):
        """KR market: 1.5% actual loss should NOT trigger 3% daily loss limit.

        KR was the primary failure surface: its daily reset was missing entirely,
        but even if reset, the capped portfolio value would double the apparent
        loss and falsely halt trading.  This test exercises the KR path to
        confirm the uncapped denominator fix is symmetric.
        """
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-1_500)
        result = rm.calculate_position_size(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="KR",  # cap = 50% → capped_portfolio = 50,000
        )
        # 1,500 / 100,000 = 1.5% < 3% limit → should be allowed
        assert result.allowed is True, (
            f"Expected allowed=True but got reason: {result.reason}"
        )

    def test_kr_loss_at_limit_blocks_trade_with_market_cap(self):
        """KR market: 3.5% actual loss SHOULD trigger 3% daily loss limit."""
        rm = self._make_rm(loss_limit=0.03)
        rm.update_daily_pnl(-3_500)
        result = rm.calculate_position_size(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
            market="KR",
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason


class TestTaskDailyReset:
    """STOCK-56: reset_all_daily_risk (called by task_daily_reset) must reset both managers."""

    def test_both_risk_managers_are_reset(self) -> None:
        """reset_all_daily_risk() calls reset_daily() on both US and KR managers.

        Tests the extracted module-level function that task_daily_reset in main.py
        delegates to (STOCK-56 fix). If kr_rm.reset_daily() is accidentally removed
        from that function, this mock-based assertion will catch it immediately.
        """
        from unittest.mock import MagicMock
        from main import reset_all_daily_risk

        us_rm = MagicMock()
        kr_rm = MagicMock()

        reset_all_daily_risk(us_rm, kr_rm)

        us_rm.reset_daily.assert_called_once()
        kr_rm.reset_daily.assert_called_once()

    def test_kr_risk_manager_reset_prevents_stale_accumulation(self) -> None:
        """KR daily PnL must not accumulate across reset boundaries.

        Before the fix, task_daily_reset only called risk_manager.reset_daily()
        and omitted kr_risk_manager.reset_daily(), so KR daily PnL kept
        accumulating across trading days and eventually triggered a permanent
        trading halt even when intra-day KR losses were within limits.
        """
        kr_rm = RiskManager(RiskParams(daily_loss_limit_pct=0.03))

        # Day 1: -1% KR loss
        kr_rm.update_daily_pnl(-1_000)

        # Simulated reset (was missing before fix)
        kr_rm.reset_daily()
        assert kr_rm.daily_pnl == 0.0

        # Day 2: another -1% KR loss
        kr_rm.update_daily_pnl(-1_000)

        # After proper reset, day-2 loss is 1% — well within 3% limit.
        # No market= param → _apply_market_cap is a no-op; uncapped_portfolio_value == portfolio_value.
        result = kr_rm.calculate_position_size(
            symbol="005930",
            price=100.0,
            portfolio_value=100_000,
            cash_available=100_000,
            current_positions=0,
        )
        assert result.allowed is True, (
            "KR trading should be allowed after daily reset; "
            f"got reason: {result.reason}"
        )


class TestVolatilityScaling:
    """Tests for risk-parity volatility scaling (apply_volatility_scaling)."""

    def _base_sizing(self, quantity: int = 10, price: float = 100.0) -> PositionSizeResult:
        return PositionSizeResult(
            quantity=quantity,
            allocation_usd=quantity * price,
            risk_per_share=price * 0.12,
            reason="OK",
            allowed=True,
        )

    def test_high_vol_reduces_position(self):
        """High ATR% → scale < 1.0 → fewer shares."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.04, price=100.0, target_risk_pct=0.02,
        )
        assert result.quantity < 10
        assert result.allowed is True
        assert "vol_scale" in result.reason

    def test_low_vol_increases_position(self):
        """Low ATR% → scale > 1.0 → more shares."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.01, price=100.0, target_risk_pct=0.02,
        )
        assert result.quantity > 10
        assert result.allowed is True

    def test_target_vol_no_change(self):
        """ATR% matches target → scale = 1.0 → same quantity."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.02, price=100.0, target_risk_pct=0.02,
        )
        assert result.quantity == 10

    def test_scale_clamped_at_min(self):
        """Extremely high volatility should be clamped at min_scale."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.20, price=100.0, target_risk_pct=0.02,
            min_scale=0.3,
        )
        assert result.quantity >= 3  # 10 * 0.3 = 3
        assert result.quantity <= 4  # rounding

    def test_scale_clamped_at_max(self):
        """Extremely low volatility should be clamped at max_scale."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.001, price=100.0, target_risk_pct=0.02,
            max_scale=1.5,
        )
        assert result.quantity == 15  # 10 * 1.5

    def test_zero_atr_no_change(self):
        """Zero ATR should return sizing unchanged."""
        sizing = self._base_sizing(quantity=10, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.0, price=100.0,
        )
        assert result.quantity == 10

    def test_not_allowed_passthrough(self):
        """Rejected sizing should pass through unchanged."""
        sizing = PositionSizeResult(
            quantity=0, allocation_usd=0, risk_per_share=0,
            reason="Max positions", allowed=False,
        )
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.03, price=100.0,
        )
        assert result.allowed is False
        assert result.quantity == 0

    def test_quantity_at_least_one(self):
        """Even extreme scaling should leave at least 1 share."""
        sizing = self._base_sizing(quantity=1, price=100.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.10, price=100.0, target_risk_pct=0.02,
            min_scale=0.1,
        )
        assert result.quantity >= 1

    def test_allocation_matches_quantity(self):
        """allocation_usd should equal quantity * price."""
        sizing = self._base_sizing(quantity=10, price=150.0)
        result = RiskManager.apply_volatility_scaling(
            sizing, atr_pct=0.03, price=150.0, target_risk_pct=0.02,
        )
        assert result.allocation_usd == result.quantity * 150.0
