"""Tests for Risk Manager."""

import pytest

from engine.risk_manager import RiskManager, RiskParams


class TestPositionSizing:
    def test_normal_sizing(self):
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL", price=150.0,
            portfolio_value=100_000, cash_available=50_000,
            current_positions=5,
        )
        assert result.allowed is True
        assert result.quantity > 0
        assert result.allocation_usd <= 100_000 * 0.10

    def test_max_positions_reached(self):
        rm = RiskManager(RiskParams(max_positions=5))
        result = rm.calculate_position_size(
            symbol="AAPL", price=150.0,
            portfolio_value=100_000, cash_available=50_000,
            current_positions=5,
        )
        assert result.allowed is False
        assert "Max positions" in result.reason

    def test_no_cash(self):
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL", price=150.0,
            portfolio_value=100_000, cash_available=0,
            current_positions=0,
        )
        assert result.allowed is False

    def test_price_too_high(self):
        rm = RiskManager(RiskParams(max_position_pct=0.01))
        result = rm.calculate_position_size(
            symbol="BRK.A", price=600_000.0,
            portfolio_value=100_000, cash_available=50_000,
            current_positions=0,
        )
        assert result.allowed is False

    def test_daily_loss_limit(self):
        rm = RiskManager(RiskParams(daily_loss_limit_pct=0.03))
        rm.update_daily_pnl(-3500)  # -3.5% of 100k
        result = rm.calculate_position_size(
            symbol="AAPL", price=150.0,
            portfolio_value=100_000, cash_available=50_000,
            current_positions=0,
        )
        assert result.allowed is False
        assert "Daily loss" in result.reason

    def test_respects_cash_over_position_limit(self):
        rm = RiskManager(RiskParams(max_position_pct=0.50))
        result = rm.calculate_position_size(
            symbol="AAPL", price=150.0,
            portfolio_value=100_000, cash_available=5_000,
            current_positions=0,
        )
        assert result.allowed is True
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
        assert rm.check_trailing_stop(
            100.0, 104.0, 110.0,
            activation_pct=0.03, trail_pct=0.05,
        ) is True


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
        return RiskManager(RiskParams(
            market_allocations={"US": us, "KR": kr},
        ))

    def test_caps_portfolio_value(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_position_size(
            symbol="AAPL", price=100.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0, market="US",
        )
        assert result.allowed is True
        # Max allocation = 50,000 * 10% = 5,000 → 50 shares
        assert result.allocation_usd <= 50_000 * 0.10 + 1

    def test_no_market_param_no_cap(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_position_size(
            symbol="AAPL", price=100.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0,
        )
        # Without market param, no cap applied → 100,000 * 10% = 10,000
        assert result.allocation_usd <= 100_000 * 0.10 + 1
        assert result.allocation_usd > 50_000 * 0.10

    def test_kr_market_capped_separately(self):
        rm = self._make_rm(us=0.7, kr=0.3)
        kr_result = rm.calculate_position_size(
            symbol="005930", price=50.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0, market="KR",
        )
        us_result = rm.calculate_position_size(
            symbol="AAPL", price=50.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0, market="US",
        )
        # KR capped at 30% → 30,000 * 10% = 3,000
        # US capped at 70% → 70,000 * 10% = 7,000
        assert kr_result.allocation_usd <= 30_000 * 0.10 + 1
        assert us_result.allocation_usd <= 70_000 * 0.10 + 1

    def test_regime_boost_bull(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        rm.set_market_regime("US", "bull")
        eff = rm.get_effective_allocation("US")
        assert eff == 0.60  # 50% + 10% boost

    def test_regime_penalty_bear(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        rm.set_market_regime("KR", "bear")
        eff = rm.get_effective_allocation("KR")
        assert eff == 0.40  # 50% - 10% penalty

    def test_regime_clamp_limits(self):
        rm = self._make_rm(us=0.8, kr=0.2)
        rm.set_market_regime("US", "bull")
        # 80% + 10% = 90% → clamped to 80%
        assert rm.get_effective_allocation("US") == 0.80
        rm.set_market_regime("KR", "bear")
        # 20% - 10% = 10% → clamped to 20%
        assert rm.get_effective_allocation("KR") == 0.20

    def test_kelly_sizing_with_market(self):
        rm = self._make_rm(us=0.5, kr=0.5)
        result = rm.calculate_kelly_position_size(
            symbol="AAPL", price=100.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0, market="US",
        )
        assert result.allowed is True
        # Capped at 50% of portfolio
        assert result.allocation_usd <= 50_000 * 0.10 + 1

    def test_no_allocations_configured(self):
        """No market_allocations → no cap applied."""
        rm = RiskManager()
        result = rm.calculate_position_size(
            symbol="AAPL", price=100.0,
            portfolio_value=100_000, cash_available=100_000,
            current_positions=0, market="US",
        )
        assert result.allocation_usd <= 100_000 * 0.10 + 1
        assert result.allocation_usd > 50_000 * 0.10
