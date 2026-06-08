"""Unit tests for ETF EW hedge mode (2026-06-08 thesis (c)).

Covers:
  - EWHedgeConfig validation (enabled requires inverse_etf + regime_proxy)
  - Regime classifier signal computation
  - Target weight calculation
  - Rebalance dispatch (skip vs act)
  - End-to-end evaluate() in ew_hedge mode
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from engine.etf_engine import ETFEngine, EWHedgeConfig


@pytest.fixture
def base_engine():
    """Minimal ETFEngine fixture with mocked deps."""
    md = AsyncMock()
    md.get_positions = AsyncMock(return_value=[])
    md.get_balance = AsyncMock(return_value=MagicMock(
        total=30_000_000, available=30_000_000, locked=0, currency="KRW",
    ))
    md.get_price = AsyncMock(return_value=15_000.0)
    md.get_ohlcv = AsyncMock(return_value=pd.DataFrame({
        "close": [100.0] * 300,
    }))
    om = MagicMock()
    om.place_buy = AsyncMock(return_value=MagicMock())
    om.place_sell = AsyncMock(return_value=MagicMock())
    universe = MagicMock()
    universe.risk_rules = MagicMock(
        max_hold_days=10, max_portfolio_pct=0.30,
        max_single_etf_pct=0.15, require_stop_loss=True,
        default_stop_loss_pct=0.08, min_hold_leveraged_hours=4,
        min_hold_sector_hours=2, sell_cooldown_hours=12,
    )
    universe.get_sector_etf_symbols = MagicMock(return_value=[
        "091160", "305720", "091180", "244580", "091170", "315930", "117680",
    ])
    universe.get_exchange = MagicMock(return_value="KRX")
    universe.is_leveraged = MagicMock(return_value=False)
    universe.get_all_sectors = MagicMock(return_value={})

    engine = ETFEngine(
        market_data=md,
        order_manager=om,
        etf_universe=universe,
        market="KR",
    )
    return engine


class TestEWHedgeConfig:
    def test_disabled_by_default(self, base_engine):
        assert base_engine._ew_hedge_cfg.enabled is False

    def test_enabled_requires_inverse_etf(self, base_engine):
        with pytest.raises(ValueError, match="inverse_etf empty"):
            base_engine.set_ew_hedge_config(EWHedgeConfig(
                enabled=True, regime_proxy="069500",
            ))

    def test_enabled_requires_regime_proxy(self, base_engine):
        with pytest.raises(ValueError, match="regime_proxy empty"):
            base_engine.set_ew_hedge_config(EWHedgeConfig(
                enabled=True, inverse_etf="114800",
            ))

    def test_hedge_ratio_bounds(self, base_engine):
        with pytest.raises(ValueError, match="hedge_ratio"):
            base_engine.set_ew_hedge_config(EWHedgeConfig(
                enabled=True, inverse_etf="114800",
                regime_proxy="069500", hedge_ratio=1.5,
            ))

    def test_min_signals_bounds(self, base_engine):
        with pytest.raises(ValueError, match="min_signals"):
            base_engine.set_ew_hedge_config(EWHedgeConfig(
                enabled=True, inverse_etf="114800",
                regime_proxy="069500", min_signals_for_hedge=4,
            ))

    def test_valid_config_accepted(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        assert base_engine._ew_hedge_cfg.enabled is True


class TestComputeEWTargets:
    def test_no_signals_full_sector_allocation(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        sectors = ["091160", "305720", "091180"]
        targets, hedge, inv = base_engine._compute_ew_targets(
            equity=10_000_000, n_signals_active=0, sector_syms=sectors,
        )
        assert hedge == 0.0
        assert inv == 0.0
        # Sectors get full allocation, equally
        for sym in sectors:
            assert abs(targets[sym] - 10_000_000 / 3) < 1.0
        # Inverse target is 0 when no signals
        assert targets["114800"] == 0.0

    def test_gate_below_threshold_no_hedge(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
            min_signals_for_hedge=2,
        ))
        sectors = ["091160", "305720"]
        targets, hedge, inv = base_engine._compute_ew_targets(
            equity=10_000_000, n_signals_active=1, sector_syms=sectors,
        )
        # 1 signal active but gate is 2 → no hedge
        assert hedge == 0.0
        assert inv == 0.0

    def test_gate_met_applies_hedge_ratio(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
            min_signals_for_hedge=2, hedge_ratio=0.50,
            inverse_vs_cash_ratio=0.5,
        ))
        sectors = ["091160", "305720"]
        targets, hedge, inv = base_engine._compute_ew_targets(
            equity=10_000_000, n_signals_active=2, sector_syms=sectors,
        )
        # 50% hedge, 25% inverse, 25% cash, 50% sectors
        assert hedge == 0.50
        assert abs(inv - 2_500_000) < 1.0   # 25% of equity
        assert abs(targets["114800"] - 2_500_000) < 1.0
        # Each sector gets (50% of equity) / 2 = 25% = 2.5M
        for sym in sectors:
            assert abs(targets[sym] - 2_500_000) < 1.0


class TestRegimeSignals:
    async def test_zero_when_no_data(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        base_engine._market_data.get_ohlcv = AsyncMock(
            return_value=pd.DataFrame({"close": []}),
        )
        s1, s2, s3 = await base_engine._compute_regime_signals()
        assert s1 is False and s2 is False and s3 is False

    async def test_roc_signal_fires_on_5pct_drop(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
            roc_5d_threshold=-0.03, breadth_threshold=0.30,
            sector_sma=50,
        ))
        # 100 bars steady at 100, then drop to 94 (= −6% > −3% threshold)
        close_series = [100.0] * 95 + [98.0, 97.0, 96.0, 95.0, 94.0]
        proxy_df = pd.DataFrame({"close": close_series})
        # All sector ETFs: rising (above SMA → high breadth, S3=False)
        sector_df = pd.DataFrame({"close": [
            *[100.0] * 50, *[100 + i for i in range(50)],
        ]})

        async def _fake_get_ohlcv(symbol, **kw):
            if symbol == "069500":
                return proxy_df
            return sector_df

        base_engine._market_data.get_ohlcv = AsyncMock(side_effect=_fake_get_ohlcv)
        s1, s2, s3 = await base_engine._compute_regime_signals()
        assert s1 is False  # vol disabled
        assert s2 is True   # ROC fires
        assert s3 is False  # breadth healthy

    async def test_breadth_signal_fires_when_few_sectors_above_sma(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
            roc_5d_threshold=-0.10, breadth_threshold=0.30,
            sector_sma=50,
        ))
        # Proxy: flat (S2 won't fire)
        proxy_df = pd.DataFrame({"close": [100.0] * 100})
        # All sectors: declining below SMA50 → low breadth (S3 fires)
        sector_df = pd.DataFrame({"close": [
            *[100.0] * 50, *[100 - i * 0.5 for i in range(50)],
        ]})

        async def _fake_get_ohlcv(symbol, **kw):
            if symbol == "069500":
                return proxy_df
            return sector_df

        base_engine._market_data.get_ohlcv = AsyncMock(side_effect=_fake_get_ohlcv)
        s1, s2, s3 = await base_engine._compute_regime_signals()
        assert s2 is False
        assert s3 is True  # all 7 sectors below SMA → 0% above < 30% → fire


class TestEvaluateDispatch:
    async def test_disabled_mode_skips_ew_hedge(self, base_engine):
        # ew_hedge disabled (default) → evaluate() runs rotation path
        market_state = MagicMock(regime=MagicMock(value="uptrend"),
                                 confidence=0.7, spy_distance_pct=10.0)
        # Mock evaluate's downstream methods so rotation doesn't blow up
        base_engine._check_hold_limits = AsyncMock(return_value=[])
        base_engine._manage_regime_etfs = AsyncMock(return_value=[])
        base_engine._manage_sector_etfs = AsyncMock(return_value=[])
        base_engine._check_exposure_limits = AsyncMock(return_value=[])
        result = await base_engine.evaluate(market_state, sector_data=None)
        # Rotation path returns regime/sector/risk keys (no ew_hedge key)
        assert "regime" in result
        assert "ew_hedge" not in result

    async def test_enabled_mode_runs_ew_hedge_path(self, base_engine):
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        # Stub signal computation to a known state (no signals active)
        base_engine._compute_regime_signals = AsyncMock(
            return_value=(False, False, False),
        )
        market_state = MagicMock()
        result = await base_engine.evaluate(market_state, sector_data=None)
        assert "ew_hedge" in result
        assert any("signals:" in s for s in result["regime"])


class TestEWHedgeBuyPath:
    """bug_001 + bug_009 regression: the BUY pass must pass a
    PositionSizeResult (not a tuple) and set skip_already_held=True."""

    async def test_bootstrap_buy_uses_position_size_result(self, base_engine):
        from engine.risk_manager import PositionSizeResult

        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        base_engine._compute_regime_signals = AsyncMock(
            return_value=(False, False, False),
        )
        # Empty portfolio → bootstrap BUYs fire for all 7 sectors
        base_engine._market_data.get_positions = AsyncMock(return_value=[])
        base_engine._market_data.get_balance = AsyncMock(return_value=MagicMock(
            total=30_000_000, available=30_000_000, locked=0, currency="KRW",
        ))
        base_engine._market_data.get_price = AsyncMock(return_value=10_000.0)
        base_engine._can_buy_etf = MagicMock(return_value=(True, ""))

        await base_engine.evaluate(MagicMock(), sector_data=None)

        # place_buy must have been called, and every call must pass a
        # PositionSizeResult (bug_001) + skip_already_held=True (bug_009)
        assert base_engine._order_manager.place_buy.await_count > 0
        for call in base_engine._order_manager.place_buy.await_args_list:
            sizing = call.kwargs.get("sizing_override")
            assert isinstance(sizing, PositionSizeResult), (
                f"sizing_override must be PositionSizeResult, got {type(sizing)}"
            )
            assert sizing.allowed is True
            assert call.kwargs.get("skip_already_held") is True

    async def test_buy_pass_does_not_raise_on_held_topup(self, base_engine):
        """A held sector below target must produce a top-up BUY, not
        an AttributeError-swallowing no-op."""
        base_engine.set_ew_hedge_config(EWHedgeConfig(
            enabled=True, inverse_etf="114800", regime_proxy="069500",
        ))
        base_engine._compute_regime_signals = AsyncMock(
            return_value=(False, False, False),
        )
        # 091160 held but well below target (1 share @ 10k = 10k vs ~4.3M target)
        held = MagicMock(symbol="091160", quantity=1, current_price=10_000.0)
        base_engine._market_data.get_positions = AsyncMock(return_value=[held])
        base_engine._market_data.get_balance = AsyncMock(return_value=MagicMock(
            total=30_000_000, available=29_990_000, locked=0, currency="KRW",
        ))
        base_engine._market_data.get_price = AsyncMock(return_value=10_000.0)
        base_engine._can_buy_etf = MagicMock(return_value=(True, ""))
        base_engine._can_sell_etf = MagicMock(return_value=(True, ""))

        result = await base_engine.evaluate(MagicMock(), sector_data=None)
        # 091160 top-up should appear as an ew_hedge BUY action
        assert any("BUY 091160" in a for a in result["ew_hedge"])
