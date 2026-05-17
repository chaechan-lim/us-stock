"""Unit tests for ETF backtest harness (#52 unblock).

Tests config defaults + risk-cap clamp logic without booting yfinance.
"""

from datetime import date
from unittest.mock import patch

import pytest

from backtest.etf_engine_backtest import (
    ETFBacktestConfig,
    ETFBacktestEngine,
    _ETFPosition,
)


class TestConfig:
    def test_defaults(self):
        cfg = ETFBacktestConfig()
        assert cfg.initial_capital == 10_000.0
        assert cfg.period == "2y"
        assert cfg.market == "US"
        assert cfg.max_portfolio_pct is None  # falls through to universe yaml
        assert cfg.max_single_etf_pct is None
        assert cfg.regime_alloc_pct["uptrend"] == pytest.approx(0.07)

    def test_overrides(self):
        cfg = ETFBacktestConfig(
            initial_capital=50_000,
            max_portfolio_pct=0.20,
            max_single_etf_pct=0.10,
        )
        assert cfg.initial_capital == 50_000
        assert cfg.max_portfolio_pct == pytest.approx(0.20)


class TestEngineInit:
    def test_caps_inherited_from_universe_yaml_when_unset(self):
        """When cfg.max_* is None, fall through to universe.risk_rules."""
        cfg = ETFBacktestConfig()
        engine = ETFBacktestEngine(cfg)
        # universe yaml currently sets US ETF caps to 10%/5% (conservative re-enable)
        assert 0 < engine._max_portfolio_pct <= 1.0
        assert 0 < engine._max_single_pct <= 1.0

    def test_cfg_overrides_universe(self):
        cfg = ETFBacktestConfig(max_portfolio_pct=0.30, max_single_etf_pct=0.15)
        engine = ETFBacktestEngine(cfg)
        assert engine._max_portfolio_pct == pytest.approx(0.30)
        assert engine._max_single_pct == pytest.approx(0.15)


class TestExecutionPureMath:
    """Verify buy/sell mutate state correctly without yfinance.

    Calls private _execute_* with hand-crafted state. Tests:
      - per-ETF cap (single_pct × equity)
      - portfolio cap (sum existing values ≤ portfolio_pct × equity)
      - cash sufficiency
      - sell returns proceeds with slippage
    """

    @pytest.fixture
    def engine(self):
        cfg = ETFBacktestConfig(
            max_portfolio_pct=0.30,
            max_single_etf_pct=0.15,
            slippage_pct=0.0,  # easier math
            commission_per_order=0.0,
        )
        return ETFBacktestEngine(cfg)

    def test_buy_respects_single_cap(self, engine):
        """Single ETF capped at 15% of $10k = $1,500 → 15 shares @ $100."""
        positions = {}
        cash = [10_000.0]
        trades = [0]
        ok = engine._execute_buy(
            "TQQQ", price=100.0, equity=10_000.0,
            reason="regime_bull", d=date(2026, 1, 1),
            positions=positions, cash_box=cash, trades_box=trades,
        )
        assert ok
        assert positions["TQQQ"].quantity == 15  # $1500 / $100
        assert cash[0] == pytest.approx(10_000 - 1500)

    def test_buy_respects_portfolio_cap(self, engine):
        """Portfolio cap 30% of $10k = $3k. Already holding $2.5k → only $500 room."""
        positions = {
            "TQQQ": _ETFPosition(
                symbol="TQQQ", quantity=25, entry_price=100.0,
                entry_date=date(2026, 1, 1), reason="regime_bull",
            ),
        }
        cash = [10_000.0]
        trades = [0]
        # Try to buy SOXL at $50 (single cap allows $1500 = 30 shares,
        # but portfolio room is only $500 = 10 shares).
        ok = engine._execute_buy(
            "SOXL", price=50.0, equity=10_000.0,
            reason="regime_bull", d=date(2026, 1, 1),
            positions=positions, cash_box=cash, trades_box=trades,
        )
        assert ok
        assert positions["SOXL"].quantity == 10
        # _latest_price stub returns entry_price → existing value = 25 * 100 = 2500
        # portfolio cap = 3000, room = 500 → 500/50 = 10 shares

    def test_buy_rejected_when_at_portfolio_cap(self, engine):
        """At portfolio cap → reject new buy."""
        positions = {
            "TQQQ": _ETFPosition(
                symbol="TQQQ", quantity=30, entry_price=100.0,  # = $3000
                entry_date=date(2026, 1, 1), reason="regime_bull",
            ),
        }
        cash = [10_000.0]
        trades = [0]
        ok = engine._execute_buy(
            "SOXL", price=50.0, equity=10_000.0,
            reason="regime_bull", d=date(2026, 1, 1),
            positions=positions, cash_box=cash, trades_box=trades,
        )
        assert ok is False
        assert "SOXL" not in positions

    def test_sell_returns_proceeds_to_cash(self, engine):
        positions = {
            "TQQQ": _ETFPosition(
                symbol="TQQQ", quantity=10, entry_price=100.0,
                entry_date=date(2026, 1, 1), reason="regime_bull",
            ),
        }
        cash = [0.0]
        trades = [0]
        engine._execute_sell("TQQQ", price=120.0, positions=positions,
                             cash_box=cash, trades_box=trades)
        assert "TQQQ" not in positions
        assert cash[0] == pytest.approx(1200.0)  # 10 * 120, no slippage/commission

    def test_sell_unknown_symbol_noop(self, engine):
        positions = {}
        cash = [100.0]
        trades = [0]
        engine._execute_sell("FAKE", price=50.0, positions=positions,
                             cash_box=cash, trades_box=trades)
        assert cash[0] == 100.0  # unchanged
        assert trades[0] == 0
