"""Tests for performance_metrics module."""

from datetime import date, timedelta

from analytics.performance_metrics import (
    compute_equity_metrics,
    compute_exposure_pct,
    compute_trade_metrics,
)


class TestTradeMetrics:
    def test_empty_returns_zero_metrics(self):
        m = compute_trade_metrics([])
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.expectancy == 0.0

    def test_basic_win_loss_distribution(self):
        trades = [
            # 3 wins, 2 losses → WR 60%
            {"side": "SELL", "pnl": 100, "market": "US", "quantity": 10,
             "filled_price": 100.0, "price": 100.0},
            {"side": "SELL", "pnl": 50, "market": "US", "quantity": 5,
             "filled_price": 50.0, "price": 50.0},
            {"side": "SELL", "pnl": 30, "market": "US", "quantity": 3,
             "filled_price": 30.0, "price": 30.0},
            {"side": "SELL", "pnl": -40, "market": "US", "quantity": 4,
             "filled_price": 40.0, "price": 40.0},
            {"side": "SELL", "pnl": -20, "market": "US", "quantity": 2,
             "filled_price": 20.0, "price": 20.0},
            # BUYs (no pnl) — counted only for cost tally
            {"side": "BUY", "pnl": None, "market": "US", "quantity": 10,
             "filled_price": 100.0, "price": 100.0},
        ]
        m = compute_trade_metrics(trades)
        assert m.total_trades == 5
        assert m.wins == 3
        assert m.losses == 2
        assert m.win_rate == 0.6
        assert m.gross_profit == 180  # 100+50+30
        assert m.gross_loss == 60  # |-40-20|
        assert m.gross_pf == 3.0
        # Expectancy = 0.6×60 - 0.4×30 = 36 - 12 = 24
        assert m.expectancy == 24
        # US fees ~ 0.05% × notional. 6 fills (5 SELL + 1 BUY).
        assert m.estimated_fees > 0
        assert m.net_pf < m.gross_pf  # costs always shrink Net PF

    def test_no_losses_returns_inf_pf(self):
        trades = [
            {"side": "SELL", "pnl": 100, "market": "US", "quantity": 10,
             "filled_price": 100.0, "price": 100.0},
        ]
        m = compute_trade_metrics(trades)
        assert m.gross_pf == float("inf")
        assert m.win_rate == 1.0

    def test_round_trip_collapses_partial_sells(self):
        """1 BUY → 3 partial SELLs (profit_taking + tier + trailing) should
        count as 1 round-trip in the round-trip metrics, not 3.
        Per-SELL WR can still report 3 wins, but rt_wins == 1."""
        trades = [
            {"symbol": "ATOM", "side": "BUY", "pnl": None, "market": "US",
             "quantity": 6, "filled_price": 10.0, "price": 10.0,
             "filled_at": "2026-05-08 10:00"},
            {"symbol": "ATOM", "side": "SELL", "pnl": 1.5, "market": "US",
             "quantity": 2, "filled_price": 10.5, "price": 10.5,
             "filled_at": "2026-05-08 11:00"},
            {"symbol": "ATOM", "side": "SELL", "pnl": 1.2, "market": "US",
             "quantity": 2, "filled_price": 10.4, "price": 10.4,
             "filled_at": "2026-05-08 12:00"},
            {"symbol": "ATOM", "side": "SELL", "pnl": 0.6, "market": "US",
             "quantity": 2, "filled_price": 10.2, "price": 10.2,
             "filled_at": "2026-05-08 13:00"},
        ]
        m = compute_trade_metrics(trades)
        # Per-SELL: 3 trades, 3 wins
        assert m.total_trades == 3
        assert m.win_rate == 1.0
        # Round-trip: 1 round-trip, 1 win, total PnL 1.5+1.2+0.6
        assert m.round_trips == 1
        assert m.round_trip_wins == 1
        assert m.round_trip_losses == 0
        assert m.round_trip_avg_pnl == 3.3

    def test_round_trip_open_position_skipped(self):
        """If position never returns to 0 (still held), no round-trip recorded."""
        trades = [
            {"symbol": "X", "side": "BUY", "pnl": None, "market": "US",
             "quantity": 5, "filled_price": 10, "price": 10,
             "filled_at": "2026-05-08 10:00"},
            {"symbol": "X", "side": "SELL", "pnl": 1.0, "market": "US",
             "quantity": 2, "filled_price": 11, "price": 11,
             "filled_at": "2026-05-08 11:00"},
            # position still 3 — no round-trip yet
        ]
        m = compute_trade_metrics(trades)
        assert m.round_trips == 0

    def test_round_trip_mixed_wins_losses(self):
        """Two complete round-trips, one positive PnL, one negative."""
        trades = [
            # Trip 1: +5 net
            {"symbol": "A", "side": "BUY", "pnl": None, "market": "US",
             "quantity": 1, "filled_price": 100, "price": 100,
             "filled_at": "2026-05-08 09:00"},
            {"symbol": "A", "side": "SELL", "pnl": 5, "market": "US",
             "quantity": 1, "filled_price": 105, "price": 105,
             "filled_at": "2026-05-08 10:00"},
            # Trip 2: -3 net
            {"symbol": "A", "side": "BUY", "pnl": None, "market": "US",
             "quantity": 1, "filled_price": 105, "price": 105,
             "filled_at": "2026-05-08 11:00"},
            {"symbol": "A", "side": "SELL", "pnl": -3, "market": "US",
             "quantity": 1, "filled_price": 102, "price": 102,
             "filled_at": "2026-05-08 12:00"},
        ]
        m = compute_trade_metrics(trades)
        assert m.round_trips == 2
        assert m.round_trip_wins == 1
        assert m.round_trip_losses == 1
        assert m.round_trip_win_rate == 0.5


class TestEquityMetrics:
    def test_flat_equity_returns_zero(self):
        d0 = date(2026, 1, 1)
        series = [(d0 + timedelta(days=i), 1000.0) for i in range(10)]
        m = compute_equity_metrics(series)
        assert m.net_return_pct == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.calmar_ratio == 0.0

    def test_uptrend_calmar(self):
        # 10 days, 0.5%/day → +5% total. No drawdown.
        d0 = date(2026, 1, 1)
        series = [(d0 + timedelta(days=i), 1000.0 * (1.005 ** i)) for i in range(10)]
        m = compute_equity_metrics(series)
        assert m.net_return_pct > 4.0
        assert m.max_drawdown_pct == 0.0
        # No DD → Calmar undefined → 0
        assert m.calmar_ratio == 0.0

    def test_drawdown_and_recovery(self):
        # Series: 100, 110, 120 (peak), 100 (-16.7% DD), 105, 115, 125 (new high)
        d0 = date(2026, 1, 1)
        values = [100, 110, 120, 100, 105, 115, 125]
        series = [(d0 + timedelta(days=i), v) for i, v in enumerate(values)]
        m = compute_equity_metrics(series)
        assert m.max_drawdown_pct < -10.0
        # Recovered: trough at index 3, fresh high at index 6 → 3 days
        assert m.max_dd_recovery_days == 3

    def test_unrecovered_drawdown(self):
        # Equity falls and never recovers in window
        d0 = date(2026, 1, 1)
        values = [100, 120, 110, 90, 95]  # peak 120 day1; never back
        series = [(d0 + timedelta(days=i), v) for i, v in enumerate(values)]
        m = compute_equity_metrics(series)
        # Trough at index 3; recovery counts days from trough to end
        assert m.max_dd_recovery_days == 1
        assert m.max_drawdown_pct < -20.0


class TestExposure:
    def test_empty_returns_zero(self):
        assert compute_exposure_pct([]) == 0.0

    def test_typical_mix(self):
        # 3 snapshots: 50%, 75%, 100% deployed → avg 75%
        snaps = [
            {"total_value": 1000, "cash": 500},
            {"total_value": 1000, "cash": 250},
            {"total_value": 1000, "cash": 0},
        ]
        assert compute_exposure_pct(snaps) == 75.0
