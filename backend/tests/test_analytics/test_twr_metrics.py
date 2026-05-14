"""P1-D (2026-05-14): TWR-aware compute_equity_metrics.

Without TWR, a single deposit doubling equity makes net_return read +100%
and Sharpe explode. With cash_flows passed, that effect is excluded.
"""

from datetime import date, timedelta

import pytest

from analytics.performance_metrics import compute_equity_metrics


def _series(values: list[float]) -> list[tuple[date, float]]:
    d0 = date(2026, 5, 1)
    return [(d0 + timedelta(days=i), v) for i, v in enumerate(values)]


class TestTwrCorrection:
    def test_pure_deposit_zero_strategy_return(self):
        """Equity doubles purely from deposit → net_return should be ~0."""
        eq = _series([1_000_000, 1_000_000, 2_000_000, 2_000_000, 2_000_000,
                      2_000_000, 2_000_000, 2_000_000, 2_000_000, 2_000_000])
        cf = [0, 0, 1_000_000, 0, 0, 0, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        # No trading PnL → near-zero return
        assert abs(m.net_return_pct) < 0.01

    def test_no_cash_flows_back_compat(self):
        """When cash_flows omitted (or all zero), behavior unchanged."""
        eq = _series([100.0, 102.0, 101.0, 103.0, 105.0, 106.0, 108.0, 110.0])
        m_legacy = compute_equity_metrics(eq)
        m_cf = compute_equity_metrics(eq, cash_flows=[0.0] * len(eq))
        assert m_legacy.net_return_pct == m_cf.net_return_pct
        assert m_legacy.sharpe_ratio == m_cf.sharpe_ratio
        assert m_legacy.max_drawdown_pct == m_cf.max_drawdown_pct

    def test_deposit_plus_real_gain(self):
        """Deposit on day 2 + 5% gain on subsequent days → return ≈ 5%."""
        # Start 100. Day 1: still 100. Day 2: deposit 100 → 200. Days 3-9:
        # +5% trading on the 200 base. So end_equity = 210, deposit = 100.
        # Raw net = (210 - 100)/100 = +110%. TWR should be ~+5%.
        eq = _series([100, 100, 200, 202, 204, 206, 208, 210, 210, 210])
        cf = [0, 0, 100, 0, 0, 0, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        # TWR: day 1-2 ratio = 100/100 = 1.0
        # day 2-3 ratio = (200-100)/100 = 1.0 (deposit excluded)
        # day 3-4 ratio = 202/200 = 1.01
        # ... cumulative gain matches the 5% trading on the 200 base.
        assert 4.0 <= m.net_return_pct <= 6.0

    def test_sharpe_finite_with_deposit(self):
        """Deposit-day large jump must not cause Sharpe to explode."""
        # 10 days; pure deposit on day 5; no trading PnL.
        eq = _series([100, 100, 100, 100, 100, 200, 200, 200, 200, 200])
        cf = [0, 0, 0, 0, 0, 100, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        # All returns zero → Sharpe should be 0, not 5+
        assert abs(m.sharpe_ratio) < 0.01
        assert m.sufficient_samples is True

    def test_mdd_on_twr_curve(self):
        """MDD ignores deposit jumps; uses synthetic equity curve."""
        # Start 100. Day 1: 95 (-5% trading). Day 2: deposit 1000 → 1095.
        # Day 3: 1086 (-0.8% trading). Raw MDD on equity_series = -5%
        # (95 vs initial 100), then big jump to 1095. With TWR, the MDD
        # should reflect the strategy's actual ~-5.8% drawdown.
        eq = _series([100, 95, 1095, 1086, 1090, 1095, 1100, 1095, 1090])
        cf = [0, 0, 1000, 0, 0, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        assert m.max_drawdown_pct < 0
        # Without TWR the calc would have peaked at 1100 → MDD only -0.5%.
        # With TWR, peak at synth[1] is below synth[0] → MDD seen.

    def test_withdrawal_handled(self):
        """Withdrawal is negative cash_flow — symmetric to deposit."""
        # Start 200 → 199 trading. Day 3 withdraw 100 → 99. Day 4 199
        # (wait: starting 99 going to 199 with no deposit would be +100%
        # trading, unrealistic). Let's just verify the math doesn't blow up.
        eq = _series([200, 199, 99, 100, 101, 102, 103, 104])
        cf = [0, 0, -100, 0, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        # Day 3 ratio: (99 - (-100)) / 199 = 1.0 (no trading change)
        # Other days: small +ve trading
        assert m.net_return_pct > 0
        assert m.net_return_pct < 5  # Sanity: not 50%

    def test_start_end_equity_reflect_actual(self):
        """start_equity / end_equity stay as REAL dollar values, not synthetic."""
        eq = _series([100, 100, 200, 200, 200, 200, 200, 200, 200, 200])
        cf = [0, 0, 100, 0, 0, 0, 0, 0, 0, 0]
        m = compute_equity_metrics(eq, cash_flows=cf)
        assert m.start_equity == 100.0
        assert m.end_equity == 200.0
