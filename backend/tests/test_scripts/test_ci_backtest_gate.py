"""Unit tests for backtest.gate — regression detection logic.

The compare function is the heart of the CI gate — if it silently stops
flagging regressions, the gate becomes a no-op without anyone noticing.
"""

import pytest

from backtest.gate import THRESHOLDS, compare


_BASELINE = {
    "ret": 6.8,
    "sharpe": 0.32,
    "mdd": -6.6,
    "pf": 1.14,
    "trades": 432,
}


def _current(**overrides) -> dict:
    return {**_BASELINE, **overrides}


class TestCompare:
    def test_no_regression_same_metrics(self):
        assert compare(_BASELINE, _current(), "KR") == []

    def test_no_regression_minor_sharpe_drop(self):
        # -0.29 drop — just under 0.30 threshold
        assert compare(_BASELINE, _current(sharpe=0.03), "KR") == []

    def test_regression_sharpe_drop(self):
        failures = compare(_BASELINE, _current(sharpe=-0.10), "KR")
        assert len(failures) == 1
        assert "Sharpe" in failures[0]

    def test_regression_pf_drop(self):
        # PF drop 0.25 — over 0.20 threshold
        failures = compare(_BASELINE, _current(pf=0.89), "KR")
        assert len(failures) == 1
        assert "PF" in failures[0]

    def test_regression_mdd_worsen(self):
        # MDD goes from -6.6% to -15% — worsens by 8.4pp, over 5pp threshold
        failures = compare(_BASELINE, _current(mdd=-15.0), "KR")
        assert len(failures) == 1
        assert "MDD" in failures[0]

    def test_regression_trades_collapse(self):
        # Trades drop from 432 to 100 — 77% drop, over 50% threshold
        failures = compare(_BASELINE, _current(trades=100), "KR")
        assert len(failures) == 1
        assert "trades" in failures[0]

    def test_multiple_regressions_reported(self):
        current = _current(sharpe=-0.5, pf=0.8, mdd=-20.0)
        failures = compare(_BASELINE, current, "KR")
        assert len(failures) == 3

    def test_return_drop_not_gated(self):
        # Big return drop alone should NOT fail (too noisy to gate)
        assert compare(_BASELINE, _current(ret=-10.0), "KR") == []

    def test_improvement_not_flagged(self):
        better = _current(sharpe=0.80, pf=1.50, mdd=-3.0, trades=500)
        assert compare(_BASELINE, better, "KR") == []

    def test_empty_baseline_reports_missing(self):
        failures = compare({}, _current(), "KR")
        assert len(failures) == 1
        assert "no baseline" in failures[0].lower()

    def test_trades_zero_baseline_no_divide_error(self):
        # Defensive: empty baseline shouldn't crash
        compare({"ret": 0, "sharpe": 0, "mdd": 0, "pf": 0, "trades": 0},
                _current(trades=0), "KR")


class TestThresholds:
    def test_thresholds_documented(self):
        """Match the values the CI gate docstring and compare() docstring
        promise — prevents silent tightening without docs update."""
        assert THRESHOLDS == {
            "sharpe_drop": 0.30,
            "pf_drop": 0.20,
            "mdd_worsen_pp": 5.0,
            "trades_drop_pct": 50.0,
        }
