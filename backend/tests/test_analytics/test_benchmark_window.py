"""P1-G (2026-05-15): benchmark_return_pct date-range alignment."""

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from analytics import performance_metrics as pm


def _hist_df(closes_by_date):
    """Build a yfinance-style daily history DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in closes_by_date.keys()])
    return pd.DataFrame({"Close": list(closes_by_date.values())}, index=idx)


def setup_function(_func):
    pm._BENCHMARK_CACHE.clear()


class TestBenchmarkDateRange:
    def test_uses_exact_window_when_dates_provided(self):
        """start/end_date overrides `days` and computes return on exact window."""
        sd = date(2026, 5, 6)
        ed = date(2026, 5, 14)
        closes = {
            sd - timedelta(days=2): 700.0,  # outside window (padding)
            sd: 733.83,                       # window start
            date(2026, 5, 10): 740.00,
            ed: 748.17,                       # window end
            ed + timedelta(days=1): 760.0,    # outside window (padding)
        }
        fake_df = _hist_df(closes)
        with patch.object(pm, "_BENCHMARK_CACHE", {}):
            with patch("yfinance.download", return_value=fake_df):
                ret = pm.benchmark_return_pct(
                    "SPY", days=30, start_date=sd, end_date=ed
                )
        # Should compute (748.17 - 733.83) / 733.83 = +1.95%
        assert ret == pytest.approx(1.95, abs=0.01)

    def test_returns_none_when_window_has_fewer_than_2_closes(self):
        sd = date(2026, 5, 14)
        ed = date(2026, 5, 14)
        closes = {sd: 700.0}
        with patch.object(pm, "_BENCHMARK_CACHE", {}):
            with patch("yfinance.download", return_value=_hist_df(closes)):
                ret = pm.benchmark_return_pct(
                    "SPY", days=30, start_date=sd, end_date=ed
                )
        assert ret is None

    def test_falls_back_to_days_when_no_dates(self):
        """Without start/end_date, behavior is backward-compatible."""
        fake_ticker = MagicMock()
        fake_ticker.history.return_value = _hist_df({
            date(2026, 5, 1): 700.0,
            date(2026, 5, 14): 720.0,
        })
        with patch.object(pm, "_BENCHMARK_CACHE", {}):
            with patch("yfinance.Ticker", return_value=fake_ticker):
                ret = pm.benchmark_return_pct("SPY", days=14)
        assert ret == pytest.approx(2.86, abs=0.01)

    def test_cache_key_includes_date_range(self):
        """Different date ranges produce different cache entries."""
        sd1, ed1 = date(2026, 5, 1), date(2026, 5, 10)
        sd2, ed2 = date(2026, 5, 5), date(2026, 5, 14)
        with patch.object(pm, "_BENCHMARK_CACHE", {}) as cache:
            with patch("yfinance.download") as dl:
                dl.return_value = _hist_df({
                    sd1: 100.0, ed1: 110.0, sd2: 105.0, ed2: 120.0,
                })
                pm.benchmark_return_pct("SPY", 30, start_date=sd1, end_date=ed1)
                pm.benchmark_return_pct("SPY", 30, start_date=sd2, end_date=ed2)
            assert len(cache) == 2

    def test_fixes_live_alpha_mismatch(self):
        """Live case: 9-day data window. Without P1-G fix, dashboard
        showed -7% alpha (compared 9-day us vs 30-day SPY). With P1-G,
        benchmark computed over the SAME 9 days."""
        sd = date(2026, 5, 6)
        ed = date(2026, 5, 14)
        # Same closes that produced +1.95% in the live computation
        closes = {sd: 733.83, ed: 748.17}
        with patch.object(pm, "_BENCHMARK_CACHE", {}):
            with patch("yfinance.download", return_value=_hist_df(closes)):
                ret = pm.benchmark_return_pct(
                    "SPY", days=30, start_date=sd, end_date=ed
                )
        # 9-day SPY = +1.95%, our return = +7.08% → apples-to-apples
        # alpha is +5.13% (not -7.1% the dashboard previously showed).
        assert ret == pytest.approx(1.95, abs=0.01)
