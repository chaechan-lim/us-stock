"""Unit tests for backtest sector strength history.

Core contract:
- confidence_multiplier maps strength+weight → sensible multiplier
- SectorHistory.score_at returns last-known scores for missing dates
- sector_for returns Unknown for unmapped symbols (neutral boost)
"""

import pytest

from backtest.sector_data import (
    SectorHistory,
    confidence_multiplier,
)


class TestConfidenceMultiplier:
    def test_zero_weight_always_neutral(self):
        assert confidence_multiplier(100, weight=0.0) == 1.0
        assert confidence_multiplier(0, weight=0.0) == 1.0
        assert confidence_multiplier(None, weight=0.0) == 1.0

    def test_none_strength_neutral(self):
        assert confidence_multiplier(None, weight=0.3) == 1.0

    def test_strength_50_is_neutral(self):
        assert confidence_multiplier(50, weight=0.3) == 1.0

    def test_full_strength_boosts_up(self):
        # weight=0.3, strength=100 → 1 + 0.3 * (100-50)/50 = 1.30
        assert confidence_multiplier(100, weight=0.3) == pytest.approx(1.30)

    def test_zero_strength_suppresses(self):
        # weight=0.3, strength=0 → 1 + 0.3 * -1 = 0.70
        assert confidence_multiplier(0, weight=0.3) == pytest.approx(0.70)

    def test_smaller_weight_smaller_effect(self):
        # weight=0.1, strength=100 → 1.10
        assert confidence_multiplier(100, weight=0.1) == pytest.approx(1.10)

    def test_multiplier_floor_at_0_1(self):
        """Even pathological weight+strength combos shouldn't kill confidence."""
        # weight=5, strength=0 → raw = 1 + 5 * -1 = -4; floor = 0.1
        assert confidence_multiplier(0, weight=5) == 0.1


class TestSectorHistory:
    def _make(self) -> SectorHistory:
        return SectorHistory(
            dates=["2026-01-05", "2026-01-06", "2026-01-07"],
            scores={
                "2026-01-05": {"Technology": 80.0, "Financials": 30.0},
                "2026-01-06": {"Technology": 75.0, "Financials": 40.0},
                "2026-01-07": {"Technology": 90.0, "Financials": 20.0},
            },
            symbol_sector={"MSFT": "Technology", "JPM": "Financials"},
        )

    def test_score_exact_date(self):
        h = self._make()
        assert h.score_at("2026-01-06") == {"Technology": 75.0, "Financials": 40.0}

    def test_score_handles_datetime(self):
        import datetime
        h = self._make()
        result = h.score_at(datetime.date(2026, 1, 6))
        assert result["Technology"] == 75.0

    def test_score_falls_back_to_prior(self):
        """Weekend / holiday lookups return the last trading day."""
        h = self._make()
        # Saturday → use Friday's (01-07 is in index; simulate 01-08 query)
        assert h.score_at("2026-01-08") == {"Technology": 90.0, "Financials": 20.0}

    def test_score_before_history_returns_empty(self):
        h = self._make()
        assert h.score_at("2025-01-01") == {}

    def test_empty_history_returns_empty(self):
        h = SectorHistory(dates=[], scores={}, symbol_sector={})
        assert h.score_at("2026-01-06") == {}

    def test_sector_for_known_symbol(self):
        h = self._make()
        assert h.sector_for("MSFT") == "Technology"

    def test_sector_for_unknown_symbol(self):
        h = self._make()
        assert h.sector_for("XYZZY") == "Unknown"

    def test_sector_for_kr_symbol_strips_suffix(self):
        """KR backtest universe uses .KS suffix but KR_SYMBOL_SECTOR keys
        are bare codes. Regression: the first sector_boost sweep produced
        identical metrics across all weights because every KR symbol
        resolved to Unknown."""
        h = SectorHistory(
            dates=[],
            scores={},
            symbol_sector={"005930": "반도체"},
        )
        assert h.sector_for("005930.KS") == "반도체"
        assert h.sector_for("005930.KQ") == "반도체"
        assert h.sector_for("005930") == "반도체"
