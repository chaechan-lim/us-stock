"""Unit tests for services.exposure_tracker (Hermes Phase 0+1)."""

import json
from datetime import date
from pathlib import Path

import pytest

from services.exposure_tracker import (
    CHRONIC_DAYS,
    CHRONIC_DEPLOYED_PCT,
    FUNNEL_CONCENTRATION_PCT,
    ExposureSnapshot,
    FunnelBreakdown,
    MarketExposure,
    SentinelFlag,
    compute_funnel_breakdown,
    compute_market_exposure,
    evaluate_sentinel,
    load_prior_idle_streak,
    write_snapshot,
)


class TestFunnelBreakdown:
    def test_empty_payload_returns_none(self):
        assert compute_funnel_breakdown("KR", None) is None
        assert compute_funnel_breakdown("KR", {}) is None

    def test_no_signals(self):
        fb = compute_funnel_breakdown("KR", {
            "buy_signals_total": 0, "buys_placed": 0, "rejections": {},
        })
        assert fb is not None
        assert fb.total_signals == 0
        assert fb.top_reason_pct == 0.0
        assert fb.top_reasons == []

    def test_top_reasons_sorted(self):
        fb = compute_funnel_breakdown("KR", {
            "buy_signals_total": 10,
            "buys_placed": 2,
            "fill_rate": 0.2,
            "rejections": {
                "sell_cooldown": 5,
                "sizing_Price too high": 2,
                "whipsaw_block": 1,
            },
        })
        assert fb.total_signals == 10
        assert fb.buys_placed == 2
        assert fb.fill_rate == pytest.approx(0.2)
        assert fb.top_reasons[0] == ("sell_cooldown", 5, pytest.approx(0.5))
        assert fb.top_reason_pct == pytest.approx(0.5)
        # Top-3 ordering preserved
        assert [r[0] for r in fb.top_reasons] == [
            "sell_cooldown", "sizing_Price too high", "whipsaw_block",
        ]

    def test_top_3_cap(self):
        fb = compute_funnel_breakdown("KR", {
            "buy_signals_total": 10,
            "buys_placed": 0,
            "rejections": {
                "a": 4, "b": 3, "c": 2, "d": 1,
            },
        })
        assert len(fb.top_reasons) == 3
        assert [r[0] for r in fb.top_reasons] == ["a", "b", "c"]


class TestMarketExposure:
    def _positions(self, *prices_qty: tuple[float, float]) -> list[dict]:
        return [
            {"symbol": f"S{i}", "market": "KR", "current_price": p, "quantity": q}
            for i, (p, q) in enumerate(prices_qty)
        ]

    def test_basic_deployed_pct(self):
        # Equity 100M KRW, 2 positions worth 10M + 5M = 15M = 15%
        exp = compute_market_exposure(
            market="KR",
            equity=100_000_000,
            cash=85_000_000,
            positions=self._positions((10_000_000, 1), (5_000_000, 1)),
            funnel=None,
            min_position_pct=0.05,
        )
        assert exp.stock_value == 15_000_000
        assert exp.deployed_pct == pytest.approx(0.15)
        assert exp.position_count == 2
        assert exp.target_slot == 5_000_000

    def test_placeholder_count(self):
        # Target slot = 5M (5% of 100M). Threshold = 30% of slot = 1.5M.
        # 1 share worth 500K → placeholder. 1 share worth 2M → not placeholder.
        exp = compute_market_exposure(
            market="KR",
            equity=100_000_000,
            cash=90_000_000,
            positions=self._positions((500_000, 1), (2_000_000, 1)),
            funnel=None,
            min_position_pct=0.05,
        )
        assert exp.placeholder_count == 1

    def test_slot_fill_ratio_caps_at_1(self):
        # Position above slot doesn't artificially inflate fill ratio
        exp = compute_market_exposure(
            market="KR",
            equity=100_000_000,
            cash=0,
            positions=self._positions((10_000_000, 1), (2_500_000, 1)),
            funnel=None,
            min_position_pct=0.05,
        )
        # Position 1 = 10M / 5M = 2.0, capped to 1.0
        # Position 2 = 2.5M / 5M = 0.5
        # Avg = (1.0 + 0.5) / 2 = 0.75
        assert exp.slot_fill_ratio == pytest.approx(0.75)

    def test_zero_equity_no_div_zero(self):
        exp = compute_market_exposure(
            market="KR",
            equity=0,
            cash=0,
            positions=self._positions((1000, 1)),
            funnel=None,
            min_position_pct=0.05,
        )
        assert exp.deployed_pct == 0.0
        assert exp.slot_fill_ratio == 0.0
        assert exp.target_slot == 0.0

    def test_idle_streak_increments_when_under_threshold(self):
        # 30% deployed < 50% threshold, prior streak 4 → new streak 5
        exp = compute_market_exposure(
            market="KR",
            equity=100_000_000,
            cash=70_000_000,
            positions=self._positions((30_000_000, 1)),
            funnel=None,
            min_position_pct=0.05,
            prior_idle_streak=4,
        )
        assert exp.cash_idle_days == 5

    def test_idle_streak_resets_when_above_threshold(self):
        # 60% deployed > 50% → streak resets
        exp = compute_market_exposure(
            market="KR",
            equity=100_000_000,
            cash=40_000_000,
            positions=self._positions((60_000_000, 1)),
            funnel=None,
            min_position_pct=0.05,
            prior_idle_streak=10,
        )
        assert exp.cash_idle_days == 0

    def test_filters_by_market_field(self):
        positions = [
            {"symbol": "A", "market": "KR", "current_price": 1000, "quantity": 5},
            {"symbol": "B", "market": "US", "current_price": 100, "quantity": 10},
        ]
        exp = compute_market_exposure(
            market="KR",
            equity=100_000,
            cash=95_000,
            positions=positions,
            funnel=None,
            min_position_pct=0.05,
        )
        assert exp.position_count == 1
        assert exp.stock_value == 5_000


class TestSentinel:
    def _exp(self, **kwargs) -> MarketExposure:
        defaults = dict(
            market="KR", equity=1.0, cash=0.0, stock_value=0.0,
            deployed_pct=0.2, position_count=0, placeholder_count=0,
            target_slot=0.05, slot_fill_ratio=0.0, cash_idle_days=0,
            funnel=None,
        )
        defaults.update(kwargs)
        return MarketExposure(**defaults)

    def test_chronic_under_deployment_fires(self):
        exp = self._exp(deployed_pct=0.20, cash_idle_days=CHRONIC_DAYS)
        flags = evaluate_sentinel(exp)
        assert any(f.flag == "chronic_under_deployment" for f in flags)

    def test_chronic_under_deployment_skips_below_day_threshold(self):
        exp = self._exp(deployed_pct=0.20, cash_idle_days=CHRONIC_DAYS - 1)
        flags = evaluate_sentinel(exp)
        assert not any(f.flag == "chronic_under_deployment" for f in flags)

    def test_funnel_concentration_fires(self):
        funnel = FunnelBreakdown(
            market="KR", total_signals=20, buys_placed=5, fill_rate=0.25,
            top_reasons=[("sell_cooldown", 12, 0.6)],
            top_reason_pct=0.6,
        )
        exp = self._exp(funnel=funnel)
        flags = evaluate_sentinel(exp)
        assert any(f.flag == "funnel_concentration" for f in flags)

    def test_funnel_concentration_skipped_when_few_signals(self):
        funnel = FunnelBreakdown(
            market="KR", total_signals=3, buys_placed=0, fill_rate=0.0,
            top_reasons=[("sell_cooldown", 3, 1.0)],
            top_reason_pct=1.0,
        )
        exp = self._exp(funnel=funnel)
        flags = evaluate_sentinel(exp)
        assert not any(f.flag == "funnel_concentration" for f in flags)

    def test_funnel_concentration_skipped_when_below_pct_threshold(self):
        funnel = FunnelBreakdown(
            market="KR", total_signals=20, buys_placed=10, fill_rate=0.5,
            top_reasons=[("sell_cooldown", 6, 0.3)],
            top_reason_pct=0.3,
        )
        exp = self._exp(funnel=funnel)
        flags = evaluate_sentinel(exp)
        assert not any(f.flag == "funnel_concentration" for f in flags)


class TestPersistence:
    def test_write_and_load_prior_streak(self, tmp_path: Path):
        snap = ExposureSnapshot(
            date="2026-05-29",
            generated_at="2026-05-29T06:00:00+09:00",
            markets={
                "KR": MarketExposure(
                    market="KR", equity=100, cash=70, stock_value=30,
                    deployed_pct=0.30, position_count=2, placeholder_count=1,
                    target_slot=5, slot_fill_ratio=0.4, cash_idle_days=3,
                    funnel=None,
                ),
            },
            flags=[
                SentinelFlag(
                    market="KR", flag="chronic_under_deployment",
                    severity="warning", detail="deployed 30% for 3d",
                ),
            ],
        )
        out = write_snapshot(snap, tmp_path)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["date"] == "2026-05-29"
        assert data["markets"]["KR"]["cash_idle_days"] == 3
        assert data["flags"][0]["flag"] == "chronic_under_deployment"

        # Load prior streak: yesterday's value was 3 → loader returns 3
        streak = load_prior_idle_streak(tmp_path, "KR", date(2026, 5, 30))
        assert streak == 3

    def test_load_prior_streak_missing_returns_zero(self, tmp_path: Path):
        streak = load_prior_idle_streak(tmp_path, "KR", date(2026, 5, 29))
        assert streak == 0

    def test_load_prior_streak_walks_back_up_to_7d(self, tmp_path: Path):
        # Snapshot from 3 days ago — loader should still find it
        snap = ExposureSnapshot(
            date="2026-05-26",
            generated_at="2026-05-26T06:00:00+09:00",
            markets={
                "KR": MarketExposure(
                    market="KR", equity=100, cash=80, stock_value=20,
                    deployed_pct=0.20, position_count=1, placeholder_count=1,
                    target_slot=5, slot_fill_ratio=0.3, cash_idle_days=2,
                    funnel=None,
                ),
            },
            flags=[],
        )
        write_snapshot(snap, tmp_path)
        streak = load_prior_idle_streak(tmp_path, "KR", date(2026, 5, 29))
        assert streak == 2

    def test_load_prior_streak_stops_at_7d(self, tmp_path: Path):
        # Snapshot from 8 days ago — loader gives up
        snap = ExposureSnapshot(
            date="2026-05-21",
            generated_at="2026-05-21T06:00:00+09:00",
            markets={
                "KR": MarketExposure(
                    market="KR", equity=100, cash=80, stock_value=20,
                    deployed_pct=0.20, position_count=1, placeholder_count=1,
                    target_slot=5, slot_fill_ratio=0.3, cash_idle_days=5,
                    funnel=None,
                ),
            },
            flags=[],
        )
        write_snapshot(snap, tmp_path)
        streak = load_prior_idle_streak(tmp_path, "KR", date(2026, 5, 29))
        assert streak == 0
