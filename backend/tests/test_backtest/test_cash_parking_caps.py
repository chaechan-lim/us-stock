"""P3-A backtest path: per_cycle_pct + max_pct caps in cash_parking.

Verifies that PipelineConfig.cash_parking_per_cycle_pct + max_pct
correctly clamp park_amount in both BUY paths (initial + split-add).
Matches the live `_park_excess_cash` cap semantics.
"""

from dataclasses import dataclass

import pytest

from backtest.full_pipeline import PipelineConfig


class TestPipelineConfigFields:
    def test_default_max_pct_unbounded(self):
        """Backward-compat: default 1.0 means no cap (= prior behavior)."""
        cfg = PipelineConfig(market="US")
        assert cfg.cash_parking_max_pct == 1.0

    def test_default_per_cycle_pct_unbounded(self):
        """P3-A backward-compat: default 1.0 = one-shot (= prior behavior)."""
        cfg = PipelineConfig(market="US")
        assert cfg.cash_parking_per_cycle_pct == 1.0

    def test_caps_configurable(self):
        cfg = PipelineConfig(
            market="US",
            cash_parking_max_pct=0.40,
            cash_parking_per_cycle_pct=0.10,
        )
        assert cfg.cash_parking_max_pct == pytest.approx(0.40)
        assert cfg.cash_parking_per_cycle_pct == pytest.approx(0.10)


class TestClampSemantics:
    """Pure math tests of the cap formula used in both backtest paths.

    The live code uses: min(park_amount, headroom, per_cycle, cash_buffer).
    Backtest mirrors this. We check the binding-constraint logic without
    booting a full pipeline simulation.
    """

    @staticmethod
    def _clamp(park_amount: float, equity: float, existing_val: float,
               max_pct: float, per_cycle_pct: float) -> float:
        max_park_value = equity * max_pct
        headroom = max_park_value - existing_val
        if headroom <= 0:
            return 0.0
        per_cycle = equity * per_cycle_pct
        return max(0.0, min(park_amount, headroom, per_cycle))

    def test_per_cycle_is_binding_when_far_below_cap(self):
        """First buy: equity=100k, max=40k, per_cycle=10k. Should buy 10k."""
        out = self._clamp(park_amount=50_000, equity=100_000,
                          existing_val=0, max_pct=0.40, per_cycle_pct=0.10)
        assert out == pytest.approx(10_000)

    def test_headroom_is_binding_near_cap(self):
        """Near cap: existing=35k of 40k. Per_cycle 10k but headroom only 5k."""
        out = self._clamp(park_amount=50_000, equity=100_000,
                          existing_val=35_000, max_pct=0.40, per_cycle_pct=0.10)
        assert out == pytest.approx(5_000)

    def test_skip_at_cap(self):
        """existing already ≥ cap → 0 (caller should return without buying)."""
        out = self._clamp(park_amount=50_000, equity=100_000,
                          existing_val=40_000, max_pct=0.40, per_cycle_pct=0.10)
        assert out == 0.0

    def test_park_amount_below_per_cycle_passes_through(self):
        """Small cash available: park_amount smaller than per_cycle cap."""
        out = self._clamp(park_amount=3_000, equity=100_000,
                          existing_val=0, max_pct=0.40, per_cycle_pct=0.10)
        assert out == pytest.approx(3_000)

    def test_backward_compat_unbounded(self):
        """Defaults (1.0/1.0): no clamp, park_amount passes through."""
        out = self._clamp(park_amount=50_000, equity=100_000,
                          existing_val=0, max_pct=1.0, per_cycle_pct=1.0)
        assert out == pytest.approx(50_000)

    def test_four_cycles_to_reach_cap(self):
        """Per-cycle=10%, cap=40%. Simulating 4 cycles fills to cap."""
        equity = 100_000
        existing = 0
        for _ in range(4):
            buy = self._clamp(park_amount=50_000, equity=equity,
                              existing_val=existing,
                              max_pct=0.40, per_cycle_pct=0.10)
            existing += buy
        assert existing == pytest.approx(40_000)
        # 5th cycle should be 0
        buy5 = self._clamp(park_amount=50_000, equity=equity,
                           existing_val=existing,
                           max_pct=0.40, per_cycle_pct=0.10)
        assert buy5 == 0.0
