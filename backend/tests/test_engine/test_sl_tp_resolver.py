"""Pure-function tests for the shared SL/TP resolver (Phase 2 #53)."""

import pandas as pd
import pytest

from engine.sl_tp_resolver import (
    SL_MAX,
    SL_MIN,
    TP_MAX,
    TP_MIN,
    resolve_strategy_sl_pct,
    resolve_strategy_tp_pct,
    resolve_strategy_trailing,
)


def _df(supertrend_long: float | None = None, n: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"close": [100.0] * n})
    if supertrend_long is not None:
        df["supertrend_long"] = [supertrend_long] * n
    return df


class TestResolveSlPct:
    def test_returns_none_for_unknown_type(self):
        assert resolve_strategy_sl_pct({"type": "moonbeam"}, 100.0, 2.0, _df()) is None

    def test_returns_none_for_empty_cfg(self):
        assert resolve_strategy_sl_pct({}, 100.0, 2.0, _df()) is None

    def test_returns_none_for_none_cfg(self):
        assert resolve_strategy_sl_pct(None, 100.0, 2.0, _df()) is None

    def test_fixed_pct(self):
        assert resolve_strategy_sl_pct(
            {"type": "fixed_pct", "max_pct": 0.05}, 100.0, 2.0, _df()
        ) == pytest.approx(0.05)

    def test_atr(self):
        # 1.5 * 2.0 / 100 = 0.03
        assert resolve_strategy_sl_pct(
            {"type": "atr", "atr_multiplier": 1.5}, 100.0, 2.0, _df()
        ) == pytest.approx(0.03)

    def test_supertrend_line_present(self):
        # line 4% below → 4% SL
        assert resolve_strategy_sl_pct(
            {"type": "supertrend"}, 100.0, 2.0, _df(supertrend_long=96.0)
        ) == pytest.approx(0.04)

    def test_supertrend_atr_fallback_when_no_line(self):
        # 2x ATR / price = 0.06
        assert resolve_strategy_sl_pct(
            {"type": "supertrend"}, 100.0, 3.0, _df()
        ) == pytest.approx(0.06)

    def test_supertrend_returns_none_when_no_line_no_atr(self):
        assert resolve_strategy_sl_pct(
            {"type": "supertrend"}, 100.0, None, _df()
        ) is None


class TestSupertrendMaxPctFloor:
    """P1 hard floor — only relevant for type=supertrend."""

    def test_floor_caps_wider_line(self):
        # line is 10% below → would yield 10% SL, but max_pct=0.07 caps it.
        assert resolve_strategy_sl_pct(
            {"type": "supertrend", "max_pct": 0.07},
            100.0, 2.0, _df(supertrend_long=90.0),
        ) == pytest.approx(0.07)

    def test_line_stays_when_tighter_than_floor(self):
        # line 3% below already tighter than 7% floor → line wins.
        assert resolve_strategy_sl_pct(
            {"type": "supertrend", "max_pct": 0.07},
            100.0, 2.0, _df(supertrend_long=97.0),
        ) == pytest.approx(0.03)

    def test_floor_applies_when_no_line(self):
        # 2xATR/price = 0.10, max_pct=0.05 caps at 0.05.
        assert resolve_strategy_sl_pct(
            {"type": "supertrend", "max_pct": 0.05},
            100.0, 5.0, _df(),
        ) == pytest.approx(0.05)


class TestClamps:
    def test_sl_clamped_to_max(self):
        assert resolve_strategy_sl_pct(
            {"type": "fixed_pct", "max_pct": 0.50}, 100.0, 2.0, _df()
        ) == SL_MAX

    def test_sl_clamped_to_min(self):
        assert resolve_strategy_sl_pct(
            {"type": "fixed_pct", "max_pct": 0.005}, 100.0, 2.0, _df()
        ) == SL_MIN


class TestResolveTrailing:
    def test_none_when_disabled(self):
        assert resolve_strategy_trailing({"enabled": False, "activation_pct": 0.05, "trail_pct": 0.03}) is None

    def test_none_when_missing_values(self):
        assert resolve_strategy_trailing({"enabled": True, "activation_pct": 0.05}) is None
        assert resolve_strategy_trailing({"enabled": True, "trail_pct": 0.03}) is None
        assert resolve_strategy_trailing(None) is None
        assert resolve_strategy_trailing({}) is None

    def test_returns_tuple_when_enabled(self):
        out = resolve_strategy_trailing({
            "enabled": True, "activation_pct": 0.05, "trail_pct": 0.03,
        })
        assert out == (0.05, 0.03)

    def test_none_when_nonpositive(self):
        assert resolve_strategy_trailing({"enabled": True, "activation_pct": 0, "trail_pct": 0.03}) is None
        assert resolve_strategy_trailing({"enabled": True, "activation_pct": 0.05, "trail_pct": -0.01}) is None


class TestResolveTpPct:
    def test_default_2x_sl_when_no_cfg(self):
        assert resolve_strategy_tp_pct({}, 0.05) == pytest.approx(0.10)

    def test_default_2x_sl_when_none(self):
        assert resolve_strategy_tp_pct(None, 0.05) == pytest.approx(0.10)

    def test_fixed_pct(self):
        assert resolve_strategy_tp_pct(
            {"type": "fixed_pct", "max_pct": 0.15}, 0.05
        ) == pytest.approx(0.15)

    def test_ratio_multiplier(self):
        assert resolve_strategy_tp_pct(
            {"type": "ratio", "risk_multiple": 3.0}, 0.04
        ) == pytest.approx(0.12)

    def test_tp_clamped(self):
        # 10 × sl=0.05 = 0.50 (at TP_MAX boundary)
        assert resolve_strategy_tp_pct(
            {"type": "ratio", "risk_multiple": 20.0}, 0.05
        ) == TP_MAX
        # tiny sl → tp=2×sl = 0.04 ≥ TP_MIN
        assert resolve_strategy_tp_pct({}, 0.02) == TP_MIN
