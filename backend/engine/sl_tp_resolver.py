"""Per-strategy SL/TP resolver — shared between live engine and backtest.

Phase 2 (#53) extraction: the live EvaluationLoop respects
`strategies.<name>.stop_loss.type` / `take_profit.type` config (PR fixing
펄어비스 SL miss, 2026-04-09). The backtest engine in
``backtest/full_pipeline.py`` did NOT — it used flat default_stop_loss_pct
or ATR-dynamic only, ignoring the per-strategy yaml block.

That gap meant per-strategy SL changes could not be validated in
backtest (every variant returned identical metrics). Phase 2 fixes that
by extracting the parser to this module so both call sites use the same
yaml-driven logic.

Pure functions only (no I/O, no logging side-effects). Tests in
``backend/tests/test_engine/test_sl_tp_resolver.py``.
"""

from __future__ import annotations

from typing import Any

try:
    import pandas as pd  # for type hint; runtime tolerates absence
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


# Clamp bounds — Sanitize the per-strategy SL to a sane operational range
# so a misconfigured yaml (0% or 50%) can't produce an absurd position.
SL_MIN = 0.02
SL_MAX = 0.20
TP_MIN = 0.04
TP_MAX = 0.50


def resolve_strategy_sl_pct(
    sl_cfg: dict | None,
    price: float,
    atr_val: float | None,
    df: "pd.DataFrame | None",
) -> float | None:
    """Resolve stop-loss percent from yaml stop_loss config.

    Returns the SL as a positive fraction (0.05 → 5%) or None when the
    config doesn't specify a usable type/value — callers fall back to
    risk_manager dynamic ATR or flat defaults.

    Supported `type` values:
        fixed_pct  — uses `max_pct` directly (e.g. 0.05 → 5%)
        atr        — `atr_multiplier` × ATR / price
        supertrend — (price - supertrend_line) / price from df, with ATR
                     fallback if the line column is missing or above price.
                     Honours `max_pct` as a hard floor: when set, the
                     returned SL is MIN(line distance, max_pct). Plugs the
                     position_cleanup leak where line never breaks but the
                     position bleeds to −5% and gets crystallized
                     (P1, 2026-05-11).

    SL is clamped to [SL_MIN, SL_MAX] (2% .. 20%) before return.
    """
    if not isinstance(sl_cfg, dict):
        return None
    sl_type = sl_cfg.get("type")

    sl_pct: float | None = None

    if sl_type == "fixed_pct":
        max_pct = sl_cfg.get("max_pct")
        if isinstance(max_pct, (int, float)) and max_pct > 0:
            sl_pct = float(max_pct)

    elif sl_type == "atr":
        mult = sl_cfg.get("atr_multiplier", 2.0)
        if atr_val and atr_val > 0 and price > 0:
            sl_pct = float(mult) * float(atr_val) / float(price)

    elif sl_type == "supertrend":
        line_value = _find_supertrend_line(df, price)
        if line_value is not None:
            sl_pct = (price - line_value) / price
        elif atr_val and atr_val > 0 and price > 0:
            sl_pct = 2.0 * float(atr_val) / float(price)
        # P1: optional hard floor — take MIN(line distance, max_pct).
        max_pct = sl_cfg.get("max_pct")
        if isinstance(max_pct, (int, float)) and max_pct > 0:
            if sl_pct is None or float(max_pct) < sl_pct:
                sl_pct = float(max_pct)

    if sl_pct is not None:
        sl_pct = max(SL_MIN, min(sl_pct, SL_MAX))
    return sl_pct


def resolve_strategy_tp_pct(
    tp_cfg: dict | None,
    sl_pct: float,
) -> float:
    """Resolve take-profit percent from yaml take_profit config.

    Always returns a value: defaults to 2x SL (1:2 RR) if config doesn't
    specify. Clamped to [TP_MIN, TP_MAX].
    """
    tp_pct: float | None = None
    if isinstance(tp_cfg, dict):
        tp_type = tp_cfg.get("type")
        if tp_type == "fixed_pct":
            mp = tp_cfg.get("max_pct")
            if isinstance(mp, (int, float)) and mp > 0:
                tp_pct = float(mp)
        elif tp_type == "ratio":
            ratio = tp_cfg.get("risk_multiple", 2.0)
            tp_pct = float(ratio) * sl_pct

    if tp_pct is None:
        tp_pct = 2.0 * sl_pct  # 1:2 RR default

    return max(TP_MIN, min(tp_pct, TP_MAX))


def resolve_strategy_trailing(
    trailing_cfg: dict | None,
) -> tuple[float, float] | None:
    """Resolve trailing-stop (activation, trail) percents from yaml.

    Returns (activation_pct, trail_pct) when the strategy yaml enables
    trailing and provides both values, or None to fall back to the
    engine-level defaults. Used by both live position_tracker and the
    backtest's risk-exits loop.
    """
    if not isinstance(trailing_cfg, dict):
        return None
    if not trailing_cfg.get("enabled", False):
        return None
    act = trailing_cfg.get("activation_pct")
    tr = trailing_cfg.get("trail_pct")
    if not isinstance(act, (int, float)) or not isinstance(tr, (int, float)):
        return None
    if act <= 0 or tr <= 0:
        return None
    return float(act), float(tr)


def _find_supertrend_line(
    df: "pd.DataFrame | None",
    price: float,
) -> float | None:
    """Locate the supertrend long-line column in df and return the latest
    value if it is below current price (uptrend). Returns None otherwise.
    """
    if df is None:
        return None
    for col in (
        "supertrend_long", "supertrend",
        "SUPERTl_7_2.0", "SUPERTl_10_3.0", "SUPERTl_14_3.0",
    ):
        if col in df.columns:
            try:
                v = float(df[col].iloc[-1])
                if v > 0 and v < price:
                    return v
            except (ValueError, TypeError):
                continue
    return None
