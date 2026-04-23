"""Backtest regression gate — pure comparison logic, no I/O or imports.

Used by scripts/ci_backtest_gate.py (live CI) and exercised directly by
tests/test_scripts/test_ci_backtest_gate.py without having to import the
script (which mutates sys.path on import).
"""

THRESHOLDS: dict[str, float] = {
    "sharpe_drop": 0.30,
    "pf_drop": 0.20,
    "mdd_worsen_pp": 5.0,
    "trades_drop_pct": 50.0,
}


def compare(baseline: dict, current: dict, market: str) -> list[str]:
    """Return list of regression messages (empty = OK).

    Return drop is deliberately NOT gated — live vs backtest return has
    high variance and regression would be too noisy.
    """
    failures: list[str] = []
    if not baseline:
        return [f"{market}: no baseline stored (run with --update-baseline first)"]

    drop_sharpe = baseline["sharpe"] - current["sharpe"]
    if drop_sharpe > THRESHOLDS["sharpe_drop"]:
        failures.append(
            f"{market}: Sharpe {baseline['sharpe']:.2f} → {current['sharpe']:.2f} "
            f"(Δ -{drop_sharpe:.2f}, threshold -{THRESHOLDS['sharpe_drop']})"
        )

    drop_pf = baseline["pf"] - current["pf"]
    if drop_pf > THRESHOLDS["pf_drop"]:
        failures.append(
            f"{market}: PF {baseline['pf']:.2f} → {current['pf']:.2f} "
            f"(Δ -{drop_pf:.2f}, threshold -{THRESHOLDS['pf_drop']})"
        )

    # MDD worsens when absolute drawdown grows
    worsen_mdd = abs(current["mdd"]) - abs(baseline["mdd"])
    if worsen_mdd > THRESHOLDS["mdd_worsen_pp"]:
        failures.append(
            f"{market}: MDD {baseline['mdd']:.1f}% → {current['mdd']:.1f}% "
            f"(worsened +{worsen_mdd:.1f}pp, threshold +{THRESHOLDS['mdd_worsen_pp']}pp)"
        )

    # Trade count collapse detects silent strategy-disable bugs
    if baseline["trades"] > 0:
        drop_pct = (baseline["trades"] - current["trades"]) / baseline["trades"] * 100
        if drop_pct > THRESHOLDS["trades_drop_pct"]:
            failures.append(
                f"{market}: trades {baseline['trades']} → {current['trades']} "
                f"(Δ -{drop_pct:.0f}%, threshold -{THRESHOLDS['trades_drop_pct']:.0f}%)"
            )

    return failures
