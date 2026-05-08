"""Performance metrics — Equity-based, cost-aware.

User feedback 2026-05-08: Profit Factor alone is misleading. Real bot
performance must be evaluated against a hierarchy:
  1순위: Net Equity Curve (Balance + Unrealized PnL, cost-adjusted)
  2순위: MDD + DD recovery time
  3순위: Net PnL / CAGR / Sharpe / Sortino / Calmar
  4순위: Net PF / Expectancy / Avg Win/Loss

Net Equity = Wallet Balance + Unrealized PnL
           - cumulative fees - cumulative slippage
           + deposits - withdrawals

This module wraps the math; the data source (snapshots + trade history)
already exists in the DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import math


# ── Cost rate estimates (per side, applied as approximation when explicit
# commission isn't recorded on the order row) ─────────────────────────────
# KIS KR domestic: ~0.015% commission + ~0.20% transaction tax (sell only).
# Conservative average: 0.10% per side.
_KR_FEE_RATE_PER_SIDE = 0.0010
# KIS US: $0.0049/share commission + small SEC/FINRA fees (sell only).
# Per-side estimate as a fraction of notional varies; use 0.05% per side.
_US_FEE_RATE_PER_SIDE = 0.0005


@dataclass
class TradeMetrics:
    total_trades: int
    wins: int
    losses: int
    win_rate: float           # 0.0–1.0
    avg_win: float            # in trade currency
    avg_loss: float           # negative number
    gross_profit: float       # sum of positive PnL
    gross_loss: float         # sum of |negative PnL|
    gross_pf: float           # gross_profit / gross_loss (∞ if no losses)
    expectancy: float         # win_rate * avg_win - loss_rate * |avg_loss|
    estimated_fees: float     # sum across all trades (per-side × 2 = round-trip)
    estimated_slippage: float # |filled_price - intended_price| × qty across trades
    net_profit: float         # gross_profit - gross_loss - fees - slippage
    net_pf: float             # (gross_profit - costs/2) / (gross_loss + costs/2)


@dataclass
class EquityMetrics:
    start_equity: float
    end_equity: float
    net_return_pct: float        # (end - start) / start × 100
    annualized_return_pct: float # CAGR computed from equity curve length
    max_drawdown_pct: float      # peak-to-trough, 0 to negative %
    max_dd_recovery_days: int    # days from MDD trough to fresh high (0 if not recovered)
    calmar_ratio: float          # annualized_return / |max_drawdown|
    sharpe_ratio: float          # daily-return-based, annualized (252)
    sortino_ratio: float         # like sharpe but only downside vol
    exposure_pct: float          # avg fraction of equity invested over time


def _estimate_trade_costs(
    market: str,
    quantity: float,
    filled_price: float | None,
    intended_price: float | None,
) -> tuple[float, float]:
    """Return (fees, slippage) estimate for one fill, in trade currency.

    Costs are conservative estimates — KR ~0.10% per side, US ~0.05% per side.
    Slippage is |filled - intended| × qty when both are known.
    """
    if not filled_price or filled_price <= 0 or quantity <= 0:
        return 0.0, 0.0
    notional = filled_price * quantity
    rate = _KR_FEE_RATE_PER_SIDE if market == "KR" else _US_FEE_RATE_PER_SIDE
    fees = notional * rate
    if intended_price and intended_price > 0:
        slippage = abs(filled_price - intended_price) * quantity
    else:
        slippage = 0.0
    return fees, slippage


def compute_trade_metrics(trades: list[dict]) -> TradeMetrics:
    """Aggregate trade-level stats. Each `trade` is a dict with keys side,
    pnl, market, quantity, filled_price, price (intended)."""
    sells = [t for t in trades if t.get("side") == "SELL" and t.get("pnl") is not None]

    if not sells:
        return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0)

    pnls = [float(t["pnl"]) for t in sells]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    n_w = len(wins)
    n_l = len(losses)
    win_rate = n_w / n if n else 0.0
    avg_win = (sum(wins) / n_w) if n_w else 0.0
    avg_loss = (sum(losses) / n_l) if n_l else 0.0  # negative
    gross_profit = sum(wins)
    gross_loss = -sum(losses)  # positive number
    gross_pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    fees_total = 0.0
    slip_total = 0.0
    for t in trades:
        market = t.get("market", "US") or "US"
        qty = float(t.get("quantity", 0) or 0)
        fp = t.get("filled_price") or t.get("price")
        ip = t.get("price")
        f, s = _estimate_trade_costs(market, qty, float(fp or 0) or None, float(ip or 0) or None)
        fees_total += f
        slip_total += s

    net_profit = gross_profit - gross_loss - fees_total - slip_total
    half_costs = (fees_total + slip_total) / 2
    if gross_loss + half_costs > 0:
        net_pf = (gross_profit - half_costs) / (gross_loss + half_costs)
    elif gross_profit > 0:
        net_pf = float("inf")
    else:
        net_pf = 0.0

    return TradeMetrics(
        total_trades=n,
        wins=n_w,
        losses=n_l,
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        gross_pf=round(gross_pf, 3) if gross_pf != float("inf") else float("inf"),
        expectancy=round(expectancy, 2),
        estimated_fees=round(fees_total, 2),
        estimated_slippage=round(slip_total, 2),
        net_profit=round(net_profit, 2),
        net_pf=round(net_pf, 3) if net_pf != float("inf") else float("inf"),
    )


def compute_equity_metrics(equity_series: list[tuple[date, float]]) -> EquityMetrics:
    """Compute equity-curve-based metrics from daily snapshots.

    Args:
        equity_series: list of (date, equity_value) sorted ascending.
                       equity_value should be cost-adjusted Net Equity.
    """
    if len(equity_series) < 2:
        zero = EquityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        if equity_series:
            zero.start_equity = zero.end_equity = equity_series[0][1]
        return zero

    start = equity_series[0][1]
    end = equity_series[-1][1]
    days = (equity_series[-1][0] - equity_series[0][0]).days
    if days <= 0 or start <= 0:
        net_return = 0.0
        cagr = 0.0
    else:
        net_return = (end - start) / start
        cagr = ((end / start) ** (365.0 / days)) - 1 if start > 0 else 0.0

    # Max drawdown + recovery
    peak = start
    peak_idx = 0
    max_dd = 0.0
    max_dd_idx = 0
    for i, (_, v) in enumerate(equity_series):
        if v > peak:
            peak = v
            peak_idx = i
        if peak > 0:
            dd = (v - peak) / peak  # ≤ 0
            if dd < max_dd:
                max_dd = dd
                max_dd_idx = i

    # DD recovery: days from trough to first equity ≥ peak
    recovery_days = 0
    if max_dd_idx > 0:
        recovery_target = equity_series[peak_idx][1]
        for j in range(max_dd_idx + 1, len(equity_series)):
            if equity_series[j][1] >= recovery_target:
                recovery_days = (equity_series[j][0] - equity_series[max_dd_idx][0]).days
                break
        else:
            # Not yet recovered: count days from trough to series end
            recovery_days = (equity_series[-1][0] - equity_series[max_dd_idx][0]).days

    # Daily returns for Sharpe / Sortino
    returns = []
    for i in range(1, len(equity_series)):
        prev = equity_series[i - 1][1]
        cur = equity_series[i][1]
        if prev > 0:
            returns.append((cur - prev) / prev)
    if returns:
        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = math.sqrt(var)
        downside = [r for r in returns if r < 0]
        if downside:
            d_var = sum(r ** 2 for r in downside) / len(downside)
            d_std = math.sqrt(d_var)
        else:
            d_std = 0.0
        sharpe = (mu / std) * math.sqrt(252) if std > 0 else 0.0
        sortino = (mu / d_std) * math.sqrt(252) if d_std > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return EquityMetrics(
        start_equity=round(start, 2),
        end_equity=round(end, 2),
        net_return_pct=round(net_return * 100, 2),
        annualized_return_pct=round(cagr * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        max_dd_recovery_days=recovery_days,
        calmar_ratio=round(calmar, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        exposure_pct=0.0,  # populated by caller from snapshots
    )


def compute_exposure_pct(snapshots: list[dict]) -> float:
    """Average position-value / equity ratio across snapshots."""
    if not snapshots:
        return 0.0
    ratios = []
    for s in snapshots:
        eq = s.get("total_value") or s.get("equity") or 0
        cash = s.get("cash") or 0
        if eq and eq > 0:
            ratios.append(max(0.0, min(1.0, (eq - cash) / eq)))
    return round((sum(ratios) / len(ratios)) * 100, 1) if ratios else 0.0
