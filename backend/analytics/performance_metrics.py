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
    win_rate: float           # 0.0–1.0  (per-SELL-event WR — partials count individually)
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
    # Round-trip view: partial SELLs (profit_taking + trailing_stop on the
    # same entry) collapsed back to single trades. Tiered trailing makes
    # the per-SELL count and WR overstate the "real" round-trip count.
    round_trips: int = 0
    round_trip_wins: int = 0
    round_trip_losses: int = 0
    round_trip_win_rate: float = 0.0
    round_trip_avg_pnl: float = 0.0


@dataclass
class EquityMetrics:
    start_equity: float
    end_equity: float
    net_return_pct: float        # (end - start) / start × 100
    annualized_return_pct: float # CAGR computed from equity curve length
    max_drawdown_pct: float      # peak-to-trough on EOD/daily series
    max_dd_recovery_days: int    # days from MDD trough to fresh high (0 if not recovered)
    calmar_ratio: float          # annualized_return / |max_drawdown|
    sharpe_ratio: float          # daily-return-based, annualized (252)
    sortino_ratio: float         # like sharpe but only downside vol
    exposure_pct: float          # avg fraction of equity invested over time
    sample_days: int = 0         # daily samples used (0 means metrics zeroed)
    sufficient_samples: bool = False  # ≥30 days needed for Sharpe/Calmar (P1-E)
    # P4: intraday MDD computed from the full sub-daily series (hourly or
    # finer). Captures peaks-and-troughs the EOD-only daily MDD misses.
    # Always ≤ max_drawdown_pct (intraday is more conservative).
    intraday_max_drawdown_pct: float = 0.0
    intraday_sample_count: int = 0  # how many sub-daily points were used


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


def _aggregate_round_trips(trades: list[dict]) -> tuple[int, int, int, float]:
    """Collapse partial SELLs (e.g. profit_taking + trailing_stop on the
    same entry) into round-trips. Walks symbol-level events in time order;
    a round-trip closes when the running position returns to 0.

    Returns (round_trips, wins, losses, total_pnl).
    """
    from collections import defaultdict

    # Group events by symbol, sort by filled_at/created_at.
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        if t.get("symbol"):
            by_sym[t["symbol"]].append(t)

    rt_count = 0
    wins = 0
    losses = 0
    total_pnl = 0.0

    for sym, events in by_sym.items():
        events.sort(key=lambda e: e.get("filled_at") or e.get("created_at") or "")
        position = 0.0
        accum_pnl = 0.0
        in_trip = False
        for e in events:
            qty = float(e.get("quantity") or 0)
            if e.get("side") == "BUY":
                if not in_trip and position == 0:
                    in_trip = True
                    accum_pnl = 0.0
                position += qty
            elif e.get("side") == "SELL":
                pnl = e.get("pnl")
                if pnl is not None:
                    try:
                        accum_pnl += float(pnl)
                    except (TypeError, ValueError):
                        pass
                position -= qty
                # Round-trip closes when we've sold down to (or past) 0.
                if position <= 0.0001 and in_trip:
                    rt_count += 1
                    total_pnl += accum_pnl
                    if accum_pnl > 0:
                        wins += 1
                    elif accum_pnl < 0:
                        losses += 1
                    # else: zero PnL — neither win nor loss
                    in_trip = False
                    position = max(0.0, position)
                    accum_pnl = 0.0
    return rt_count, wins, losses, total_pnl


def compute_trade_metrics(trades: list[dict]) -> TradeMetrics:
    """Aggregate trade-level stats. Each `trade` is a dict with keys side,
    pnl, market, quantity, filled_price, price (intended)."""
    sells = [t for t in trades if t.get("side") == "SELL" and t.get("pnl") is not None]

    if not sells:
        return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,
                            round_trips=0, round_trip_wins=0,
                            round_trip_losses=0, round_trip_win_rate=0.0,
                            round_trip_avg_pnl=0.0)

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

    rt_count, rt_wins, rt_losses, rt_total_pnl = _aggregate_round_trips(trades)
    rt_win_rate = (rt_wins / rt_count) if rt_count else 0.0
    rt_avg_pnl = (rt_total_pnl / rt_count) if rt_count else 0.0

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
        round_trips=rt_count,
        round_trip_wins=rt_wins,
        round_trip_losses=rt_losses,
        round_trip_win_rate=round(rt_win_rate, 4),
        round_trip_avg_pnl=round(rt_avg_pnl, 2),
    )


def _compute_intraday_dd(intraday_series: list[float]) -> float:
    """Peak-to-trough on a sub-daily series. Returns negative %."""
    if len(intraday_series) < 2:
        return 0.0
    peak = intraday_series[0]
    max_dd = 0.0
    for v in intraday_series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return round(max_dd * 100, 2)


def compute_equity_metrics(
    equity_series: list[tuple[date, float]],
    intraday_values: list[float] | None = None,
    cash_flows: list[float] | None = None,
) -> EquityMetrics:
    """Compute equity-curve-based metrics from daily snapshots.

    Args:
        equity_series: list of (date, equity_value) sorted ascending — one
                       point per trading day (last sample of the day).
        intraday_values: optional ordered list of equity values at higher
                       frequency. Used for intraday MDD.
        cash_flows: optional list aligned with equity_series — external
                       deposit (+) / withdrawal (-) amount that landed in
                       the equity between the prior snapshot and this one.
                       When provided, metrics are computed via Time-
                       Weighted Return (TWR), excluding the dollar impact
                       of deposits/withdrawals from the strategy's return.
                       P1-D (2026-05-14) fix — without this, a single
                       deposit doubling equity makes net_return read +100%
                       and Sharpe explode.
    """
    n = len(equity_series)
    if n < 2:
        zero = EquityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             sample_days=n, sufficient_samples=False,
                             intraday_max_drawdown_pct=0.0,
                             intraday_sample_count=0)
        if equity_series:
            zero.start_equity = zero.end_equity = equity_series[0][1]
        if intraday_values and len(intraday_values) >= 2:
            zero.intraday_max_drawdown_pct = _compute_intraday_dd(intraday_values)
            zero.intraday_sample_count = len(intraday_values)
        return zero

    start = equity_series[0][1]
    end = equity_series[-1][1]
    days = (equity_series[-1][0] - equity_series[0][0]).days

    # P1-E (2026-05-14): minimum window for annualized stats. Annualizing
    # 9 days of data via (1+r)^(365/9) explodes any non-trivial return
    # to thousands of percent (Sharpe also ×√252 → noise). Require ≥30
    # days of window before showing annualized return / Sharpe / Sortino /
    # Calmar — under that, daily-basis net_return is still shown but
    # derived ratios stay 0.
    MIN_DAYS_FOR_ANNUALIZED = 30

    # TWR path: build a synthetic equity curve where cash-flow days don't
    # produce a return. Then run MDD / Sharpe / etc on the synthetic curve.
    use_twr = (
        cash_flows is not None
        and len(cash_flows) == n
        and any(cf != 0.0 for cf in cash_flows)
    )
    if use_twr:
        synth: list[tuple[date, float]] = [(equity_series[0][0], 1.0)]
        for i in range(1, n):
            prev_eq = equity_series[i - 1][1]
            cur_eq = equity_series[i][1]
            cf = float(cash_flows[i])
            if prev_eq <= 0:
                synth.append((equity_series[i][0], synth[-1][1]))
                continue
            r = (cur_eq - cf - prev_eq) / prev_eq
            synth.append((equity_series[i][0], synth[-1][1] * (1.0 + r)))
        # Returns + total from synthetic curve.
        net_return = synth[-1][1] - 1.0
        if days >= MIN_DAYS_FOR_ANNUALIZED:
            cagr = ((1.0 + net_return) ** (365.0 / days)) - 1.0
        else:
            cagr = 0.0
        curve_for_mdd = synth
    else:
        if days <= 0 or start <= 0:
            net_return = 0.0
            cagr = 0.0
        else:
            net_return = (end - start) / start
            if days >= MIN_DAYS_FOR_ANNUALIZED and start > 0:
                cagr = ((end / start) ** (365.0 / days)) - 1
            else:
                cagr = 0.0
        curve_for_mdd = equity_series

    # Max drawdown + recovery on whichever curve we're using.
    peak = curve_for_mdd[0][1]
    peak_idx = 0
    max_dd = 0.0
    max_dd_idx = 0
    for i, (_, v) in enumerate(curve_for_mdd):
        if v > peak:
            peak = v
            peak_idx = i
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
                max_dd_idx = i

    recovery_days = 0
    if max_dd_idx > 0:
        recovery_target = curve_for_mdd[peak_idx][1]
        for j in range(max_dd_idx + 1, len(curve_for_mdd)):
            if curve_for_mdd[j][1] >= recovery_target:
                recovery_days = (curve_for_mdd[j][0] - curve_for_mdd[max_dd_idx][0]).days
                break
        else:
            recovery_days = (curve_for_mdd[-1][0] - curve_for_mdd[max_dd_idx][0]).days

    # Daily returns for Sharpe / Sortino. For TWR, exclude cash-flow effect.
    returns = []
    for i in range(1, len(equity_series)):
        prev = equity_series[i - 1][1]
        cur = equity_series[i][1]
        if prev > 0:
            if use_twr:
                cf = float(cash_flows[i])
                returns.append((cur - cf - prev) / prev)
            else:
                returns.append((cur - prev) / prev)
    # P1-E (2026-05-14): annualized ratios (×√252) need ≥30 daily returns
    # to be even loosely meaningful — under that, the √252 scaling turns
    # a few lucky days into a Sharpe of 6+. The 7-day floor stays as
    # `sufficient_samples` (raw daily-return computation), but Sharpe /
    # Sortino are zeroed below 30 to avoid misleading the operator.
    if len(returns) >= MIN_DAYS_FOR_ANNUALIZED:
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

    intraday_dd = (
        _compute_intraday_dd(intraday_values) if intraday_values else round(max_dd * 100, 2)
    )

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
        sample_days=n,
        sufficient_samples=(len(returns) >= MIN_DAYS_FOR_ANNUALIZED),
        intraday_max_drawdown_pct=intraday_dd,
        intraday_sample_count=len(intraday_values) if intraday_values else 0,
    )


_BENCHMARK_CACHE: dict[str, tuple[float, float]] = {}  # key → (return_pct, fetched_at_ts)
_BENCHMARK_TTL_SEC = 300  # 5 min


def benchmark_return_pct(
    symbol: str,
    days: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> float | None:
    """Fetch the benchmark's % return over a window.

    P1-G (2026-05-15): when ``start_date`` and ``end_date`` are provided,
    use them as the EXACT comparison window — overriding ``days``. This
    aligns the benchmark to the actual data range so alpha = our_return -
    benchmark_return is an apples-to-apples comparison. Without alignment,
    a 9-day live history compared against SPY's 30d return falsely showed
    -7% alpha when same-window comparison was +5% alpha.

    ``days`` is kept for backward compat / caching key + fallback when
    date range isn't available.

    Cached for 5 min so the dashboard's 60s refetch doesn't hammer yfinance.
    Returns None on fetch failure.
    """
    import time as _t
    cache_key = (
        f"{symbol}|{start_date.isoformat()}|{end_date.isoformat()}"
        if start_date and end_date
        else f"{symbol}|{days}"
    )
    cached = _BENCHMARK_CACHE.get(cache_key)
    if cached and (_t.time() - cached[1]) < _BENCHMARK_TTL_SEC:
        return cached[0]

    try:
        import yfinance as yf

        if start_date and end_date:
            # Fetch the exact window. Pad +/- a few days to handle
            # weekends/holidays — pick first/last close inside the range.
            pad_start = start_date - timedelta(days=5)
            pad_end = end_date + timedelta(days=2)
            hist = yf.download(
                symbol, start=pad_start, end=pad_end,
                progress=False, auto_adjust=False, threads=False,
            )
            if hist is None or hist.empty:
                return None
            if hasattr(hist.columns, "nlevels") and hist.columns.nlevels > 1:
                hist.columns = hist.columns.get_level_values(0)
            # Closes on or before start_date for first, on or before end_date for last
            import pandas as _pd
            idx = _pd.to_datetime(hist.index).date
            in_window = [(d, p) for d, p in zip(idx, hist["Close"]) if start_date <= d <= end_date]
            if len(in_window) < 2:
                return None
            first = float(in_window[0][1])
            last = float(in_window[-1][1])
        else:
            t = yf.Ticker(symbol)
            hist = t.history(period=f"{max(days + 7, 14)}d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                return None
            window = hist.tail(days + 1) if len(hist) > days + 1 else hist
            first = float(window["Close"].iloc[0])
            last = float(window["Close"].iloc[-1])

        if first <= 0:
            return None
        ret = (last - first) / first * 100
        _BENCHMARK_CACHE[cache_key] = (ret, _t.time())
        return round(ret, 2)
    except Exception:
        return None


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
