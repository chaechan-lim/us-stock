"""Portfolio API endpoints."""

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request

from api.accounts import validate_account_id_or_404
from data.stock_name_service import get_name, resolve_names

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
logger = logging.getLogger(__name__)

# Cached exchange rate (refreshed via summary calls)
_cached_usd_krw: float = 1450.0  # sensible default

# Session factory (set from main.py init)
_session_factory = None


def init_portfolio(session_factory):
    """Set session factory for portfolio endpoints."""
    global _session_factory
    _session_factory = session_factory


def _get_num(obj: object, attr: str, default: float = 0.0) -> float:
    val = getattr(obj, attr, None) if obj else None
    return float(val) if val else default


def _snapshot_equity_krw(snapshot, usd_krw: float, combined: bool = False) -> float:
    """Convert a snapshot to KRW equity.

    For combined ALL-market returns we align with the live KIS-style total:
    KR total evaluation + US position evaluation only. US cash is shared with KR
    in integrated-margin accounts, so summing US total_value again would double-count it.
    """
    if snapshot is None:
        return 0.0

    if getattr(snapshot, "market", None) == "KR":
        return float(getattr(snapshot, "total_value_usd", 0.0) or 0.0)

    rate = float(getattr(snapshot, "usd_krw_rate", None) or usd_krw or 0.0)
    total_usd = float(getattr(snapshot, "total_value_usd", 0.0) or 0.0)
    if combined:
        cash_usd = float(getattr(snapshot, "cash_usd", 0.0) or 0.0)
        total_usd = max(0.0, total_usd - cash_usd)
    return total_usd * rate


def _snapshot_cash_flow_krw(snapshot, usd_krw: float, combined: bool = False) -> float:
    """Convert a snapshot cash-flow to KRW.

    In combined integrated-margin mode, deposits/withdrawals should come from the KR side
    because KR total already includes the shared cash pool. US snapshot cash flows are
    derived from US total account value and can misclassify shared-cash moves.
    """
    if snapshot is None:
        return 0.0

    cash_flow = float(getattr(snapshot, "cash_flow", 0.0) or 0.0)
    if cash_flow == 0.0:
        return 0.0

    if getattr(snapshot, "market", None) == "US":
        if combined:
            return 0.0
        rate = float(getattr(snapshot, "usd_krw_rate", None) or usd_krw or 0.0)
        return cash_flow * rate
    return cash_flow


@router.get("/summary")
async def portfolio_summary(
    request: Request,
    market: str = "ALL",
    account_id: Optional[str] = Depends(validate_account_id_or_404),
) -> dict:
    """Get portfolio summary: balance + positions + PnL.

    market=ALL returns unified view with total_equity in KRW (USD converted).
    account_id is validated against configured accounts; unknown IDs return 404.
    account_id omitted → all-accounts summary (current single-adapter behaviour).
    NOTE: per-account data isolation requires multi-adapter support (future work);
    until then account_id is accepted for validation only — the returned data
    reflects all accounts regardless of which account_id is provided.
    """
    if account_id is not None:
        logger.warning(
            "account_id=%s provided to /portfolio/summary but per-account "
            "filtering is not yet implemented; returning all-accounts view.",
            account_id,
        )
    if market == "ALL":
        return await _combined_summary(request)

    md = get_market_data(request, market)
    if not md:
        return {"error": f"Market {market} not configured"}

    balance = await md.get_balance()
    positions = await md.get_positions()

    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    total_cost = sum(p.avg_price * p.quantity for p in positions if p.avg_price > 0)
    total_unrealized_pnl_pct = (
        round(total_unrealized_pnl / total_cost * 100, 2) if total_cost > 0 else 0.0
    )

    return {
        "market": market,
        "balance": {
            "currency": balance.currency,
            "total": balance.total,
            "available": balance.available,
            "locked": balance.locked,
        },
        "positions_count": len(positions),
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "total_equity": balance.total,
    }


async def _combined_summary(request: Request) -> dict:
    """Build unified summary from both US and KR adapters."""
    global _cached_usd_krw

    us_md = getattr(request.app.state, "market_data", None)
    kr_md = getattr(request.app.state, "kr_market_data", None)

    kr_balance = None
    us_balance = None
    kr_positions = []
    us_positions = []

    if kr_md:
        try:
            kr_balance = await kr_md.get_balance()
            kr_positions = await kr_md.get_positions()
        except Exception as e:
            logger.warning("KR balance/positions fetch failed: %s", e)
    if us_md:
        try:
            us_balance = await us_md.get_balance()
            us_positions = await us_md.get_positions()
        except Exception as e:
            logger.warning("US balance/positions fetch failed: %s", e)

    krw_total = kr_balance.total if kr_balance else 0
    krw_available = kr_balance.available if kr_balance else 0
    usd_total = us_balance.total if us_balance else 0
    usd_available = us_balance.available if us_balance else 0

    # Fetch exchange rate — prefer MarketDataService cache (5-min TTL),
    # fall back to adapter's cached rate from last balance fetch.
    adapter = getattr(request.app.state, "adapter", None)
    if us_md:
        try:
            rate = await us_md.get_exchange_rate()
            if rate > 0:
                _cached_usd_krw = rate
        except Exception as e:
            logger.warning("Exchange rate fetch failed: %s", e)
    elif adapter:
        _cached_usd_krw = getattr(adapter, "_last_exchange_rate", _cached_usd_krw)
    if _cached_usd_krw <= 0:
        _cached_usd_krw = 1450.0

    # Total equity — combine KR and US totals, avoiding deposit double-count.
    # In 통합증거금 accounts, the KRW deposit (예수금) appears in BOTH:
    #   - KR domestic API (VTTC8434R): tot_evlu_amt includes deposit + KR stocks + overseas
    #   - US overseas API (CTRP6504R): tot_asst_amt includes deposit + US stocks
    # So: total = kr_tot_evlu_amt + us_tot_asst_amt - shared_deposit
    adapter = getattr(request.app.state, "adapter", None)
    kr_adapter = getattr(request.app.state, "kr_adapter", None)

    us_tot_asst = _get_num(adapter, "_tot_asst_krw")
    us_tot_dncl = _get_num(adapter, "_tot_dncl_krw")
    us_position_value_krw = _get_num(adapter, "_us_position_value_krw")
    withdrawable_total_krw = _get_num(adapter, "_withdrawable_total_krw")
    kr_tot_evlu = _get_num(kr_adapter, "_tot_evlu_amt")
    kr_stock_eval_krw = _get_num(kr_adapter, "_scts_evlu_amt")
    kr_deposit_krw = _get_num(kr_adapter, "_dnca_tot_amt")
    # Also read full_us_usd — still needed for available_cash (STOCK-42) below.
    full_us_usd = _get_num(adapter, "_full_account_usd")

    equity_formula = "krw_total + usd_total * exchange_rate"
    cash_formula = "total_equity - total_position_value"
    kr_total_krw = krw_total
    us_total_krw = us_position_value_krw if us_position_value_krw > 0 else (
        us_tot_asst if us_tot_asst > 0 else usd_total * _cached_usd_krw
    )
    shared_deposit_krw = us_tot_dncl

    if kr_stock_eval_krw > 0 and usd_total > 0:
        # 2026-04-16: 통합증거금 총자산 = 국내주식평가 + US계좌총액(USD) × 환율.
        # US balance.total (CTRP6504R)은 예수금 + 해외주식평가를 이미 포함.
        # 국내주식만 별도 더하면 KIS 앱 총자산과 1% 이내 일치.
        # 이전 공식들이 실패한 이유: kr_deposit/kr_tot_evlu/us_position 조합은
        # 통합증거금의 예수금 공유 구조에서 이중계산 또는 누락 발생.
        equity_formula = "kr_stock_eval + us_total_usd * rate"
        total_equity = kr_stock_eval_krw + usd_total * _cached_usd_krw
    elif krw_total > 0 and us_position_value_krw > 0:
        equity_formula = "kr_total_krw + us_position_value_krw"
        total_equity = krw_total + us_position_value_krw
    elif us_tot_asst > 0:
        # US-only or non-통합증거금: use US total + KR total separately
        equity_formula = "krw_total + us_tot_asst"
        total_equity = krw_total + us_tot_asst
    else:
        # Fallback: separate KR total + US total at market rate.
        total_equity = krw_total + usd_total * _cached_usd_krw

    # Available cash = total_equity - total_position_value.
    # Previous STOCK-42 approach used US adapter's _full_available_usd * rate,
    # but this overstates available cash in 통합증거금 because:
    #   - US frcr_ord_psbl_amt1 assumes ALL KRW can convert to USD
    #   - KR positions already occupy part of that shared deposit pool
    # Deriving from equity - positions avoids this cross-market mismatch.
    kr_position_value = sum(
        p.current_price * p.quantity for p in kr_positions if p.current_price > 0
    )
    us_position_value = sum(
        p.current_price * p.quantity for p in us_positions if p.current_price > 0
    )
    total_position_value = kr_position_value + us_position_value * _cached_usd_krw
    if kr_tot_evlu > 0 and us_position_value_krw > 0:
        # 통합증거금에서는 "가용현금"을 자체 산식이 아니라 KIS 주문가능예수금에 맞춘다.
        cash_formula = "kis_orderable_cash"
        available_cash = withdrawable_total_krw or krw_available
    else:
        available_cash = max(0.0, total_equity - total_position_value)

    all_positions = kr_positions + us_positions
    total_unrealized_pnl_krw = sum(p.unrealized_pnl for p in kr_positions)
    total_unrealized_pnl_usd = sum(p.unrealized_pnl for p in us_positions)

    # Calculate weighted-average unrealized PnL % (by cost basis, in KRW)
    total_cost_krw = sum(p.avg_price * p.quantity for p in kr_positions if p.avg_price > 0)
    total_cost_usd = sum(p.avg_price * p.quantity for p in us_positions if p.avg_price > 0)
    total_cost_combined = total_cost_krw + total_cost_usd * _cached_usd_krw
    total_unrealized_pnl_combined = (
        total_unrealized_pnl_krw + total_unrealized_pnl_usd * _cached_usd_krw
    )
    total_unrealized_pnl_pct = (
        round(total_unrealized_pnl_combined / total_cost_combined * 100, 2)
        if total_cost_combined > 0
        else 0.0
    )

    return {
        "market": "ALL",
        "balance": {
            "currency": "KRW",
            "total": krw_total,
            "available": krw_available,
        },
        "usd_balance": {
            "total": usd_total,
            "available": usd_available,
        },
        "exchange_rate": _cached_usd_krw,
        "positions_count": len(all_positions),
        "total_unrealized_pnl": total_unrealized_pnl_krw,
        "total_unrealized_pnl_usd": total_unrealized_pnl_usd,
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "total_equity": total_equity,
        "available_cash": available_cash,
        "equity_breakdown": {
            "formula": equity_formula,
            "kr_total_krw": kr_total_krw,
            "us_total_krw": us_total_krw,
            "shared_deposit_krw": shared_deposit_krw,
            "kr_deposit_krw": kr_deposit_krw,
            "kr_stock_eval_krw": kr_stock_eval_krw,
            "kr_tot_evlu_krw": kr_tot_evlu,
        },
        "cash_breakdown": {
            "formula": cash_formula,
            "combined_cash_krw": available_cash,
            "kr_orderable_cash_krw": krw_available,
            "us_orderable_cash_usd": usd_available,
            "us_orderable_cash_krw": usd_available * _cached_usd_krw,
            "withdrawable_cash_krw": withdrawable_total_krw,
            "total_position_value_krw": total_position_value,
        },
    }


@router.get("/positions")
async def list_positions(
    request: Request,
    market: str = "ALL",
    account_id: Optional[str] = Depends(validate_account_id_or_404),
):
    """List all current positions. market=ALL returns both US and KR.

    account_id is validated against configured accounts; unknown IDs return 404.
    NOTE: per-account data isolation requires multi-adapter support (future work);
    until then account_id is accepted for validation only — the returned data
    reflects all accounts regardless of which account_id is provided.
    """
    if account_id is not None:
        logger.warning(
            "account_id=%s provided to /portfolio/positions but per-account "
            "filtering is not yet implemented; returning all-accounts view.",
            account_id,
        )
    if market == "ALL":
        results = []
        for m in ("US", "KR"):
            md = get_market_data(request, m)
            if not md:
                continue
            try:
                positions = await md.get_positions()
                results.extend(await enrich_positions(positions, m, request))
            except Exception as e:
                logger.warning("Position fetch failed for %s market: %s", m, e)
                continue
        return results

    md = get_market_data(request, market)
    if not md:
        return []

    positions = await md.get_positions()
    return await enrich_positions(positions, market, request)


async def enrich_positions(positions, market: str, request: Request) -> list[dict]:
    """Build position dicts with names and SL/TP info from position tracker."""
    # Resolve missing names in background
    unknown = [p.symbol for p in positions if not get_name(p.symbol, market)]
    if unknown:
        try:
            await resolve_names(unknown, market)
        except Exception as e:
            logger.warning("Stock name resolution failed: %s", e)

    # Get tracked position info (SL/TP/trailing stop)
    tracker = _get_position_tracker(request, market)
    tracked_map = {}
    if tracker:
        for t in tracker.get_status():
            tracked_map[t["symbol"]] = t

    results = []
    for p in positions:
        entry = {
            "symbol": p.symbol,
            "name": get_name(p.symbol, market) or "",
            "exchange": p.exchange,
            "quantity": p.quantity,
            "avg_price": p.avg_price,
            "current_price": p.current_price,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pnl_pct": p.unrealized_pnl_pct,
            "market": market,
        }
        # Add SL/TP tracking info if available
        tracked = tracked_map.get(p.symbol)
        if tracked:
            entry["stop_loss_pct"] = tracked.get("stop_loss_pct")
            entry["take_profit_pct"] = tracked.get("take_profit_pct")
            entry["highest_price"] = tracked.get("highest_price")
            entry["trailing_active"] = tracked.get("trailing_active", False)
        results.append(entry)
    return results


@router.get("/returns")
async def portfolio_returns(request: Request):
    """Get realized P&L by period (1d/1w/1m).

    2026-04-17: Switched from snapshot-based equity diff to realized P&L
    aggregation. Snapshot-based diffs were unreliable because the equity
    formula has changed multiple times and snapshot timing creates phantom
    swings. Realized P&L from the trades table is deterministic.

    Returns combined KRW equivalent. USD trades are converted at the
    current cached USD/KRW rate (close enough for display).
    """
    if not _session_factory:
        return {"daily": None, "weekly": None, "monthly": None}

    from sqlalchemy import select, func
    from core.models import Order

    now = datetime.utcnow()
    periods = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    rate = _cached_usd_krw if _cached_usd_krw > 0 else 1450.0

    async with _session_factory() as session:
        result = {}
        for period_name, since in periods.items():
            stmt = (
                select(Order.market, func.sum(Order.pnl).label("total_pnl"))
                .where(
                    Order.side == "SELL",
                    Order.status == "filled",
                    Order.pnl.isnot(None),
                    Order.is_paper == False,  # noqa: E712
                    Order.created_at >= since,
                )
                .group_by(Order.market)
            )
            rows = (await session.execute(stmt)).all()
            us_pnl = 0.0
            kr_pnl = 0.0
            for market, total_pnl in rows:
                if market == "US":
                    us_pnl = float(total_pnl or 0.0)
                else:
                    kr_pnl = float(total_pnl or 0.0)

            change_krw = kr_pnl + us_pnl * rate
            result[period_name] = {
                "change": round(change_krw, 0),
                "pct": 0.0,  # not meaningful for realized-only
                "simple_pct": 0.0,
                "base_equity": 0,
                "has_cash_flows": False,
                "realized_kr": round(kr_pnl, 0),
                "realized_us": round(us_pnl, 2),
            }

        return result


async def _get_oldest_snapshot(session, since, market: str):
    """Get the oldest snapshot after a given time for a market."""
    from sqlalchemy import select

    from core.models import PortfolioSnapshot

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.recorded_at >= since)
        .where(PortfolioSnapshot.market == market)
        .order_by(PortfolioSnapshot.recorded_at.asc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_latest_snapshot_before_or_at(session, when: datetime, market: str):
    """Get the most recent snapshot at or before a given time for a market."""
    from sqlalchemy import desc, select

    from core.models import PortfolioSnapshot

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.recorded_at <= when)
        .where(PortfolioSnapshot.market == market)
        .order_by(desc(PortfolioSnapshot.recorded_at))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_latest_snapshot(session, market: str):
    """Get the most recent snapshot for a market."""
    from sqlalchemy import desc, select

    from core.models import PortfolioSnapshot

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.market == market)
        .order_by(desc(PortfolioSnapshot.recorded_at))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_snapshots_in_range(session, since: datetime, market: str) -> list:
    """Get all snapshots for a market in a time range, ordered by time."""
    from sqlalchemy import select

    from core.models import PortfolioSnapshot

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.recorded_at >= since)
        .where(PortfolioSnapshot.market == market)
        .order_by(PortfolioSnapshot.recorded_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _has_cash_flows(snapshots: list) -> bool:
    """Check if any snapshot in the list has a non-zero cash_flow."""
    return any((getattr(s, "cash_flow", 0.0) or 0.0) != 0.0 for s in snapshots)


async def _has_cash_flows_db(session, since: datetime, market: str) -> bool:
    """Lightweight DB check: does any snapshot in range have non-zero cash_flow?

    Avoids fetching full snapshot rows in the common case (no deposits/withdrawals).
    """
    from sqlalchemy import func, select

    from core.models import PortfolioSnapshot

    stmt = (
        select(func.count())
        .select_from(PortfolioSnapshot)
        .where(
            PortfolioSnapshot.recorded_at >= since,
            PortfolioSnapshot.market == market,
            PortfolioSnapshot.cash_flow != 0.0,
        )
    )
    result = await session.execute(stmt)
    count = result.scalar() or 0
    return count > 0


def _prepend_snapshot(seed, snapshots: list) -> list:
    """Prepend a boundary snapshot when it is not already included."""
    if seed is None:
        return snapshots
    if snapshots:
        first = snapshots[0]
        if (
            getattr(first, "id", None) == getattr(seed, "id", None)
            or getattr(first, "recorded_at", None) == getattr(seed, "recorded_at", None)
        ):
            return snapshots
    return [seed, *snapshots]


def _find_base_index(timeline: list[tuple[datetime, float, float]], since: datetime) -> int | None:
    """Pick the last combined point at/before `since`, else the first after it."""
    if not timeline:
        return None

    base_idx = None
    for idx, (ts, _, _) in enumerate(timeline):
        if ts <= since:
            base_idx = idx
            continue
        break
    return base_idx if base_idx is not None else 0


def _append_live_equity(
    timeline: list[tuple[datetime, float, float]],
    at: datetime,
    equity_krw: float,
) -> list[tuple[datetime, float, float]]:
    """Append the current live total_equity to a snapshot timeline."""
    if equity_krw <= 0:
        return timeline

    if not timeline:
        return [(at, equity_krw, 0.0)]

    last_ts, last_equity, _ = timeline[-1]
    if at <= last_ts and abs(last_equity - equity_krw) < 0.5:
        return timeline
    if abs(last_equity - equity_krw) < 0.5:
        return timeline
    return [*timeline, (at, equity_krw, 0.0)]


def _build_equity_timeline(
    us_snapshots: list, kr_snapshots: list, usd_krw: float, combined: bool = False
) -> list[tuple[datetime, float, float]]:
    """Build a combined equity timeline using carry-forward for mismatched timestamps.

    In production, US and KR save_snapshot() each call datetime.utcnow()
    independently, so their recorded_at values differ by at least milliseconds.
    Exact-timestamp aggregation would compare single-market equity values against
    each other, producing nonsensical sub-period returns.

    Instead, carry forward the last-known equity from each market: at each snapshot
    event, total portfolio equity = this_market_equity + last_known_other_market_equity.
    Entries are skipped until at least one snapshot from every active market has been
    seen, ensuring the initial equity denominator reflects the full portfolio.

    Returns list of (timestamp, total_equity_krw, cash_flow_krw) sorted by time.
    """
    events: list[tuple[datetime, str, float, float]] = []
    integrated_combined = combined and bool(us_snapshots) and bool(kr_snapshots)

    for s in us_snapshots:
        equity = _snapshot_equity_krw(s, usd_krw, combined=integrated_combined)
        cf = _snapshot_cash_flow_krw(s, usd_krw, combined=integrated_combined)
        events.append((s.recorded_at, "US", equity, cf))

    for s in kr_snapshots:
        equity = _snapshot_equity_krw(s, usd_krw, combined=integrated_combined)
        cf = _snapshot_cash_flow_krw(s, usd_krw, combined=integrated_combined)
        events.append((s.recorded_at, "KR", equity, cf))

    events.sort(key=lambda x: x[0])

    if not events:
        return []

    has_us = bool(us_snapshots)
    has_kr = bool(kr_snapshots)
    last_equity: dict[str, float | None] = {"US": None, "KR": None}
    timeline: list[tuple[datetime, float, float]] = []

    for ts, market, equity, cf in events:
        last_equity[market] = equity

        # Skip until we have an initial equity reading for every active market
        if has_us and last_equity["US"] is None:
            continue
        if has_kr and last_equity["KR"] is None:
            continue

        total_equity = (last_equity["US"] or 0.0) + (last_equity["KR"] or 0.0)
        timeline.append((ts, total_equity, cf))

    return timeline


def _calculate_twr_from_timeline(timeline: list[tuple[datetime, float, float]]) -> float:
    """Calculate TWR from a pre-built combined timeline."""
    if len(timeline) < 2:
        return 0.0

    compound = 1.0
    for idx in range(1, len(timeline)):
        prev_eq = timeline[idx - 1][1]
        curr_eq = timeline[idx][1]
        curr_cf = timeline[idx][2]

        if prev_eq <= 0:
            logger.warning(
                "[TWR] Skipping sub-period %d: prev_eq=%.2f (zero or negative equity)",
                idx,
                prev_eq,
            )
            continue

        sub_return = (curr_eq - curr_cf - prev_eq) / prev_eq
        compound *= 1.0 + sub_return

    return (compound - 1.0) * 100.0


def _calculate_twr(
    us_snapshots: list,
    kr_snapshots: list,
    usd_krw: float,
    combined: bool = False,
) -> float:
    """Calculate TWR (Time-Weighted Return) across a period.

    TWR splits the period at each cash flow event and chains sub-period returns:
      TWR = prod(1 + Ri) - 1, where Ri = (end_equity - start_equity - cf) / start_equity

    Uses carry-forward per-market equity (via _build_equity_timeline) so that
    near-simultaneous US and KR snapshots with slightly different timestamps are
    aggregated correctly rather than being compared against each other directly.
    """
    timeline = _build_equity_timeline(us_snapshots, kr_snapshots, usd_krw, combined=combined)
    return _calculate_twr_from_timeline(timeline)


@router.delete("/snapshots")
async def delete_snapshots(request: Request, ids: str = "", market: str = "KR"):
    """Delete anomalous portfolio snapshots by ID.

    Admin endpoint for correcting bad data (STOCK-45).
    Pass comma-separated IDs, e.g. ?ids=196,197,198,199,200&market=KR
    """
    from fastapi.responses import JSONResponse

    if market not in ("US", "KR"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid market: {market}. Use 'US' or 'KR'."},
        )

    if not ids:
        return JSONResponse(status_code=400, content={"error": "No IDs provided"})

    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid ID format — use comma-separated integers"},
        )

    if market == "KR":
        pm = getattr(request.app.state, "kr_portfolio_manager", None)
    else:
        pm = getattr(request.app.state, "portfolio_manager", None)

    if not pm:
        return JSONResponse(
            status_code=503,
            content={"error": f"Portfolio manager not configured for {market}"},
        )

    deleted = await pm.delete_snapshots_by_ids(id_list)
    return {"deleted": deleted, "ids": id_list, "market": market}


@router.get("/equity-history")
async def equity_history(request: Request, days: int = 30, market: str = "US"):
    """Get portfolio equity history for charting."""
    if market == "KR":
        pm = getattr(request.app.state, "kr_portfolio_manager", None)
    else:
        pm = getattr(request.app.state, "portfolio_manager", None)
    if not pm:
        return []
    return await pm.get_equity_history(days=days)


@router.get("/trade-summary")
async def trade_summary_periods(request: Request, market: str | None = None):
    """Get trade P&L summary by period: today, this week, this month, all-time."""
    from api.trades import _session_factory

    if not _session_factory:
        return _empty_summary()

    try:
        from db.trade_repository import TradeRepository

        async with _session_factory() as session:
            repo = TradeRepository(session)
            all_orders = await repo.get_trade_history(limit=500)

            # Filter by market if specified
            if market:
                all_orders = [o for o in all_orders if getattr(o, "market", "US") == market]

            # STOCK-37: Include "not_found" orders that have PnL — these were
            # actually filled but KIS API couldn't find them during reconciliation
            # (date boundary / API delay). PnL existence proves the fill happened.
            sells = [
                o
                for o in all_orders
                if o.side == "SELL" and o.status in ("filled", "not_found") and o.pnl is not None
            ]

            # Use each trade's market-local timezone for period classification
            # US trades → America/New_York, KR trades → Asia/Seoul
            _market_tz = {
                "US": ZoneInfo("America/New_York"),
                "KR": ZoneInfo("Asia/Seoul"),
            }

            def _trade_market_date(s) -> date | None:
                """Get the local market date for a trade."""
                t = s.filled_at or s.created_at
                if not t:
                    return None
                mkt = getattr(s, "market", "US")
                tz = _market_tz.get(mkt, ZoneInfo("UTC"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=ZoneInfo("UTC"))
                return t.astimezone(tz).date()

            def _market_today(mkt: str) -> date:
                tz = _market_tz.get(mkt, ZoneInfo("UTC"))
                return datetime.now(tz).date()

            def _in_today(s) -> bool:
                d = _trade_market_date(s)
                if not d:
                    return False
                return d == _market_today(getattr(s, "market", "US"))

            def _in_week(s) -> bool:
                d = _trade_market_date(s)
                if not d:
                    return False
                today = _market_today(getattr(s, "market", "US"))
                week_start = today - timedelta(days=today.weekday())
                return d >= week_start

            def _in_month(s) -> bool:
                d = _trade_market_date(s)
                if not d:
                    return False
                today = _market_today(getattr(s, "market", "US"))
                return d.year == today.year and d.month == today.month

            def _calc(trades, convert_currency: bool = False):
                wins = [t for t in trades if t.pnl > 0]
                losses = [t for t in trades if t.pnl <= 0]
                if convert_currency:
                    # Convert US PnL (USD) to KRW for combined summary
                    total_pnl = sum(
                        t.pnl * _cached_usd_krw if getattr(t, "market", "US") == "US" else t.pnl
                        for t in trades
                    )
                else:
                    total_pnl = sum(t.pnl for t in trades)
                # Cost-weighted PnL %: total_pnl / total_cost_basis
                total_cost = 0.0
                for t in trades:
                    qty = getattr(t, "filled_quantity", None) or getattr(t, "quantity", 0) or 0
                    entry = getattr(t, "entry_price", None)
                    if not entry:
                        # Infer entry from pnl and pnl_pct: entry = price - pnl/qty
                        # or from pnl_pct: entry = sell_price / (1 + pnl_pct/100)
                        pct = getattr(t, "pnl_pct", None)
                        sell_price = (
                            getattr(t, "filled_price", None) or getattr(t, "price", None) or 0
                        )
                        if pct and sell_price and pct != -100:
                            entry = sell_price / (1 + pct / 100)
                    if entry and qty:
                        cost = abs(entry * qty)
                        if convert_currency and getattr(t, "market", "US") == "US":
                            cost *= _cached_usd_krw
                        total_cost += cost
                pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost > 0 else None
                return {
                    "trades": len(trades),
                    "wins": len(wins),
                    "losses": len(losses),
                    "pnl": round(total_pnl, 2),
                    "pnl_pct": pnl_pct,
                    "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
                }

            today_sells = [s for s in sells if _in_today(s)]
            week_sells = [s for s in sells if _in_week(s)]
            month_sells = [s for s in sells if _in_month(s)]

            # When no market filter, convert US PnL to KRW for combined totals
            convert = not market

            return {
                "today": _calc(today_sells, convert_currency=convert),
                "week": _calc(week_sells, convert_currency=convert),
                "month": _calc(month_sells, convert_currency=convert),
                "all_time": _calc(sells, convert_currency=convert),
                "total_buys": sum(
                    1 for o in all_orders if o.side == "BUY" and o.status in ("filled", "not_found")
                ),
                "total_sells": len(sells),
            }
    except Exception as e:
        logger.warning("Trade summary failed: %s", e)
        return _empty_summary()


def _empty_summary():
    empty = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0, "pnl_pct": None, "win_rate": 0}
    return {
        "today": empty,
        "week": empty,
        "month": empty,
        "all_time": empty,
        "total_buys": 0,
        "total_sells": 0,
    }


def get_market_data(request: Request, market: str = "US"):
    """Get market data service for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_market_data", None)
    return getattr(request.app.state, "market_data", None)


def _get_position_tracker(request: Request, market: str = "US"):
    """Get position tracker for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_position_tracker", None)
    return getattr(request.app.state, "position_tracker", None)
