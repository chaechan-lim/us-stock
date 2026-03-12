"""Portfolio API endpoints."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Request

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


@router.get("/summary")
async def portfolio_summary(request: Request, market: str = "ALL"):
    """Get portfolio summary: balance + positions + PnL.

    market=ALL returns unified view with total_equity in KRW (USD converted).
    """
    if market == "ALL":
        return await _combined_summary(request)

    md = _get_market_data(request, market)
    if not md:
        return {"error": f"Market {market} not configured"}

    balance = await md.get_balance()
    positions = await md.get_positions()

    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)

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
        except Exception:
            pass
    if us_md:
        try:
            us_balance = await us_md.get_balance()
            us_positions = await us_md.get_positions()
        except Exception:
            pass

    krw_total = kr_balance.total if kr_balance else 0
    krw_available = kr_balance.available if kr_balance else 0
    usd_total = us_balance.total if us_balance else 0
    usd_available = us_balance.available if us_balance else 0

    # Fetch exchange rate from KIS adapter
    adapter = getattr(request.app.state, "adapter", None)
    if adapter:
        # Try live exchange rate first, fall back to cached rate from balance fetch
        try:
            rate = await adapter._fetch_exchange_rate()
            if rate > 0:
                _cached_usd_krw = rate
        except Exception:
            pass
        if _cached_usd_krw <= 0:
            _cached_usd_krw = getattr(adapter, "_last_exchange_rate", 1450.0)

    # Total equity: avoid double-counting the shared deposit (통합증거금)
    # US present-balance total already includes KRW deposit + US positions.
    # KR total also includes the same KRW deposit + KR positions.
    # So only add KR's position value (total - available = invested portion).
    usd_in_krw = usd_total * _cached_usd_krw
    kr_position_value = max(0, krw_total - krw_available) if krw_total else 0
    total_equity = usd_in_krw + kr_position_value

    all_positions = kr_positions + us_positions
    total_unrealized_pnl_krw = sum(p.unrealized_pnl for p in kr_positions)
    total_unrealized_pnl_usd = sum(p.unrealized_pnl for p in us_positions)

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
        "total_equity": total_equity,
    }


@router.get("/positions")
async def list_positions(request: Request, market: str = "ALL"):
    """List all current positions. market=ALL returns both US and KR."""
    if market == "ALL":
        results = []
        for m in ("US", "KR"):
            md = _get_market_data(request, m)
            if not md:
                continue
            try:
                positions = await md.get_positions()
                results.extend(await _enrich_positions(positions, m, request))
            except Exception:
                continue
        return results

    md = _get_market_data(request, market)
    if not md:
        return []

    positions = await md.get_positions()
    return await _enrich_positions(positions, market, request)


async def _enrich_positions(positions, market: str, request: Request) -> list[dict]:
    """Build position dicts with names and SL/TP info from position tracker."""
    # Resolve missing names in background
    unknown = [p.symbol for p in positions if not get_name(p.symbol, market)]
    if unknown:
        try:
            await resolve_names(unknown, market)
        except Exception:
            pass

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
    """Get cumulative returns: daily, weekly, monthly (from equity snapshots)."""
    from core.models import PortfolioSnapshot
    from sqlalchemy import select

    if not _session_factory:
        return {"daily": None, "weekly": None, "monthly": None}

    now = datetime.utcnow()
    periods = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    # Get the latest snapshot per market as "current" equity
    async with _session_factory() as session:
        result = {}
        for period_name, since in periods.items():
            # Get oldest snapshot after `since` for each market
            us_old = await _get_oldest_snapshot(session, since, "US")
            kr_old = await _get_oldest_snapshot(session, since, "KR")
            us_new = await _get_latest_snapshot(session, "US")
            kr_new = await _get_latest_snapshot(session, "KR")

            old_equity = 0.0
            new_equity = 0.0

            if us_old and us_new:
                old_equity += us_old.total_value_usd * _cached_usd_krw
                new_equity += us_new.total_value_usd * _cached_usd_krw
            if kr_old and kr_new:
                old_equity += kr_old.total_value_usd
                new_equity += kr_new.total_value_usd

            if old_equity > 0:
                change = new_equity - old_equity
                pct = (change / old_equity) * 100
                result[period_name] = {
                    "change": round(change, 0),
                    "pct": round(pct, 2),
                    "base_equity": round(old_equity, 0),
                }
            else:
                result[period_name] = None

        return result


async def _get_oldest_snapshot(session, since, market: str):
    """Get the oldest snapshot after a given time for a market."""
    from core.models import PortfolioSnapshot
    from sqlalchemy import select

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.recorded_at >= since)
        .where(PortfolioSnapshot.market == market)
        .order_by(PortfolioSnapshot.recorded_at.asc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_latest_snapshot(session, market: str):
    """Get the most recent snapshot for a market."""
    from core.models import PortfolioSnapshot
    from sqlalchemy import select, desc

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.market == market)
        .order_by(desc(PortfolioSnapshot.recorded_at))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


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

            sells = [
                o for o in all_orders
                if o.side == "SELL" and o.status == "filled" and o.pnl is not None
            ]

            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)

            def _calc(trades):
                wins = [t for t in trades if t.pnl > 0]
                losses = [t for t in trades if t.pnl <= 0]
                total_pnl = sum(t.pnl for t in trades)
                return {
                    "trades": len(trades),
                    "wins": len(wins),
                    "losses": len(losses),
                    "pnl": round(total_pnl, 2),
                    "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
                }

            today_sells = [s for s in sells if s.filled_at and s.filled_at >= today_start]
            week_sells = [s for s in sells if s.filled_at and s.filled_at >= week_start]
            month_sells = [s for s in sells if s.filled_at and s.filled_at >= month_start]

            return {
                "today": _calc(today_sells),
                "week": _calc(week_sells),
                "month": _calc(month_sells),
                "all_time": _calc(sells),
                "total_buys": sum(1 for o in all_orders if o.side == "BUY" and o.status == "filled"),
                "total_sells": len(sells),
            }
    except Exception as e:
        logger.warning("Trade summary failed: %s", e)
        return _empty_summary()


def _empty_summary():
    empty = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0, "win_rate": 0}
    return {
        "today": empty,
        "week": empty,
        "month": empty,
        "all_time": empty,
        "total_buys": 0,
        "total_sells": 0,
    }


def _get_market_data(request: Request, market: str = "US"):
    """Get market data service for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_market_data", None)
    return getattr(request.app.state, "market_data", None)


def _get_position_tracker(request: Request, market: str = "US"):
    """Get position tracker for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_position_tracker", None)
    return getattr(request.app.state, "position_tracker", None)
