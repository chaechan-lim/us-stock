"""Portfolio API endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/summary")
async def portfolio_summary(request: Request, market: str = "ALL"):
    """Get portfolio summary: balance + positions + PnL.

    market=ALL returns unified view (KRW primary + USD positions).
    market=US or market=KR returns single-market view.
    """
    if market == "ALL":
        return await _combined_summary(request)

    md = _get_market_data(request, market)
    if not md:
        return {"error": f"Market {market} not configured"}

    balance = await md.get_balance()
    positions = await md.get_positions()

    total_position_value = sum(p.current_price * p.quantity for p in positions)
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
        "total_position_value": total_position_value,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_equity": balance.total,
    }


async def _combined_summary(request: Request) -> dict:
    """Build unified summary from both US and KR adapters."""
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

    # KRW is the base currency (single account)
    krw_total = kr_balance.total if kr_balance else 0
    krw_available = kr_balance.available if kr_balance else 0
    usd_total = us_balance.total if us_balance else 0
    usd_available = us_balance.available if us_balance else 0

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
        "positions_count": len(all_positions),
        "total_unrealized_pnl": total_unrealized_pnl_krw,
        "total_unrealized_pnl_usd": total_unrealized_pnl_usd,
        "total_equity": krw_total,
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
                for p in positions:
                    results.append({
                        "symbol": p.symbol,
                        "exchange": p.exchange,
                        "quantity": p.quantity,
                        "avg_price": p.avg_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                        "market": m,
                    })
            except Exception:
                continue
        return results

    md = _get_market_data(request, market)
    if not md:
        return []

    positions = await md.get_positions()
    return [
        {
            "symbol": p.symbol,
            "exchange": p.exchange,
            "quantity": p.quantity,
            "avg_price": p.avg_price,
            "current_price": p.current_price,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pnl_pct": p.unrealized_pnl_pct,
            "market": market,
        }
        for p in positions
    ]


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


def _get_market_data(request: Request, market: str = "US"):
    """Get market data service for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_market_data", None)
    return getattr(request.app.state, "market_data", None)
