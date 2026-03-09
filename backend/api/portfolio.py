"""Portfolio API endpoints."""

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_market_data
from data.market_data_service import MarketDataService

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _get_market_data(request: Request, market: str = "US") -> MarketDataService:
    """Get market data service for the specified market."""
    if market == "KR":
        return getattr(request.app.state, "kr_market_data", None)
    return getattr(request.app.state, "market_data", None)


@router.get("/summary")
async def portfolio_summary(request: Request, market: str = "US"):
    """Get portfolio summary: balance + positions + PnL."""
    md = _get_market_data(request, market)
    if not md:
        return {"error": f"Market {market} not configured"}

    balance = await md.get_balance()
    positions = await md.get_positions()

    total_position_value = sum(p.current_price * p.quantity for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)

    result = {
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

    # When viewing US market, also include KRW balance for reference
    if market == "US":
        kr_md = getattr(request.app.state, "kr_market_data", None)
        if kr_md:
            try:
                kr_balance = await kr_md.get_balance()
                result["krw_balance"] = {
                    "total": kr_balance.total,
                    "available": kr_balance.available,
                }
            except Exception:
                pass

    return result


@router.get("/positions")
async def list_positions(request: Request, market: str = "US"):
    """List all current positions."""
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
