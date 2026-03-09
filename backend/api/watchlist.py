"""Watchlist API endpoints (DB-backed)."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from data.stock_name_service import resolve_names
from db.session import get_session_factory
from db.trade_repository import TradeRepository

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


class WatchlistAdd(BaseModel):
    symbol: str
    exchange: str = "NASD"
    market: str = "US"


async def _get_symbols(market: str = "US") -> list[dict]:
    """Get active watchlist from DB, with names resolved."""
    factory = get_session_factory()
    async with factory() as session:
        repo = TradeRepository(session)
        items = await repo.get_watchlist(active_only=True, market=market)

    symbols = [w.symbol for w in items]
    names = await resolve_names(symbols, market)

    return [
        {
            "symbol": w.symbol,
            "exchange": w.exchange,
            "market": w.market,
            "name": w.name or names.get(w.symbol, ""),
            "sector": w.sector,
            "source": w.source,
            "added_at": w.added_at.isoformat() if w.added_at else None,
        }
        for w in items
    ]


@router.get("/")
async def get_watchlist(market: str = Query("US")):
    """Get current watchlist."""
    items = await _get_symbols(market)
    return {
        "symbols": [w["symbol"] for w in items],
        "items": items,
    }


@router.post("/")
async def add_symbol(req: WatchlistAdd):
    """Add a symbol to watchlist."""
    symbol = req.symbol.upper()
    factory = get_session_factory()
    async with factory() as session:
        repo = TradeRepository(session)
        await repo.add_to_watchlist(
            symbol=symbol, exchange=req.exchange, market=req.market,
        )
    items = await _get_symbols(req.market)
    return {
        "symbols": [w["symbol"] for w in items],
        "items": items,
    }


@router.delete("/{symbol}")
async def remove_symbol(symbol: str, market: str = Query("US")):
    """Remove a symbol from watchlist."""
    symbol = symbol.upper()
    factory = get_session_factory()
    async with factory() as session:
        repo = TradeRepository(session)
        await repo.remove_from_watchlist(symbol, market=market)
    items = await _get_symbols(market)
    return {
        "symbols": [w["symbol"] for w in items],
        "items": items,
    }
