"""Trade history API endpoints.

Maintains an in-memory trade log AND persists to DB via TradeRepository.
The in-memory log is the primary fast-path; DB is the durable store.
"""

import logging

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/trades", tags=["trades"])
logger = logging.getLogger(__name__)

# In-memory trade log (fast reads, lost on restart)
_trade_log: list[dict] = []

# Session factory — set at startup via init_trades()
_session_factory = None


def init_trades(session_factory) -> None:
    """Wire DB session factory. Called from main.py lifespan."""
    global _session_factory
    _session_factory = session_factory


@router.get("/")
async def get_trades(
    limit: int = Query(50, ge=1, le=200),
    symbol: str | None = None,
    market: str | None = None,
):
    """Get trade history (in-memory + DB fallback)."""
    # If in-memory has data, use it (fast path)
    if _trade_log:
        trades = _trade_log
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol.upper()]
        if market:
            trades = [t for t in trades if t.get("market", "US") == market]
        return trades[-limit:]

    # Fallback: read from DB if available
    if _session_factory:
        try:
            from db.trade_repository import TradeRepository
            async with _session_factory() as session:
                repo = TradeRepository(session)
                orders = await repo.get_trade_history(limit=limit, symbol=symbol)
                return [
                    {
                        "symbol": o.symbol, "side": o.side,
                        "quantity": o.quantity, "price": o.price,
                        "filled_price": o.filled_price,
                        "status": o.status, "strategy": o.strategy_name,
                        "pnl": o.pnl,
                        "created_at": str(o.created_at),
                    }
                    for o in orders
                ]
        except Exception as e:
            logger.warning("DB trade history fallback failed: %s", e)

    return []


@router.get("/summary")
async def trade_summary(market: str | None = None):
    """Get aggregated trade stats."""
    trades = _trade_log
    if market:
        trades = [t for t in trades if t.get("market", "US") == market]

    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }

    sells = [t for t in trades if t.get("side") == "SELL" and t.get("pnl") is not None]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in sells)

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl": total_pnl,
        "win_rate": len(wins) / len(sells) * 100 if sells else 0.0,
    }


def record_trade(trade: dict) -> None:
    """Record a trade to in-memory log AND persist to DB."""
    _trade_log.append(trade)

    # Async DB persist (fire-and-forget via session factory)
    if _session_factory:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_persist_trade(trade))
        except RuntimeError:
            pass  # No event loop — skip DB persist


async def _persist_trade(trade: dict) -> None:
    """Persist trade to orders table."""
    try:
        from db.trade_repository import TradeRepository
        async with _session_factory() as session:
            repo = TradeRepository(session)
            await repo.save_order(
                symbol=trade.get("symbol", ""),
                side=trade.get("side", ""),
                order_type="market",
                quantity=trade.get("quantity", 0),
                price=trade.get("price"),
                filled_quantity=trade.get("filled_quantity", 0),
                filled_price=trade.get("filled_price"),
                status=trade.get("status", "filled"),
                strategy_name=trade.get("strategy", ""),
                pnl=trade.get("pnl"),
            )
    except Exception as e:
        logger.warning("Failed to persist trade to DB: %s", e)
