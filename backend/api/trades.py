"""Trade history API endpoints.

Maintains an in-memory trade log AND persists to DB via TradeRepository.
The in-memory log is the primary fast-path; DB is the durable store.
On startup, the trade log is restored from DB so history survives restarts.
"""

import logging

from data.stock_name_service import get_name
from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/trades", tags=["trades"])
logger = logging.getLogger(__name__)

# In-memory trade log (restored from DB on startup)
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
    offset: int = Query(0, ge=0),
    symbol: str | None = None,
    market: str | None = None,
):
    """Get trade history (in-memory + DB fallback).

    Returns trades sorted newest-first (descending by created_at).
    Supports offset-based pagination for browsing older trades.
    """
    # If in-memory has data, use it (fast path)
    if _trade_log:
        trades = _trade_log
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol.upper()]
        if market:
            trades = [t for t in trades if t.get("market", "US") == market]
        # Reverse to newest-first, then apply offset+limit pagination
        newest_first = list(reversed(trades))
        result = newest_first[offset : offset + limit]

        # Batch-resolve missing names (fills cache for future calls)
        missing = [t for t in result if not t.get("name")]
        if missing:
            from data.stock_name_service import resolve_names as _resolve

            syms_by_market: dict[str, list[str]] = {}
            for t in missing:
                mkt = t.get("market", "US")
                syms_by_market.setdefault(mkt, []).append(t["symbol"])
            for mkt, syms in syms_by_market.items():
                await _resolve(list(set(syms)), mkt)

        for t in result:
            if not t.get("name"):
                t["name"] = get_name(t.get("symbol", ""), t.get("market", "US")) or ""
        return result

    # Fallback: read from DB if available
    if _session_factory:
        try:
            from db.trade_repository import TradeRepository

            async with _session_factory() as session:
                repo = TradeRepository(session)
                orders = await repo.get_trade_history(
                    limit=limit, offset=offset, symbol=symbol,
                )
                return [
                    {
                        "symbol": o.symbol,
                        "side": o.side,
                        "quantity": o.quantity,
                        "price": o.price,
                        "filled_price": o.filled_price,
                        "status": o.status,
                        "strategy": o.strategy_name,
                        "pnl": o.pnl,
                        "pnl_pct": getattr(o, "pnl_pct", None),
                        "market": getattr(o, "market", "US"),
                        "session": getattr(o, "session", "regular") or "regular",
                        "name": get_name(o.symbol, getattr(o, "market", "US")) or "",
                        "created_at": str(o.created_at),
                    }
                    for o in orders
                ]
        except Exception as e:
            logger.warning("DB trade history fallback failed: %s", e)

    return []


@router.get("/summary")
async def trade_summary(market: str | None = None):
    """Get aggregated trade stats (excludes paper orders)."""
    # Exclude paper orders from summary to avoid PnL distortion
    trades = [t for t in _trade_log if not t.get("is_paper", False)]
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
    """Record a trade to in-memory log AND persist to DB.

    Idempotent: if an entry with the same non-empty order_id already exists
    in the trade log, merges new data into the existing entry (preserving
    PnL if already set). This prevents duplicate rows when both place_sell()
    and reconciliation record the same order (STOCK-33).
    """
    order_id = trade.get("order_id", "")

    # Dedup: merge into existing entry if order_id matches
    if order_id:
        for existing in _trade_log:
            if existing.get("order_id") == order_id:
                _merge_trade_entry(existing, trade)
                # Still persist to DB — save_order UPSERT handles DB-level dedup
                if _session_factory:
                    import asyncio

                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(_persist_trade(existing))
                    except RuntimeError:
                        pass
                return

    _trade_log.append(trade)

    # Async DB persist (fire-and-forget via session factory)
    if _session_factory:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_persist_trade(trade))
        except RuntimeError:
            pass  # No event loop — skip DB persist


def _merge_trade_entry(existing: dict, new: dict) -> None:
    """Merge new trade data into an existing entry, preserving PnL.

    Rules:
    - PnL/pnl_pct: never overwrite a real value with None
    - status: never downgrade from 'filled'
    - All other fields: overwrite with new value if provided
    """
    for key, value in new.items():
        if key in ("pnl", "pnl_pct") and value is None and existing.get(key) is not None:
            continue  # Don't overwrite real PnL with None
        if key == "status" and existing.get("status") == "filled" and value != "filled":
            continue  # Don't downgrade from filled
        existing[key] = value


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
                buy_strategy=trade.get("buy_strategy", ""),
                kis_order_id=trade.get("order_id", ""),
                pnl=trade.get("pnl"),
                pnl_pct=trade.get("pnl_pct"),
                exchange=trade.get("exchange", "NASD"),
                market=trade.get("market", "US"),
                session=trade.get("session", "regular"),
                is_paper=trade.get("is_paper", False),
            )
    except Exception as e:
        logger.warning("Failed to persist trade to DB: %s", e)


def _order_to_dict(o) -> dict:
    """Convert an Order ORM object to trade log dict."""
    return {
        "order_id": o.kis_order_id or "",
        "symbol": o.symbol,
        "side": o.side,
        "quantity": o.quantity,
        "price": o.price,
        "filled_price": o.filled_price,
        "filled_quantity": o.filled_quantity,
        "status": o.status,
        "strategy": o.strategy_name or "",
        "buy_strategy": getattr(o, "buy_strategy", "") or "",
        "pnl": o.pnl,
        "pnl_pct": getattr(o, "pnl_pct", None),
        "is_paper": getattr(o, "is_paper", False),
        "market": getattr(o, "market", "US"),
        "session": getattr(o, "session", "regular") or "regular",
        "created_at": str(o.created_at) if o.created_at else "",
        "db_id": o.id,
    }


async def restore_trade_log(exclude_paper: bool = True) -> int:
    """Restore in-memory trade log from DB on startup.

    Deduplicates by kis_order_id: when multiple rows share the same non-empty
    kis_order_id, keeps only the one with PnL data (or the newest if neither
    has PnL). This cleans up any duplicates that slipped into the DB (STOCK-33).

    Args:
        exclude_paper: If True (default), paper orders are excluded from the
            restored trade log to prevent position/PnL distortion.

    Returns count of restored trades.
    """
    if not _session_factory:
        return 0
    try:
        from db.trade_repository import TradeRepository

        async with _session_factory() as session:
            repo = TradeRepository(session)
            orders = await repo.get_trade_history(
                limit=200,
                exclude_paper=exclude_paper,
            )
            _trade_log.clear()
            seen_order_ids: dict[str, int] = {}  # order_id -> index in _trade_log
            for o in reversed(orders):  # oldest first
                entry = _order_to_dict(o)
                oid = entry.get("order_id", "")
                if oid and oid in seen_order_ids:
                    # Duplicate kis_order_id — keep the one with PnL
                    idx = seen_order_ids[oid]
                    existing = _trade_log[idx]
                    if entry.get("pnl") is not None and existing.get("pnl") is None:
                        _trade_log[idx] = entry  # Replace with the one that has PnL
                    # else: keep existing (it has PnL or both have it)
                    continue
                if oid:
                    seen_order_ids[oid] = len(_trade_log)
                _trade_log.append(entry)
        logger.info("Restored %d trades from DB into trade log", len(_trade_log))
        return len(_trade_log)
    except Exception as e:
        logger.warning("Failed to restore trade log from DB: %s", e)
        return 0


async def reconcile_pending_orders(held_symbols: set[str]) -> int:
    """Reconcile pending DB orders using current exchange positions.

    Only reconciles live (non-paper) orders to avoid position distortion.

    Heuristic for orders without kis_order_id:
    - BUY + symbol in held_symbols → filled
    - SELL + symbol NOT in held_symbols → filled
    - Otherwise → cancelled (orphan cleanup already ran)

    Returns count of updated orders.
    """
    if not _session_factory:
        return 0
    updated = 0
    try:
        from db.trade_repository import TradeRepository

        async with _session_factory() as session:
            repo = TradeRepository(session)
            pending = await repo.get_open_orders(exclude_paper=True)
            if not pending:
                return 0

            for o in pending:
                if o.side == "BUY" and o.symbol in held_symbols:
                    # We hold this stock → BUY was filled
                    await repo.update_order_status(
                        o.id,
                        "filled",
                        filled_price=o.price,
                        filled_quantity=o.quantity,
                    )
                    updated += 1
                elif o.side == "SELL" and o.symbol not in held_symbols:
                    # We don't hold this stock → SELL was filled
                    await repo.update_order_status(
                        o.id,
                        "filled",
                        filled_price=o.price,
                        filled_quantity=o.quantity,
                    )
                    updated += 1
                else:
                    # BUY but not held / SELL but still held → cancelled
                    await repo.update_order_status(o.id, "cancelled")
                    updated += 1

            if updated:
                logger.info(
                    "Reconciled %d pending DB orders (held=%s)",
                    updated,
                    ", ".join(sorted(held_symbols)) or "none",
                )
    except Exception as e:
        logger.warning("Failed to reconcile pending DB orders: %s", e)
    return updated


async def update_order_in_db(
    kis_order_id: str,
    status: str,
    filled_price: float | None = None,
    filled_quantity: float | None = None,
) -> bool:
    """Update a specific order's status in DB by KIS order ID.

    Called by reconciliation task when exchange confirms status change.
    Also updates the in-memory trade log entry.
    """
    if not _session_factory or not kis_order_id:
        return False
    try:
        from db.trade_repository import TradeRepository

        async with _session_factory() as session:
            repo = TradeRepository(session)
            order = await repo.find_by_kis_order_id(kis_order_id)
            if order:
                await repo.update_order_status(
                    order.id,
                    status,
                    filled_price=filled_price,
                    filled_quantity=filled_quantity,
                )
                # Update in-memory trade log too
                for t in _trade_log:
                    if t.get("order_id") == kis_order_id:
                        t["status"] = status
                        if filled_price is not None:
                            t["filled_price"] = filled_price
                        if filled_quantity is not None:
                            t["filled_quantity"] = filled_quantity
                        break
                return True
    except Exception as e:
        logger.warning("Failed to update order %s in DB: %s", kis_order_id, e)
    return False
