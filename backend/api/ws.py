"""WebSocket endpoints for real-time log streaming and price updates."""

import asyncio
import json
import logging
from collections import deque
from typing import Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Connected log clients
_log_clients: Set[WebSocket] = set()

# In-memory ring buffer for HTTP polling fallback
_log_buffer: deque[dict] = deque(maxlen=500)


class WebSocketLogHandler(logging.Handler):
    """Logging handler that broadcasts structured log entries to WebSocket clients."""

    # Standard LogRecord attributes to exclude from extra fields
    _STANDARD_FIELDS = frozenset({
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "pathname", "thread", "threadName",
        "process", "processName", "levelname", "levelno", "message",
        "msecs", "taskName",
    })

    # Python logging level names → frontend-friendly names
    _LEVEL_MAP = {"WARNING": "WARN", "CRITICAL": "ERROR"}

    def emit(self, record: logging.LogRecord):
        from datetime import datetime, timezone

        level = self._LEVEL_MAP.get(record.levelname, record.levelname)
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": level,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields attached to the record
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._STANDARD_FIELDS:
                continue
            payload[key] = value

        # Always store in ring buffer (for HTTP polling)
        _log_buffer.append(payload)

        if not _log_clients:
            return

        entry = json.dumps(payload, default=str)

        # Schedule broadcast (non-blocking)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No event loop running
        for ws in list(_log_clients):
            loop.create_task(_safe_send(ws, entry))


async def _safe_send(ws: WebSocket, data: str):
    try:
        await ws.send_text(data)
    except Exception as e:
        logger.debug("WebSocket send failed, removing client: %s", e)
        _log_clients.discard(ws)


def install_log_handler():
    """Install the WebSocket log handler on the root logger."""
    handler = WebSocketLogHandler()
    logging.getLogger().addHandler(handler)


@router.get("/logs")
async def get_logs(limit: int = Query(200, ge=1, le=500), level: str = Query("ALL")):
    """Get recent logs from in-memory buffer (HTTP fallback for WebSocket)."""
    logs = list(_log_buffer)
    if level != "ALL":
        logs = [l for l in logs if l.get("level") == level]
    return logs[-limit:]


@router.websocket("/logs")
async def ws_logs(websocket: WebSocket):
    """Stream server logs to connected clients."""
    await websocket.accept()
    _log_clients.add(websocket)
    try:
        while True:
            # Keep connection alive; client doesn't send data
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _log_clients.discard(websocket)


@router.websocket("/prices")
async def ws_prices(websocket: WebSocket):
    """Stream price updates for subscribed symbols.

    Client sends: {"subscribe": ["AAPL", "TSLA"]}
    Server sends: {"symbol": "AAPL", "price": 180.5, "change_pct": 1.2, "volume": 50000000}
    """
    await websocket.accept()
    symbols: list[str] = []
    interval = 10  # seconds between updates

    async def price_loop():
        while True:
            if symbols and hasattr(websocket.app.state, "market_data"):
                market_data = websocket.app.state.market_data
                for sym in list(symbols):
                    try:
                        ticker = await market_data.get_ticker(sym)
                        await websocket.send_text(json.dumps({
                            "symbol": ticker.symbol,
                            "price": ticker.price,
                            "change_pct": ticker.change_pct,
                            "volume": ticker.volume,
                        }))
                    except Exception as e:
                        logger.debug("Price update failed for %s: %s", sym, e)
            await asyncio.sleep(interval)

    task = asyncio.create_task(price_loop())
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if "subscribe" in data:
                    symbols = [s.upper() for s in data["subscribe"] if s]
                    logging.getLogger(__name__).info(
                        "Price WS subscribed: %s", symbols
                    )
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
