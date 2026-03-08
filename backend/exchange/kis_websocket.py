"""KIS WebSocket client for real-time market data.

Constraints (per KIS dev portal):
- Max 41 subscriptions per session
- 1 connection per approval_key
- PINGPONG keepalive required
- US stock data is delayed (~15min)
- Do NOT keep connections alive indefinitely — connect only when needed
- Do NOT loop connect/disconnect or subscribe/unsubscribe rapidly

Lifecycle:
- connect() during market hours only
- Auto-disconnect after max_session_sec (default 4h)
- Graceful close() when market closes
- Reconnect with exponential backoff on unexpected disconnect
- Min interval between connect attempts to avoid IP ban
"""

import json
import asyncio
import logging
import time
from typing import Callable, Any

import websockets

from exchange.kis_auth import KISAuth

logger = logging.getLogger(__name__)

MAX_SUBSCRIPTIONS = 41
DEFAULT_MAX_SESSION_SEC = 4 * 3600  # 4 hours
MIN_RECONNECT_INTERVAL_SEC = 30     # min gap between connection attempts
MAX_RECONNECT_DELAY_SEC = 300       # 5 min max backoff

# WebSocket TR_IDs
WS_TR_EXECUTION = "HDFSCNT0"   # Real-time (delayed) execution price
WS_TR_ORDERBOOK = "HDFSASP0"   # Real-time bid/ask

# Exchange prefix mapping for WebSocket tr_key
_EXCHANGE_PREFIX = {
    "NASD": "DNAS",  # NASDAQ
    "NYSE": "DNYS",  # NYSE
    "AMEX": "DAMS",  # AMEX
}


class KISWebSocket:
    """KIS WebSocket client for real-time overseas stock data."""

    def __init__(
        self,
        auth: KISAuth,
        ws_url: str,
        max_session_sec: int = DEFAULT_MAX_SESSION_SEC,
    ):
        self._auth = auth
        self._ws_url = ws_url
        self._ws = None
        self._subscriptions: set[str] = set()
        self._callbacks: dict[str, list[Callable]] = {
            "price": [],
            "orderbook": [],
            "execution": [],
        }
        self._running = False
        self._reconnect_delay = MIN_RECONNECT_INTERVAL_SEC
        self._listen_task: asyncio.Task | None = None
        self._max_session_sec = max_session_sec
        self._connected_at: float = 0
        self._last_connect_attempt: float = 0
        self._intentional_close = False

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    @property
    def available_slots(self) -> int:
        return MAX_SUBSCRIPTIONS - len(self._subscriptions)

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    @property
    def session_age_sec(self) -> float:
        if not self._connected_at:
            return 0
        return time.monotonic() - self._connected_at

    async def connect(self) -> None:
        """Establish WebSocket connection with rate limiting."""
        # Enforce minimum interval between connect attempts
        now = time.monotonic()
        since_last = now - self._last_connect_attempt
        if self._last_connect_attempt > 0 and since_last < MIN_RECONNECT_INTERVAL_SEC:
            wait = MIN_RECONNECT_INTERVAL_SEC - since_last
            logger.info("WS connect throttle: waiting %.1fs", wait)
            await asyncio.sleep(wait)

        self._last_connect_attempt = time.monotonic()
        self._intentional_close = False

        approval_key = await self._auth.get_approval_key()
        self._ws = await websockets.connect(self._ws_url, ping_interval=None)
        self._running = True
        self._connected_at = time.monotonic()
        self._reconnect_delay = MIN_RECONNECT_INTERVAL_SEC
        logger.info("KIS WebSocket connected (max %d subs, session limit %ds)",
                     MAX_SUBSCRIPTIONS, self._max_session_sec)

        # Start listener
        self._listen_task = asyncio.create_task(self._listen())

    async def close(self) -> None:
        """Gracefully close WebSocket connection."""
        self._intentional_close = True
        self._running = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._subscriptions.clear()
        self._connected_at = 0
        logger.info("KIS WebSocket closed (intentional)")

    async def subscribe(
        self, symbol: str, data_type: str = "price", exchange: str = "NASD",
    ) -> bool:
        """Subscribe to real-time data for a symbol."""
        sub_key = f"{symbol}:{data_type}"
        if sub_key in self._subscriptions:
            return True

        if len(self._subscriptions) >= MAX_SUBSCRIPTIONS:
            logger.warning(
                "WebSocket subscription limit reached (%d/%d). Cannot subscribe %s",
                len(self._subscriptions), MAX_SUBSCRIPTIONS, symbol,
            )
            return False

        if not self._ws:
            return False

        tr_id = WS_TR_EXECUTION if data_type == "price" else WS_TR_ORDERBOOK
        approval_key = await self._auth.get_approval_key()
        prefix = _EXCHANGE_PREFIX.get(exchange, "DNAS")

        msg = {
            "header": {
                "approval_key": approval_key,
                "custtype": "P",
                "tr_type": "1",  # 1=subscribe
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": tr_id,
                    "tr_key": f"{prefix}{symbol}",
                },
            },
        }

        await self._ws.send(json.dumps(msg))
        self._subscriptions.add(sub_key)
        logger.info("Subscribed to %s (%d/%d)", sub_key,
                     len(self._subscriptions), MAX_SUBSCRIPTIONS)
        return True

    async def unsubscribe(
        self, symbol: str, data_type: str = "price", exchange: str = "NASD",
    ) -> None:
        """Unsubscribe from a symbol."""
        sub_key = f"{symbol}:{data_type}"
        if sub_key not in self._subscriptions:
            return

        if not self._ws:
            self._subscriptions.discard(sub_key)
            return

        tr_id = WS_TR_EXECUTION if data_type == "price" else WS_TR_ORDERBOOK
        approval_key = await self._auth.get_approval_key()
        prefix = _EXCHANGE_PREFIX.get(exchange, "DNAS")

        msg = {
            "header": {
                "approval_key": approval_key,
                "custtype": "P",
                "tr_type": "2",  # 2=unsubscribe
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id": tr_id,
                    "tr_key": f"{prefix}{symbol}",
                },
            },
        }

        await self._ws.send(json.dumps(msg))
        self._subscriptions.discard(sub_key)
        logger.info("Unsubscribed from %s", sub_key)

    def on_price(self, callback: Callable[[dict], Any]) -> None:
        self._callbacks["price"].append(callback)

    def on_orderbook(self, callback: Callable[[dict], Any]) -> None:
        self._callbacks["orderbook"].append(callback)

    def on_execution(self, callback: Callable[[dict], Any]) -> None:
        self._callbacks["execution"].append(callback)

    async def update_subscriptions(
        self,
        priority_symbols: list[str],
        watch_symbols: list[str],
    ) -> None:
        """Update subscriptions based on priority.

        Priority order:
        1. Held positions (priority_symbols) - always subscribed
        2. Watch list (watch_symbols) - fill remaining slots
        """
        desired = set()
        for s in priority_symbols:
            desired.add(f"{s}:price")
        remaining = MAX_SUBSCRIPTIONS - len(desired)
        for s in watch_symbols[:remaining]:
            desired.add(f"{s}:price")

        # Unsubscribe removed
        to_remove = self._subscriptions - desired
        for sub_key in to_remove:
            symbol, dtype = sub_key.split(":", 1)
            await self.unsubscribe(symbol, dtype)

        # Subscribe new
        to_add = desired - self._subscriptions
        for sub_key in to_add:
            symbol, dtype = sub_key.split(":", 1)
            await self.subscribe(symbol, dtype)

    async def refresh_session(self) -> None:
        """Gracefully reconnect: unsubscribe all, close, wait, reconnect, re-subscribe.

        Called when session exceeds max_session_sec to avoid indefinite connections.
        """
        if not self.is_connected:
            return

        subs_copy = list(self._subscriptions)
        logger.info("WS session refresh: saving %d subscriptions", len(subs_copy))

        # Graceful: unsubscribe all first
        for sub_key in list(self._subscriptions):
            symbol, dtype = sub_key.split(":", 1)
            try:
                await self.unsubscribe(symbol, dtype)
            except Exception:
                pass

        # Close connection
        await self.close()

        # Wait before reconnecting (avoid rapid connect/disconnect)
        await asyncio.sleep(MIN_RECONNECT_INTERVAL_SEC)

        # Reconnect and re-subscribe
        try:
            await self.connect()
            for sub_key in subs_copy:
                symbol, dtype = sub_key.split(":", 1)
                await self.subscribe(symbol, dtype)
            logger.info("WS session refreshed: %d subscriptions restored",
                        len(self._subscriptions))
        except Exception as e:
            logger.error("WS session refresh failed: %s", e)

    def get_status(self) -> dict:
        """Get WebSocket status for monitoring."""
        return {
            "connected": self.is_connected,
            "subscriptions": self.subscription_count,
            "available_slots": self.available_slots,
            "max_subscriptions": MAX_SUBSCRIPTIONS,
            "session_age_sec": round(self.session_age_sec),
            "max_session_sec": self._max_session_sec,
            "symbols": sorted(
                s.split(":")[0] for s in self._subscriptions
            ),
        }

    # -- Private --

    async def _listen(self) -> None:
        """Main WebSocket listener loop with session timeout."""
        while self._running and self._ws:
            try:
                # Check session age — refresh if exceeded
                if self.session_age_sec > self._max_session_sec:
                    logger.info("WS session expired (%.0fs > %ds), refreshing",
                                self.session_age_sec, self._max_session_sec)
                    asyncio.create_task(self.refresh_session())
                    return

                message = await asyncio.wait_for(
                    self._ws.recv(), timeout=60.0,
                )
                await self._handle_message(message)
                self._reconnect_delay = MIN_RECONNECT_INTERVAL_SEC

            except asyncio.TimeoutError:
                # No message in 60s — normal, just loop
                continue

            except websockets.ConnectionClosed:
                if self._intentional_close:
                    break
                if self._running:
                    logger.warning(
                        "WebSocket disconnected unexpectedly. "
                        "Reconnecting in %.0fs...", self._reconnect_delay,
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2, MAX_RECONNECT_DELAY_SEC,
                    )
                    subs_copy = list(self._subscriptions)
                    try:
                        self._subscriptions.clear()
                        self._ws = None
                        await self.connect()
                        for sub_key in subs_copy:
                            symbol, dtype = sub_key.split(":", 1)
                            await self.subscribe(symbol, dtype)
                    except Exception as e:
                        # Restore subscriptions list so next reconnect can retry
                        self._subscriptions = set(subs_copy)
                        logger.error("Reconnection failed: %s", e)
                break

            except Exception as e:
                if not self._intentional_close:
                    logger.error("WebSocket listener error: %s", e)

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch incoming WebSocket message."""
        # PINGPONG keepalive
        if raw.startswith("PINGPONG"):
            if self._ws:
                await self._ws.send(raw)
            return

        try:
            # KIS sends pipe-delimited data for execution prices
            if "|" in raw:
                parts = raw.split("|")
                if len(parts) >= 4:
                    tr_id = parts[1]
                    data_str = parts[3]
                    fields = data_str.split("^")

                    if tr_id == WS_TR_EXECUTION and len(fields) >= 3:
                        price_data = {
                            "symbol": fields[0] if fields else "",
                            "price": float(fields[2]) if len(fields) > 2 else 0,
                            "volume": float(fields[3]) if len(fields) > 3 else 0,
                        }
                        for cb in self._callbacks["price"]:
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    await cb(price_data)
                                else:
                                    cb(price_data)
                            except Exception as cb_err:
                                logger.error("WS callback error: %s", cb_err)

            else:
                data = json.loads(raw)
                # Handle structured JSON responses (subscription confirmations, etc.)
                header = data.get("header", {})
                if header.get("tr_id") == "PINGPONG":
                    if self._ws:
                        await self._ws.send(raw)

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.debug("Failed to parse WS message: %s", e)
