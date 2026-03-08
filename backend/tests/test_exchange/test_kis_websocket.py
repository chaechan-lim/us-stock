"""Tests for KIS WebSocket client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exchange.kis_websocket import (
    KISWebSocket,
    MAX_SUBSCRIPTIONS,
    WS_TR_EXECUTION,
    WS_TR_ORDERBOOK,
    _EXCHANGE_PREFIX,
)


@pytest.fixture
def mock_auth():
    auth = AsyncMock()
    auth.get_approval_key = AsyncMock(return_value="test-approval-key")
    return auth


@pytest.fixture
def ws_client(mock_auth):
    return KISWebSocket(auth=mock_auth, ws_url="ws://test:21000")


# ── Properties ───────────────────────────────────────────────────────

def test_initial_state(ws_client):
    assert ws_client.subscription_count == 0
    assert ws_client.available_slots == MAX_SUBSCRIPTIONS
    assert ws_client.is_connected is False


def test_max_subscriptions_constant():
    assert MAX_SUBSCRIPTIONS == 41


def test_exchange_prefix_mapping():
    assert _EXCHANGE_PREFIX["NASD"] == "DNAS"
    assert _EXCHANGE_PREFIX["NYSE"] == "DNYS"
    assert _EXCHANGE_PREFIX["AMEX"] == "DAMS"


# ── Subscribe ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscribe_success(ws_client, mock_auth):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    result = await ws_client.subscribe("AAPL")
    assert result is True
    assert ws_client.subscription_count == 1
    assert "AAPL:price" in ws_client._subscriptions

    # Verify message sent
    sent = json.loads(ws_client._ws.send.call_args[0][0])
    assert sent["header"]["tr_type"] == "1"
    assert sent["body"]["input"]["tr_id"] == WS_TR_EXECUTION
    assert sent["body"]["input"]["tr_key"] == "DNASAAPL"


@pytest.mark.asyncio
async def test_subscribe_orderbook(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    result = await ws_client.subscribe("TSLA", data_type="orderbook")
    assert result is True
    assert "TSLA:orderbook" in ws_client._subscriptions

    sent = json.loads(ws_client._ws.send.call_args[0][0])
    assert sent["body"]["input"]["tr_id"] == WS_TR_ORDERBOOK


@pytest.mark.asyncio
async def test_subscribe_nyse_exchange(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    await ws_client.subscribe("IBM", exchange="NYSE")
    sent = json.loads(ws_client._ws.send.call_args[0][0])
    assert sent["body"]["input"]["tr_key"] == "DNYSIBM"


@pytest.mark.asyncio
async def test_subscribe_amex_exchange(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    await ws_client.subscribe("SPY", exchange="AMEX")
    sent = json.loads(ws_client._ws.send.call_args[0][0])
    assert sent["body"]["input"]["tr_key"] == "DAMSSPY"


@pytest.mark.asyncio
async def test_subscribe_duplicate_returns_true(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    await ws_client.subscribe("AAPL")
    result = await ws_client.subscribe("AAPL")
    assert result is True
    assert ws_client.subscription_count == 1  # Not doubled


@pytest.mark.asyncio
async def test_subscribe_limit_reached(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    # Fill up subscriptions
    ws_client._subscriptions = {f"SYM{i}:price" for i in range(MAX_SUBSCRIPTIONS)}

    result = await ws_client.subscribe("OVERFLOW")
    assert result is False
    assert ws_client.subscription_count == MAX_SUBSCRIPTIONS


@pytest.mark.asyncio
async def test_subscribe_no_ws_returns_false(ws_client):
    ws_client._ws = None
    result = await ws_client.subscribe("AAPL")
    assert result is False


# ── Unsubscribe ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unsubscribe(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True
    ws_client._subscriptions = {"AAPL:price"}

    await ws_client.unsubscribe("AAPL")
    assert ws_client.subscription_count == 0

    sent = json.loads(ws_client._ws.send.call_args[0][0])
    assert sent["header"]["tr_type"] == "2"


@pytest.mark.asyncio
async def test_unsubscribe_not_subscribed(ws_client):
    ws_client._ws = AsyncMock()
    await ws_client.unsubscribe("NEVER")
    ws_client._ws.send.assert_not_called()


# ── Callbacks ────────────────────────────────────────────────────────

def test_on_price_callback(ws_client):
    cb = MagicMock()
    ws_client.on_price(cb)
    assert cb in ws_client._callbacks["price"]


def test_on_orderbook_callback(ws_client):
    cb = MagicMock()
    ws_client.on_orderbook(cb)
    assert cb in ws_client._callbacks["orderbook"]


def test_on_execution_callback(ws_client):
    cb = MagicMock()
    ws_client.on_execution(cb)
    assert cb in ws_client._callbacks["execution"]


# ── Message handling ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_pingpong(ws_client):
    ws_client._ws = AsyncMock()
    await ws_client._handle_message("PINGPONG")
    ws_client._ws.send.assert_called_once_with("PINGPONG")


@pytest.mark.asyncio
async def test_handle_price_data(ws_client):
    received = []
    ws_client.on_price(lambda d: received.append(d))

    # Pipe-delimited format: len|tr_id|data_flag|data
    raw = f"100|{WS_TR_EXECUTION}|0|AAPL^100^185.50^5000"
    await ws_client._handle_message(raw)

    assert len(received) == 1
    assert received[0]["symbol"] == "AAPL"
    assert received[0]["price"] == 185.50
    assert received[0]["volume"] == 5000


@pytest.mark.asyncio
async def test_handle_price_async_callback(ws_client):
    received = []

    async def async_cb(data):
        received.append(data)

    ws_client.on_price(async_cb)

    raw = f"100|{WS_TR_EXECUTION}|0|MSFT^200^420.00^3000"
    await ws_client._handle_message(raw)

    assert len(received) == 1
    assert received[0]["symbol"] == "MSFT"


@pytest.mark.asyncio
async def test_handle_json_confirmation(ws_client):
    """JSON subscription confirmations should not crash."""
    msg = json.dumps({
        "header": {"tr_id": "SUBSCRIBE_OK", "msg_cd": "0000"},
        "body": {},
    })
    await ws_client._handle_message(msg)  # Should not raise


@pytest.mark.asyncio
async def test_handle_malformed_message(ws_client):
    """Bad data should be logged but not crash."""
    await ws_client._handle_message("not-valid-at-all")
    await ws_client._handle_message("")
    await ws_client._handle_message("a|b")
    # No exceptions


# ── Update subscriptions ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_subscriptions(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    await ws_client.update_subscriptions(
        priority_symbols=["AAPL", "TSLA"],
        watch_symbols=["MSFT", "GOOG"],
    )
    assert ws_client.subscription_count == 4


@pytest.mark.asyncio
async def test_update_subscriptions_removes_old(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True
    ws_client._subscriptions = {"OLD:price"}

    await ws_client.update_subscriptions(
        priority_symbols=["AAPL"],
        watch_symbols=[],
    )
    assert "OLD:price" not in ws_client._subscriptions
    assert "AAPL:price" in ws_client._subscriptions


@pytest.mark.asyncio
async def test_update_subscriptions_respects_limit(ws_client):
    ws_client._ws = AsyncMock()
    ws_client._running = True

    # 5 priority + many watch → capped at MAX
    priority = [f"P{i}" for i in range(5)]
    watch = [f"W{i}" for i in range(100)]

    await ws_client.update_subscriptions(priority, watch)
    assert ws_client.subscription_count <= MAX_SUBSCRIPTIONS


# ── Status ───────────────────────────────────────────────────────────

def test_get_status(ws_client):
    ws_client._subscriptions = {"AAPL:price", "TSLA:price"}
    ws_client._running = True
    ws_client._ws = MagicMock()

    status = ws_client.get_status()
    assert status["connected"] is True
    assert status["subscriptions"] == 2
    assert status["available_slots"] == MAX_SUBSCRIPTIONS - 2
    assert "AAPL" in status["symbols"]
    assert "TSLA" in status["symbols"]


def test_get_status_disconnected(ws_client):
    status = ws_client.get_status()
    assert status["connected"] is False
    assert status["subscriptions"] == 0


# ── Close ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close(ws_client):
    mock_ws = AsyncMock()
    ws_client._ws = mock_ws
    ws_client._running = True
    ws_client._subscriptions = {"AAPL:price"}

    await ws_client.close()
    assert ws_client._running is False
    assert ws_client.subscription_count == 0
    assert ws_client._ws is None
    mock_ws.close.assert_called_once()
