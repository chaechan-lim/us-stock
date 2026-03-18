"""Tests for NotificationService.

Covers alert categories, throttling, multi-provider dispatch,
alert history, template formatting, disabled service, adapter pattern,
and CRITICAL throttle bypass.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.notification import (
    AlertCategory,
    AlertLevel,
    AlertRecord,
    DiscordAdapter,
    NotificationAdapter,
    NotificationService,
    SlackAdapter,
    TelegramAdapter,
)


# ── Mock adapter ─────────────────────────────────────────────────────

class MockAdapter(NotificationAdapter):
    """In-memory adapter for testing dispatch logic."""

    def __init__(self, name_val="mock", configured=True, fail=False):
        self._name = name_val
        self._configured = configured
        self._fail = fail
        self.sent: list[tuple[str, AlertLevel]] = []
        self.sent_rich: list[tuple[str, str, AlertLevel, dict | None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if self._fail:
            raise ConnectionError("mock fail")
        self.sent.append((message, level))
        return True

    async def send_rich(
        self, title: str, body: str, level: AlertLevel, fields: dict | None = None,
    ) -> bool:
        if self._fail:
            raise ConnectionError("mock fail")
        self.sent_rich.append((title, body, level, fields))
        return True


# ── Helpers ──────────────────────────────────────────────────────────

def _mock_session(telegram_status=200, discord_status=204):
    """Return a mock aiohttp.ClientSession that fakes HTTP responses."""

    def _make_resp(status, text="OK"):
        resp = AsyncMock()
        resp.status = status
        resp.text = AsyncMock(return_value=text)
        return resp

    def _post_side_effect(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if "discord.com" in str(url):
            status = discord_status
        else:
            status = telegram_status
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=_make_resp(status))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(side_effect=_post_side_effect)
    return session


def _svc_with_mock(name="mock", configured=True, fail=False) -> tuple[NotificationService, MockAdapter]:
    """Create a service with a single mock adapter."""
    svc = NotificationService(enabled=True, throttle_seconds=0)
    adapter = MockAdapter(name_val=name, configured=configured, fail=fail)
    svc.add_adapter(adapter)
    return svc, adapter


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def disabled_svc():
    return NotificationService(enabled=False)


@pytest.fixture
def telegram_svc():
    return NotificationService(
        enabled=True,
        provider="telegram",
        telegram_bot_token="fake-token",
        telegram_chat_id="12345",
    )


@pytest.fixture
def discord_svc():
    return NotificationService(
        enabled=True,
        provider="discord",
        discord_webhook_url="https://discord.com/api/webhooks/fake",
    )


@pytest.fixture
def multi_svc():
    """Service with both Telegram and Discord configured."""
    return NotificationService(
        enabled=True,
        provider="telegram",
        telegram_bot_token="fake-token",
        telegram_chat_id="12345",
        discord_webhook_url="https://discord.com/api/webhooks/fake",
        throttle_seconds=300,
        max_history=100,
    )


@pytest.fixture
def throttled_svc():
    """Service with a short throttle window for testing."""
    return NotificationService(
        enabled=True,
        provider="telegram",
        telegram_bot_token="fake-token",
        telegram_chat_id="12345",
        throttle_seconds=60,
    )


# ── Enum sanity tests ───────────────────────────────────────────────

def test_alert_level_values():
    assert AlertLevel.DEBUG == "debug"
    assert AlertLevel.INFO == "info"
    assert AlertLevel.WARNING == "warning"
    assert AlertLevel.CRITICAL == "critical"


def test_alert_category_values():
    assert AlertCategory.TRADE == "trade"
    assert AlertCategory.POSITION == "position"
    assert AlertCategory.RISK == "risk"
    assert AlertCategory.SYSTEM == "system"
    assert AlertCategory.MARKET == "market"
    assert AlertCategory.REPORT == "report"


# ── Disabled service ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_send_returns_false(disabled_svc):
    result = await disabled_svc.send("test message")
    assert result is False


@pytest.mark.asyncio
async def test_disabled_notify_trade_executed(disabled_svc):
    result = await disabled_svc.notify_trade_executed(
        "AAPL", "BUY", 10, 180.0, "trend_following"
    )
    assert result is False


@pytest.mark.asyncio
async def test_disabled_notify_stop_loss(disabled_svc):
    result = await disabled_svc.notify_stop_loss("AAPL", 10, 180.0, 165.0, -150.0)
    assert result is False


@pytest.mark.asyncio
async def test_disabled_notify_error(disabled_svc):
    result = await disabled_svc.notify_error("Connection timeout")
    assert result is False


@pytest.mark.asyncio
async def test_disabled_notify_risk_breach(disabled_svc):
    result = await disabled_svc.notify_risk_breach(
        "daily_loss_limit", "Daily loss limit exceeded"
    )
    assert result is False


@pytest.mark.asyncio
async def test_disabled_notify_daily_summary(disabled_svc):
    result = await disabled_svc.notify_daily_summary(50000.0, 250.0, 3, 0.65)
    assert result is False


# ── Backward compatibility ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_legacy_send_telegram(telegram_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await telegram_svc.send("Hello", AlertLevel.INFO)
    assert result is True


@pytest.mark.asyncio
async def test_legacy_send_telegram_failure(telegram_svc):
    session = _mock_session(telegram_status=400)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await telegram_svc.send("Hello")
    assert result is False


@pytest.mark.asyncio
async def test_legacy_send_discord(discord_svc):
    session = _mock_session(discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await discord_svc.send("Hello", AlertLevel.WARNING)
    assert result is True


@pytest.mark.asyncio
async def test_legacy_notify_trade(telegram_svc):
    """notify_trade is a backward-compat alias for notify_trade_executed."""
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await telegram_svc.notify_trade(
            "AAPL", "BUY", 10, 180.0, "trend_following"
        )
    assert result is True


@pytest.mark.asyncio
async def test_telegram_no_credentials():
    svc = NotificationService(enabled=True, provider="telegram")
    result = await svc.send("test")
    assert result is False


@pytest.mark.asyncio
async def test_discord_no_url():
    svc = NotificationService(enabled=True, provider="discord")
    result = await svc.send("test")
    assert result is False


@pytest.mark.asyncio
async def test_unknown_provider():
    svc = NotificationService(enabled=True, provider="slack")
    result = await svc.send("test")
    assert result is False


# ── Multi-provider dispatch ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_provider_both_succeed(multi_svc):
    session = _mock_session(telegram_status=200, discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await multi_svc.notify_trade_executed(
            "TSLA", "BUY", 5, 250.0, "momentum"
        )
    assert result is True
    # session.post should have been called twice (telegram + discord)
    assert session.post.call_count == 2


@pytest.mark.asyncio
async def test_multi_provider_telegram_fails_discord_ok(multi_svc):
    session = _mock_session(telegram_status=500, discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await multi_svc.notify_error("db connection lost")
    # Still True because Discord succeeded
    assert result is True


@pytest.mark.asyncio
async def test_multi_provider_both_fail(multi_svc):
    session = _mock_session(telegram_status=500, discord_status=500)
    with patch("aiohttp.ClientSession", return_value=session):
        result = await multi_svc.notify_system_event("startup", "Engine started")
    assert result is False


@pytest.mark.asyncio
async def test_no_providers_configured():
    svc = NotificationService(enabled=True, provider="telegram")
    # No token/chat_id and no discord url -> no providers configured
    result = await svc.notify_trade_executed("AAPL", "BUY", 1, 100.0, "test")
    assert result is False


# ── Throttling ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_throttle_suppresses_duplicate(throttled_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        r1 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 10, 180.0, "trend"
        )
        r2 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 5, 181.0, "trend"
        )
    assert r1 is True
    assert r2 is False  # throttled


@pytest.mark.asyncio
async def test_throttle_different_symbols_not_throttled(throttled_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        r1 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 10, 180.0, "trend"
        )
        r2 = await throttled_svc.notify_trade_executed(
            "TSLA", "BUY", 5, 250.0, "momentum"
        )
    assert r1 is True
    assert r2 is True


@pytest.mark.asyncio
async def test_throttle_different_categories_not_throttled(throttled_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        r1 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 10, 180.0, "trend"
        )
        r2 = await throttled_svc.notify_stop_loss("AAPL", 10, 180.0, 165.0, -150.0)
    assert r1 is True
    assert r2 is True  # different category (TRADE vs POSITION)


@pytest.mark.asyncio
async def test_throttle_bypass_for_critical(throttled_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        r1 = await throttled_svc.notify_error("First error")
        r2 = await throttled_svc.notify_error("Second error")
    # Both should succeed because CRITICAL bypasses throttle
    assert r1 is True
    assert r2 is True


@pytest.mark.asyncio
async def test_throttle_expires(throttled_svc):
    """After throttle window passes, same alert should go through."""
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        r1 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 10, 180.0, "trend"
        )
        assert r1 is True

        # Manually expire the throttle by manipulating _recent timestamps
        for key in throttled_svc._recent:
            throttled_svc._recent[key] -= 120  # shift back 120 seconds (> 60s throttle)

        r2 = await throttled_svc.notify_trade_executed(
            "AAPL", "BUY", 5, 181.0, "trend"
        )
        assert r2 is True


# ── Alert history ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_records_alerts(multi_svc):
    session = _mock_session(telegram_status=200, discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        await multi_svc.notify_trade_executed("AAPL", "BUY", 10, 180.0, "trend")
        await multi_svc.notify_stop_loss("TSLA", 5, 250.0, 230.0, -100.0)

    history = multi_svc.history
    assert len(history) == 2
    # Newest first
    assert history[0].category == AlertCategory.POSITION
    assert history[0].symbol == "TSLA"
    assert history[1].category == AlertCategory.TRADE
    assert history[1].symbol == "AAPL"


@pytest.mark.asyncio
async def test_history_max_size():
    svc = NotificationService(
        enabled=True,
        provider="telegram",
        telegram_bot_token="tok",
        telegram_chat_id="123",
        max_history=3,
        throttle_seconds=0,  # no throttle
    )
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        for i in range(5):
            await svc.notify_trade_executed(
                f"SYM{i}", "BUY", 1, 100.0 + i, "test"
            )
    assert len(svc.history) == 3
    # Should have the last 3 (SYM2, SYM3, SYM4), newest first
    assert svc.history[0].symbol == "SYM4"
    assert svc.history[1].symbol == "SYM3"
    assert svc.history[2].symbol == "SYM2"


@pytest.mark.asyncio
async def test_history_not_recorded_when_disabled(disabled_svc):
    await disabled_svc.notify_trade_executed("AAPL", "BUY", 10, 180.0, "test")
    assert len(disabled_svc.history) == 0


@pytest.mark.asyncio
async def test_history_not_recorded_when_throttled(throttled_svc):
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        await throttled_svc.notify_trade_executed("AAPL", "BUY", 10, 180.0, "test")
        await throttled_svc.notify_trade_executed("AAPL", "BUY", 5, 181.0, "test")
    # Only 1 recorded (the second was throttled)
    assert len(throttled_svc.history) == 1


def test_alert_record_to_dict():
    from datetime import datetime, timezone
    ts = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    record = AlertRecord(
        timestamp=ts,
        category=AlertCategory.TRADE,
        level=AlertLevel.INFO,
        symbol="AAPL",
        message="Trade Executed",
        data={"price": 180.0},
    )
    d = record.to_dict()
    assert d["category"] == "trade"
    assert d["level"] == "info"
    assert d["symbol"] == "AAPL"
    assert d["data"]["price"] == 180.0
    assert "2026-01-15" in d["timestamp"]


# ── Template / formatting tests (adapter-agnostic) ──────────────────

@pytest.mark.asyncio
async def test_trade_executed_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_trade_executed("NVDA", "BUY", 20, 450.50, "breakout")

    assert len(adapter.sent_rich) == 1
    title, body, level, fields = adapter.sent_rich[0]
    assert "Trade Executed" in title
    assert "NVDA" in body
    assert "x20" in body
    assert "$450.50" in body
    assert "breakout" in body
    assert level == AlertLevel.INFO
    assert fields["Symbol"] == "NVDA"
    assert fields["Side"] == "BUY"


@pytest.mark.asyncio
async def test_stop_loss_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_stop_loss("AAPL", 10, 180.0, 165.0, -150.0)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Stop-Loss" in title
    assert "AAPL" in body
    assert "x10" in body
    assert "$180.00" in body
    assert "$165.00" in body
    assert "-$150.00" in body
    assert level == AlertLevel.WARNING


@pytest.mark.asyncio
async def test_take_profit_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_take_profit("MSFT", 15, 300.0, 350.0, 750.0)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Take-Profit" in title
    assert "+$750.00" in body


@pytest.mark.asyncio
async def test_profit_taking_content():
    """notify_profit_taking shows partial sell details with gain %."""
    svc, adapter = _svc_with_mock()
    # Sold 10 shares, 10 remaining, entry $100 -> exit $115 (15% gain), P&L $150
    await svc.notify_profit_taking("AAPL", 10, 100.0, 115.0, 150.0, 10)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Profit-Taking" in title
    assert "Partial Sell" in title
    assert "AAPL" in body
    assert "x10" in body
    assert "remaining 10" in body
    assert "$100.00" in body
    assert "$115.00" in body
    assert "+15.0%" in body
    assert "+$150.00" in body
    assert level == AlertLevel.INFO
    # Check fields
    assert fields["Sold"] == "10"
    assert fields["Remaining"] == "10"
    assert fields["Gain"] == "+15.0%"
    assert fields["P&L"] == "+$150.00"


@pytest.mark.asyncio
async def test_profit_taking_kr_market():
    """notify_profit_taking formats KR market prices with won symbol."""
    svc, adapter = _svc_with_mock()
    # KR stock (numeric symbol), sold 5, remaining 15, entry 50000 -> exit 57500 (15%)
    await svc.notify_profit_taking("005930", 5, 50000.0, 57500.0, 37500.0, 15)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Profit-Taking" in title
    assert "005930" in body
    assert "x5" in body
    assert "remaining 15" in body
    assert "+15.0%" in body
    assert "+\u20a937,500" in body  # won currency


@pytest.mark.asyncio
async def test_profit_taking_negative_gain():
    """notify_profit_taking handles negative gain (shouldn't normally happen but be safe)."""
    svc, adapter = _svc_with_mock()
    await svc.notify_profit_taking("TSLA", 5, 200.0, 190.0, -50.0, 5)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Profit-Taking" in title
    assert "-5.0%" in body
    assert "-$50.00" in body


@pytest.mark.asyncio
async def test_profit_taking_zero_entry():
    """notify_profit_taking handles zero entry price without division error."""
    svc, adapter = _svc_with_mock()
    await svc.notify_profit_taking("TEST", 1, 0.0, 10.0, 10.0, 1)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Profit-Taking" in title
    assert "+0.0%" in body


@pytest.mark.asyncio
async def test_profit_taking_data_dict():
    """notify_profit_taking stores gain_pct and remaining_qty in alert data."""
    svc, adapter = _svc_with_mock()
    await svc.notify_profit_taking("MSFT", 10, 200.0, 240.0, 400.0, 10)

    record = svc.history[0]
    assert record.data["qty"] == 10
    assert record.data["remaining_qty"] == 10
    assert record.data["gain_pct"] == pytest.approx(20.0)
    assert record.data["entry"] == 200.0
    assert record.data["exit"] == 240.0
    assert record.data["pnl"] == 400.0


@pytest.mark.asyncio
async def test_trailing_stop_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_trailing_stop("GOOG", 8, 140.0, 155.0, 170.0, 120.0)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Trailing-Stop" in title
    assert "$170.00" in body  # highest
    assert "+$120.00" in body


@pytest.mark.asyncio
async def test_order_rejected_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_order_rejected("AMZN", "Insufficient buying power")

    title, body, level, fields = adapter.sent_rich[0]
    assert "Order Rejected" in title
    assert "AMZN" in body
    assert "Insufficient buying power" in body


@pytest.mark.asyncio
async def test_risk_breach_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_risk_breach(
        "daily_loss_limit",
        "Daily loss exceeded 3%",
        {"current_loss": "-3.2%", "limit": "3.0%"},
    )

    title, body, level, fields = adapter.sent_rich[0]
    assert "daily_loss_limit" in title
    assert "Daily loss exceeded 3%" in body
    assert level == AlertLevel.CRITICAL
    assert fields is not None
    assert "current_loss" in fields


@pytest.mark.asyncio
async def test_risk_breach_no_details():
    svc, adapter = _svc_with_mock()
    await svc.notify_risk_breach("max_drawdown", "Drawdown limit breached")

    title, body, level, fields = adapter.sent_rich[0]
    assert "max_drawdown" in title or "max_drawdown" in body


@pytest.mark.asyncio
async def test_system_event_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_system_event("startup", "Engine v2.1 started")

    title, body, level, fields = adapter.sent_rich[0]
    assert "Startup" in title
    assert "Engine v2.1 started" in body
    assert level == AlertLevel.INFO


@pytest.mark.asyncio
async def test_system_event_error_level():
    """Error and health_degraded events should use WARNING level."""
    svc, adapter = _svc_with_mock()
    await svc.notify_system_event("error", "Something bad")
    await svc.notify_system_event("startup", "Restarted")

    assert adapter.sent_rich[0][2] == AlertLevel.WARNING
    assert adapter.sent_rich[1][2] == AlertLevel.INFO


@pytest.mark.asyncio
async def test_market_event_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_market_event(
        "regime_change", {"from": "bull", "to": "bear", "vix": 35.2}
    )

    title, body, level, fields = adapter.sent_rich[0]
    assert "Regime Change" in title
    assert "bull" in body
    assert "bear" in body


@pytest.mark.asyncio
async def test_market_event_no_details():
    svc, adapter = _svc_with_mock()
    await svc.notify_market_event("market_open")

    title, body, level, fields = adapter.sent_rich[0]
    assert "Market Open" in title


@pytest.mark.asyncio
async def test_daily_summary_content():
    svc, adapter = _svc_with_mock()
    await svc.notify_daily_summary(52000.0, 350.0, 5, 0.72)

    title, body, level, fields = adapter.sent_rich[0]
    assert "Daily Summary" in title
    assert "$52,000.00" in body
    assert "+$350.00" in body
    assert "72.0%" in body
    assert fields is not None
    assert fields["Positions"] == "5"


@pytest.mark.asyncio
async def test_daily_summary_negative_pnl():
    svc, adapter = _svc_with_mock()
    await svc.notify_daily_summary(48000.0, -500.0, 3, 0.40)

    _, body, _, _ = adapter.sent_rich[0]
    assert "-$500.00" in body


@pytest.mark.asyncio
async def test_error_with_traceback():
    svc, adapter = _svc_with_mock()
    tb = "Traceback (most recent call last):\n  File ...\nValueError: bad value"
    await svc.notify_error("Something broke", traceback_str=tb)

    title, body, level, fields = adapter.sent_rich[0]
    assert "System Error" in title
    assert "Something broke" in body
    assert "ValueError" in body
    assert level == AlertLevel.CRITICAL


@pytest.mark.asyncio
async def test_error_without_traceback():
    svc, adapter = _svc_with_mock()
    await svc.notify_error("Connection timeout")

    title, body, level, fields = adapter.sent_rich[0]
    assert "System Error" in title
    assert "Connection timeout" in body


# ── Adapter-specific formatting ─────────────────────────────────────

@pytest.mark.asyncio
async def test_telegram_adapter_html_formatting():
    """TelegramAdapter.send_rich wraps title in <b> tags."""
    adapter = TelegramAdapter(bot_token="tok", chat_id="123")
    session = _mock_session(telegram_status=200)
    with patch("aiohttp.ClientSession", return_value=session):
        await adapter.send_rich("Trade Executed", "BUY AAPL", AlertLevel.INFO,
                                {"Symbol": "AAPL"})

    # Inspect the payload sent to Telegram
    call_args = session.post.call_args
    payload = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["json"]
    text = payload["text"]
    assert "<b>Trade Executed</b>" in text
    assert "BUY AAPL" in text
    assert "<b>Symbol:</b>" in text
    assert payload["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_discord_adapter_uses_embeds():
    """DiscordAdapter.send_rich uses Discord embeds, no HTML."""
    adapter = DiscordAdapter(webhook_url="https://discord.com/api/webhooks/fake")
    session = _mock_session(discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        await adapter.send_rich("Trade Executed", "BUY AAPL", AlertLevel.INFO,
                                {"Symbol": "AAPL"})

    call_args = session.post.call_args
    payload = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["json"]
    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert "Trade Executed" in embed["title"]
    assert "BUY AAPL" in embed["description"]
    # No HTML tags in Discord
    assert "<b>" not in embed["description"]


@pytest.mark.asyncio
async def test_discord_plain_text_no_html():
    """Discord adapter send() produces plain text without HTML."""
    adapter = DiscordAdapter(webhook_url="https://discord.com/api/webhooks/fake")
    session = _mock_session(discord_status=204)
    with patch("aiohttp.ClientSession", return_value=session):
        await adapter.send("BUY AAPL x10 @ $180.00", AlertLevel.INFO)

    call_args = session.post.call_args
    payload = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["json"]
    assert "content" in payload
    assert "<b>" not in payload["content"]
    assert "AAPL" in payload["content"]


@pytest.mark.asyncio
async def test_slack_adapter_uses_attachments():
    """SlackAdapter.send_rich uses Slack attachments."""
    adapter = SlackAdapter(webhook_url="https://hooks.slack.com/services/fake")
    session = _mock_session(telegram_status=200)  # Slack returns 200
    with patch("aiohttp.ClientSession", return_value=session):
        await adapter.send_rich("Trade Executed", "BUY AAPL", AlertLevel.WARNING,
                                {"Symbol": "AAPL"})

    call_args = session.post.call_args
    payload = call_args[1].get("json") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["json"]
    assert "attachments" in payload
    att = payload["attachments"][0]
    assert "Trade Executed" in att["title"]
    assert att["color"] == "#F59E0B"  # WARNING color


# ── Multi-adapter dispatch with mock adapters ───────────────────────

@pytest.mark.asyncio
async def test_multi_adapter_dispatch():
    """Both adapters receive the send_rich call."""
    svc = NotificationService(enabled=True, throttle_seconds=0)
    a1 = MockAdapter(name_val="primary")
    a2 = MockAdapter(name_val="secondary")
    svc.add_adapter(a1)
    svc.add_adapter(a2)

    await svc.notify_trade_executed("AAPL", "BUY", 10, 180.0, "trend")

    assert len(a1.sent_rich) == 1
    assert len(a2.sent_rich) == 1
    # Both got the same content
    assert a1.sent_rich[0][1] == a2.sent_rich[0][1]


@pytest.mark.asyncio
async def test_adapter_exception_does_not_block_others():
    """If one adapter raises, other adapters still receive the message."""
    svc = NotificationService(enabled=True, throttle_seconds=0)
    failing = MockAdapter(name_val="failing", fail=True)
    working = MockAdapter(name_val="working")
    svc.add_adapter(failing)
    svc.add_adapter(working)

    result = await svc.notify_system_event("startup", "test")

    assert result is True
    assert len(working.sent_rich) == 1


@pytest.mark.asyncio
async def test_unconfigured_adapter_skipped():
    """Unconfigured adapters are not called."""
    svc = NotificationService(enabled=True, throttle_seconds=0)
    configured = MockAdapter(name_val="ok", configured=True)
    unconfigured = MockAdapter(name_val="skip", configured=False)
    svc.add_adapter(configured)
    svc.add_adapter(unconfigured)

    await svc.notify_trade_executed("AAPL", "BUY", 1, 100.0, "test")

    assert len(configured.sent_rich) == 1
    assert len(unconfigured.sent_rich) == 0


# ── Adapter management ──────────────────────────────────────────────

def test_add_adapter():
    svc = NotificationService(enabled=True)
    svc.add_adapter(MockAdapter(name_val="test"))
    assert "test" in svc.adapter_names


def test_remove_adapter():
    svc = NotificationService(enabled=True)
    svc.add_adapter(MockAdapter(name_val="test"))
    svc.remove_adapter("test")
    assert "test" not in svc.adapter_names


def test_adapter_names_only_configured():
    svc = NotificationService(enabled=True)
    svc.add_adapter(MockAdapter(name_val="ok", configured=True))
    svc.add_adapter(MockAdapter(name_val="no", configured=False))
    assert svc.adapter_names == ["ok"]


# ── Adapter is_configured ───────────────────────────────────────────

def test_discord_adapter_not_configured():
    assert DiscordAdapter().is_configured is False
    assert DiscordAdapter("").is_configured is False
    assert DiscordAdapter("https://url").is_configured is True


def test_telegram_adapter_not_configured():
    assert TelegramAdapter().is_configured is False
    assert TelegramAdapter("tok", "").is_configured is False
    assert TelegramAdapter("tok", "123").is_configured is True


def test_slack_adapter_not_configured():
    assert SlackAdapter().is_configured is False
    assert SlackAdapter("https://url").is_configured is True


# ── Edge cases ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_exception_does_not_propagate(telegram_svc):
    """If the HTTP call raises, send should return False, not propagate."""
    with patch("aiohttp.ClientSession", side_effect=Exception("network down")):
        result = await telegram_svc.send("test", AlertLevel.INFO)
    assert result is False


@pytest.mark.asyncio
async def test_pnl_sign_formatting():
    assert NotificationService._pnl_sign(100.0) == "+$100.00"
    assert NotificationService._pnl_sign(-50.5) == "-$50.50"
    assert NotificationService._pnl_sign(0.0) == "+$0.00"


@pytest.mark.asyncio
async def test_pnl_sign_formatting_kr():
    assert NotificationService._pnl_sign(100000.0, "KR") == "+₩100,000"
    assert NotificationService._pnl_sign(-50000.0, "KR") == "-₩50,000"
    assert NotificationService._pnl_sign(0.0, "KR") == "+₩0"


@pytest.mark.asyncio
async def test_fmt_price():
    assert NotificationService._fmt_price(150.50) == "$150.50"
    assert NotificationService._fmt_price(72300.0, "KR") == "₩72,300"
    assert NotificationService._fmt_price(1500000.0, "KR") == "₩1,500,000"


@pytest.mark.asyncio
async def test_detect_market():
    assert NotificationService._detect_market("AAPL") == "US"
    assert NotificationService._detect_market("005930") == "KR"
    assert NotificationService._detect_market("TSLA") == "US"
    assert NotificationService._detect_market("069500") == "KR"


@pytest.mark.asyncio
async def test_kr_trade_notification_uses_won():
    svc, adapter = _svc_with_mock()
    await svc.notify_trade_executed("005930", "BUY", 10, 72300.0, "trend", market="KR")
    title, body, level, fields = adapter.sent_rich[0]
    assert "₩72,300" in body
    assert "$" not in body


@pytest.mark.asyncio
async def test_kr_trade_auto_detected():
    """KR symbol (numeric) auto-detects market without explicit param."""
    svc, adapter = _svc_with_mock()
    await svc.notify_trade_executed("005930", "BUY", 10, 72300.0, "trend")
    title, body, level, fields = adapter.sent_rich[0]
    assert "₩72,300" in body


@pytest.mark.asyncio
async def test_us_trade_still_uses_dollar():
    svc, adapter = _svc_with_mock()
    await svc.notify_trade_executed("AAPL", "BUY", 10, 150.50, "trend")
    title, body, level, fields = adapter.sent_rich[0]
    assert "$150.50" in body


@pytest.mark.asyncio
async def test_pct_formatting():
    assert NotificationService._pct(0.65) == "65.0%"
    assert NotificationService._pct(1.0) == "100.0%"
    assert NotificationService._pct(0.0) == "0.0%"


# ── Constructor defaults ────────────────────────────────────────────

def test_default_throttle_seconds():
    svc = NotificationService()
    assert svc.throttle_seconds == 300


def test_custom_throttle_seconds():
    svc = NotificationService(throttle_seconds=60)
    assert svc.throttle_seconds == 60


def test_default_history_empty():
    svc = NotificationService()
    assert svc.history == []


# ── Throttle internal logic unit tests ──────────────────────────────

def test_is_throttled_returns_false_first_time():
    svc = NotificationService(enabled=True, throttle_seconds=300)
    assert svc._is_throttled(AlertCategory.TRADE, "AAPL", AlertLevel.INFO) is False


def test_is_throttled_returns_true_second_time():
    svc = NotificationService(enabled=True, throttle_seconds=300)
    svc._is_throttled(AlertCategory.TRADE, "AAPL", AlertLevel.INFO)
    assert svc._is_throttled(AlertCategory.TRADE, "AAPL", AlertLevel.INFO) is True


def test_is_throttled_critical_bypass():
    svc = NotificationService(enabled=True, throttle_seconds=300)
    svc._is_throttled(AlertCategory.SYSTEM, "", AlertLevel.CRITICAL)
    # Second CRITICAL should NOT be throttled
    assert svc._is_throttled(AlertCategory.SYSTEM, "", AlertLevel.CRITICAL) is False


def test_is_throttled_different_key_not_blocked():
    svc = NotificationService(enabled=True, throttle_seconds=300)
    svc._is_throttled(AlertCategory.TRADE, "AAPL", AlertLevel.INFO)
    assert svc._is_throttled(AlertCategory.TRADE, "TSLA", AlertLevel.INFO) is False


# ── Legacy constructor auto-registers adapters ──────────────────────

def test_legacy_constructor_registers_discord():
    svc = NotificationService(
        enabled=True,
        discord_webhook_url="https://discord.com/api/webhooks/test",
    )
    assert "discord" in svc.adapter_names


def test_legacy_constructor_registers_telegram():
    svc = NotificationService(
        enabled=True,
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    assert "telegram" in svc.adapter_names


def test_legacy_constructor_registers_both():
    svc = NotificationService(
        enabled=True,
        telegram_bot_token="tok",
        telegram_chat_id="123",
        discord_webhook_url="https://discord.com/api/webhooks/test",
    )
    names = svc.adapter_names
    assert "discord" in names
    assert "telegram" in names


# ── notify_system_error ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notify_system_error():
    svc, adapter = _svc_with_mock()
    await svc.notify_system_error("scheduler", "Task failed", "retry exhausted")

    title, body, level, fields = adapter.sent_rich[0]
    assert "scheduler" in title
    assert "Task failed" in body
    assert "retry exhausted" in body
    assert level == AlertLevel.CRITICAL
