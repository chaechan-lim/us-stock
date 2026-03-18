"""Notification service with pluggable provider adapters.

Architecture:
  NotificationService (orchestrator)
    └── NotificationAdapter (abstract)
         ├── DiscordAdapter   (primary)
         ├── TelegramAdapter
         └── SlackAdapter

Features:
- Multi-provider: sends to all registered adapters simultaneously.
- Throttling: suppresses duplicate category+symbol alerts within a window.
  CRITICAL alerts bypass throttling.
- Alert history: keeps the last *max_history* alerts in memory.
- Adapter pattern: add new providers by subclassing NotificationAdapter.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    TRADE = "trade"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    MARKET = "market"
    REPORT = "report"


_LEVEL_PREFIX = {
    AlertLevel.DEBUG: "[DEBUG]",
    AlertLevel.INFO: "[INFO]",
    AlertLevel.WARNING: "[WARNING]",
    AlertLevel.CRITICAL: "[CRITICAL]",
}


# ── Alert record ───────────────────────────────────────────────────────

@dataclass
class AlertRecord:
    timestamp: datetime
    category: AlertCategory
    level: AlertLevel
    symbol: str
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "level": self.level.value,
            "symbol": self.symbol,
            "message": self.message,
            "data": self.data,
        }


# ── Adapter interface ──────────────────────────────────────────────────

class NotificationAdapter(ABC):
    """Abstract base for notification providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'discord', 'telegram', 'slack')."""
        ...

    @abstractmethod
    async def send(self, message: str, level: AlertLevel) -> bool:
        """Send a plain-text message. Returns True on success."""
        ...

    @abstractmethod
    async def send_rich(
        self, title: str, body: str, level: AlertLevel, fields: dict | None = None,
    ) -> bool:
        """Send a rich/formatted message. Returns True on success."""
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this adapter has valid credentials."""
        ...


# ── Discord adapter ───────────────────────────────────────────────────

_DISCORD_COLORS = {
    AlertLevel.DEBUG: 0x808080,
    AlertLevel.INFO: 0x3B82F6,
    AlertLevel.WARNING: 0xF59E0B,
    AlertLevel.CRITICAL: 0xEF4444,
}


class DiscordAdapter(NotificationAdapter):
    """Discord webhook adapter with embed support."""

    def __init__(self, webhook_url: str = ""):
        self._webhook_url = webhook_url

    @property
    def name(self) -> str:
        return "discord"

    @property
    def is_configured(self) -> bool:
        return bool(self._webhook_url)

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if not self.is_configured:
            return False
        payload = {"content": f"{_LEVEL_PREFIX.get(level, '')} {message}"}
        return await self._post(payload)

    async def send_rich(
        self, title: str, body: str, level: AlertLevel, fields: dict | None = None,
    ) -> bool:
        if not self.is_configured:
            return False

        embed: dict[str, Any] = {
            "title": f"{_LEVEL_PREFIX.get(level, '')} {title}",
            "description": body,
            "color": _DISCORD_COLORS.get(level, 0x3B82F6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in fields.items()
            ]

        return await self._post({"embeds": [embed]})

    async def _post(self, payload: dict) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 204):
                        return True
                    body = await resp.text()
                    logger.error("Discord error %d: %s", resp.status, body)
                    return False
        except Exception as e:
            logger.error("Discord send failed: %s", e)
            return False


# ── Telegram adapter ──────────────────────────────────────────────────

class TelegramAdapter(NotificationAdapter):
    """Telegram Bot API adapter with HTML formatting."""

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self._bot_token = bot_token
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_configured(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if not self.is_configured:
            return False
        prefix = _LEVEL_PREFIX.get(level, "")
        return await self._post(f"{prefix} {message}")

    async def send_rich(
        self, title: str, body: str, level: AlertLevel, fields: dict | None = None,
    ) -> bool:
        if not self.is_configured:
            return False
        prefix = _LEVEL_PREFIX.get(level, "")
        html = f"{prefix} <b>{title}</b>\n{body}"
        if fields:
            html += "\n" + "\n".join(f"<b>{k}:</b> {v}" for k, v in fields.items())
        return await self._post(html)

    async def _post(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.error("Telegram error %d: %s", resp.status, body)
                    return False
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False


# ── Slack adapter ─────────────────────────────────────────────────────

_SLACK_COLORS = {
    AlertLevel.DEBUG: "#808080",
    AlertLevel.INFO: "#3B82F6",
    AlertLevel.WARNING: "#F59E0B",
    AlertLevel.CRITICAL: "#EF4444",
}


class SlackAdapter(NotificationAdapter):
    """Slack incoming webhook adapter with attachment support."""

    def __init__(self, webhook_url: str = ""):
        self._webhook_url = webhook_url

    @property
    def name(self) -> str:
        return "slack"

    @property
    def is_configured(self) -> bool:
        return bool(self._webhook_url)

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if not self.is_configured:
            return False
        prefix = _LEVEL_PREFIX.get(level, "")
        payload = {"text": f"{prefix} {message}"}
        return await self._post(payload)

    async def send_rich(
        self, title: str, body: str, level: AlertLevel, fields: dict | None = None,
    ) -> bool:
        if not self.is_configured:
            return False
        prefix = _LEVEL_PREFIX.get(level, "")
        attachment: dict[str, Any] = {
            "color": _SLACK_COLORS.get(level, "#3B82F6"),
            "title": f"{prefix} {title}",
            "text": body,
            "ts": int(time.time()),
        }
        if fields:
            attachment["fields"] = [
                {"title": k, "value": str(v), "short": True}
                for k, v in fields.items()
            ]
        return await self._post({"attachments": [attachment]})

    async def _post(self, payload: dict) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.error("Slack error %d: %s", resp.status, body)
                    return False
        except Exception as e:
            logger.error("Slack send failed: %s", e)
            return False


# ── Notification service (orchestrator) ────────────────────────────────

class NotificationService:
    """Multi-provider notification orchestrator.

    Usage:
        svc = NotificationService(enabled=True)
        svc.add_adapter(DiscordAdapter(webhook_url="..."))
        svc.add_adapter(TelegramAdapter(bot_token="...", chat_id="..."))
        await svc.notify_trade_executed("AAPL", "BUY", 10, 150.0, "trend")

    Backward-compatible constructor also accepts legacy kwargs.
    """

    def __init__(
        self,
        enabled: bool = False,
        # Legacy kwargs (still work for backward compat)
        provider: str = "",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook_url: str = "",
        slack_webhook_url: str = "",
        *,
        throttle_seconds: int = 300,
        max_history: int = 100,
    ):
        self._enabled = enabled
        self._adapters: list[NotificationAdapter] = []

        # Throttling
        self._throttle_seconds = throttle_seconds
        self._recent: dict[str, float] = {}

        # Alert history
        self._history: deque[AlertRecord] = deque(maxlen=max_history)

        # Auto-register adapters from legacy kwargs
        if discord_webhook_url:
            self._adapters.append(DiscordAdapter(discord_webhook_url))
        if telegram_bot_token and telegram_chat_id:
            self._adapters.append(TelegramAdapter(telegram_bot_token, telegram_chat_id))
        if slack_webhook_url:
            self._adapters.append(SlackAdapter(slack_webhook_url))

    # ── Adapter management ─────────────────────────────────────────────

    def add_adapter(self, adapter: NotificationAdapter) -> None:
        """Register an additional notification adapter."""
        self._adapters.append(adapter)
        logger.info("Notification adapter added: %s", adapter.name)

    def remove_adapter(self, name: str) -> None:
        """Remove adapter by name."""
        self._adapters = [a for a in self._adapters if a.name != name]

    @property
    def adapter_names(self) -> list[str]:
        return [a.name for a in self._adapters if a.is_configured]

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def history(self) -> list[AlertRecord]:
        return list(reversed(self._history))

    @property
    def throttle_seconds(self) -> int:
        return self._throttle_seconds

    # ── Throttle logic ─────────────────────────────────────────────────

    def _is_throttled(
        self, category: AlertCategory, symbol: str, level: AlertLevel,
    ) -> bool:
        if level == AlertLevel.CRITICAL:
            return False
        key = f"{category.value}:{symbol or level.value}"
        now = time.monotonic()
        last = self._recent.get(key)
        if last is not None and (now - last) < self._throttle_seconds:
            return True
        self._recent[key] = now
        return False

    # ── Core dispatch ──────────────────────────────────────────────────

    async def _dispatch(
        self,
        category: AlertCategory,
        level: AlertLevel,
        symbol: str,
        title: str,
        plain_text: str,
        data: dict | None = None,
        fields: dict | None = None,
    ) -> bool:
        """Send to all configured adapters, record history, apply throttle."""
        if not self._enabled:
            logger.debug("Notification disabled, skipping: %s", plain_text[:80])
            return False

        if self._is_throttled(category, symbol, level):
            return False

        # Record
        self._history.append(AlertRecord(
            timestamp=datetime.now(timezone.utc),
            category=category,
            level=level,
            symbol=symbol,
            message=plain_text,
            data=data or {},
        ))

        configured = [a for a in self._adapters if a.is_configured]
        if not configured:
            logger.warning("No notification adapters configured")
            return False

        success = False
        for adapter in configured:
            try:
                result = await adapter.send_rich(title, plain_text, level, fields)
                if result:
                    success = True
            except Exception as e:
                logger.error("Adapter %s failed: %s", adapter.name, e)

        if not success and configured:
            logger.critical(
                "ALL notification adapters failed for: %s — %s",
                title, plain_text[:100],
            )

        return success

    # ── Backward-compatible send ───────────────────────────────────────

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Legacy send interface."""
        if not self._enabled:
            return False
        configured = [a for a in self._adapters if a.is_configured]
        if not configured:
            return False
        success = False
        for adapter in configured:
            try:
                if await adapter.send(message, level):
                    success = True
            except Exception as e:
                logger.error("Adapter %s send failed: %s", adapter.name, e)
        return success

    # ── Formatting helpers ─────────────────────────────────────────────

    @staticmethod
    def _fmt_price(value: float, market: str = "US") -> str:
        """Format price with currency symbol based on market."""
        if market == "KR":
            return f"₩{value:,.0f}"
        return f"${value:,.2f}"

    @staticmethod
    def _pnl_sign(value: float, market: str = "US") -> str:
        if market == "KR":
            return f"+₩{value:,.0f}" if value >= 0 else f"-₩{abs(value):,.0f}"
        return f"+${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"

    @staticmethod
    def _pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    # ── Symbol label helper ────────────────────────────────────────────

    @staticmethod
    def _symbol_label(symbol: str) -> str:
        """Return 'SYMBOL (Name)' if name is known, else just 'SYMBOL'."""
        from data.stock_name_service import get_name
        name = get_name(symbol) or get_name(symbol, "KR")
        return f"{symbol} ({name})" if name else symbol

    @staticmethod
    def _detect_market(symbol: str) -> str:
        """Detect market from symbol (KR stocks are numeric)."""
        return "KR" if symbol.isdigit() else "US"

    # ── Convenience methods ────────────────────────────────────────────

    _SESSION_TAGS = {
        "regular": "",
        "pre_market": "[PRE-MKT] ",
        "after_hours": "[AFTER-HRS] ",
        "extended_nxt": "[NXT] ",
    }

    async def notify_trade_executed(
        self, symbol: str, side: str, qty: int, price: float, strategy: str,
        market: str = "", stop_loss_pct: float = 0.0, take_profit_pct: float = 0.0,
        filled_qty: int = 0, filled_price: float = 0.0, session: str = "regular",
    ) -> bool:
        mkt = market or self._detect_market(symbol)
        label = self._symbol_label(symbol)
        p = self._fmt_price(price, mkt)
        sl_tp = ""
        partial = ""
        session_tag = self._SESSION_TAGS.get(session, "")
        fields: dict = {"Symbol": label, "Side": side.upper(), "Qty": qty,
             "Price": p, "Strategy": strategy}
        if session != "regular":
            fields["Session"] = session_tag.strip()
        if filled_qty and 0 < filled_qty < qty:
            partial = f" | PARTIAL FILL {filled_qty}/{qty}"
            fields["Fill"] = f"{filled_qty}/{qty}"
        if filled_price and filled_price != price:
            fields["Filled Price"] = self._fmt_price(filled_price, mkt)
        if side.upper() == "BUY" and stop_loss_pct > 0:
            sl_price = price * (1 - stop_loss_pct)
            tp_price = price * (1 + take_profit_pct) if take_profit_pct > 0 else 0
            sl_tp = f" | SL {self._fmt_price(sl_price, mkt)} (-{stop_loss_pct*100:.0f}%)"
            if tp_price > 0:
                sl_tp += f" / TP {self._fmt_price(tp_price, mkt)} (+{take_profit_pct*100:.0f}%)"
            fields["SL"] = f"{self._fmt_price(sl_price, mkt)} (-{stop_loss_pct*100:.0f}%)"
            if tp_price > 0:
                fields["TP"] = f"{self._fmt_price(tp_price, mkt)} (+{take_profit_pct*100:.0f}%)"
        return await self._dispatch(
            AlertCategory.TRADE, AlertLevel.INFO, symbol,
            f"{session_tag}Trade Executed",
            f"{session_tag}{side.upper()} {label} x{qty} @ {p} | Strategy: {strategy}{sl_tp}{partial}",
            {"side": side.upper(), "qty": qty, "price": price, "strategy": strategy},
            fields,
        )

    async def notify_order_rejected(
        self, symbol: str, reason: str, market: str = "",
    ) -> bool:
        label = self._symbol_label(symbol)
        return await self._dispatch(
            AlertCategory.TRADE, AlertLevel.WARNING, symbol,
            "Order Rejected",
            f"Order Rejected: {label} | Reason: {reason}",
            {"reason": reason},
            {"Symbol": label, "Reason": reason},
        )

    async def notify_stop_loss(
        self, symbol: str, qty: int, entry: float, exit_price: float, pnl: float,
        market: str = "",
    ) -> bool:
        mkt = market or self._detect_market(symbol)
        label = self._symbol_label(symbol)
        pnl_str = self._pnl_sign(pnl, mkt)
        return await self._dispatch(
            AlertCategory.POSITION, AlertLevel.WARNING, symbol,
            "Stop-Loss Triggered",
            f"SELL {label} x{qty} | {self._fmt_price(entry, mkt)} -> {self._fmt_price(exit_price, mkt)} | P&L: {pnl_str}",
            {"qty": qty, "entry": entry, "exit": exit_price, "pnl": pnl},
            {"Symbol": label, "Entry": self._fmt_price(entry, mkt),
             "Exit": self._fmt_price(exit_price, mkt), "P&L": pnl_str},
        )

    async def notify_take_profit(
        self, symbol: str, qty: int, entry: float, exit_price: float, pnl: float,
        market: str = "",
    ) -> bool:
        mkt = market or self._detect_market(symbol)
        label = self._symbol_label(symbol)
        pnl_str = self._pnl_sign(pnl, mkt)
        return await self._dispatch(
            AlertCategory.POSITION, AlertLevel.INFO, symbol,
            "Take-Profit Hit",
            f"SELL {label} x{qty} | {self._fmt_price(entry, mkt)} -> {self._fmt_price(exit_price, mkt)} | P&L: {pnl_str}",
            {"qty": qty, "entry": entry, "exit": exit_price, "pnl": pnl},
            {"Symbol": label, "Entry": self._fmt_price(entry, mkt),
             "Exit": self._fmt_price(exit_price, mkt), "P&L": pnl_str},
        )

    async def notify_profit_taking(
        self, symbol: str, qty: int, entry: float, exit_price: float, pnl: float,
        remaining_qty: int, market: str = "",
    ) -> bool:
        """Notify partial profit-taking sell with sold/remaining details."""
        mkt = market or self._detect_market(symbol)
        label = self._symbol_label(symbol)
        pnl_str = self._pnl_sign(pnl, mkt)
        gain_pct = ((exit_price - entry) / entry * 100) if entry else 0.0
        fp = self._fmt_price
        return await self._dispatch(
            AlertCategory.POSITION, AlertLevel.INFO, symbol,
            "Profit-Taking (Partial Sell)",
            f"SELL {label} x{qty} (remaining {remaining_qty}) | "
            f"{fp(entry, mkt)} -> {fp(exit_price, mkt)} | "
            f"Gain {gain_pct:+.1f}% | P&L: {pnl_str}",
            {"qty": qty, "remaining_qty": remaining_qty, "entry": entry,
             "exit": exit_price, "pnl": pnl, "gain_pct": gain_pct},
            {"Symbol": label, "Sold": str(qty), "Remaining": str(remaining_qty),
             "Entry": fp(entry, mkt), "Exit": fp(exit_price, mkt),
             "Gain": f"{gain_pct:+.1f}%", "P&L": pnl_str},
        )

    async def notify_trailing_stop(
        self, symbol: str, qty: int, entry: float, exit_price: float,
        highest: float, pnl: float, market: str = "",
    ) -> bool:
        mkt = market or self._detect_market(symbol)
        label = self._symbol_label(symbol)
        pnl_str = self._pnl_sign(pnl, mkt)
        fp = self._fmt_price
        return await self._dispatch(
            AlertCategory.POSITION, AlertLevel.WARNING, symbol,
            "Trailing-Stop Triggered",
            f"SELL {label} x{qty} | Entry {fp(entry, mkt)} | High {fp(highest, mkt)} | "
            f"Exit {fp(exit_price, mkt)} | P&L: {pnl_str}",
            {"qty": qty, "entry": entry, "exit": exit_price, "highest": highest, "pnl": pnl},
            {"Symbol": label, "Entry": fp(entry, mkt), "Highest": fp(highest, mkt),
             "Exit": fp(exit_price, mkt), "P&L": pnl_str},
        )

    async def notify_risk_breach(
        self, category: str, reason: str, details: dict[str, Any] | None = None,
    ) -> bool:
        detail_str = " | ".join(f"{k}={v}" for k, v in (details or {}).items())
        plain = f"Risk Breach [{category}]: {reason}"
        if detail_str:
            plain += f" | {detail_str}"
        return await self._dispatch(
            AlertCategory.RISK, AlertLevel.CRITICAL, "",
            f"Risk Breach: {category}",
            plain,
            {"category": category, "reason": reason, **(details or {})},
            details,
        )

    async def notify_system_event(self, event_type: str, message: str) -> bool:
        level = AlertLevel.WARNING if event_type in ("error", "health_degraded") else AlertLevel.INFO
        return await self._dispatch(
            AlertCategory.SYSTEM, level, "",
            f"System: {event_type.replace('_', ' ').title()}",
            f"System [{event_type}]: {message}",
            {"event_type": event_type},
        )

    async def notify_system_error(
        self, component: str, error: str, details: str = "",
    ) -> bool:
        """Notify of a component error (CRITICAL, bypasses throttle)."""
        plain = f"Error in {component}: {error}"
        if details:
            plain += f" | {details}"
        return await self._dispatch(
            AlertCategory.SYSTEM, AlertLevel.CRITICAL, "",
            f"System Error: {component}",
            plain,
            {"component": component, "error": error, "details": details},
            {"Component": component, "Error": error},
        )

    async def notify_market_event(
        self, event_type: str, details: dict[str, Any] | None = None,
    ) -> bool:
        detail_str = " | ".join(f"{k}={v}" for k, v in (details or {}).items())
        plain = f"Market [{event_type}]"
        if detail_str:
            plain += f": {detail_str}"
        return await self._dispatch(
            AlertCategory.MARKET, AlertLevel.INFO, "",
            f"Market: {event_type.replace('_', ' ').title()}",
            plain,
            {"event_type": event_type, **(details or {})},
            details,
        )

    async def notify_daily_summary(
        self, equity: float, daily_pnl: float, positions: int, win_rate: float = 0.0,
    ) -> bool:
        pnl_str = self._pnl_sign(daily_pnl)
        wr_str = self._pct(win_rate)
        return await self._dispatch(
            AlertCategory.REPORT, AlertLevel.INFO, "",
            "Daily Summary",
            f"Equity: ${equity:,.2f} | P&L: {pnl_str} | Positions: {positions} | Win Rate: {wr_str}",
            {"equity": equity, "daily_pnl": daily_pnl, "positions": positions, "win_rate": win_rate},
            {"Equity": f"${equity:,.2f}", "Daily P&L": pnl_str,
             "Positions": str(positions), "Win Rate": wr_str},
        )

    async def notify_error(self, error: str, traceback_str: str | None = None) -> bool:
        plain = f"System Error: {error}"
        if traceback_str:
            plain += f"\n{traceback_str[-500:]}"
        return await self._dispatch(
            AlertCategory.SYSTEM, AlertLevel.CRITICAL, "",
            "System Error",
            plain,
            {"error": error},
        )

    # ── Legacy alias ───────────────────────────────────────────────────

    async def notify_trade(
        self, symbol: str, side: str, quantity: int, price: float, strategy: str,
    ) -> bool:
        return await self.notify_trade_executed(symbol, side, quantity, price, strategy)
