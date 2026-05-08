"""Centralized structured logging configuration for the trading engine."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 30
    backup_count: int = 3
    enable_file: bool = True
    enable_json: bool = True
    enable_console: bool = True


# ANSI color codes for console output
_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes based on log level."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        # Temporarily modify levelname to include color
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{_RESET}"
        result = super().format(record)
        record.levelname = original_levelname
        return result


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log entries."""

    # Fields that are part of the standard LogRecord and should not appear in extras
    _STANDARD_FIELDS = frozenset({
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "pathname", "thread", "threadName",
        "process", "processName", "levelname", "levelno", "message",
        "msecs", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Collect extra fields attached to the record
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._STANDARD_FIELDS:
                continue
            entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def setup_logging(config: LogConfig | None = None) -> None:
    """Configure the root logger with console, file, and JSON handlers.

    Args:
        config: Logging configuration. Uses defaults if None.
    """
    if config is None:
        config = LogConfig()

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Remove any existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    human_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Console handler with colored output
    if config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(fmt=human_fmt))
        root.addHandler(console_handler)

    # File-based handlers need the log directory
    if config.enable_file or config.enable_json:
        os.makedirs(config.log_dir, exist_ok=True)

    max_bytes = config.max_file_size_mb * 1024 * 1024

    # Rotating human-readable file handler
    if config.enable_file:
        file_handler = RotatingFileHandler(
            filename=os.path.join(config.log_dir, "trading.log"),
            maxBytes=max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(human_fmt))
        root.addHandler(file_handler)

    # Rotating JSON structured file handler
    if config.enable_json:
        json_handler = RotatingFileHandler(
            filename=os.path.join(config.log_dir, "trading.json"),
            maxBytes=max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(JSONFormatter())
        root.addHandler(json_handler)

    # 2026-05-08: Suppress yfinance ERROR-level "possibly delisted" /
    # "No fundamentals data found" noise. yfinance routinely logs ERROR
    # for ETFs (no fundamentals exist), SPACs (no fundamentals yet), and
    # symbols flaky in Yahoo's coverage (some KR tickers). Our scanner /
    # adapter wrappers already swallow the exceptions and decide whether
    # to skip the symbol, so yfinance's own error logs are pure noise.
    # CRITICAL level keeps only true crash-class messages.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class TradingLogger:
    """Convenience wrapper that attaches structured fields to log records.

    Usage::

        log = get_trading_logger(__name__)
        log.trade("AAPL", "BUY", 10, 185.50, "momentum")
        log.signal("TSLA", "ENTRY", 0.87, "mean_reversion")
        log.risk("Daily loss limit approaching", pnl=-450.0)
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def logger(self) -> logging.Logger:
        """Access the underlying stdlib logger."""
        return self._logger

    def trade(
        self,
        symbol: str,
        side: str,
        qty: int | float,
        price: float,
        strategy: str,
        **extra: Any,
    ) -> None:
        """Log a trade execution at INFO level."""
        extra.update(
            symbol=symbol, side=side, qty=qty, price=price, strategy=strategy
        )
        self._log(
            logging.INFO,
            "TRADE %s %s qty=%s price=%.2f strategy=%s",
            symbol, side, qty, price, strategy,
            extra=extra,
        )

    def signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strategy: str,
        **extra: Any,
    ) -> None:
        """Log a trading signal at INFO level."""
        extra.update(
            symbol=symbol, signal_type=signal_type,
            confidence=confidence, strategy=strategy,
        )
        self._log(
            logging.INFO,
            "SIGNAL %s %s confidence=%.2f strategy=%s",
            symbol, signal_type, confidence, strategy,
            extra=extra,
        )

    def risk(self, message: str, **extra: Any) -> None:
        """Log a risk event at WARNING level."""
        self._log(logging.WARNING, message, extra=extra)

    def error(
        self,
        message: str,
        exc_info: Optional[BaseException | bool] = None,
        **extra: Any,
    ) -> None:
        """Log an error event at ERROR level."""
        self._log(logging.ERROR, message, exc_info=exc_info, extra=extra)

    def market(self, message: str, **extra: Any) -> None:
        """Log a market context event at INFO level."""
        extra.setdefault("context", "market")
        self._log(logging.INFO, message, extra=extra)

    # ------------------------------------------------------------------

    def _log(
        self,
        level: int,
        msg: str,
        *args: Any,
        exc_info: Optional[BaseException | bool] = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Emit a log record with extra fields attached."""
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=extra)


def get_trading_logger(name: str) -> TradingLogger:
    """Factory function returning a TradingLogger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
    """
    return TradingLogger(name)
