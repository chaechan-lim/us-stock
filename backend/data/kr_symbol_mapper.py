"""Korean stock symbol mapper.

Maps between internal 6-digit codes and external ticker formats:
- Internal (KIS API): 005930
- yfinance: 005930.KS (KOSPI) or 005930.KQ (KOSDAQ)
- pykrx: 005930 (same as internal)
"""

import logging
import re

logger = logging.getLogger(__name__)

# yfinance suffix by exchange
_YF_SUFFIX = {"KRX": ".KS", "KOSDAQ": ".KQ"}


def to_yfinance(symbol: str, exchange: str = "KRX") -> str:
    """Convert internal code to yfinance ticker (005930 -> 005930.KS)."""
    suffix = _YF_SUFFIX.get(exchange, ".KS")
    return f"{symbol}{suffix}"


def from_yfinance(yf_ticker: str) -> tuple[str, str]:
    """Convert yfinance ticker to (symbol, exchange).

    005930.KS -> ('005930', 'KRX')
    035420.KQ -> ('035420', 'KOSDAQ')
    """
    if yf_ticker.endswith(".KS"):
        return yf_ticker[:-3], "KRX"
    elif yf_ticker.endswith(".KQ"):
        return yf_ticker[:-3], "KOSDAQ"
    return yf_ticker, "KRX"


def is_kr_symbol(symbol: str) -> bool:
    """Check if a symbol looks like a Korean stock code (6-digit number)."""
    return bool(re.match(r"^\d{6}$", symbol))


def normalize_kr_symbol(symbol: str) -> str:
    """Ensure Korean stock code is zero-padded to 6 digits."""
    if symbol.isdigit():
        return symbol.zfill(6)
    return symbol
