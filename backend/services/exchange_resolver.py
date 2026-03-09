"""Exchange code resolver for KIS API.

Determines the correct KIS exchange code (NASD/NYSE/AMEX) for a given
US stock/ETF symbol. Uses yfinance exchange info with in-memory caching.
"""

import logging

import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance exchange code → KIS exchange code
_YF_TO_KIS = {
    "NMS": "NASD",   # NASDAQ Global Select Market
    "NGM": "NASD",   # NASDAQ Global Market
    "NCM": "NASD",   # NASDAQ Capital Market
    "NYQ": "NYSE",   # New York Stock Exchange
    "PCX": "AMEX",   # NYSE Arca (formerly Pacific Exchange)
    "ASE": "AMEX",   # NYSE American (formerly AMEX)
    "BTS": "NASD",   # BATS → route via NASD
}


class ExchangeResolver:
    """Resolve KIS exchange codes with caching."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def resolve(self, symbol: str) -> str:
        """Get KIS exchange code for a symbol.

        Returns NASD/NYSE/AMEX. Results are cached in memory.
        """
        if symbol in self._cache:
            return self._cache[symbol]

        exchange = self._lookup(symbol)
        self._cache[symbol] = exchange
        return exchange

    def _lookup(self, symbol: str) -> str:
        """Look up exchange via yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            yf_exchange = getattr(ticker.fast_info, "exchange", None)
            if yf_exchange and yf_exchange in _YF_TO_KIS:
                kis_code = _YF_TO_KIS[yf_exchange]
                logger.debug("Exchange resolved: %s → %s (%s)", symbol, kis_code, yf_exchange)
                return kis_code
            if yf_exchange:
                logger.warning(
                    "Unknown yfinance exchange '%s' for %s, defaulting to NASD",
                    yf_exchange, symbol,
                )
        except Exception as e:
            logger.warning("Exchange lookup failed for %s: %s, defaulting to NASD", symbol, e)

        return "NASD"

    def set(self, symbol: str, exchange: str) -> None:
        """Manually set exchange code for a symbol."""
        self._cache[symbol] = exchange

    def preload(self, mapping: dict[str, str]) -> None:
        """Preload known exchange mappings (e.g., from ETFUniverse)."""
        self._cache.update(mapping)
