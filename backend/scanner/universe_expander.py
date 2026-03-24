"""Dynamic Universe Expander.

Discovers stocks dynamically using:
1. yfinance predefined screeners (most_actives, day_gainers, growth_technology, etc.)
2. Sector rotation — expand holdings in top-performing sectors
3. ETF Universe config — sector top_holdings as baseline
4. KIS ranking APIs — volume surge, gainers, new highs (rate-limit aware)

Replaces the hardcoded 30-stock base_universe in after_hours_scan.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yfinance as yf

from scanner.etf_universe import ETFUniverse
from scanner.sector_analyzer import SectorAnalyzer

if TYPE_CHECKING:
    from exchange.kis_adapter import KISAdapter
    from exchange.kis_kr_adapter import KISKRAdapter
    from services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# yfinance predefined screener queries
# Research shows growth > value for stock selection (IC: growth=0.23, value=-0.08)
SCREENER_QUERIES = [
    "most_actives",
    "day_gainers",
    "growth_technology_stocks",
    "undervalued_growth_stocks",  # growth focus, not pure value
    "aggressive_small_caps",
    "small_cap_gainers",
    "undervalued_large_caps",
]

# KIS screening: max symbols per API call (first page only, no pagination)
KIS_SCREEN_LIMIT = 15


@dataclass
class UniverseResult:
    """Result of universe expansion."""
    symbols: list[str]
    etf_symbols: list[str] = field(default_factory=list)
    sources: dict[str, list[str]] = field(default_factory=dict)
    total_discovered: int = 0


class UniverseExpander:
    """Dynamically discover and expand the stock universe."""

    def __init__(
        self,
        etf_universe: ETFUniverse | None = None,
        sector_analyzer: SectorAnalyzer | None = None,
        kis_adapter: "KISAdapter | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        max_per_screener: int = 15,
        max_total: int = 80,
    ):
        self._etf = etf_universe or ETFUniverse()
        self._sector_analyzer = sector_analyzer or SectorAnalyzer()
        self._kis = kis_adapter
        self._rate_limiter = rate_limiter
        self._max_per_screener = max_per_screener
        self._max_total = max_total

    async def expand(
        self,
        existing_watchlist: list[str] | None = None,
        sector_data: dict[str, dict] | None = None,
    ) -> UniverseResult:
        """Build an expanded universe from multiple sources.

        Args:
            existing_watchlist: Current watchlist symbols to always include.
            sector_data: Sector performance data for sector-weighted expansion.

        Returns:
            UniverseResult with deduplicated symbols and source tracking.
        """
        sources: dict[str, list[str]] = {}
        all_symbols: set[str] = set()

        # Source 1: Existing watchlist (always included)
        if existing_watchlist:
            sources["watchlist"] = list(existing_watchlist)
            all_symbols.update(existing_watchlist)

        # Source 2: ETF Universe sector holdings (baseline)
        sector_holdings = self._get_sector_holdings(sector_data)
        sources["sector_holdings"] = sector_holdings
        all_symbols.update(sector_holdings)

        # Source 3: yfinance screeners (dynamic discovery)
        screener_symbols = self._run_screeners()
        sources["screeners"] = screener_symbols
        all_symbols.update(screener_symbols)

        # Source 4: KIS ranking APIs (optional, rate-limit aware)
        kis_symbols = await self._run_kis_screening()
        if kis_symbols:
            sources["kis_ranking"] = kis_symbols
            all_symbols.update(kis_symbols)

        # Filter: US stocks only, remove ETFs/non-equity
        filtered = self._filter_symbols(all_symbols)

        # Collect ETF symbols from config for watchlist inclusion
        etf_syms = self._etf.all_etf_symbols

        result = UniverseResult(
            symbols=sorted(filtered)[: self._max_total],
            etf_symbols=etf_syms,
            sources=sources,
            total_discovered=len(filtered),
        )
        logger.info(
            "Universe expanded: %d symbols (watchlist=%d, sectors=%d, screeners=%d, kis=%d)",
            len(result.symbols),
            len(sources.get("watchlist", [])),
            len(sector_holdings),
            len(screener_symbols),
            len(kis_symbols),
        )
        return result

    def _get_sector_holdings(
        self, sector_data: dict[str, dict] | None = None,
    ) -> list[str]:
        """Get sector holdings, weighted by sector strength.

        Strong sectors get all top + mid holdings; weak sectors get fewer.
        """
        sectors = self._etf.get_all_sectors()
        if not sectors:
            return []

        # If we have sector performance, prioritize strong sectors
        sector_scores: dict[str, float] = {}
        if sector_data:
            scored = self._sector_analyzer.analyze(sector_data)
            sector_scores = {s.name: s.strength_score for s in scored}

        holdings: list[str] = []
        for name, sector_etf in sectors.items():
            strength = sector_scores.get(name, 50.0)
            mid = sector_etf.mid_holdings or []

            # Strong sectors (>60): all top + 3 mid
            # Medium sectors (30-60): 3 top + 2 mid
            # Weak sectors (<30): 1 top + 1 mid
            if strength >= 60:
                holdings.extend(sector_etf.top_holdings)
                holdings.extend(mid[:3])
            elif strength >= 30:
                holdings.extend(sector_etf.top_holdings[:3])
                holdings.extend(mid[:2])
            else:
                holdings.extend(sector_etf.top_holdings[:1])
                holdings.extend(mid[:1])

        return holdings

    def _run_screeners(self) -> list[str]:
        """Run yfinance predefined screeners to discover stocks."""
        discovered: list[str] = []

        for query in SCREENER_QUERIES:
            try:
                result = yf.screen(query)
                if not result or "quotes" not in result:
                    continue

                quotes = result["quotes"]
                count = 0
                for q in quotes:
                    if count >= self._max_per_screener:
                        break
                    symbol = q.get("symbol", "")
                    # Only US-listed stocks (no suffix like .L, .TO)
                    if symbol and "." not in symbol and symbol.isalpha():
                        discovered.append(symbol)
                        count += 1

                logger.debug(
                    "Screener '%s': found %d symbols", query, count,
                )
            except Exception as e:
                logger.debug("Screener '%s' failed: %s", query, e)

        return list(dict.fromkeys(discovered))  # dedupe preserving order

    async def _run_kis_screening(self) -> list[str]:
        """Run KIS ranking APIs for additional stock discovery.

        Rate-limit aware: acquires tokens before each call, uses lowest
        priority. Only runs if KIS adapter is configured.
        Returns symbols only (first page, no pagination).
        """
        if not self._kis:
            return []

        discovered: list[str] = []
        calls = [
            ("volume_surge", self._kis.fetch_volume_surge),
            ("gainers", lambda exch="NAS": self._kis.fetch_updown_rate(exch, "up")),
            ("new_highs", lambda exch="NAS": self._kis.fetch_new_highlow(exch, True)),
        ]

        for name, fetch_fn in calls:
            try:
                if self._rate_limiter:
                    await self._rate_limiter.acquire()
                stocks = await fetch_fn()
                symbols = [
                    s.symbol for s in stocks[:KIS_SCREEN_LIMIT]
                    if s.symbol and "." not in s.symbol
                ]
                discovered.extend(symbols)
                logger.debug("KIS '%s': found %d symbols", name, len(symbols))
            except Exception as e:
                logger.debug("KIS '%s' failed: %s", name, e)

        return list(dict.fromkeys(discovered))  # dedupe preserving order

    def _filter_symbols(self, symbols: set[str]) -> set[str]:
        """Filter out ETFs, leveraged products, and invalid symbols."""
        # Known ETF symbols to exclude
        etf_symbols = set(self._etf.all_etf_symbols)
        etf_symbols.update(self._etf.safe_haven)

        filtered = set()
        for sym in symbols:
            sym = sym.upper().strip()
            if not sym:
                continue
            # Skip ETFs and leveraged products
            if sym in etf_symbols:
                continue
            # Skip symbols with special characters (warrants, units)
            if not sym.replace("-", "").isalpha():
                continue
            # Skip single-char symbols (likely indices)
            if len(sym) < 2:
                continue
            filtered.add(sym)

        return filtered


# ---------------------------------------------------------------------------
# KR Universe Expander
# ---------------------------------------------------------------------------

# KIS KR screening: max symbols per API call (first page, no pagination)
KR_KIS_SCREEN_LIMIT = 15


@dataclass
class KRUniverseResult:
    """Result of KR universe expansion."""

    symbols: list[str]
    sources: dict[str, list[str]] = field(default_factory=dict)
    total_discovered: int = 0
    exchange_map: dict[str, str] = field(default_factory=dict)


class KRUniverseExpander:
    """Dynamically discover Korean stocks using KIS domestic ranking APIs.

    Sources:
    1. Existing watchlist (always preserved)
    2. Curated seed list (kr_screener._KR_UNIVERSE — 56 major KOSPI/KOSDAQ stocks)
    3. KIS domestic ranking APIs:
       - 거래량 급등 (volume surge) — KOSPI + KOSDAQ
       - 등락률 상위 (top gainers) — KOSPI + KOSDAQ
       - 신고가 (new highs) — KOSPI + KOSDAQ

    Replaces the static KRScreener-only approach with dynamic universe
    expansion similar to the US UniverseExpander.
    """

    def __init__(
        self,
        kis_kr_adapter: "KISKRAdapter | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        max_total: int = 80,
    ):
        self._kis_kr = kis_kr_adapter
        self._rate_limiter = rate_limiter
        self._max_total = max_total

    async def expand_kr(
        self,
        existing_watchlist: list[str] | None = None,
    ) -> KRUniverseResult:
        """Build an expanded KR universe from multiple sources.

        Args:
            existing_watchlist: Current KR watchlist symbols to always include.

        Returns:
            KRUniverseResult with deduplicated symbols, source tracking,
            and exchange map (symbol → "KRX" or "KOSDAQ").
        """
        from scanner.kr_screener import _EXCHANGE_MAP, _KR_UNIVERSE

        sources: dict[str, list[str]] = {}
        exchange_map: dict[str, str] = dict(_EXCHANGE_MAP)  # start from seed map

        # Source 1: Existing watchlist (always included)
        if existing_watchlist:
            sources["watchlist"] = list(existing_watchlist)

        # Source 2: Curated seed list (56 major KOSPI/KOSDAQ stocks)
        seed = [s[0] for s in _KR_UNIVERSE]
        sources["seed"] = seed

        # Source 3: KIS domestic ranking APIs (dynamic discovery)
        kis_symbols, kis_exchange_map = await self._run_kis_kr_screening()
        if kis_symbols:
            sources["kis_kr_ranking"] = kis_symbols
            exchange_map.update(kis_exchange_map)

        # Combine: watchlist → seed → KIS ranking (dedup, order-preserving)
        all_symbols: list[str] = list(dict.fromkeys(
            (existing_watchlist or []) + seed + kis_symbols
        ))

        # Filter: valid KR 6-digit numeric codes only
        filtered = [s for s in all_symbols if _is_valid_kr_symbol(s)]

        result = KRUniverseResult(
            symbols=filtered[: self._max_total],
            sources=sources,
            total_discovered=len(filtered),
            exchange_map=exchange_map,
        )
        logger.info(
            "KR universe expanded: %d symbols "
            "(watchlist=%d, seed=%d, kis_kr=%d)",
            len(result.symbols),
            len(sources.get("watchlist", [])),
            len(seed),
            len(kis_symbols),
        )
        return result

    async def _run_kis_kr_screening(
        self,
    ) -> tuple[list[str], dict[str, str]]:
        """Run KIS domestic ranking APIs for dynamic KR stock discovery.

        Calls volume surge, top gainers, and new highs for both
        KOSPI (market="J") and KOSDAQ (market="K").
        Rate-limit aware: acquires token before each API call.

        Returns:
            Tuple of (deduplicated symbol list, exchange map dict).
        """
        if not self._kis_kr:
            return [], {}

        discovered: list[str] = []
        exchange_map: dict[str, str] = {}

        # (name, coroutine_factory, market_code, exchange_label)
        calls: list[tuple[str, Any, str]] = [
            ("kr_volume_surge_kospi",
             lambda: self._kis_kr.fetch_volume_surge("J"), "KRX"),        # type: ignore[union-attr]
            ("kr_volume_surge_kosdaq",
             lambda: self._kis_kr.fetch_volume_surge("K"), "KOSDAQ"),     # type: ignore[union-attr]
            ("kr_gainers_kospi",
             lambda: self._kis_kr.fetch_updown_rate("J", "up"), "KRX"),   # type: ignore[union-attr]
            ("kr_gainers_kosdaq",
             lambda: self._kis_kr.fetch_updown_rate("K", "up"), "KOSDAQ"),# type: ignore[union-attr]
            ("kr_new_highs_kospi",
             lambda: self._kis_kr.fetch_new_highlow("J", True), "KRX"),   # type: ignore[union-attr]
            ("kr_new_highs_kosdaq",
             lambda: self._kis_kr.fetch_new_highlow("K", True), "KOSDAQ"),# type: ignore[union-attr]
        ]

        for name, fetch_fn, exchange_label in calls:
            try:
                if self._rate_limiter:
                    await self._rate_limiter.acquire()
                stocks = await fetch_fn()
                for s in stocks[:KR_KIS_SCREEN_LIMIT]:
                    if _is_valid_kr_symbol(s.symbol):
                        discovered.append(s.symbol)
                        # Prefer the stock's own exchange field, fall back to label
                        exchange_map[s.symbol] = s.exchange or exchange_label
                logger.debug(
                    "KIS KR '%s': found %d symbols", name, len(stocks),
                )
            except Exception as e:
                logger.debug("KIS KR '%s' failed: %s", name, e)

        return list(dict.fromkeys(discovered)), exchange_map


def _is_valid_kr_symbol(symbol: str) -> bool:
    """Return True if symbol is a valid KR 6-digit numeric stock code."""
    return bool(symbol) and len(symbol) == 6 and symbol.isdigit()
