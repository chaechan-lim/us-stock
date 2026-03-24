"""Korean stock screener using yfinance + curated universe.

Since KRX direct scraping (pykrx) is blocked from this server,
we use yfinance for Korean stock data and maintain a curated
universe of major KOSPI/KOSDAQ stocks.

Discovers KR stocks by:
1. Curated universe (대형주 + 업종대표주)
2. yfinance screening (market cap, volume, fundamentals)
"""

import logging
from dataclasses import dataclass, field

import yfinance as yf

from data.kr_symbol_mapper import to_yfinance

logger = logging.getLogger(__name__)


# Curated KR stock universe — major KOSPI/KOSDAQ stocks
# Format: (symbol, exchange, name)
_KR_UNIVERSE = [
    # KOSPI 대형주
    ("005930", "KRX", "삼성전자"),
    ("000660", "KRX", "SK하이닉스"),
    ("373220", "KRX", "LG에너지솔루션"),
    ("207940", "KRX", "삼성바이오로직스"),
    ("005935", "KRX", "삼성전자우"),
    ("006400", "KRX", "삼성SDI"),
    ("051910", "KRX", "LG화학"),
    ("005380", "KRX", "현대차"),
    ("000270", "KRX", "기아"),
    ("035420", "KRX", "NAVER"),
    ("035720", "KRX", "카카오"),
    ("068270", "KRX", "셀트리온"),
    ("105560", "KRX", "KB금융"),
    ("055550", "KRX", "신한지주"),
    ("012330", "KRX", "현대모비스"),
    ("066570", "KRX", "LG전자"),
    ("003550", "KRX", "LG"),
    ("034730", "KRX", "SK"),
    ("032830", "KRX", "삼성생명"),
    ("015760", "KRX", "한국전력"),
    ("003670", "KRX", "포스코퓨처엠"),
    ("009150", "KRX", "삼성전기"),
    ("028260", "KRX", "삼성물산"),
    ("086790", "KRX", "하나금융지주"),
    ("017670", "KRX", "SK텔레콤"),
    ("010130", "KRX", "고려아연"),
    ("018260", "KRX", "삼성에스디에스"),
    ("011200", "KRX", "HMM"),
    ("033780", "KRX", "KT&G"),
    ("034020", "KRX", "두산에너빌리티"),
    # KOSDAQ 대형주
    ("247540", "KOSDAQ", "에코프로비엠"),
    ("086520", "KOSDAQ", "에코프로"),
    ("377300", "KOSDAQ", "카카오페이"),
    ("263750", "KOSDAQ", "펄어비스"),
    ("196170", "KOSDAQ", "알테오젠"),
    ("328130", "KOSDAQ", "루닛"),
    ("041510", "KOSDAQ", "에스엠"),
    ("293490", "KOSDAQ", "카카오게임즈"),
    ("112040", "KOSDAQ", "위메이드"),
    ("403870", "KOSDAQ", "HPSP"),
    # KOSDAQ 추가 (반도체/IT/바이오/엔터)
    ("058470", "KOSDAQ", "리노공업"),
    ("357780", "KOSDAQ", "솔브레인"),
    ("036930", "KOSDAQ", "주성엔지니어링"),
    ("005290", "KOSDAQ", "동진쎄미켐"),
    ("078600", "KOSDAQ", "대주전자재료"),
    ("067160", "KOSDAQ", "아프리카TV"),
    ("145020", "KOSDAQ", "휴젤"),
    ("091990", "KOSDAQ", "셀트리온헬스케어"),
    ("240810", "KOSDAQ", "원익IPS"),
    ("095340", "KOSDAQ", "ISC"),
    ("039030", "KOSDAQ", "이오테크닉스"),
    ("352820", "KOSDAQ", "하이브"),
    ("222080", "KOSDAQ", "씨아이에스"),
    ("950160", "KOSDAQ", "코오롱티슈진"),
    ("141080", "KOSDAQ", "레고켐바이오"),
]

# Quick lookup: symbol -> exchange
_EXCHANGE_MAP = {sym: ex for sym, ex, _ in _KR_UNIVERSE}


@dataclass
class KRScreenResult:
    """Result from KR stock screening."""
    symbols: list[str] = field(default_factory=list)
    sources: dict[str, list[str]] = field(default_factory=dict)
    total_discovered: int = 0


class KRScreener:
    """Korean stock screener using yfinance + curated universe."""

    def __init__(
        self,
        max_per_source: int = 20,
        max_total: int = 60,
        min_market_cap: int = 500_000_000_000,  # 5000억원 (configurable)
        min_avg_volume: int = 100_000,
    ):
        self._max_per_source = max_per_source
        self._max_total = max_total
        self._min_market_cap = min_market_cap
        self._min_avg_volume = min_avg_volume

    def screen(
        self,
        dynamic_symbols: list[str] | None = None,
        **kwargs,
    ) -> KRScreenResult:
        """Screen Korean stocks from curated universe using yfinance data.

        Args:
            dynamic_symbols: Optional list of dynamically discovered symbols
                (e.g. from KRUniverseExpander / KIS ranking APIs) to include
                alongside the curated seed list.
            **kwargs: Accepted for backward compatibility (date, markets ignored).

        The curated list serves as a seed/baseline. Dynamic symbols from
        KRUniverseExpander are merged in so the scanner benefits from both
        the curated large-caps and newly discovered opportunities.
        """
        result = KRScreenResult()

        # Source 1: Curated large-caps (seed / baseline — always included)
        curated = [s[0] for s in _KR_UNIVERSE]
        result.sources["curated"] = curated

        # Source 2: Dynamic symbols from KRUniverseExpander / KIS ranking
        dynamic = list(dynamic_symbols) if dynamic_symbols else []
        if dynamic:
            result.sources["dynamic"] = dynamic

        # Source 3: yfinance screening (filter by market cap + volume)
        # Screen over union of seed + dynamic to apply quality filter
        all_candidates = list(dict.fromkeys(curated + dynamic))
        screened = self._screen_by_yfinance(all_candidates)
        result.sources["yfinance_filtered"] = screened

        # Combine: screened first (quality-filtered), then curated, then dynamic
        seen: set[str] = set()
        combined: list[str] = []
        for s in screened:
            if s not in seen:
                seen.add(s)
                combined.append(s)
        for s in curated:
            if s not in seen:
                seen.add(s)
                combined.append(s)
        for s in dynamic:
            if s not in seen:
                seen.add(s)
                combined.append(s)

        result.symbols = combined[: self._max_total]
        result.total_discovered = len(combined)
        return result

    def _screen_by_yfinance(self, symbols: list[str]) -> list[str]:
        """Filter symbols using yfinance market data."""
        qualified = []
        for symbol in symbols:
            try:
                yf_sym = to_yfinance(symbol, self._get_exchange(symbol))
                ticker = yf.Ticker(yf_sym)
                info = ticker.fast_info
                market_cap = getattr(info, "market_cap", 0) or 0
                avg_vol = getattr(info, "three_month_average_volume", 0) or 0

                if market_cap >= self._min_market_cap and avg_vol >= self._min_avg_volume:
                    qualified.append(symbol)
            except Exception as e:
                logger.debug("yfinance screening failed for %s: %s", symbol, e)

        return qualified[:self._max_per_source]

    def _get_exchange(self, symbol: str) -> str:
        """Look up exchange from curated universe."""
        return _EXCHANGE_MAP.get(symbol, "KRX")


def get_kr_exchange(symbol: str) -> str:
    """Public helper: get exchange code for a KR symbol from curated universe."""
    return _EXCHANGE_MAP.get(symbol, "KRX")
