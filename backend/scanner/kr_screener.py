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
    # 091990 셀트리온헬스케어 — 2024-12 흡수합병으로 상장폐지 (셀트리온 0.4885주로 전환)
    ("240810", "KOSDAQ", "원익IPS"),
    ("095340", "KOSDAQ", "ISC"),
    ("039030", "KOSDAQ", "이오테크닉스"),
    ("352820", "KOSDAQ", "하이브"),
    ("222080", "KOSDAQ", "씨아이에스"),
    ("950160", "KOSDAQ", "코오롱티슈진"),
    ("141080", "KOSDAQ", "레고켐바이오"),
    # 2026-04-14: 중소형 burst 후보 추가 (방산/조선/반도체장비/바이오/로봇/테마)
    # 이 종목들은 KIS 랭킹에 자주 올라오는 중소형 모멘텀 종목으로,
    # seed에 있어야 dual_momentum이 평가 대상에 넣을 수 있음.
    # KOSPI 방산/조선
    ("012450", "KRX", "한화에어로스페이스"),
    ("329180", "KRX", "현대로템"),
    ("079550", "KRX", "LIG넥스원"),
    ("009540", "KRX", "HD한국조선해양"),
    ("010620", "KRX", "HD현대미포"),
    ("042660", "KRX", "한화오션"),
    ("010140", "KRX", "삼성중공업"),
    # KOSPI 반도체장비/소재
    ("042700", "KRX", "한미반도체"),
    ("140860", "KRX", "파크시스템스"),
    ("000990", "KRX", "DB하이텍"),
    # KOSPI 바이오/제약
    ("000100", "KRX", "유한양행"),
    ("128940", "KRX", "한미약품"),
    # KOSPI 2차전지/소재
    ("003670", "KRX", "포스코퓨처엠"),  # 이미 위에 있으면 dedup됨
    ("006260", "KRX", "LS"),
    ("011790", "KRX", "SKC"),
    # KOSPI 로봇/자율주행
    ("454910", "KRX", "두산로보틱스"),
    ("277810", "KRX", "레인보우로보틱스"),
    ("018880", "KRX", "한온시스템"),
    # KOSDAQ 반도체장비
    ("131970", "KOSDAQ", "테스나"),
    ("089030", "KOSDAQ", "테크윙"),
    ("322510", "KOSDAQ", "제이엘케이"),
    # KOSDAQ 바이오
    ("214150", "KOSDAQ", "클래시스"),
    ("178320", "KOSDAQ", "서진시스템"),
    # KOSDAQ 엔터/컨텐츠
    ("035900", "KOSDAQ", "JYP Ent."),
    ("122870", "KOSDAQ", "와이지엔터테인먼트"),
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
        exchange_map: dict[str, str] | None = None,
        **kwargs,
    ) -> KRScreenResult:
        """Screen Korean stocks from curated universe using yfinance data.

        Args:
            dynamic_symbols: Optional list of dynamically discovered symbols
                (e.g. from KRUniverseExpander / KIS ranking APIs) to include
                alongside the curated seed list.
            exchange_map: Optional symbol→exchange mapping for dynamic symbols
                (from KRUniverseResult.exchange_map). Required to get the
                correct yfinance suffix (.KS vs .KQ) for KOSDAQ symbols not
                in the curated universe.
            **kwargs: Accepted for backward compatibility (date, markets ignored).

        The curated list serves as a seed/baseline. Dynamic symbols from
        KRUniverseExpander are merged in so the scanner benefits from both
        the curated large-caps and newly discovered opportunities.

        Dynamic symbols are ONLY included if they pass yfinance quality
        screening (market cap + volume). Curated symbols are always included
        as pre-vetted large-caps.
        """
        result = KRScreenResult()
        merged_exchange_map = dict(exchange_map) if exchange_map else {}

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
        screened = self._screen_by_yfinance(
            all_candidates, merged_exchange_map,
        )
        result.sources["yfinance_filtered"] = screened

        # Combine: screened first (quality-filtered), then curated.
        # Dynamic symbols are ONLY included if they passed screening —
        # curated symbols are pre-vetted large-caps that bypass filtering,
        # but dynamic symbols from KIS ranking could be penny stocks.
        screened_set = set(screened)
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
        # Dynamic symbols: only if they passed quality screening
        for s in dynamic:
            if s not in seen and s in screened_set:
                seen.add(s)
                combined.append(s)

        result.symbols = combined[: self._max_total]
        result.total_discovered = len(combined)
        return result

    def _screen_by_yfinance(
        self,
        symbols: list[str],
        extra_exchange_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Filter symbols using yfinance market data."""
        qualified = []
        for symbol in symbols:
            try:
                exchange = self._get_exchange(
                    symbol, extra_exchange_map,
                )
                yf_sym = to_yfinance(symbol, exchange)
                ticker = yf.Ticker(yf_sym)
                info = ticker.fast_info
                market_cap = getattr(info, "market_cap", 0) or 0
                avg_vol = (
                    getattr(info, "three_month_average_volume", 0) or 0
                )

                if (
                    market_cap >= self._min_market_cap
                    and avg_vol >= self._min_avg_volume
                ):
                    qualified.append(symbol)
            except Exception as e:
                logger.debug(
                    "yfinance screening failed for %s: %s", symbol, e,
                )

        return qualified[:self._max_per_source]

    def _get_exchange(
        self,
        symbol: str,
        extra_map: dict[str, str] | None = None,
    ) -> str:
        """Look up exchange for symbol.

        Checks extra_map (dynamic discoveries) first, then curated universe.
        Falls back to "KRX" if unknown.
        """
        if extra_map and symbol in extra_map:
            return extra_map[symbol]
        return _EXCHANGE_MAP.get(symbol, "KRX")


def get_kr_exchange(symbol: str) -> str:
    """Public helper: get exchange code for a KR symbol from curated universe."""
    return _EXCHANGE_MAP.get(symbol, "KRX")
