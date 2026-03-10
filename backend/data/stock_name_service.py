"""Lightweight stock name resolution with caching."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Static KR stock names (ETFs + screener universe + sector holdings)
_KR_NAMES: dict[str, str] = {
    # ETFs — leveraged pairs
    "069500": "KODEX 200",
    "122630": "KODEX 레버리지",
    "114800": "KODEX 인버스",
    "229200": "KODEX 코스닥150",
    "233740": "KODEX 코스닥150레버리지",
    "251340": "KODEX 코스닥150선물인버스",
    # ETFs — sectors
    "091160": "KODEX 반도체",
    "305720": "KODEX 2차전지산업",
    "091180": "KODEX 자동차",
    "244580": "KODEX 바이오",
    "091170": "KODEX 은행",
    "315930": "KODEX 한국대만IT프리미어",
    "117680": "KODEX 철강",
    # ETFs — safe haven
    "148070": "KOSEF 국고채10년",
    "132030": "KODEX 골드선물(H)",
    "261240": "KODEX 미국달러선물",
    # KOSPI 대형주
    "005930": "삼성전자",
    "005935": "삼성전자우",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스",
    "006400": "삼성SDI",
    "051910": "LG화학",
    "005380": "현대차",
    "000270": "기아",
    "035420": "NAVER",
    "035720": "카카오",
    "068270": "셀트리온",
    "105560": "KB금융",
    "055550": "신한지주",
    "012330": "현대모비스",
    "066570": "LG전자",
    "003550": "LG",
    "034730": "SK",
    "032830": "삼성생명",
    "015760": "한국전력",
    "003670": "포스코퓨처엠",
    "009150": "삼성전기",
    "028260": "삼성물산",
    "086790": "하나금융지주",
    "017670": "SK텔레콤",
    "010130": "고려아연",
    "018260": "삼성에스디에스",
    "011200": "HMM",
    "033780": "KT&G",
    "034020": "두산에너빌리티",
    "005490": "POSCO홀딩스",
    "004020": "현대제철",
    "042700": "한미반도체",
    # KOSDAQ 대형주
    "247540": "에코프로비엠",
    "086520": "에코프로",
    "377300": "카카오페이",
    "263750": "펄어비스",
    "196170": "알테오젠",
    "328130": "루닛",
    "041510": "에스엠",
    "293490": "카카오게임즈",
    "112040": "위메이드",
    "403870": "HPSP",
    # 섹터 대표주 (kr_etf_universe.yaml top_holdings)
    "326030": "SK바이오사이언스",
}

# Static US ETF names (avoid yfinance lookups)
_US_ETF_NAMES: dict[str, str] = {
    "SPY": "SPDR S&P 500",
    "QQQ": "Invesco QQQ Trust",
    "SOXX": "iShares Semiconductor",
    "ARKK": "ARK Innovation",
    "TQQQ": "ProShares UltraPro QQQ",
    "SQQQ": "ProShares UltraPro Short QQQ",
    "UPRO": "ProShares UltraPro S&P 500",
    "SPXU": "ProShares UltraPro Short S&P",
    "SOXL": "Direxion Semiconductor Bull 3X",
    "SOXS": "Direxion Semiconductor Bear 3X",
    "TECL": "Direxion Technology Bull 3X",
    "TECS": "Direxion Technology Bear 3X",
    "SARK": "Tuttle Capital Short ARKK",
    "FAS": "Direxion Financial Bull 3X",
    "ERX": "Direxion Energy Bull 2X",
    "LABU": "Direxion Biotech Bull 3X",
    "XLK": "Technology Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLY": "Consumer Discretionary SPDR",
    "XLP": "Consumer Staples SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR",
    "XLU": "Utilities Select Sector SPDR",
    "XLRE": "Real Estate Select Sector SPDR",
    "XLC": "Communication Services SPDR",
    "SHY": "iShares 1-3 Year Treasury",
    "TLT": "iShares 20+ Year Treasury",
    "GLD": "SPDR Gold Shares",
    "UUP": "Invesco DB US Dollar Index",
    "VXX": "Barclays iPath VIX Short-Term",
    "UVXY": "ProShares Ultra VIX Short-Term",
    "SVXY": "ProShares Short VIX Short-Term",
}

# In-memory cache for dynamically resolved names
_cache: dict[str, str] = {}


def get_name(symbol: str, market: str = "US") -> Optional[str]:
    """Get stock name from cache or static mapping."""
    if market == "KR" and symbol in _KR_NAMES:
        return _KR_NAMES[symbol]
    if symbol in _US_ETF_NAMES:
        return _US_ETF_NAMES[symbol]
    return _cache.get(symbol)


def set_name(symbol: str, name: str) -> None:
    """Cache a stock name."""
    if name:
        _cache[symbol] = name


async def resolve_names(symbols: list[str], market: str = "US") -> dict[str, str]:
    """Resolve stock names for a list of symbols.

    Uses static mapping for KR, yfinance for unknown symbols.
    """
    result: dict[str, str] = {}
    unknown: list[str] = []

    for sym in symbols:
        name = get_name(sym, market)
        if name:
            result[sym] = name
        else:
            unknown.append(sym)

    if unknown:
        await _resolve_via_yfinance(unknown, market, result)

    return result


async def _resolve_via_yfinance(
    symbols: list[str], market: str, result: dict[str, str],
) -> None:
    """Resolve names using yfinance (run in thread to avoid blocking)."""
    import asyncio

    def _fetch():
        try:
            import yfinance as yf
            for sym in symbols[:20]:  # limit batch size
                try:
                    if market == "KR":
                        # Try KOSPI (.KS) first, then KOSDAQ (.KQ)
                        name = ""
                        for suffix in (".KS", ".KQ"):
                            ticker = yf.Ticker(f"{sym}{suffix}")
                            info = ticker.info or {}
                            name = info.get("shortName") or info.get("longName", "")
                            if name:
                                break
                    else:
                        ticker = yf.Ticker(sym)
                        info = ticker.info or {}
                        name = info.get("shortName") or info.get("longName", "")
                    if name:
                        result[sym] = name
                        set_name(sym, name)
                except Exception:
                    continue
        except ImportError:
            pass

    try:
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _fetch),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning("yfinance name resolution timed out")
