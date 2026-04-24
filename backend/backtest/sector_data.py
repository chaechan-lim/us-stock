"""Historical sector strength for backtest sector-boost.

Loads daily OHLC for 11 US sector SPDRs / 7 KR KODEX sector ETFs and
computes return_1w / return_1m / return_3m at each bar, then runs the
live `SectorAnalyzer` over the window to produce a strength_score 0-100.

Backtest calls `score_at_date(date)` to get {sector: strength_score} and
resolves each symbol to its sector via `sector_for_symbol(sym)`.
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from scanner.sector_analyzer import SectorAnalyzer

logger = logging.getLogger(__name__)


US_SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communications",
}

KR_SECTOR_ETFS: dict[str, str] = {
    "091160": "반도체",
    "305720": "2차전지",
    "091180": "자동차",
    "244580": "바이오",
    "091170": "금융",
    "315930": "IT",
    "117680": "철강소재",
}

# Approximate symbol → sector mapping for the backtest universe.
# Not exhaustive — unmapped symbols fall through to "Unknown" and receive
# a neutral multiplier (no boost, no suppression).
US_SYMBOL_SECTOR: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "QCOM": "Technology", "INTC": "Technology",
    "AMD": "Technology", "CSCO": "Technology", "IBM": "Technology",
    "MU": "Technology", "NOW": "Technology", "TXN": "Technology",
    "PANW": "Technology", "SNOW": "Technology", "PLTR": "Technology",
    "ANET": "Technology", "CRWD": "Technology", "DELL": "Technology",
    "SMCI": "Technology",
    # Communications
    "GOOGL": "Communications", "GOOG": "Communications", "META": "Communications",
    "NFLX": "Communications", "DIS": "Communications", "CMCSA": "Communications",
    "T": "Communications", "VZ": "Communications", "TMUS": "Communications",
    # Consumer Disc.
    "AMZN": "Consumer Disc.", "TSLA": "Consumer Disc.", "HD": "Consumer Disc.",
    "MCD": "Consumer Disc.", "NKE": "Consumer Disc.", "LOW": "Consumer Disc.",
    "SBUX": "Consumer Disc.", "BKNG": "Consumer Disc.", "TJX": "Consumer Disc.",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "MO": "Consumer Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "MS": "Financials",
    "GS": "Financials", "C": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials", "FITB": "Financials",
    "CFG": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare", "CVS": "Healthcare",
    "AMGN": "Healthcare", "ISRG": "Healthcare", "ALGN": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "PSX": "Energy", "OXY": "Energy",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "LMT": "Industrials", "DE": "Industrials",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "FR": "Real Estate",
}

# KR mapping (yfinance uses codes w/o suffix in our internal format)
KR_SYMBOL_SECTOR: dict[str, str] = {
    # 반도체 (semiconductors)
    "005930": "반도체", "005935": "반도체", "000660": "반도체", "042700": "반도체",
    # 2차전지
    "006400": "2차전지", "373220": "2차전지", "247540": "2차전지", "003670": "2차전지",
    # 자동차
    "005380": "자동차", "000270": "자동차", "012330": "자동차",
    # 바이오
    "068270": "바이오", "207940": "바이오", "196170": "바이오", "214450": "바이오",
    # 금융
    "055550": "금융", "105560": "금융", "086790": "금융", "316140": "금융",
    "024110": "금융", "138040": "금융", "011200": "금융",
    # IT (플랫폼/소프트웨어)
    "035420": "IT", "035720": "IT", "036570": "IT", "259960": "IT",
    # 철강소재
    "005490": "철강소재", "010140": "철강소재", "010130": "철강소재",
    "034020": "철강소재",  # 두산에너빌리티 (plant/heavy)
    # 2026-04-23 추가 매핑 (KR backtest universe 커버리지 개선)
    # 확실한 매핑만 추가. 유틸리티/통신/엔터/홀딩스는 KR sector ETF에 해당 없어서 Unknown 유지.
    "051910": "2차전지",  # LG화학
    "032830": "금융",     # 삼성생명
    "066570": "IT",       # LG전자 (가전/IT)
    "009150": "IT",       # 삼성전기 (전자부품)
    "018260": "IT",       # 삼성SDS
    "086520": "2차전지",  # 에코프로 (배터리 소재)
    "263750": "IT",       # 펄어비스 (게임 → IT)
    "403870": "반도체",   # HPSP (반도체 장비)
    "058470": "반도체",   # 리노공업
    "357780": "반도체",   # 솔브레인
    "036930": "반도체",   # 주성엔지니어링
    "039030": "반도체",   # 이오테크닉스
    "145020": "바이오",   # 휴젤
    # 매핑 안 하는 심볼 (KR 섹터 ETF 커버리지 밖):
    #   015760 한국전력 (utility), 033780 KT&G (staples), 017670 SKT (telecom),
    #   028260 삼성물산 (holding), 003550 LG (holding), 352820 하이브 (엔터)
}


@dataclass
class SectorHistory:
    """Per-date sector strength scores for one market.

    dates  : sorted list of 'YYYY-MM-DD' strings (one per trading day).
    scores : {date_str: {sector_name: strength_score_0_100}}
    """
    dates: list[str]
    scores: dict[str, dict[str, float]]
    symbol_sector: dict[str, str]

    def score_at(self, date: Any) -> dict[str, float]:
        """Return sector scores for the requested date (or the nearest prior).

        Falls back to an empty dict when the date is before any history —
        callers get no boost for those days.
        """
        key = self._date_key(date)
        if key in self.scores:
            return self.scores[key]
        # Fallback: find the latest date <= key
        if not self.dates:
            return {}
        idx = _bisect_left(self.dates, key) - 1
        if idx < 0:
            return {}
        return self.scores[self.dates[idx]]

    def sector_for(self, symbol: str) -> str:
        """Map symbol → sector. KR backtest uses .KS / .KQ suffixes; strip them
        before lookup since KR_SYMBOL_SECTOR keys are bare codes."""
        key = symbol
        if symbol.endswith(".KS") or symbol.endswith(".KQ"):
            key = symbol[:-3]
        return self.symbol_sector.get(key, "Unknown")

    @staticmethod
    def _date_key(date: Any) -> str:
        if isinstance(date, str):
            return date[:10]
        if hasattr(date, "strftime"):
            return date.strftime("%Y-%m-%d")
        return str(date)[:10]


def _bisect_left(sorted_list: list[str], target: str) -> int:
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def build_sector_history(
    data_loader: Any, market: str, period: str = "2y",
) -> SectorHistory:
    """Fetch sector ETF OHLC and compute per-date strength scores."""
    if market == "KR":
        etf_map = {f"{code}.KS": name for code, name in KR_SECTOR_ETFS.items()}
        symbol_map = KR_SYMBOL_SECTOR
    else:
        etf_map = US_SECTOR_ETFS
        symbol_map = US_SYMBOL_SECTOR

    etf_data = data_loader.load_multiple(list(etf_map.keys()), period=period)
    if not etf_data:
        logger.warning("Sector ETF data empty for %s — boost will be neutral", market)
        return SectorHistory(dates=[], scores={}, symbol_sector=symbol_map)

    # Normalize: {sector_name: pd.Series(close prices by date)}
    closes_by_sector: dict[str, pd.Series] = {}
    for etf_sym, sector_name in etf_map.items():
        if etf_sym not in etf_data:
            continue
        df = etf_data[etf_sym].df
        if df.empty or "close" not in df.columns:
            continue
        closes_by_sector[sector_name] = df["close"]

    if not closes_by_sector:
        logger.warning("No usable sector ETF closes for %s", market)
        return SectorHistory(dates=[], scores={}, symbol_sector=symbol_map)

    # Common index
    index = sorted(
        {d for s in closes_by_sector.values() for d in s.index}
    )
    all_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]

    analyzer = SectorAnalyzer()
    scores: dict[str, dict[str, float]] = {}

    # For each bar, compute returns and analyzer scores
    # Need ~63 bars (3mo) history before first meaningful score
    for i, bar_date in enumerate(index):
        if i < 63:
            continue

        sector_data: dict[str, dict[str, float]] = {}
        for sector_name, closes in closes_by_sector.items():
            try:
                # Align by index to guard sparse series
                # (e.g. XLC starts later than XLK)
                aligned = closes.reindex(index).ffill()
                if pd.isna(aligned.iloc[i]):
                    continue
                price = float(aligned.iloc[i])
                if price <= 0:
                    continue

                def _ret(n: int) -> float:
                    if i - n < 0:
                        return 0.0
                    past = aligned.iloc[i - n]
                    if pd.isna(past) or past <= 0:
                        return 0.0
                    return float((price - past) / past * 100)

                sector_data[sector_name] = {
                    "symbol": sector_name,
                    "return_1w": _ret(5),
                    "return_1m": _ret(21),
                    "return_3m": _ret(63),
                }
            except (IndexError, KeyError):
                continue

        if not sector_data:
            continue

        date_str = all_dates[i]
        sector_scores = analyzer.analyze(sector_data)
        scores[date_str] = {s.name: s.strength_score for s in sector_scores}

    return SectorHistory(
        dates=sorted(scores.keys()),
        scores=scores,
        symbol_sector=symbol_map,
    )


def confidence_multiplier(
    strength: float | None, weight: float,
) -> float:
    """Map strength score [0..100] + boost weight → confidence multiplier.

    weight=0 → always 1.0 (no boost).
    weight=0.3:
        strength=100 → 1.15
        strength=50  → 1.00
        strength=0   → 0.85
    Unknown sector → 1.0 (neutral).
    """
    if weight <= 0 or strength is None:
        return 1.0
    return max(0.1, 1.0 + weight * (strength - 50) / 50)
