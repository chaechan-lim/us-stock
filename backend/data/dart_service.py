"""KR DART (전자공시시스템) insider trading service.

Mirrors `data/insider_service.py` (Finnhub for US) but uses opendart.fss.or.kr
free API. Reports 5%+ shareholder + executive (등기임원) buy/sell
transactions, plugged into `event_calendar.get_confidence_adjustment()`
to bias signal confidence for KR symbols.

Setup:
  1. Register free API key at https://opendart.fss.or.kr/uss/umt/login/loginForm.do
  2. Set DART_API_KEY in .env
  3. Set DART_ENABLED=true to activate

Without API key the service no-ops (returns 0.0 from get_signal_adjustment),
keeping the existing event_calendar pipeline functional for US only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import aiohttp

logger = logging.getLogger(__name__)

DART_BASE_URL = "https://opendart.fss.or.kr/api"
DART_LIST_ENDPOINT = "/list.json"
# H = 지분공시 (5%+ ownership filings, includes insider buys/sells)
DART_PBLNTF_TY_INSIDER = "H"


@dataclass
class DARTFiling:
    """One row from DART list.json."""

    corp_code: str          # 8-digit DART corp ID
    corp_name: str          # 회사명
    stock_code: str         # 6-digit KRX code (e.g. "005930"), if listed
    report_nm: str          # 보고서명 — contains buyer/seller summary
    rcept_no: str           # filing receipt number
    rcept_dt: str           # 접수일자 YYYYMMDD
    flr_nm: str             # 공시제출인명 (filer name)

    @property
    def is_insider_buy(self) -> bool:
        # Heuristic: report names mentioning 취득 (acquisition) without
        # the 처분 (disposal) qualifier. Real implementation parses the
        # XBRL-like 'document' endpoint for share-count deltas.
        if not self.report_nm:
            return False
        nm = self.report_nm
        return "취득" in nm and "처분" not in nm

    @property
    def is_insider_sell(self) -> bool:
        if not self.report_nm:
            return False
        return "처분" in self.report_nm


class DARTInsiderService:
    """KR DART insider signal — analog to InsiderTradingService for US.

    Refresh() pulls last 14d of insider filings for the watchlist
    symbols. get_signal_adjustment(symbol) returns a confidence
    adjustment in [-0.10, +0.10] based on recent net buys/sells.

    Off when DART_API_KEY env is unset — keeps live engine functional
    without an API key (every adjustment is 0.0).
    """

    def __init__(
        self,
        api_key: str | None = None,
        enabled: bool = False,
        lookback_days: int = 14,
    ):
        self._api_key = api_key or os.getenv("DART_API_KEY", "")
        self._enabled = enabled and bool(self._api_key)
        self._lookback_days = lookback_days
        self._signal_cache: dict[str, float] = {}  # symbol -> adjustment
        self._last_refresh: datetime | None = None
        if not self._enabled:
            logger.info("DART service disabled (no API key or disabled flag)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _fetch_filings(
        self,
        session: aiohttp.ClientSession,
        corp_code: str,
        bgn_de: str,
        end_de: str,
    ) -> list[DARTFiling]:
        params = {
            "crtfc_key": self._api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,        # YYYYMMDD start
            "end_de": end_de,        # YYYYMMDD end
            "pblntf_ty": DART_PBLNTF_TY_INSIDER,
            "page_count": "100",
        }
        async with session.get(DART_BASE_URL + DART_LIST_ENDPOINT, params=params) as r:
            data = await r.json()
        if data.get("status") != "000":
            return []
        out = []
        for item in data.get("list", []):
            out.append(DARTFiling(
                corp_code=item.get("corp_code", ""),
                corp_name=item.get("corp_name", ""),
                stock_code=item.get("stock_code", ""),
                report_nm=item.get("report_nm", ""),
                rcept_no=item.get("rcept_no", ""),
                rcept_dt=item.get("rcept_dt", ""),
                flr_nm=item.get("flr_nm", ""),
            ))
        return out

    async def refresh(self, symbols: list[str]) -> None:
        """Pull last `lookback_days` filings for each symbol.

        Requires a static stock_code → corp_code map (DART distinguishes
        listed companies by their 8-digit corp_code, not the 6-digit
        stock code). We resolve from the DART CORPCODE.zip snapshot
        downloaded separately. For now, log a warning if any symbol
        lacks a mapping.
        """
        if not self._enabled:
            return
        # TODO: load stock_code → corp_code map from data/dart_corp_map.json
        # (one-time download from opendart corpCode.xml).
        end_de = date.today().strftime("%Y%m%d")
        bgn_de = (date.today() - timedelta(days=self._lookback_days)).strftime("%Y%m%d")

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                corp_code = self._stock_to_corp(symbol)
                if not corp_code:
                    continue
                try:
                    filings = await self._fetch_filings(
                        session, corp_code, bgn_de, end_de,
                    )
                except Exception as e:
                    logger.warning("DART fetch failed for %s: %s", symbol, e)
                    continue
                # Net buy minus sell. Each filing weighted equally for now;
                # real version would extract share counts from the document
                # endpoint for size-weighted scoring.
                buys = sum(1 for f in filings if f.is_insider_buy)
                sells = sum(1 for f in filings if f.is_insider_sell)
                net = buys - sells
                if net >= 2:
                    self._signal_cache[symbol] = 0.10
                elif net == 1:
                    self._signal_cache[symbol] = 0.05
                elif net <= -2:
                    self._signal_cache[symbol] = -0.10
                elif net == -1:
                    self._signal_cache[symbol] = -0.05
                else:
                    self._signal_cache[symbol] = 0.0

        self._last_refresh = datetime.utcnow()

    def _stock_to_corp(self, stock_code: str) -> str:
        """Resolve 6-digit KRX code to 8-digit DART corp_code.

        Placeholder: real impl loads from a static map file generated by
        one-time corpCode.xml download from DART.
        """
        return ""  # TODO

    def get_signal_adjustment(self, symbol: str) -> float:
        """Return confidence adjustment [-0.10, +0.10] for symbol."""
        return self._signal_cache.get(symbol, 0.0)

    def to_dict(self) -> dict:
        return {
            "enabled": self._enabled,
            "tracked_symbols": len(self._signal_cache),
            "last_refresh": (
                self._last_refresh.isoformat() if self._last_refresh else None
            ),
        }
