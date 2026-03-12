"""KIS REST API adapter for Korean domestic stock trading.

Wraps 한국투자증권 Open API domestic endpoints for:
- Market data (현재가, 일봉, 호가)
- Order management (매수, 매도, 취소)
- Account (잔고, 보유종목)

Shares the same KISAuth (OAuth token) as the US adapter.
"""

import json
import logging
from typing import Any

import aiohttp
from aiohttp import ContentTypeError

from config import KISConfig
from exchange.base import (
    Balance,
    Candle,
    ExchangeAdapter,
    OrderBook,
    OrderResult,
    Position,
    Ticker,
)
from exchange.kis_auth import KISAuth

logger = logging.getLogger(__name__)

# TR_ID mappings for Korean domestic stock operations
TR_ID_KR_LIVE = {
    # Market data (same for live/paper)
    "PRICE": "FHKST01010100",
    "DAILY_CANDLE": "FHKST03010100",
    "ORDERBOOK": "FHKST01010200",
    # Orders
    "BUY": "TTTC0802U",
    "SELL": "TTTC0801U",
    "CANCEL": "TTTC0803U",
    # Account
    "BALANCE": "TTTC8434R",
    "PENDING_ORDERS": "TTTC8036R",
    "EXECUTED_ORDERS": "TTTC8001R",
    "BUYING_POWER": "TTTC8908R",
}

TR_ID_KR_PAPER = {
    **TR_ID_KR_LIVE,
    "BUY": "VTTC0802U",
    "SELL": "VTTC0801U",
    "CANCEL": "VTTC0803U",
    "BALANCE": "VTTC8434R",
    "PENDING_ORDERS": "VTTC8036R",
    "EXECUTED_ORDERS": "VTTC8001R",
    "BUYING_POWER": "VTTC8908R",
}

# Exchange code mapping: our enum -> KIS FID_COND_MRKT_DIV_CODE
_MRKT_DIV = {"KRX": "J", "KOSDAQ": "K"}


class KISKRAdapter(ExchangeAdapter):
    """KIS REST API adapter for Korean domestic stocks."""

    def __init__(self, config: KISConfig, auth: KISAuth):
        self._config = config
        self._auth = auth
        self._session: aiohttp.ClientSession | None = None
        self._is_paper = "vts" in config.base_url
        self._tr = TR_ID_KR_PAPER if self._is_paper else TR_ID_KR_LIVE

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._auth.initialize()
        logger.info(
            "KIS KR adapter initialized (mode=%s)",
            "paper" if self._is_paper else "live",
        )

    async def close(self) -> None:
        if self._session:
            await self._session.close()
        await self._auth.close()

    @staticmethod
    def _detect_exchange(symbol: str) -> str:
        """Detect exchange from symbol using kr_screener universe map."""
        try:
            from scanner.kr_screener import get_kr_exchange
            return get_kr_exchange(symbol)
        except Exception:
            return "KRX"

    # -- Market Data --

    async def fetch_ticker(self, symbol: str, exchange: str = "KRX") -> Ticker:
        await self._auth.ensure_valid_token()
        params = {
            "FID_COND_MRKT_DIV_CODE": _MRKT_DIV.get(exchange, "J"),
            "FID_INPUT_ISCD": symbol,
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            self._tr["PRICE"],
            params,
        )
        output = data.get("output", {})
        return Ticker(
            symbol=symbol,
            price=float(output.get("stck_prpr", 0)),
            change_pct=float(output.get("prdy_ctrt", 0)),
            volume=float(output.get("acml_vol", 0)),
        )

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
        exchange: str = "KRX",
    ) -> list[Candle]:
        await self._auth.ensure_valid_token()

        period_map = {"1D": "D", "1W": "W", "1M": "M"}
        period = period_map.get(timeframe, "D")

        # Date range: fetch enough history
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y%m%d")
        days_back = {"D": limit * 2, "W": limit * 10, "M": limit * 40}
        start_date = (
            datetime.now() - timedelta(days=days_back.get(period, limit * 2))
        ).strftime("%Y%m%d")

        params = {
            "FID_COND_MRKT_DIV_CODE": _MRKT_DIV.get(exchange, "J"),
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0",  # 수정주가
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            self._tr["DAILY_CANDLE"],
            params,
        )
        candles = []
        for item in data.get("output2", []):
            try:
                date_str = item.get("stck_bsop_date", "0")
                if not date_str or date_str == "0":
                    continue
                candles.append(
                    Candle(
                        timestamp=int(date_str),
                        open=float(item.get("stck_oprc", 0)),
                        high=float(item.get("stck_hgpr", 0)),
                        low=float(item.get("stck_lwpr", 0)),
                        close=float(item.get("stck_clpr", 0)),
                        volume=float(item.get("acml_vol", 0)),
                    )
                )
            except (ValueError, KeyError):
                continue
        candles.reverse()  # oldest first
        return candles[-limit:]

    async def fetch_orderbook(
        self, symbol: str, exchange: str = "KRX", limit: int = 20
    ) -> OrderBook:
        await self._auth.ensure_valid_token()
        params = {
            "FID_COND_MRKT_DIV_CODE": _MRKT_DIV.get(exchange, "J"),
            "FID_INPUT_ISCD": symbol,
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            self._tr["ORDERBOOK"],
            params,
        )
        output = data.get("output1", {})
        bids = []
        asks = []
        for i in range(1, min(limit, 11)):
            bp = float(output.get(f"bidp{i}", 0))
            bq = float(output.get(f"bidp_rsqn{i}", 0))
            ap = float(output.get(f"askp{i}", 0))
            aq = float(output.get(f"askp_rsqn{i}", 0))
            if bp > 0:
                bids.append((bp, bq))
            if ap > 0:
                asks.append((ap, aq))
        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    # -- Account --

    async def fetch_balance(self) -> Balance:
        await self._auth.ensure_valid_token()
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",   # 종목별
            "UNPR_DVSN": "01",   # 기준가
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            self._tr["BALANCE"],
            params,
        )

        output2 = data.get("output2", [])
        if isinstance(output2, list) and output2:
            output2 = output2[0]
        elif not isinstance(output2, dict):
            output2 = {}

        total = float(output2.get("tot_evlu_amt", 0))         # 총평가금액
        invested = float(output2.get("pchs_amt_smtl_amt", 0)) # 매입금액합계
        deposit = float(output2.get("dnca_tot_amt", 0))       # 예수금총금액

        # Get actual orderable amount via 주문가능조회
        # dnca_tot_amt includes unsettled US stock buys — not actual buying power
        available = await self._fetch_orderable_amount()
        if available is None:
            available = deposit  # fallback

        logger.debug(
            "KR balance: deposit=%.0f, orderable=%.0f, total=%.0f, invested=%.0f",
            deposit, available, total, invested,
        )

        return Balance(
            currency="KRW",
            total=total or (available + invested),
            available=available,
            locked=invested,
        )

    async def _fetch_orderable_amount(self) -> float | None:
        """Query actual orderable cash via 주문가능조회 API.

        Tests multiple parameter combinations to find the correct
        domestic buying power. OVRS_ICLD_YN=Y can over-deduct due to
        overseas settlement obligations.
        """
        try:
            # Try OVRS_ICLD_YN=N first (domestic-only buying power)
            for ovrs in ("N", "Y"):
                params = {
                    "CANO": self._config.account_no,
                    "ACNT_PRDT_CD": self._config.account_product,
                    "PDNO": "005930",       # Reference stock (삼성전자)
                    "ORD_UNPR": "0",        # 0 = market order context
                    "ORD_DVSN": "01",       # 01 = 시장가
                    "CMA_EVLU_AMT_ICLD_YN": "Y",
                    "OVRS_ICLD_YN": ovrs,
                }
                data = await self._get(
                    "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                    self._tr["BUYING_POWER"],
                    params,
                )
                if data.get("rt_cd") != "0":
                    continue
                output = data.get("output", {})
                cash = float(output.get("ord_psbl_cash", 0))
                logger.info(
                    "KR 주문가능 (OVRS=%s): cash=%.0f, max_buy=%.0f",
                    ovrs, cash, float(output.get("max_buy_amt", 0)),
                )
                if cash > 0:
                    return cash

            # Both returned non-positive — use the OVRS=N result as best guess
            logger.warning("KR orderable cash is non-positive, using OVRS=N result: %.0f", cash)
            return cash
        except Exception as e:
            logger.warning("Failed to fetch KR orderable amount: %s", e)
            return None

    async def fetch_positions(self) -> list[Position]:
        await self._auth.ensure_valid_token()
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            self._tr["BALANCE"],
            params,
        )
        positions = []
        for item in data.get("output1", []):
            qty = float(item.get("hldg_qty", 0))
            if qty <= 0:
                continue
            avg_price = float(item.get("pchs_avg_pric", 0))
            cur_price = float(item.get("prpr", 0))
            pnl = float(item.get("evlu_pfls_amt", 0))
            positions.append(
                Position(
                    symbol=item.get("pdno", ""),
                    exchange=self._detect_exchange(item.get("pdno", "")),
                    quantity=qty,
                    avg_price=avg_price,
                    current_price=cur_price,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=(
                        ((cur_price - avg_price) / avg_price * 100)
                        if avg_price > 0
                        else 0
                    ),
                )
            )
        return positions

    async def fetch_buying_power(self) -> float:
        balance = await self.fetch_balance()
        return balance.available

    # -- Orders --

    async def create_buy_order(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
        exchange: str = "KRX",
    ) -> OrderResult:
        return await self._place_order(
            symbol, "buy", quantity, price, order_type, self._tr["BUY"]
        )

    async def create_sell_order(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
        exchange: str = "KRX",
    ) -> OrderResult:
        return await self._place_order(
            symbol, "sell", quantity, price, order_type, self._tr["SELL"]
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self._auth.ensure_valid_token()
        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02 = cancel
            "ORD_QTY": "0",
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
        }
        hashkey = await self._auth.get_hashkey(body)
        data = await self._post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            self._tr["CANCEL"],
            body,
            hashkey,
        )
        return data.get("rt_cd") == "0"

    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        # 1) Check pending (unfilled) orders
        orders = await self.fetch_pending_orders()
        for o in orders:
            if o.order_id == order_id:
                return o
        # 2) Check today's executed orders (체결내역)
        executed = await self.fetch_executed_orders()
        for o in executed:
            if o.order_id == order_id:
                return o
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side="unknown",
            order_type="unknown",
            quantity=0,
            status="not_found",
        )

    async def fetch_executed_orders(self) -> list[OrderResult]:
        """Fetch today's executed (filled) orders — 주식일별주문체결조회."""
        await self._auth.ensure_valid_token()
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "INQR_STRT_DT": today,
            "INQR_END_DT": today,
            "SLL_BUY_DVSN_CD": "00",  # 전체 (매수+매도)
            "INQR_DVSN": "00",  # 역순
            "PDNO": "",
            "CCLD_DVSN": "01",  # 체결만
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            self._tr["EXECUTED_ORDERS"],
            params,
        )
        results = []
        for item in data.get("output1", []):
            total_qty = float(item.get("ord_qty", 0))
            filled_qty = float(item.get("tot_ccld_qty", 0))
            filled_price = float(item.get("avg_prvs", 0)) or None
            if filled_qty <= 0:
                continue
            results.append(
                OrderResult(
                    order_id=item.get("odno", ""),
                    symbol=item.get("pdno", ""),
                    side="buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                    order_type="limit",
                    quantity=total_qty,
                    price=float(item.get("ord_unpr", 0)),
                    filled_quantity=filled_qty,
                    filled_price=filled_price,
                    status="filled" if filled_qty >= total_qty else "partial",
                )
            )
        return results

    async def fetch_pending_orders(self) -> list[OrderResult]:
        await self._auth.ensure_valid_token()
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1": "0",  # 전체
            "INQR_DVSN_2": "0",  # 전체
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
            self._tr["PENDING_ORDERS"],
            params,
        )
        raw_output = data.get("output", [])
        if raw_output:
            logger.info(
                "KR pending orders raw: %d items, first=%s",
                len(raw_output),
                {k: v for k, v in raw_output[0].items() if v and v != "0"} if raw_output else {},
            )
        results = []
        for item in raw_output:
            qty = float(item.get("ord_qty", 0))
            # psbl_qty = 취소가능수량 (cancellable/unfilled quantity)
            psbl = float(item.get("psbl_qty", 0))
            if psbl <= 0:
                continue
            results.append(
                OrderResult(
                    order_id=item.get("odno", ""),
                    symbol=item.get("pdno", ""),
                    side="buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                    order_type="limit",
                    quantity=qty,
                    price=float(item.get("ord_unpr", 0)),
                    filled_quantity=qty - psbl,
                    status="open",
                )
            )
        return results

    # -- Private helpers --

    async def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None,
        order_type: str,
        tr_id: str,
    ) -> OrderResult:
        await self._auth.ensure_valid_token()

        ord_dvsn = "00" if order_type == "limit" else "01"
        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)) if price else "0",
        }

        hashkey = await self._auth.get_hashkey(body)
        data = await self._post(
            "/uapi/domestic-stock/v1/trading/order-cash",
            tr_id,
            body,
            hashkey,
        )

        output = data.get("output", {})
        success = data.get("rt_cd") == "0"

        return OrderResult(
            order_id=output.get("ODNO", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="pending" if success else "failed",
        )

    async def _get(
        self, path: str, tr_id: str, params: dict[str, str],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        import asyncio
        url = f"{self._config.base_url}{path}"
        data: dict[str, Any] = {}
        for attempt in range(max_retries):
            headers = self._auth.get_auth_headers(tr_id)
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status >= 400:
                    # Try to parse body — KIS returns EGW00201 rate-limit as HTTP 500
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"rt_cd": "-1", "msg1": f"HTTP {resp.status}"}
                    logger.warning("KIS KR HTTP %d for GET %s: %s", resp.status, path, data.get("msg1", ""))
                    msg_cd = data.get("msg_cd", "")
                    if msg_cd == "EGW00201" and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.3 * (attempt + 1))
                        continue
                    return data
                try:
                    data = await resp.json()
                except (json.JSONDecodeError, ContentTypeError, ValueError):
                    data = {"rt_cd": "-1", "msg1": "Invalid JSON response"}
                    return data
            if data.get("rt_cd") == "0":
                return data
            msg_cd = data.get("msg_cd", "")
            if msg_cd == "EGW00201" and attempt < max_retries - 1:
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
            logger.warning("KIS KR API error: %s %s", msg_cd, data.get("msg1"))
            return data
        return data

    async def _post(
        self, path: str, tr_id: str, body: dict, hashkey: str = "",
        max_retries: int = 3,
    ) -> dict[str, Any]:
        import asyncio
        url = f"{self._config.base_url}{path}"
        data: dict[str, Any] = {}
        for attempt in range(max_retries):
            headers = self._auth.get_auth_headers(tr_id, hashkey)
            async with self._session.post(url, headers=headers, json=body) as resp:
                if resp.status >= 400:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"rt_cd": "-1", "msg1": f"HTTP {resp.status}"}
                    logger.warning("KIS KR HTTP %d for POST %s: %s", resp.status, path, data.get("msg1", ""))
                    msg_cd = data.get("msg_cd", "")
                    if msg_cd == "EGW00201" and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.3 * (attempt + 1))
                        continue
                    return data
                try:
                    data = await resp.json()
                except (json.JSONDecodeError, ContentTypeError, ValueError):
                    data = {"rt_cd": "-1", "msg1": "Invalid JSON response"}
                    return data
            if data.get("rt_cd") == "0":
                return data
            msg_cd = data.get("msg_cd", "")
            if msg_cd == "EGW00201" and attempt < max_retries - 1:
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
            logger.warning("KIS KR API error: %s %s", msg_cd, data.get("msg1"))
            return data
        return data
