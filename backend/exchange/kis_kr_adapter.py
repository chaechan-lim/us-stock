"""KIS REST API adapter for Korean domestic stock trading.

Wraps 한국투자증권 Open API domestic endpoints for:
- Market data (현재가, 일봉, 호가)
- Order management (매수, 매도, 취소)
- Account (잔고, 보유종목)
- Scanner / Ranking (거래량 급등, 등락률, 신고가/신저가)

Shares the same KISAuth (OAuth token) as the US adapter.
"""

import json
import logging
from dataclasses import dataclass
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
from exchange.kis_auth import KISAuth, is_token_error as _is_token_error
from exchange.utils import safe_float as _safe_float

logger = logging.getLogger(__name__)


@dataclass
class KRRankedStock:
    """A KR stock from KIS domestic ranking API.

    Uses @dataclass (not Pydantic) — intentional for lightweight internal DTO
    matching the US RankedStock pattern. No validation needed for transport.
    """

    symbol: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    exchange: str = "KRX"  # KRX (KOSPI) or KOSDAQ
    source: str = ""


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
    "ACCOUNT_BALANCE": "CTRP6548R",  # 투자계좌자산현황 (통합 총자산)
    "PENDING_ORDERS": "TTTC8036R",
    "EXECUTED_ORDERS": "TTTC8001R",
    "BUYING_POWER": "TTTC8908R",
    # Scanner / Ranking (국내 주식 랭킹)
    "KR_VOLUME_SURGE": "FHPST01720000",   # 거래량 급등
    "KR_UPDOWN_RATE": "FHPST01700000",    # 등락률 순위
    "KR_NEW_HIGHLOW": "FHPST01600000",    # 신고가/신저가
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
        self._tot_evlu_amt: float = 0.0  # 총평가금액 (통합증거금: KR+overseas)
        self._scts_evlu_amt: float = 0.0  # 국내주식 평가금액
        self._dnca_tot_amt: float = 0.0   # 예수금 총액
        self._integrated_total_asset: float = 0.0  # CTRP6548R 통합 총자산
        self._integrated_total_asset_ts: float = 0.0  # last fetch time

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
        except Exception as e:
            logger.warning("Exchange detection failed for %s, defaulting to KRX: %s", symbol, e)
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

        # STOCK-53: Don't use tot_evlu_amt — in integrated margin (통합증거금)
        # accounts it includes overseas assets, inflating KR portfolio value
        # and causing 100% exposure calculation that blocks all KR buys.
        # Use scts_evlu_amt (domestic stock evaluation only) instead.
        stock_eval = float(output2.get("scts_evlu_amt", 0))   # 유가증권평가금액 (domestic only)
        # Store tot_evlu_amt for combined total equity calculation (includes overseas in 통합증거금)
        self._tot_evlu_amt = float(output2.get("tot_evlu_amt", 0))
        invested = float(output2.get("pchs_amt_smtl_amt", 0)) # 매입금액합계
        deposit = float(output2.get("dnca_tot_amt", 0))       # 예수금총금액
        self._scts_evlu_amt = stock_eval
        self._dnca_tot_amt = deposit

        # Get actual orderable amount via 주문가능조회
        # dnca_tot_amt includes unsettled US stock buys — not actual buying power
        available = await self._fetch_orderable_amount()
        if available is None:
            available = deposit  # fallback

        # Domestic-only total asset: deposit + domestic stock market value.
        # available/orderable cash can be lower due to unsettled funds, but that
        # should affect buying power only, not total asset valuation.
        total = deposit + stock_eval
        if total <= 0:
            # Fallback: use purchase cost if stock evaluation not available
            total = deposit + invested

        logger.info(
            "KR balance: deposit=%.0f, orderable=%.0f, stock_eval=%.0f, "
            "total=%.0f, invested=%.0f",
            deposit, available, stock_eval, total, invested,
        )

        # Fetch integrated total asset (CTRP6548R) — this is the KIS app's
        # "총자산" number that includes deposit + domestic + overseas stocks.
        await self._fetch_integrated_total_asset()

        return Balance(
            currency="KRW",
            total=total,
            available=available,
            locked=stock_eval or invested,
        )

    async def _fetch_integrated_total_asset(self) -> None:
        """Fetch 통합 총자산 via CTRP6548R (투자계좌자산현황조회).

        This is the single API that returns the KIS app's "총자산" number:
        deposit + domestic stocks + overseas stocks in one field.
        Previously we tried to compute this from VTTC8434R + CTRP6504R
        but the fields don't add up correctly in 통합증거금 accounts.
        """
        import time as _time
        # 30-min cache — avoid extra API call on every balance fetch
        if _time.time() - self._integrated_total_asset_ts < 1800 and self._integrated_total_asset > 0:
            return
        try:
            await self._auth.ensure_valid_token()
            params = {
                "CANO": self._config.account_no[:8],
                "ACNT_PRDT_CD": self._config.account_no[8:] or "01",
                "INQR_DVSN_1": "",
                "BSPR_BF_DT_APLY_YN": "",
            }
            data = await self._get(
                "/uapi/domestic-stock/v1/trading/inquire-account-balance",
                self._tr.get("ACCOUNT_BALANCE", "CTRP6548R"),
                params,
            )
            output2 = data.get("output2", {})
            if isinstance(output2, list) and output2:
                output2 = output2[0]
            elif not isinstance(output2, dict):
                output2 = {}

            tot_asst = float(output2.get("tot_asst_amt", 0))
            if tot_asst > 0:
                self._integrated_total_asset = tot_asst
                self._integrated_total_asset_ts = _time.time()
                logger.info(
                    "CTRP6548R 통합총자산: %.0f (해외: %s, 예수금: %s)",
                    tot_asst,
                    output2.get("ovrs_stck_evlu_amt1", "-"),
                    output2.get("tot_dncl_amt", "-"),
                )
        except Exception as e:
            logger.warning("CTRP6548R fetch failed: %s", e)

    async def _fetch_orderable_amount(self) -> float | None:
        """Query actual orderable cash via 주문가능조회 API.

        Tests multiple parameter combinations to find the correct
        domestic buying power. OVRS_ICLD_YN=Y can over-deduct due to
        overseas settlement obligations.
        """
        try:
            cash = 0.0
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
                    logger.warning(
                        "KR 주문가능조회 failed (OVRS=%s): msg_cd=%s msg=%s",
                        ovrs, data.get("msg_cd", ""), data.get("msg1", ""),
                    )
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
            logger.warning("Failed to fetch KR orderable amount: %s", e, exc_info=True)
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
        session: str = "regular",
    ) -> OrderResult:
        if session == "pre_market":
            # 장전시간외 (07:30~08:30): 전일종가 only, ORD_DVSN=05
            return await self._place_order(
                symbol, "buy", quantity, price, "limit", self._tr["BUY"],
                ord_dvsn_override="05",
            )
        elif session == "after_hours":
            # 장후시간외 (15:40~16:00): 당일종가 only, ORD_DVSN=06
            return await self._place_order(
                symbol, "buy", quantity, price, "limit", self._tr["BUY"],
                ord_dvsn_override="06",
            )
        elif session == "extended_nxt":
            return await self._place_nxt_order(symbol, "buy", quantity, price)
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
        session: str = "regular",
    ) -> OrderResult:
        if session == "pre_market":
            return await self._place_order(
                symbol, "sell", quantity, price, "limit", self._tr["SELL"],
                ord_dvsn_override="05",
            )
        elif session == "after_hours":
            return await self._place_order(
                symbol, "sell", quantity, price, "limit", self._tr["SELL"],
                ord_dvsn_override="06",
            )
        elif session == "extended_nxt":
            return await self._place_nxt_order(symbol, "sell", quantity, price)
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

    async def _place_nxt_order(
        self, symbol: str, side: str, quantity: int, price: float | None,
    ) -> OrderResult:
        """Place order on NXT (넥스트레이드) exchange.

        NXT TR_IDs: Buy TTTC0012U (paper: VTTTC0012U), Sell TTTC0011U (paper: VTTTC0011U).
        EXCG_ID_DVSN_CD: NXT (direct) or SOR (smart order routing).
        Limit orders only. Trading hours: pre 08:00~09:00, after 15:40~20:00.
        """
        await self._auth.ensure_valid_token()

        if side == "buy":
            tr_id = "VTTC0012U" if self._is_paper else "TTTC0012U"
        else:
            tr_id = "VTTC0011U" if self._is_paper else "TTTC0011U"

        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "PDNO": symbol,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)) if price else "0",
            "EXCG_ID_DVSN_CD": "NXT",
            "ORD_DVSN": "00",  # Limit only
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

        if not success:
            logger.warning(
                "KIS KR NXT order failed: %s %s qty=%d msg=%s",
                side, symbol, quantity, data.get("msg1", ""),
            )

        return OrderResult(
            order_id=output.get("ODNO", ""),
            symbol=symbol,
            side=side,
            order_type="limit",
            quantity=quantity,
            price=price,
            status="pending" if success else "failed",
        )

    async def _place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None,
        order_type: str,
        tr_id: str,
        ord_dvsn_override: str | None = None,
    ) -> OrderResult:
        await self._auth.ensure_valid_token()

        ord_dvsn = ord_dvsn_override or ("00" if order_type == "limit" else "01")
        # KIS KR market orders (ord_dvsn="01") require ORD_UNPR to be "0"
        if ord_dvsn == "01":
            order_price = "0"
        else:
            order_price = str(int(price)) if price else "0"
        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": order_price,
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

        if not success:
            msg_cd = data.get("msg_cd", "")
            msg = data.get("msg1", "")
            logger.warning(
                "KIS KR order failed: %s %s %s qty=%d msg_cd=%s msg=%s",
                side, symbol, order_type, quantity, msg_cd, msg,
            )

        return OrderResult(
            order_id=output.get("ODNO", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="pending" if success else "failed",
        )

    # -- Scanner / Ranking --

    async def fetch_volume_surge(
        self, market: str = "J", limit: int = 20,
    ) -> list[KRRankedStock]:
        """Fetch domestic stocks with surging volume.

        Args:
            market: "J" = KOSPI (KRX), "K" = KOSDAQ
            limit: Max number of results to return.
        """
        await self._auth.ensure_valid_token()
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_COND_SCR_DIV_CODE": "20171",  # 거래량 급등 화면
            "FID_INPUT_ISCD": "0000",           # 전체 종목
            "FID_RANK_SORT_CLS_CODE": "0",      # 급등률 내림차순
            "FID_BLNG_CLS_CODE": "0",           # 전체 업종
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/ranking/volume-surge",
            self._tr["KR_VOLUME_SURGE"],
            params,
        )
        exchange = "KRX" if market == "J" else "KOSDAQ"
        return self._parse_kr_ranked(data, "kr_volume_surge", limit, exchange)

    async def fetch_updown_rate(
        self, market: str = "J", direction: str = "up", limit: int = 20,
    ) -> list[KRRankedStock]:
        """Fetch domestic stocks by price change rate (gainers or losers).

        Args:
            market: "J" = KOSPI (KRX), "K" = KOSDAQ
            direction: "up" for gainers, "down" for losers.
            limit: Max number of results to return.
        """
        await self._auth.ensure_valid_token()
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_COND_SCR_DIV_CODE": "20170",  # 등락률 순위 화면
            "FID_INPUT_ISCD": "0000",           # 전체 종목
            "FID_RANK_SORT_CLS_CODE": "0" if direction == "up" else "1",
            "FID_PRCSTEP_RCNT_CLS_CODE": "1",  # 상승/하락
            "FID_BLNG_CLS_CODE": "0",           # 전체 업종
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/ranking/updown-rate",
            self._tr["KR_UPDOWN_RATE"],
            params,
        )
        exchange = "KRX" if market == "J" else "KOSDAQ"
        return self._parse_kr_ranked(data, f"kr_updown_{direction}", limit, exchange)

    async def fetch_new_highlow(
        self, market: str = "J", high: bool = True, limit: int = 20,
    ) -> list[KRRankedStock]:
        """Fetch domestic stocks hitting new highs or new lows.

        Args:
            market: "J" = KOSPI (KRX), "K" = KOSDAQ
            high: True for new highs, False for new lows.
            limit: Max number of results to return.
        """
        await self._auth.ensure_valid_token()
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_COND_SCR_DIV_CODE": "20160",  # 신고가/신저가 화면
            "FID_INPUT_ISCD": "0000",           # 전체 종목
            "FID_RANK_SORT_CLS_CODE": "0",      # 내림차순
            "FID_BLNG_CLS_CODE": "0",           # 전체 업종
            "FID_DIV_CLS_CODE": "1" if high else "2",  # 1=신고가, 2=신저가
        }
        data = await self._get(
            "/uapi/domestic-stock/v1/ranking/new-highlow",
            self._tr["KR_NEW_HIGHLOW"],
            params,
        )
        exchange = "KRX" if market == "J" else "KOSDAQ"
        return self._parse_kr_ranked(
            data, "kr_new_high" if high else "kr_new_low", limit, exchange,
        )

    def _parse_kr_ranked(
        self,
        data: dict[str, Any],
        source: str,
        limit: int,
        exchange: str = "KRX",
    ) -> list[KRRankedStock]:
        """Parse KIS domestic ranking API response into KRRankedStock list."""
        results: list[KRRankedStock] = []
        for item in data.get("output", [])[:limit]:
            if not isinstance(item, dict):
                continue
            # Domestic stock code field (종목코드)
            symbol = item.get("stck_shrn_iscd", item.get("mksc_shrn_iscd", "")).strip()
            if not symbol:
                continue
            results.append(KRRankedStock(
                symbol=symbol,
                name=item.get("hts_kor_isnm", ""),
                price=_safe_float(item.get("stck_prpr", 0)),
                change_pct=_safe_float(item.get("prdy_ctrt", 0)),
                volume=_safe_float(item.get("acml_vol", 0)),
                exchange=exchange,
                source=source,
            ))
        return results

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
                    # Server-rejected token: force re-issue, retry once
                    if _is_token_error(data) and attempt < max_retries - 1:
                        await self._auth.force_refresh()
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
            if _is_token_error(data) and attempt < max_retries - 1:
                await self._auth.force_refresh()
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
                    # Token errors on POST (orders): refresh + retry is safe
                    # because the server rejected the request before processing.
                    if _is_token_error(data) and attempt < max_retries - 1:
                        await self._auth.force_refresh()
                        continue
                    # Non-rate-limit HTTP errors: return immediately without retry
                    # to prevent duplicate orders (server may have processed the request)
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
            if _is_token_error(data) and attempt < max_retries - 1:
                await self._auth.force_refresh()
                continue
            logger.warning("KIS KR API error: %s %s", msg_cd, data.get("msg1"))
            return data
        return data
