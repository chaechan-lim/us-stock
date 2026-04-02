"""KIS REST API adapter for US stock trading.

Wraps 한국투자증권 Open API endpoints for:
- Market data (quotes, OHLCV, orderbook)
- Order management (buy, sell, cancel, modify)
- Account (balance, positions, buying power)
- Scanner (volume surge, price movers, new highs/lows)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Union

import aiohttp
from aiohttp import ContentTypeError

from config import KISConfig
from config.accounts import AccountConfig
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


@dataclass
class RankedStock:
    """A stock from KIS ranking/screening API."""

    symbol: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    source: str = ""


logger = logging.getLogger(__name__)

# TR_ID mappings for US stock operations
TR_ID_LIVE = {
    # Market data (same for live/paper)
    "PRICE": "HHDFS00000300",
    "PRICE_DETAIL": "HHDFS76200200",
    "ORDERBOOK": "HHDFS76200100",
    "MINUTE_CANDLE": "HHDFS76950200",
    "DAILY_CANDLE": "HHDFS76240000",
    # Orders (regular session)
    "BUY_US": "TTTT1002U",
    "SELL_US": "TTTT1006U",
    "CANCEL_US": "TTTT1004U",
    # Orders (daytime / extended hours)
    "BUY_DAYTIME": "TTTS6036U",
    "SELL_DAYTIME": "TTTS6037U",
    "CANCEL_DAYTIME": "TTTS6038U",
    # Account
    "BALANCE": "TTTS3012R",
    "BUYING_POWER": "TTTS3007R",
    "ORDER_HISTORY": "TTTS3035R",
    "PENDING_ORDERS": "TTTS3018R",
    "PRESENT_BALANCE": "CTRP6504R",
    # Scanner / Ranking
    "VOLUME_SURGE": "HHDFS76270000",
    "UPDOWN_RATE": "HHDFS76290000",
    "NEW_HIGHLOW": "HHDFS76300000",
}

TR_ID_PAPER = {
    **TR_ID_LIVE,
    # Paper-specific overrides (TT -> VT prefix)
    "BUY_US": "VTTT1002U",
    "SELL_US": "VTTT1006U",
    "CANCEL_US": "VTTT1004U",
    "BUY_DAYTIME": "VTTS6036U",  # Paper daytime (may not be supported)
    "SELL_DAYTIME": "VTTS6037U",
    "CANCEL_DAYTIME": "VTTS6038U",
    "BALANCE": "VTTS3012R",
    "BUYING_POWER": "VTTS3007R",
    "ORDER_HISTORY": "VTTS3035R",
    "PENDING_ORDERS": "VTTS3018R",
}

# Daytime (extended hours) exchange codes: regular NASD->BAQ, NYSE->BAY, AMEX->BAA
_DAYTIME_EXCHANGE = {"NASD": "BAQ", "NYSE": "BAY", "AMEX": "BAA"}


# Market data endpoints use 3-char codes; trading endpoints use 4-char codes
_EXCD_MARKET = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}


class KISAdapter(ExchangeAdapter):
    """KIS REST API adapter."""

    def __init__(
        self,
        config: Union[KISConfig, AccountConfig],
        auth: KISAuth,
        account_id: str = "ACC001",
    ):
        self._config = config
        self._auth = auth
        self._account_id = account_id
        self._session: aiohttp.ClientSession | None = None
        self._is_paper = "vts" in config.base_url
        self._tr = TR_ID_PAPER if self._is_paper else TR_ID_LIVE
        self._last_exchange_rate: float = 1450.0  # USD/KRW
        self._usd_deposit_krw: float = 0.0  # 달러예수금 (KRW equivalent)
        self._tot_asst_krw: float = 0.0  # CTRP6504R tot_asst_amt (해외자산+예수금)
        self._tot_dncl_krw: float = 0.0  # CTRP6504R tot_dncl_amt (예수금, 통합증거금 공유)

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._auth.initialize()
        logger.info("KIS adapter initialized (mode=%s)", "paper" if self._is_paper else "live")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
        await self._auth.close()

    # -- Market Data --

    async def fetch_ticker(self, symbol: str, exchange: str = "NASD") -> Ticker:
        await self._auth.ensure_valid_token()
        params = {
            "AUTH": "",
            "EXCD": _EXCD_MARKET.get(exchange, exchange),
            "SYMB": symbol,
        }
        data = await self._get(
            "/uapi/overseas-price/v1/quotations/price",
            self._tr["PRICE"],
            params,
        )
        output = data.get("output", {})
        return Ticker(
            symbol=symbol,
            price=float(output.get("last", 0)),
            change_pct=float(output.get("rate", 0)),
            volume=float(output.get("tvol", 0)),
        )

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
        exchange: str = "NASD",
    ) -> list[Candle]:
        await self._auth.ensure_valid_token()

        # Map timeframe to KIS period type
        period_map = {"1D": "D", "1W": "W", "1M": "M"}
        period = period_map.get(timeframe, "D")

        params = {
            "AUTH": "",
            "EXCD": _EXCD_MARKET.get(exchange, exchange),
            "SYMB": symbol,
            "GUBN": period,
            "BYMD": "",  # empty = latest
            "MODP": "1",
        }
        data = await self._get(
            "/uapi/overseas-price/v1/quotations/dailyprice",
            self._tr["DAILY_CANDLE"],
            params,
        )
        candles = []
        for item in data.get("output2", [])[:limit]:
            try:
                candles.append(
                    Candle(
                        timestamp=int(item.get("xymd", "0")),
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("clos", 0)),
                        volume=float(item.get("tvol", 0)),
                    )
                )
            except (ValueError, KeyError):
                continue
        candles.reverse()  # oldest first
        return candles

    async def fetch_orderbook(
        self, symbol: str, exchange: str = "NASD", limit: int = 20
    ) -> OrderBook:
        await self._auth.ensure_valid_token()
        # KIS orderbook endpoint for overseas stocks
        params = {
            "AUTH": "",
            "EXCD": _EXCD_MARKET.get(exchange, exchange),
            "SYMB": symbol,
        }
        data = await self._get(
            "/uapi/overseas-price/v1/quotations/inquire-asking-price",
            self._tr["ORDERBOOK"],
            params,
        )
        # Parse bid/ask from response
        output = data.get("output", {})
        bids = []
        asks = []
        for i in range(1, min(limit, 11)):
            bp = float(output.get(f"bidp{i}", 0))
            bq = float(output.get(f"bidq{i}", 0))
            ap = float(output.get(f"askp{i}", 0))
            aq = float(output.get(f"askq{i}", 0))
            if bp > 0:
                bids.append((bp, bq))
            if ap > 0:
                asks.append((ap, aq))
        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    # -- Account --

    async def fetch_balance(self) -> Balance:
        await self._auth.ensure_valid_token()

        # 1. Present balance (includes KRW deposits + USD positions/cash)
        pb_params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "WCRC_FRCR_DVSN_CD": "01",  # 외화
            "NATN_CD": "840",  # USA
            "TR_MKET_CD": "00",  # 전체
            "INQR_DVSN_CD": "00",
        }
        pb_data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-present-balance",
            self._tr.get("PRESENT_BALANCE", "CTRP6504R"),
            pb_params,
        )
        # output3 has total asset in KRW (tot_asst_amt) and exchange rate
        pb_o3 = pb_data.get("output3", {})
        if isinstance(pb_o3, list) and pb_o3:
            pb_o3 = pb_o3[0]

        # Get exchange rate and total KRW deposits
        exrt = float(pb_o3.get("frst_bltn_exrt", 0)) if pb_o3 else 0
        if exrt <= 0:
            exrt = 1450.0  # fallback
        tot_asst_krw = float(pb_o3.get("tot_asst_amt", 0)) if pb_o3 else 0
        tot_dncl_krw = float(pb_o3.get("tot_dncl_amt", 0)) if pb_o3 else 0

        # 2. Position value from inquire-balance (more accurate per-position data)
        bal_params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            self._tr["BALANCE"],
            bal_params,
        )
        position_value = 0.0
        for item in data.get("output1", []):
            qty = float(item.get("ovrs_cblc_qty", 0))
            cur_price = float(item.get("now_pric2", 0))
            if qty > 0 and cur_price > 0:
                position_value += qty * cur_price

        # 3. Buying power (available cash for orders)
        bp_params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": "NASD",
            "OVRS_ORD_UNPR": "1",
            "ITEM_CD": "AAPL",
        }
        bp_data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            self._tr["BUYING_POWER"],
            bp_params,
        )
        bp_output = bp_data.get("output", {})

        # Use frcr_ord_psbl_amt1 (includes KRW auto-conversion) as available
        available = float(bp_output.get("frcr_ord_psbl_amt1", 0))
        if available <= 0:
            available = float(bp_output.get("ord_psbl_frcr_amt", 0))
        if available <= 0:
            available = await self._estimate_usd_from_krw()

        # Cache exchange rate, total, and USD deposit for cross-market calculations
        self._last_exchange_rate = exrt
        self._tot_asst_krw = tot_asst_krw  # 해외자산 + 예수금 (KR stocks 미포함)
        self._tot_dncl_krw = tot_dncl_krw  # 예수금 (통합증거금 공유)
        self._usd_deposit_krw = float(pb_o3.get("frcr_evlu_tota", 0)) if pb_o3 else 0
        # STOCK-53: Store uncapped buying power + positions for total_equity calc.
        # frcr_ord_psbl_amt1 includes KRW auto-conversion, reflecting the full
        # account capacity (통합증거금). Used by portfolio summary for accurate total.
        self._full_account_usd = available + position_value
        # STOCK-42: Store uncapped frcr_ord_psbl_amt1 for available_cash calc.
        # In 통합증거금 accounts, this already includes KRW auto-conversion,
        # so portfolio.py should use this directly instead of summing KR+US available.
        self._full_available_usd = available

        # 4. Total: use present-balance total if available, else fallback
        if tot_asst_krw > 0:
            total = tot_asst_krw / exrt  # Convert KRW total to USD
            # STOCK-53: frcr_ord_psbl_amt1 can exceed tot_asst_amt/exrt because
            # it includes KRW auto-conversion to USD. Cap available at total
            # to prevent negative invested values in exposure calculations.
            if available > total:
                logger.info(
                    "US available ($%.2f) exceeds total ($%.2f), "
                    "capping to prevent negative exposure",
                    available,
                    total,
                )
                available = total - position_value
                if available < 0:
                    available = 0.0
            locked = total - available
            logger.info(
                "US balance: total=₩%s ($%.2f), deposits=₩%s, available=$%.2f (rate=%.1f)",
                f"{tot_asst_krw:,.0f}",
                total,
                f"{tot_dncl_krw:,.0f}",
                available,
                exrt,
            )
        else:
            total = available + position_value
            locked = position_value

        return Balance(
            currency="USD",
            total=total,
            available=available,
            locked=locked,
        )

    async def _estimate_usd_from_krw(self) -> float:
        """Estimate USD buying power from KRW deposit balance."""
        try:
            # Query domestic deposit balance (KRW)
            params = {
                "CANO": self._config.account_no,
                "ACNT_PRDT_CD": self._config.account_product,
                "OVRS_EXCG_CD": "NASD",
                "TR_CRCY_CD": "KRW",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            }
            data = await self._get(
                "/uapi/overseas-stock/v1/trading/inquire-balance",
                self._tr["BALANCE"],
                params,
            )
            output2 = data.get("output2", {})
            if isinstance(output2, list) and output2:
                output2 = output2[0]
            logger.debug("KRW balance output2: %s", output2)

            # Try various KRW balance fields
            krw_balance = 0.0
            for field in (
                "tot_evlu_pfls_amt",
                "frcr_buy_amt_smtl1",
                "ovrs_rlzt_pfls_amt",
                "tot_ord_psbl_amt",
            ):
                val = output2.get(field, "")
                if val and float(val) > 0:
                    krw_balance = float(val)
                    break

            if krw_balance <= 0:
                # Get total KRW from all available fields
                for field in ("dnca_tot_amt", "nass_amt", "tot_evlu_amt"):
                    val = output2.get(field, "")
                    if val and float(val) > 0:
                        krw_balance = float(val)
                        break

            if krw_balance > 0:
                rate = await self._fetch_exchange_rate()
                if rate <= 0:
                    rate = 1380.0  # fallback approximate rate
                usd_equiv = krw_balance / rate * 0.99  # 1% buffer for FX spread
                logger.info(
                    "Estimated USD buying power from KRW: ₩%.0f / %.1f = $%.2f",
                    krw_balance,
                    rate,
                    usd_equiv,
                )
                return usd_equiv
        except Exception as e:
            logger.warning("Failed to estimate USD from KRW: %s", e)
        return 0.0

    async def _fetch_exchange_rate(self) -> float:
        """Fetch USD/KRW exchange rate from KIS API."""
        try:
            params = {
                "AUTH": "",
                "EXCD": "NAS",
                "SYMB": "AAPL",
            }
            data = await self._get(
                "/uapi/overseas-price/v1/quotations/price-detail",
                self._tr.get("PRICE_DETAIL", "HHDFS76200200"),
                params,
            )
            output = data.get("output", {})
            # t_rate = today's exchange rate
            rate = float(output.get("t_rate", 0))
            if rate > 0:
                logger.debug("Exchange rate from KIS: %.2f", rate)
                return rate
        except Exception as e:
            logger.debug("Exchange rate fetch failed: %s", e)
        return 0.0

    async def fetch_positions(self) -> list[Position]:
        await self._auth.ensure_valid_token()
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            self._tr["BALANCE"],
            params,
        )
        positions = []
        for item in data.get("output1", []):
            qty = float(item.get("ovrs_cblc_qty", 0))
            if qty <= 0:
                continue
            avg_price = float(item.get("pchs_avg_pric", 0))
            cur_price = float(item.get("now_pric2", 0))
            pnl = float(item.get("frcr_evlu_pfls_amt", 0))
            positions.append(
                Position(
                    symbol=item.get("ovrs_pdno", ""),
                    exchange=item.get("ovrs_excg_cd", "NASD"),
                    quantity=qty,
                    avg_price=avg_price,
                    current_price=cur_price,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=(
                        ((cur_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
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
        exchange: str = "NASD",
        session: str = "regular",
    ) -> OrderResult:
        if session in ("pre_market", "after_hours"):
            return await self._place_daytime_order(
                symbol,
                "buy",
                quantity,
                price,
                exchange,
            )
        return await self._place_order(
            symbol, "buy", quantity, price, order_type, exchange, self._tr["BUY_US"]
        )

    async def create_sell_order(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
        exchange: str = "NASD",
        session: str = "regular",
    ) -> OrderResult:
        if session in ("pre_market", "after_hours"):
            return await self._place_daytime_order(
                symbol,
                "sell",
                quantity,
                price,
                exchange,
            )
        return await self._place_order(
            symbol, "sell", quantity, price, order_type, exchange, self._tr["SELL_US"]
        )

    async def cancel_order(self, order_id: str, symbol: str, exchange: str = "NASD") -> bool:
        await self._auth.ensure_valid_token()
        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": exchange,
            "PDNO": symbol,
            "ORGN_ODNO": order_id,
            "RVSE_CNCL_DVSN_CD": "02",  # 02 = cancel
            "ORD_QTY": "0",  # full cancel
            "OVRS_ORD_UNPR": "0",
        }
        hashkey = await self._auth.get_hashkey(body)
        data = await self._post(
            "/uapi/overseas-stock/v1/trading/order-rvsecncl",
            self._tr["CANCEL_US"],
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
        """Fetch today's executed (filled) orders — 해외주식 체결내역."""
        await self._auth.ensure_valid_token()
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # Use US Eastern time for query date
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today = now_et.strftime("%Y%m%d")
        params = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "PDNO": "%",  # all symbols
            "ORD_STRT_DT": today,
            "ORD_END_DT": today,
            "SLL_BUY_DVSN_CD": "00",  # all (buy+sell)
            "CCLD_NCCS_DVSN": "01",  # 체결만
            "OVRS_EXCG_CD": "NASD",
            "SORT_SQN": "DS",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-ccnl",
            self._tr["ORDER_HISTORY"],
            params,
        )
        results = []
        for item in data.get("output", []):
            total_qty = float(item.get("ft_ord_qty", 0))
            filled_qty = float(item.get("ft_ccld_qty", 0))
            filled_price = float(item.get("ft_ccld_unpr3", 0)) or None
            if filled_qty <= 0:
                continue
            results.append(
                OrderResult(
                    order_id=item.get("odno", ""),
                    symbol=item.get("pdno", ""),
                    side="buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                    order_type="limit",
                    quantity=total_qty,
                    price=float(item.get("ft_ord_unpr3", 0)),
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
            "OVRS_EXCG_CD": "NASD",
            "SORT_SQN": "DS",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/trading/inquire-nccs",
            self._tr["PENDING_ORDERS"],
            params,
        )
        results = []
        for item in data.get("output", []):
            results.append(
                OrderResult(
                    order_id=item.get("odno", ""),
                    symbol=item.get("pdno", ""),
                    side="buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                    order_type="limit",
                    quantity=float(item.get("ft_ord_qty", 0)),
                    price=float(item.get("ft_ord_unpr3", 0)),
                    filled_quantity=float(item.get("ft_ccld_qty", 0)),
                    status="open",
                )
            )
        return results

    # -- Scanner / Ranking --

    async def fetch_volume_surge(
        self,
        exchange: str = "NAS",
        limit: int = 20,
    ) -> list[RankedStock]:
        """Fetch stocks with surging volume."""
        await self._auth.ensure_valid_token()
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "MINX": "5",  # 10 min window
            "VOL_RANG": "1",  # >= 100 shares
            "KEYB": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/ranking/volume-surge",
            self._tr["VOLUME_SURGE"],
            params,
        )
        return self._parse_ranked(data, "volume_surge", limit)

    async def fetch_updown_rate(
        self,
        exchange: str = "NAS",
        direction: str = "up",
        limit: int = 20,
    ) -> list[RankedStock]:
        """Fetch stocks by price change rate (gainers or losers)."""
        await self._auth.ensure_valid_token()
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "GUBN": "1" if direction == "up" else "0",
            "VOL_RANG": "1",  # >= 100 shares
            "KEYB": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/ranking/updown-rate",
            self._tr["UPDOWN_RATE"],
            params,
        )
        return self._parse_ranked(data, f"updown_{direction}", limit)

    async def fetch_new_highlow(
        self,
        exchange: str = "NAS",
        high: bool = True,
        limit: int = 20,
    ) -> list[RankedStock]:
        """Fetch stocks hitting new highs or new lows."""
        await self._auth.ensure_valid_token()
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "MINX": "9",  # 120 min window
            "VOL_RANG": "1",  # >= 100 shares
            "GUBN": "1" if high else "0",
            "GUBN2": "1",  # sustained (not momentary)
            "KEYB": "",
        }
        data = await self._get(
            "/uapi/overseas-stock/v1/ranking/new-highlow",
            self._tr["NEW_HIGHLOW"],
            params,
        )
        return self._parse_ranked(data, "new_high" if high else "new_low", limit)

    def _parse_ranked(
        self,
        data: dict[str, Any],
        source: str,
        limit: int,
    ) -> list[RankedStock]:
        """Parse KIS ranking API response into RankedStock list."""
        results: list[RankedStock] = []
        for item in data.get("output2", data.get("output", []))[:limit]:
            if isinstance(item, dict):
                symbol = item.get("symb", item.get("stck_shrn_iscd", "")).strip()
                if not symbol:
                    continue
                results.append(
                    RankedStock(
                        symbol=symbol,
                        name=item.get("name", item.get("hts_kor_isnm", "")),
                        price=float(item.get("last", item.get("stck_prpr", 0)) or 0),
                        change_pct=float(item.get("rate", item.get("prdy_ctrt", 0)) or 0),
                        volume=float(item.get("tvol", item.get("acml_vol", 0)) or 0),
                        source=source,
                    )
                )
        return results

    # -- Private helpers --

    async def _place_daytime_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None,
        exchange: str,
    ) -> OrderResult:
        """Place a daytime (extended hours) order via KIS API.

        Uses separate endpoint/TR_ID for US pre-market/after-hours trading.
        - Endpoint: /uapi/overseas-stock/v1/trading/daytime-order
        - Buy TR_ID: TTTS6036U, Sell TR_ID: TTTS6037U
        - Exchange codes: BAQ (NASD), BAY (NYSE), BAA (AMEX)
        - Limit orders only (ORD_DVSN=00)
        - Trading hours: KST 10:00~23:20 (DST: 10:00~22:20)
        """
        await self._auth.ensure_valid_token()

        if price is None or price <= 0:
            logger.warning("Daytime order requires limit price, got %s", price)
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                order_type="limit",
                quantity=quantity,
                status="failed",
            )

        daytime_exchange = _DAYTIME_EXCHANGE.get(exchange, "BAQ")
        tr_id = self._tr["BUY_DAYTIME"] if side == "buy" else self._tr["SELL_DAYTIME"]

        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": daytime_exchange,
            "PDNO": symbol,
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # Limit only
        }

        hashkey = await self._auth.get_hashkey(body)
        data = await self._post(
            "/uapi/overseas-stock/v1/trading/daytime-order",
            tr_id,
            body,
            hashkey,
        )

        output = data.get("output", {})
        success = data.get("rt_cd") == "0"

        if success:
            logger.info(
                "Daytime %s order placed: %s %d @ $%.2f (exchange=%s)",
                side,
                symbol,
                quantity,
                price,
                daytime_exchange,
            )
        else:
            logger.warning(
                "Daytime %s order failed: %s — %s",
                side,
                symbol,
                data.get("msg1", "unknown"),
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
        exchange: str,
        tr_id: str,
    ) -> OrderResult:
        await self._auth.ensure_valid_token()

        ord_dvsn = "00" if order_type == "limit" else "01"
        # Reject limit orders with invalid price (same guard as _place_daytime_order)
        if ord_dvsn == "00" and (price is None or price <= 0):
            logger.warning("Limit order requires price > 0, got %s", price)
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                order_type="limit",
                quantity=quantity,
                status="failed",
            )
        # KIS US market orders (ord_dvsn="01") require OVRS_ORD_UNPR to be "0"
        if ord_dvsn == "01":
            order_price = "0"
        else:
            order_price = f"{price:.2f}" if price is not None else "0"
        sll_type = "00" if side == "sell" else ""
        body = {
            "CANO": self._config.account_no,
            "ACNT_PRDT_CD": self._config.account_product,
            "OVRS_EXCG_CD": exchange,
            "PDNO": symbol,
            "ORD_QTY": str(quantity),
            "OVRS_ORD_UNPR": order_price,
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": sll_type,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": ord_dvsn,
        }

        hashkey = await self._auth.get_hashkey(body)
        data = await self._post(
            "/uapi/overseas-stock/v1/trading/order",
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
        self,
        path: str,
        tr_id: str,
        params: dict[str, str],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        import asyncio

        url = f"{self._config.base_url}{path}"
        data = {}
        for attempt in range(max_retries):
            headers = self._auth.get_auth_headers(tr_id)
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status >= 400:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"rt_cd": "-1", "msg1": f"HTTP {resp.status}"}
                    logger.warning(
                        "KIS HTTP %d for GET %s: %s", resp.status, path, data.get("msg1", "")
                    )
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
            logger.warning("KIS API error: %s %s", msg_cd, data.get("msg1"))
            return data
        return data

    async def _post(
        self,
        path: str,
        tr_id: str,
        body: dict,
        hashkey: str = "",
        max_retries: int = 3,
    ) -> dict[str, Any]:
        import asyncio

        url = f"{self._config.base_url}{path}"
        data = {}
        for attempt in range(max_retries):
            headers = self._auth.get_auth_headers(tr_id, hashkey)
            async with self._session.post(url, headers=headers, json=body) as resp:
                if resp.status >= 400:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"rt_cd": "-1", "msg1": f"HTTP {resp.status}"}
                    logger.warning(
                        "KIS HTTP %d for POST %s: %s", resp.status, path, data.get("msg1", "")
                    )
                    msg_cd = data.get("msg_cd", "")
                    if msg_cd == "EGW00201" and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
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
            logger.warning("KIS API error: %s %s", msg_cd, data.get("msg1"))
            return data
        return data
