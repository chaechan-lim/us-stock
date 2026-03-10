"""External data service: yfinance + FRED.

Provides fundamental data, analyst consensus, and macro indicators
that KIS API does not offer for overseas stocks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsensusData:
    analyst_count: int = 0
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    target_mean: float = 0.0
    target_high: float = 0.0
    target_low: float = 0.0
    target_upside_pct: float = 0.0
    recent_upgrades: int = 0
    recent_downgrades: int = 0


@dataclass
class FundamentalData:
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    profit_margin: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    ps_ratio: float | None = None


@dataclass
class SmartMoneyData:
    institutional_pct: float | None = None
    insider_buy_count_90d: int = 0
    insider_sell_count_90d: int = 0
    short_ratio: float | None = None


@dataclass
class StockInfo:
    symbol: str = ""
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: int = 0
    beta: float | None = None
    high_52w: float = 0.0
    low_52w: float = 0.0
    avg_volume: int = 0
    earnings_date: str | None = None


@dataclass
class StockProfile:
    """Combined profile for a stock: technical + fundamental + consensus."""
    symbol: str
    info: StockInfo = field(default_factory=StockInfo)
    consensus: ConsensusData = field(default_factory=ConsensusData)
    fundamentals: FundamentalData = field(default_factory=FundamentalData)
    smart_money: SmartMoneyData = field(default_factory=SmartMoneyData)


class ExternalDataService:
    """yfinance-based fundamental + consensus data provider."""

    async def get_stock_profile(self, symbol: str, current_price: float = 0.0) -> StockProfile:
        """Build a full StockProfile from yfinance data."""
        profile = StockProfile(symbol=symbol)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Basic info
            profile.info = StockInfo(
                symbol=symbol,
                name=info.get("shortName", ""),
                sector=info.get("sector", ""),
                industry=info.get("industry", ""),
                market_cap=info.get("marketCap", 0),
                beta=info.get("beta"),
                high_52w=info.get("fiftyTwoWeekHigh", 0),
                low_52w=info.get("fiftyTwoWeekLow", 0),
                avg_volume=info.get("averageVolume", 0),
            )

            # Fundamentals
            profile.fundamentals = FundamentalData(
                revenue_growth=info.get("revenueGrowth"),
                earnings_growth=info.get("earningsGrowth"),
                profit_margin=info.get("profitMargins"),
                roe=info.get("returnOnEquity"),
                debt_to_equity=info.get("debtToEquity"),
                free_cash_flow=info.get("freeCashflow"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                peg_ratio=info.get("pegRatio"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
            )

            # Consensus
            profile.consensus = await self._get_consensus(ticker, current_price)

            # Smart money
            profile.smart_money = await self._get_smart_money(ticker, info)

        except Exception as e:
            logger.warning("Failed to fetch yfinance data for %s: %s", symbol, e)

        return profile

    async def get_history(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"stock splits": "stock_splits"})
            return df
        except Exception as e:
            logger.warning("Failed to fetch history for %s: %s", symbol, e)
            return pd.DataFrame()

    async def get_sector_performance(self) -> dict[str, dict[str, float]]:
        """Get 11 GICS sector ETF performance."""
        sector_etfs = {
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

        result = {}
        for etf_symbol, sector_name in sector_etfs.items():
            try:
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(period="3mo")
                if hist.empty:
                    continue

                close = hist["Close"]
                result[sector_name] = {
                    "symbol": etf_symbol,
                    "return_1d": float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0,
                    "return_1w": float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0,
                    "return_1m": float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0,
                    "return_3m": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
                }
            except Exception as e:
                logger.warning("Failed to fetch sector data for %s: %s", etf_symbol, e)

        return result

    async def get_kr_sector_performance(self) -> dict[str, dict[str, float]]:
        """Get KR sector performance via KODEX sector ETFs."""
        kr_sector_etfs = {
            "091160": "반도체",
            "305720": "2차전지",
            "091180": "자동차",
            "244580": "바이오",
            "091170": "금융",
            "315930": "IT",
            "117680": "철강소재",
        }

        result = {}
        for code, sector_name in kr_sector_etfs.items():
            try:
                yf_sym = f"{code}.KS"
                ticker = yf.Ticker(yf_sym)
                hist = ticker.history(period="3mo")
                if hist.empty:
                    continue

                close = hist["Close"]
                result[sector_name] = {
                    "symbol": code,
                    "return_1d": float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0,
                    "return_1w": float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0,
                    "return_1m": float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0,
                    "return_3m": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
                }
            except Exception as e:
                logger.warning("Failed to fetch KR sector data for %s: %s", code, e)

        return result

    async def get_multiple_profiles(
        self, symbols: list[str], current_prices: dict[str, float] | None = None
    ) -> list[StockProfile]:
        """Get profiles for multiple symbols."""
        profiles = []
        prices = current_prices or {}
        for symbol in symbols:
            profile = await self.get_stock_profile(symbol, prices.get(symbol, 0))
            profiles.append(profile)
        return profiles

    # -- Private --

    async def _get_consensus(
        self, ticker: yf.Ticker, current_price: float
    ) -> ConsensusData:
        data = ConsensusData()
        try:
            recs = ticker.recommendations
            if recs is not None and not recs.empty:
                latest = recs.iloc[-1] if len(recs) > 0 else None
                if latest is not None:
                    data.strong_buy = int(latest.get("strongBuy", 0))
                    data.buy = int(latest.get("buy", 0))
                    data.hold = int(latest.get("hold", 0))
                    data.sell = int(latest.get("sell", 0))
                    data.strong_sell = int(latest.get("strongSell", 0))
                    data.analyst_count = (
                        data.strong_buy + data.buy + data.hold + data.sell + data.strong_sell
                    )

            targets = ticker.analyst_price_targets
            if targets is not None:
                if isinstance(targets, dict):
                    data.target_mean = targets.get("mean", 0) or 0
                    data.target_high = targets.get("high", 0) or 0
                    data.target_low = targets.get("low", 0) or 0

            if current_price > 0 and data.target_mean > 0:
                data.target_upside_pct = (data.target_mean - current_price) / current_price * 100

            upgrades = ticker.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                recent = upgrades.tail(30)
                data.recent_upgrades = int((recent.get("Action", pd.Series()) == "upgrade").sum())
                data.recent_downgrades = int((recent.get("Action", pd.Series()) == "downgrade").sum())

        except Exception as e:
            logger.debug("Consensus data unavailable for ticker: %s", e)

        return data

    async def _get_smart_money(
        self, ticker: yf.Ticker, info: dict[str, Any]
    ) -> SmartMoneyData:
        data = SmartMoneyData()
        try:
            data.institutional_pct = info.get("heldPercentInstitutions")
            data.short_ratio = info.get("shortRatio")

            insiders = ticker.insider_transactions
            if insiders is not None and not insiders.empty:
                buys = insiders[insiders.get("Transaction", pd.Series()).str.contains("Purchase|Buy", case=False, na=False)]
                sells = insiders[insiders.get("Transaction", pd.Series()).str.contains("Sale|Sell", case=False, na=False)]
                data.insider_buy_count_90d = len(buys)
                data.insider_sell_count_90d = len(sells)
        except Exception as e:
            logger.debug("Smart money data unavailable: %s", e)

        return data
