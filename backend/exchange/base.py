from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Ticker:
    symbol: str
    price: float
    change_pct: float = 0.0
    volume: float = 0.0
    timestamp: int = 0


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBook:
    symbol: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # [(price, qty)]
    asks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Balance:
    currency: str
    total: float
    available: float
    locked: float = 0.0


@dataclass
class Position:
    symbol: str
    exchange: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None = None
    filled_quantity: float = 0.0
    filled_price: float | None = None
    status: str = "pending"
    timestamp: int = 0


class ExchangeAdapter(ABC):
    """Abstract interface for KIS exchange operations."""

    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    # -- Market Data --

    @abstractmethod
    async def fetch_ticker(self, symbol: str, exchange: str = "NASD") -> Ticker:
        ...

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
        exchange: str = "NASD",
    ) -> list[Candle]:
        ...

    @abstractmethod
    async def fetch_orderbook(
        self, symbol: str, exchange: str = "NASD", limit: int = 20
    ) -> OrderBook:
        ...

    # -- Account --

    @abstractmethod
    async def fetch_balance(self) -> Balance:
        ...

    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        ...

    @abstractmethod
    async def fetch_buying_power(self) -> float:
        ...

    # -- Orders --

    @abstractmethod
    async def create_buy_order(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
        exchange: str = "NASD",
        session: str = "regular",
    ) -> OrderResult:
        ...

    @abstractmethod
    async def create_sell_order(
        self,
        symbol: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
        exchange: str = "NASD",
        session: str = "regular",
    ) -> OrderResult:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str, **kwargs) -> bool:
        ...

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> OrderResult:
        ...

    @abstractmethod
    async def fetch_pending_orders(self) -> list[OrderResult]:
        ...
