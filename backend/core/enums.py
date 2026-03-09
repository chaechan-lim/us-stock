from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketState(str, Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    CRASH = "crash"


class MacroRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class RiskLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Market(str, Enum):
    US = "US"
    KR = "KR"


class Exchange(str, Enum):
    # US
    NASD = "NASD"
    NYSE = "NYSE"
    AMEX = "AMEX"
    # KR
    KRX = "KRX"        # KOSPI
    KOSDAQ = "KOSDAQ"


class RecommendGrade(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    PASS = "PASS"


class Conviction(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EntryTiming(str, Enum):
    IMMEDIATE = "immediate"
    PULLBACK = "pullback_to_ema20"
    BREAKOUT_WAIT = "breakout_wait"


class PositionSizeGrade(str, Enum):
    SMALL = "small"      # 5% of portfolio
    MEDIUM = "medium"    # 10%
    LARGE = "large"      # 15%


class ExitAction(str, Enum):
    HOLD = "HOLD"
    TRIM = "TRIM"
    EXIT = "EXIT"


class ScanType(str, Enum):
    VOLUME_SURGE = "volume_surge"
    PRICE_MOVER = "price_mover"
    NEW_HIGH = "new_high"
    NEW_LOW = "new_low"
    MARKET_CAP = "market_cap"
    TRADING_VALUE = "trading_value"
