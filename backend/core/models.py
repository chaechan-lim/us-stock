"""SQLAlchemy ORM models for US stock trading system."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    BigInteger,
    Index,
    UniqueConstraint,
)
from sqlalchemy import JSON as JSONB  # Use generic JSON for SQLite compatibility
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(2), nullable=False, default="US")  # US or KR
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(10), nullable=False, default="NASD")
    side = Column(String(4), nullable=False)
    order_type = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    filled_quantity = Column(Float, default=0)
    filled_price = Column(Float)
    status = Column(String(20), nullable=False, default="pending")
    strategy_name = Column(String(50))
    buy_strategy = Column(String(50))   # original buy strategy (for SELL attribution)
    kis_order_id = Column(String(50))
    pnl = Column(Float)
    session = Column(String(20), nullable=True, default="regular")  # regular/pre_market/after_hours/extended_nxt
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime)

    __table_args__ = (
        Index("idx_orders_symbol", "symbol"),
        Index("idx_orders_created", "created_at"),
        Index("idx_orders_status", "status"),
    )


class PositionRecord(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(2), nullable=False, default="US")
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(10), nullable=False, default="NASD")
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop = Column(Float)
    strategy_name = Column(String(50))
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("market", "symbol", name="uq_positions_market_symbol"),
        Index("idx_positions_symbol", "symbol"),
    )


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(2), nullable=False, default="US")
    total_value_usd = Column(Float, nullable=False)
    cash_usd = Column(Float, nullable=False)
    invested_usd = Column(Float, nullable=False)
    realized_pnl = Column(Float)
    unrealized_pnl = Column(Float)
    daily_pnl = Column(Float)
    drawdown_pct = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_snapshots_recorded", "recorded_at"),)


class StrategyLog(Base):
    __tablename__ = "strategy_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    indicators = Column(JSONB)
    market_state = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_strategy_logs_created", "created_at"),)


class ScannerResult(Base):
    __tablename__ = "scanner_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_type = Column(String(30), nullable=False)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(10))
    score = Column(Float)
    details = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_scanner_created", "created_at"),)


class SectorAnalysis(Base):
    __tablename__ = "sector_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sector_code = Column(String(10), nullable=False)
    sector_name = Column(String(30), nullable=False)
    strength_score = Column(Float)
    return_1w = Column(Float)
    return_1m = Column(Float)
    return_3m = Column(Float)
    trend = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_type = Column(String(30), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentMemory(Base):
    """Persistent memory for AI agents — stores insights for future context."""
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_type = Column(String(30), nullable=False)    # market_analyst, risk, trade_review
    category = Column(String(30), nullable=False)      # symbol, sector, market, lesson
    symbol = Column(String(20))                        # null for market/lesson
    content = Column(Text, nullable=False)             # insight text
    token_count = Column(Integer, nullable=False, default=0)
    importance = Column(Integer, nullable=False, default=5)  # 1-10 priority
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)      # auto-cleanup

    __table_args__ = (
        Index("idx_agent_memory_lookup", "agent_type", "category", "symbol"),
        Index("idx_agent_memory_expires", "expires_at"),
    )


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(30), nullable=False)
    severity = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    config = Column(JSONB, nullable=False)
    metrics = Column(JSONB, nullable=False)
    trades = Column(JSONB)
    equity_curve = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String(2), nullable=False, default="US")
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(10), nullable=False, default="NASD")
    name = Column(String(100))
    sector = Column(String(30))
    market_cap = Column(BigInteger)
    source = Column(String(20))
    score = Column(Float)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("market", "symbol", name="uq_watchlist_market_symbol"),
        Index("idx_watchlist_active", "is_active"),
    )
