from pydantic_settings import BaseSettings
from pydantic import Field


class KISConfig(BaseSettings):
    app_key: str = ""
    app_secret: str = ""
    account_no: str = ""
    account_product: str = "01"
    base_url: str = "https://openapivts.koreainvestment.com:29443"  # default: paper
    ws_url: str = "ws://ops.koreainvestment.com:21000"

    model_config = {"env_prefix": "KIS_"}


class TradingConfig(BaseSettings):
    mode: str = "paper"  # "paper" or "live"
    evaluation_interval_sec: int = 600  # 10 min (was 5 min — too frequent)
    initial_balance_usd: float = 10_000
    min_confidence: float = 0.50
    max_positions: int = 10
    cooldown_after_sell_sec: int = 14400  # 4 hours
    daily_buy_limit: int = 5  # Max buys per day per market
    pending_order_ttl_min: int = 15  # Auto-cancel unfilled orders after N minutes

    model_config = {"env_prefix": "TRADING_"}


class ETFConfig(BaseSettings):
    enabled: bool = True
    max_portfolio_pct: float = 0.30
    max_hold_days: int = 10
    confirmation_days: int = 2  # regime switch confirmation

    model_config = {"env_prefix": "ETF_"}


class RiskConfig(BaseSettings):
    max_single_stock_pct: float = 0.20
    max_invested_pct: float = 0.80
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.03
    max_sector_pct: float = 0.40
    vix_threshold: float = 35.0
    market_allocation_us: float = 0.50  # US market max share of portfolio
    market_allocation_kr: float = 0.50  # KR market max share of portfolio

    model_config = {"env_prefix": "RISK_"}


class DatabaseConfig(BaseSettings):
    url: str = "postgresql+asyncpg://usstock:usstock@localhost:5432/us_stock_trading"
    echo: bool = False

    model_config = {"env_prefix": "DB_"}


class RedisConfig(BaseSettings):
    url: str = "redis://localhost:6379/1"

    model_config = {"env_prefix": "REDIS_"}


class NotificationConfig(BaseSettings):
    enabled: bool = False
    provider: str = "discord"  # primary: discord
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    slack_webhook_url: str = ""

    model_config = {"env_prefix": "NOTIFY_"}


class LLMConfig(BaseSettings):
    enabled: bool = False
    api_key: str = ""
    model: str = "claude-haiku-4-5-20251001"
    fallback_model: str = "claude-sonnet-4-6"
    gemini_api_key: str = ""
    gemini_fallback_model: str = "gemini-2.5-flash"
    max_tokens: int = 4096
    # Cost control
    max_daily_calls: int = 100  # Daily LLM call budget (0=unlimited)
    pre_trade_risk_enabled: bool = False  # AI pre-trade risk check (expensive)
    news_use_gemini: bool = True  # Use Gemini (free tier) for news sentiment
    # Cooldown after 429/quota errors (seconds). Ultra tier recovers fast → 5 min default
    cooldown_seconds: int = 300
    # Articles per LLM call for news sentiment (larger = fewer calls)
    news_batch_size: int = 25

    model_config = {"env_prefix": "LLM_"}


class ExtendedHoursConfig(BaseSettings):
    """Extended hours trading configuration (kill switch + risk params).

    Optimized via backtest (3y, 2023-2026):
      Best: CAGR 14.12% vs baseline 11.64% (+2.48%p), Sharpe 1.07, MDD -9.0%
      Extended trades: 196, WR 40%, PnL +$14,347
    """
    enabled: bool = False  # Master kill switch
    us_enabled: bool = False  # US pre-market / after-hours
    kr_enabled: bool = False  # KR 시간외 / NXT
    # Risk parameters (backtest-optimized)
    max_position_pct: float = 0.05  # 5% per position (backtest optimal)
    max_positions: int = 5
    min_confidence: float = 0.55  # Lower bar OK: only triggers on dip/spillover
    slippage_multiplier: float = 2.0  # 2x slippage vs regular
    # Step-by-step activation
    sl_tp_monitoring_only: bool = True  # Phase 1: only monitor SL/TP, no new buys

    model_config = {"env_prefix": "EXTENDED_HOURS_"}


class ExternalDataConfig(BaseSettings):
    fred_api_key: str = ""
    finnhub_api_key: str = ""

    model_config = {"env_prefix": "EXTERNAL_"}


class AppConfig:
    def __init__(self) -> None:
        self.kis = KISConfig()
        self.trading = TradingConfig()
        self.etf = ETFConfig()
        self.risk = RiskConfig()
        self.extended_hours = ExtendedHoursConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.notification = NotificationConfig()
        self.llm = LLMConfig()
        self.external = ExternalDataConfig()

    @property
    def is_live(self) -> bool:
        return self.trading.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.trading.mode == "paper"
