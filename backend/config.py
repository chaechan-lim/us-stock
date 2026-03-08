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
    evaluation_interval_sec: int = 300
    initial_balance_usd: float = 10_000
    min_confidence: float = 0.50
    max_positions: int = 10
    cooldown_after_sell_sec: int = 14400  # 4 hours
    daily_buy_limit: int = 10

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
    gemini_fallback_model: str = "gemini-3-flash-preview"
    max_tokens: int = 4096

    model_config = {"env_prefix": "LLM_"}


class ExternalDataConfig(BaseSettings):
    fred_api_key: str = ""

    model_config = {"env_prefix": "EXTERNAL_"}


class AuthConfig(BaseSettings):
    api_token: str = ""  # Bearer token for API auth; empty = auth disabled

    model_config = {"env_prefix": "AUTH_"}


class AppConfig:
    def __init__(self) -> None:
        self.kis = KISConfig()
        self.trading = TradingConfig()
        self.etf = ETFConfig()
        self.risk = RiskConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.notification = NotificationConfig()
        self.llm = LLMConfig()
        self.external = ExternalDataConfig()
        self.auth = AuthConfig()

    @property
    def is_live(self) -> bool:
        return self.trading.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.trading.mode == "paper"
