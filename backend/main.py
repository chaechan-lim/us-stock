"""US Stock Auto-Trading Engine - FastAPI Application."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import AppConfig
from core.models import Base
from db.session import get_engine
from api.router import api_router
from api.ws import install_log_handler
from services.log_manager import LogConfig, setup_logging
from services.rate_limiter import RateLimiter
from data.market_data_service import MarketDataService
from data.indicator_service import IndicatorService
from strategies.registry import StrategyRegistry
from strategies.combiner import SignalCombiner
from engine.risk_manager import RiskManager
from engine.order_manager import OrderManager, set_trade_recorder, set_db_recorder
from engine.position_tracker import PositionTracker
from engine.scheduler import TradingScheduler, MarketPhase
from engine.evaluation_loop import EvaluationLoop
from engine.portfolio_manager import PortfolioManager
from engine.recovery import RecoveryManager
from scanner.pipeline import ScannerPipeline
from scanner.fundamental_enricher import FundamentalEnricher
from scanner.stock_scanner import StockScanner
from scanner.sector_analyzer import SectorAnalyzer
from scanner.universe_expander import UniverseExpander
from scanner.etf_universe import ETFUniverse
from engine.etf_engine import ETFEngine
from data.market_state import MarketStateDetector
from data.external_data_service import ExternalDataService
from data.fred_service import FREDService
from db.session import get_session_factory
from services.cache import CacheService
from services.health import HealthMonitor
from services.notification import NotificationService
from services.exchange_resolver import ExchangeResolver

setup_logging(LogConfig())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    config = AppConfig()
    app.state.config = config

    # Redis cache
    cache = CacheService(url=config.redis.url)
    await cache.initialize()
    app.state.cache = cache

    # Create database tables + add any missing columns on existing tables
    engine = get_engine(config.database)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # Auto-migrate: add columns/indexes that exist in the ORM model but are
    # missing from the physical DB (e.g. is_paper, buy_strategy added after
    # initial deploy)
    from db.migrations import ensure_columns, ensure_indexes

    added_cols = await ensure_columns(engine)
    if added_cols:
        logger.info("Auto-migration added columns: %s", ", ".join(added_cols))
    created_idxs = await ensure_indexes(engine)
    if created_idxs:
        logger.info("Auto-migration created indexes: %s", ", ".join(created_idxs))

    # Initialize exchange adapter
    if config.is_paper:
        from exchange.paper_adapter import PaperAdapter

        adapter = PaperAdapter(config.trading.initial_balance_usd)
        await adapter.initialize()
    else:
        from exchange.kis_adapter import KISAdapter
        from exchange.kis_auth import KISAuth

        auth = KISAuth(
            app_key=config.kis.app_key,
            app_secret=config.kis.app_secret,
            base_url=config.kis.base_url,
            redis_client=cache if cache.available else None,
        )
        adapter = KISAdapter(config.kis, auth)
        await adapter.initialize()

    app.state.adapter = adapter
    logger.info("Exchange adapter initialized (mode=%s)", config.trading.mode)

    # KR adapter (shares same KIS auth — same app_key/secret/account)
    if config.is_paper:
        from exchange.paper_adapter import PaperAdapter as PaperAdapterCls
        kr_adapter = PaperAdapterCls(
            config.trading.initial_balance_usd * 50_000, currency="KRW",
        )
        await kr_adapter.initialize()
    else:
        from exchange.kis_kr_adapter import KISKRAdapter
        kr_adapter = KISKRAdapter(config.kis, auth)
        await kr_adapter.initialize()
    app.state.kr_adapter = kr_adapter
    logger.info("KR adapter initialized (mode=%s)", config.trading.mode)

    # KIS WebSocket (live mode only)
    kis_ws = None
    if not config.is_paper:
        from exchange.kis_websocket import KISWebSocket

        kis_ws = KISWebSocket(auth=auth, ws_url=config.kis.ws_url)
    app.state.kis_ws = kis_ws

    # Initialize services
    rate_limiter = RateLimiter(
        max_per_second=5 if config.is_paper else 20
    )
    market_data = MarketDataService(adapter=adapter, rate_limiter=rate_limiter)
    app.state.market_data = market_data
    app.state.indicator_svc = IndicatorService()

    # Strategy registry
    registry = StrategyRegistry()
    app.state.registry = registry

    # Notification service
    notif_cfg = config.notification
    notification = NotificationService(
        enabled=notif_cfg.enabled,
        provider=notif_cfg.provider,
        telegram_bot_token=notif_cfg.telegram_bot_token,
        telegram_chat_id=notif_cfg.telegram_chat_id,
        discord_webhook_url=notif_cfg.discord_webhook_url,
        slack_webhook_url=notif_cfg.slack_webhook_url,
    )
    app.state.notification = notification

    # Engine components
    from engine.risk_manager import RiskParams
    market_allocs = {
        "US": config.risk.market_allocation_us,
        "KR": config.risk.market_allocation_kr,
    }

    # Load tiered trailing stop and breakeven stop config from strategies.yaml (STOCK-24)
    tiered_cfg = registry._config_loader.get_tiered_trailing_stop_config()
    tiered_tiers = None
    if tiered_cfg.get("enabled", False):
        raw_tiers = tiered_cfg.get("tiers", [])
        tiered_tiers = [(t["gain_pct"], t["trail_pct"]) for t in raw_tiers]

    be_cfg = registry._config_loader.get_breakeven_stop_config()
    be_enabled = be_cfg.get("enabled", True)
    be_activation = be_cfg.get("activation_ratio", 0.50)
    be_lock_ratio = be_cfg.get("lock_ratio", 0.75)
    be_lock_pct = be_cfg.get("lock_pct", 0.50)

    risk_params = RiskParams(
        market_allocations=market_allocs,
        max_position_pct=0.08,          # 8% per position (diversified, backtest optimal)
        max_positions=20,               # More positions, better diversification
        default_stop_loss_pct=0.12,     # Wider SL: more room for volatility
        default_take_profit_pct=0.50,   # Wide TP: let winners run
        tiered_trailing_tiers=tiered_tiers,
        breakeven_stop_enabled=be_enabled,
        breakeven_stop_activation_ratio=be_activation,
        breakeven_stop_lock_ratio=be_lock_ratio,
        breakeven_stop_lock_pct=be_lock_pct,
    )
    risk_manager = RiskManager(params=risk_params)

    # KR-specific risk params: wider SL for ±30% daily limit, tighter TP
    kr_risk_params = RiskParams(
        market_allocations=market_allocs,
        default_stop_loss_pct=0.12,     # 12% (wider for KR volatility)
        default_take_profit_pct=0.25,   # 25% (allow larger moves)
        tiered_trailing_tiers=tiered_tiers,
        breakeven_stop_enabled=be_enabled,
        breakeven_stop_activation_ratio=be_activation,
        breakeven_stop_lock_ratio=be_lock_ratio,
        breakeven_stop_lock_pct=be_lock_pct,
    )
    kr_risk_manager = RiskManager(params=kr_risk_params)
    order_manager = OrderManager(adapter=adapter, risk_manager=risk_manager, notification=notification, market_data=market_data, is_paper=config.is_paper)
    app.state.risk_manager = risk_manager
    app.state.order_manager = order_manager
    consensus_cfg = registry._config_loader.get_consensus_config()
    app.state.combiner = SignalCombiner(consensus_config=consensus_cfg)

    # Wire trade recording
    from api.trades import record_trade, persist_trade_to_db
    set_trade_recorder(record_trade)
    # STOCK-38: Awaited DB persist ensures filled_price/status are saved
    # at order time, not relying solely on fire-and-forget or reconciliation
    set_db_recorder(persist_trade_to_db)

    # Health monitor
    health = HealthMonitor()

    async def check_adapter():
        await adapter.fetch_balance()
        return {"mode": config.trading.mode}

    health.register_check("adapter", check_adapter)
    health.set_notification(notification)
    app.state.health = health

    # Position tracker (session_factory set below after DB init)
    position_tracker = PositionTracker(
        adapter=adapter,
        risk_manager=risk_manager,
        order_manager=order_manager,
        notification=notification,
        market_data=market_data,
        market="US",
    )
    app.state.position_tracker = position_tracker

    # Adaptive weight manager (per-stock strategy selection)
    from engine.adaptive_weights import AdaptiveWeightManager
    adaptive_cfg = registry._config_loader.get_adaptive_config()
    stock_profiles = registry._config_loader.get_stock_profiles() or None
    adaptive_weights = AdaptiveWeightManager(
        alpha=adaptive_cfg.get("alpha", 0.6),
        ema_decay=adaptive_cfg.get("ema_decay", 0.1),
        min_signals_for_adaptation=adaptive_cfg.get("min_signals", 5),
        stock_profiles=stock_profiles,
    )

    # Portfolio manager
    session_factory = get_session_factory(config.database)
    portfolio_manager = PortfolioManager(
        market_data=market_data, session_factory=session_factory,
    )
    app.state.portfolio_manager = portfolio_manager

    # Wire position trackers to DB for persistence
    position_tracker._session_factory = session_factory

    # Wire trade DB persistence
    from api.trades import init_trades
    init_trades(session_factory)

    # Wire portfolio DB access
    from api.portfolio import init_portfolio
    init_portfolio(session_factory)

    # LLM client (multi-provider with fallback)
    llm_client = None
    if config.llm.enabled and (config.llm.api_key or config.llm.gemini_api_key):
        from services.llm import LLMClient
        llm_client = LLMClient(config.llm)
        providers = []
        if config.llm.api_key:
            providers.append(f"anthropic({config.llm.model})")
        if config.llm.gemini_api_key:
            providers.append(f"gemini({config.llm.gemini_fallback_model})")
        logger.info("LLM client enabled: %s", " -> ".join(providers))
    app.state.llm_client = llm_client

    # Agent context service (persistent memory for AI agents)
    agent_ctx = None
    if llm_client:
        from services.agent_context import AgentContextService
        agent_ctx = AgentContextService(session_factory)
        logger.info("Agent context service enabled")
    app.state.agent_context = agent_ctx

    # AI agents (market analyst, risk assessment, trade review, news sentiment)
    ai_agent = None
    risk_agent = None
    trade_review_agent = None
    news_sentiment_agent = None
    kr_news_sentiment_agent = None
    if llm_client:
        from agents.market_analyst import MarketAnalystAgent
        from agents.risk_assessment import RiskAssessmentAgent
        from agents.trade_review import TradeReviewAgent
        from agents.news_sentiment_agent import NewsSentimentAgent
        ai_agent = MarketAnalystAgent(llm_client=llm_client, context_service=agent_ctx)
        risk_agent = RiskAssessmentAgent(llm_client=llm_client, context_service=agent_ctx)
        trade_review_agent = TradeReviewAgent(llm_client=llm_client, context_service=agent_ctx)
        # News sentiment: use Gemini (free tier) to save Anthropic credits
        news_model = None
        if config.llm.news_use_gemini and config.llm.gemini_fallback_model:
            news_model = config.llm.gemini_fallback_model
        news_batch_size = config.llm.news_batch_size
        news_sentiment_agent = NewsSentimentAgent(
            llm_client=llm_client, context_service=agent_ctx,
            model_override=news_model,
            batch_size=news_batch_size,
        )
        # KR agent: also use Gemini to save costs (Korean supported by Gemini)
        kr_news_sentiment_agent = NewsSentimentAgent(
            llm_client=llm_client,
            model_override=news_model,
            batch_size=news_batch_size,
        )
        logger.info("AI agents enabled (analyst, risk, trade_review, news_sentiment)")
    app.state.risk_agent = risk_agent
    app.state.trade_review_agent = trade_review_agent
    app.state.news_sentiment_agent = news_sentiment_agent

    # Resolve services from app.state
    indicator_svc = app.state.indicator_svc
    combiner = app.state.combiner

    # News service (Finnhub)
    from data.news_service import FinnhubNewsService
    from scanner.news_enricher import NewsEnricher
    news_service = FinnhubNewsService(api_key=config.external.finnhub_api_key)
    app.state.news_service = news_service
    news_enricher = NewsEnricher() if news_service.available else None
    if news_service.available:
        logger.info("Finnhub news service enabled")

    # Event calendar (earnings, macro, insider)
    from data.earnings_service import EarningsCalendarService
    from data.macro_calendar import MacroCalendarService
    from data.insider_service import InsiderTradingService
    from data.event_calendar import EventCalendarService
    earnings_svc = EarningsCalendarService(api_key=config.external.finnhub_api_key)
    macro_svc = MacroCalendarService()
    insider_svc = InsiderTradingService(api_key=config.external.finnhub_api_key)
    event_calendar = EventCalendarService(earnings_svc, macro_svc, insider_svc)
    app.state.event_calendar = event_calendar
    position_tracker._event_calendar = event_calendar

    from data.kr_macro_calendar import KRMacroCalendarService
    kr_macro_calendar = KRMacroCalendarService()
    app.state.kr_macro_calendar = kr_macro_calendar
    logger.info("Event calendar services initialized (US + KR)")

    # Scanner pipeline (with AI agent + news enricher if available)
    enricher = FundamentalEnricher()
    scanner_pipeline = ScannerPipeline(
        market_data=market_data,
        indicator_svc=indicator_svc,
        enricher=enricher,
        ai_agent=ai_agent,
        news_enricher=news_enricher,
    )
    app.state.scanner_pipeline = scanner_pipeline

    # Exchange resolver (caches yfinance exchange lookups for KIS API)
    exchange_resolver = ExchangeResolver()
    app.state.exchange_resolver = exchange_resolver

    # Evaluation loop (after agents — risk_agent used for pre-trade check)
    # Pre-trade AI risk check disabled by default (biggest LLM cost driver)
    pre_trade_agent = risk_agent if config.llm.pre_trade_risk_enabled else None
    evaluation_loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=combiner,
        order_manager=order_manager,
        risk_manager=risk_manager,
        adaptive_weights=adaptive_weights,
        risk_agent=pre_trade_agent,
        exchange_resolver=exchange_resolver,
        position_tracker=position_tracker,
        market="US",
        event_calendar=event_calendar,
    )
    evaluation_loop._daily_buy_limit = config.trading.daily_buy_limit
    # STOCK-43: Apply config cooldown + Redis persistence + PositionTracker callback
    evaluation_loop._sell_cooldown_secs = config.trading.cooldown_after_sell_sec
    evaluation_loop.set_cache(cache)
    position_tracker.register_on_sell(evaluation_loop.register_sell_cooldown)
    app.state.evaluation_loop = evaluation_loop

    # Stock scanner & sector analyzer
    stock_scanner = StockScanner(adapter=adapter, market_data=market_data)
    sector_analyzer = SectorAnalyzer()
    external_data = ExternalDataService()
    etf_universe = ETFUniverse()
    # Preload ETF exchange codes so evaluation_loop doesn't need yfinance lookups
    for sym in etf_universe.all_etf_symbols:
        exchange_resolver.set(sym, etf_universe.get_exchange(sym))
    universe_expander = UniverseExpander(
        etf_universe=etf_universe, sector_analyzer=sector_analyzer,
        kis_adapter=adapter, rate_limiter=rate_limiter,
    )
    app.state.stock_scanner = stock_scanner
    app.state.sector_analyzer = sector_analyzer
    app.state.external_data = external_data
    app.state.universe_expander = universe_expander

    # ETF Engine (leveraged pair switching + sector ETF rotation)
    etf_engine = ETFEngine(
        market_data=market_data,
        order_manager=order_manager,
        etf_universe=etf_universe,
        sector_analyzer=sector_analyzer,
        notification=notification,
        market="US",
        risk_manager=risk_manager,
    )
    app.state.etf_engine = etf_engine
    logger.info("ETF Engine initialized")

    # Market state detector
    market_state_detector = MarketStateDetector()
    app.state.market_state_detector = market_state_detector

    # FRED macro data service
    fred_service = FREDService(api_key=config.external.fred_api_key)
    app.state.fred_service = fred_service

    # KR engine components (separate instances, same classes)
    from data.kr_symbol_mapper import to_yfinance as kr_to_yfinance
    from scanner.kr_screener import get_kr_exchange
    # Share rate limiter with US — same KIS credentials, same rate limit
    kr_market_data = MarketDataService(
        adapter=kr_adapter, rate_limiter=rate_limiter,
        yf_symbol_mapper=lambda s: kr_to_yfinance(s, get_kr_exchange(s)),
    )
    kr_order_manager = OrderManager(
        adapter=kr_adapter, risk_manager=kr_risk_manager, notification=notification,
        market_data=kr_market_data, market="KR", is_paper=config.is_paper,
    )
    kr_position_tracker = PositionTracker(
        adapter=kr_adapter,
        risk_manager=kr_risk_manager,
        order_manager=kr_order_manager,
        notification=notification,
        market_data=kr_market_data,
        session_factory=session_factory,
        market="KR",
    )
    app.state.kr_market_data = kr_market_data
    app.state.kr_order_manager = kr_order_manager
    app.state.kr_position_tracker = kr_position_tracker

    # KR portfolio manager (for equity history)
    kr_portfolio_manager = PortfolioManager(
        market_data=kr_market_data, session_factory=session_factory,
        market="KR",
    )
    app.state.kr_portfolio_manager = kr_portfolio_manager

    # KR ETF Engine
    from pathlib import Path
    kr_etf_config_path = Path(__file__).resolve().parent.parent / "config" / "kr_etf_universe.yaml"
    kr_etf_universe = ETFUniverse(config_path=kr_etf_config_path)
    kr_etf_engine = ETFEngine(
        market_data=kr_market_data,
        order_manager=kr_order_manager,
        etf_universe=kr_etf_universe,
        sector_analyzer=sector_analyzer,
        notification=notification,
        market="KR",
        risk_manager=kr_risk_manager,
    )
    app.state.kr_etf_engine = kr_etf_engine
    kr_market_state_detector = MarketStateDetector()
    app.state.kr_market_state_detector = kr_market_state_detector
    logger.info("KR engine components initialized (incl. ETF Engine)")

    # Recovery manager for circuit breakers
    recovery_mgr = RecoveryManager(notification=notification)
    app.state.recovery = recovery_mgr

    # Trading scheduler with tasks
    scheduler = TradingScheduler(recovery_manager=recovery_mgr)

    async def task_health_check():
        await health.check_all()

    async def task_system_status_report():
        """Send periodic system status to Discord (every 30 min)."""
        try:
            report = await health.get_status_report()
            sys = report["system"]
            uptime_h = sys["uptime_seconds"] / 3600

            # Check statuses
            checks = report.get("checks", {})
            check_lines = []
            for name, info in checks.items():
                icon = "OK" if info["status"] == "ok" else "FAIL"
                check_lines.append(f"{icon} {name}")

            # Scheduler stats
            sched_info = scheduler.get_status() if hasattr(scheduler, "get_status") else {}
            tasks = sched_info.get("tasks", [])
            active_tasks = sum(1 for t in tasks if t.get("active"))
            total_tasks = len(tasks)
            phase = sched_info.get("market_phase", "-")
            kr_phase = sched_info.get("kr_market_phase", "-")

            msg = (
                f"**System Status Report**\n"
                f"Status: **{report['status'].upper()}** | Uptime: {uptime_h:.1f}h\n"
                f"CPU: {sys['cpu_percent']}% | "
                f"RAM: {sys['memory_used_gb']}/{sys['memory_total_gb']}GB ({sys['memory_percent']}%) | "
                f"Disk: {sys['disk_percent']}%\n"
                f"US: {phase} | KR: {kr_phase} | Tasks: {active_tasks}/{total_tasks} active\n"
                f"Checks: {' | '.join(check_lines)}"
            )
            await notification.notify_system_event("status_report", msg)
        except Exception as e:
            logger.error("System status report failed: %s", e)

    async def task_position_check():
        await position_tracker.check_all()

    async def task_position_db_sync():
        """Periodic reconciliation: sync in-memory positions to DB."""
        await position_tracker.sync_to_db()

    async def task_daily_reset():
        risk_manager.reset_daily()
        logger.info("Daily risk counters reset")

    async def task_evaluation_loop():
        # Sync exchange rate for combined portfolio allocation
        exrt = getattr(adapter, "_last_exchange_rate", 0)
        if exrt > 0:
            evaluation_loop.set_exchange_rate(exrt)
            kr_evaluation_loop.set_exchange_rate(exrt)
        await evaluation_loop._evaluate_all()

    async def task_daily_scan():
        from db.trade_repository import TradeRepository
        try:
            async with session_factory() as session:
                repo = TradeRepository(session)
                watchlist = await repo.get_watchlist(active_only=True, market="US")
                symbols = [w.symbol for w in watchlist]
            if symbols:
                evaluation_loop.set_watchlist(symbols)
                await scanner_pipeline.run_full_scan(symbols)
        except Exception as e:
            logger.error("Daily scan failed: %s", e)

    scheduler.add_task(
        "health_check", task_health_check,
        interval_sec=120, phases=None,  # always
    )
    scheduler.add_task(
        "system_status_report", task_system_status_report,
        interval_sec=1800, phases=None,  # always, every 30 min
    )
    scheduler.add_task(
        "position_check", task_position_check,
        interval_sec=60, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "position_db_sync", task_position_db_sync,
        interval_sec=300, phases=[MarketPhase.REGULAR],  # sync every 5 min
    )
    scheduler.add_task(
        "daily_reset", task_daily_reset,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET],
    )
    scheduler.add_task(
        "evaluation_loop", task_evaluation_loop,
        interval_sec=300, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "daily_scan", task_daily_scan,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET],
    )

    async def task_update_watchlist_names():
        """Batch update stock names in watchlist DB (daily)."""
        from db.trade_repository import TradeRepository
        from data.stock_name_service import resolve_names, get_name
        try:
            for mkt in ("US", "KR"):
                async with session_factory() as session:
                    repo = TradeRepository(session)
                    items = await repo.get_watchlist(active_only=True, market=mkt)
                    nameless = [w.symbol for w in items if not w.name]

                if not nameless:
                    continue

                names = await resolve_names(nameless, mkt)

                async with session_factory() as session:
                    repo = TradeRepository(session)
                    updated = 0
                    for sym, name in names.items():
                        if name and await repo.update_watchlist_name(sym, name, mkt):
                            updated += 1
                    logger.info("Watchlist names updated: %s %d/%d", mkt, updated, len(nameless))
        except Exception as e:
            logger.error("Watchlist name update failed: %s", e)

    scheduler.add_task(
        "update_watchlist_names", task_update_watchlist_names,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET],
    )

    async def task_portfolio_snapshot():
        await portfolio_manager.save_snapshot()
        logger.debug("Portfolio snapshot saved")

    async def task_order_reconciliation():
        changes = await order_manager.reconcile_all()
        if changes:
            logger.info("Order reconciliation: %d status changes", len(changes))
            # Persist status changes to DB
            from api.trades import update_order_in_db, record_trade
            for change in changes:
                # Update existing DB row status (also updates in-memory log entry)
                await update_order_in_db(
                    kis_order_id=change.get("order_id", ""),
                    status=change["new_status"],
                    filled_price=change.get("filled_price"),
                    filled_quantity=change.get("filled_quantity"),
                )
                # STOCK-33: If newly filled, ensure it appears in trade log.
                # record_trade() is now idempotent: if the order_id already
                # exists in _trade_log (from place_sell), it merges without
                # creating a duplicate. PnL is preserved during merge.
                if change["new_status"] == "filled" and change["old_status"] != "filled":
                    record_trade({
                        "order_id": change.get("order_id", ""),
                        "symbol": change["symbol"],
                        "side": change["side"],
                        "quantity": change.get("quantity", 0),
                        "price": change.get("price"),
                        "filled_price": change.get("filled_price"),
                        "filled_quantity": change.get("filled_quantity", 0),
                        "strategy": change.get("strategy", ""),
                        "status": "filled",
                        "market": "US",
                        "created_at": "",
                    })
        # Cancel stale unfilled orders
        stale = await order_manager.cancel_stale_orders(config.trading.pending_order_ttl_min)
        if stale:
            from api.trades import update_order_in_db
            for s in stale:
                await update_order_in_db(
                    kis_order_id=s["order_id"], status="cancelled",
                )
            if notification:
                symbols = ", ".join(f"{s['side']} {s['symbol']}" for s in stale)
                await notification.notify_system_event(
                    "stale_order_cancel",
                    f"미체결 주문 자동 취소 ({len(stale)}건, "
                    f"TTL={config.trading.pending_order_ttl_min}분): {symbols}",
                )

    async def task_market_state_update():
        """T0: Update market regime from SPY data."""
        try:
            spy_df = await market_data.get_ohlcv("SPY", limit=250)
            if not spy_df.empty:
                state = market_state_detector.detect(spy_df)
                evaluation_loop.set_market_state(state.regime.value)
                risk_manager.set_market_regime("US", state.regime.value)
                logger.info(
                    "Market state: %s (SPY=%.2f, SMA200=%.2f, VIX=%.1f)",
                    state.regime.value, state.spy_price,
                    state.spy_sma200, state.vix_level,
                )
        except Exception as e:
            logger.error("Market state update failed: %s", e)

    async def task_etf_evaluation():
        """ETF Engine: regime-based leveraged pair + sector ETF rotation."""
        try:
            # Get current market state
            spy_df = await market_data.get_ohlcv("SPY", limit=250)
            if spy_df.empty:
                return
            state = market_state_detector.detect(spy_df)

            # Get sector data for sector rotation
            sector_data = await external_data.get_sector_performance()

            # Run ETF engine evaluation
            actions = await etf_engine.evaluate(
                market_state=state,
                sector_data=sector_data,
            )

            total = sum(len(v) for v in actions.values())
            if total > 0:
                logger.info(
                    "ETF Engine: %d actions (regime=%d, sector=%d, risk=%d)",
                    total, len(actions["regime"]),
                    len(actions["sector"]), len(actions["risk"]),
                )
        except Exception as e:
            logger.error("ETF evaluation failed: %s", e)

    async def task_intraday_hot_scan():
        """T2: Intraday hot scan — find active stocks during session."""
        from db.trade_repository import TradeRepository
        try:
            async with session_factory() as session:
                repo = TradeRepository(session)
                watchlist = await repo.get_watchlist(active_only=True, market="US")
                symbols = [w.symbol for w in watchlist]
            if symbols:
                scan_result = await stock_scanner.run_all_scans(symbols)
                hot_symbols = stock_scanner.get_symbols(max_results=10)
                if hot_symbols:
                    logger.info("Hot scan found %d symbols: %s", len(hot_symbols), hot_symbols[:5])
        except Exception as e:
            logger.error("Intraday hot scan failed: %s", e)

    async def task_sector_analysis():
        """T3: Periodic sector strength analysis."""
        try:
            sector_data = await external_data.get_sector_performance()
            if sector_data:
                scores = sector_analyzer.analyze(sector_data)
                top = sector_analyzer.get_top_sectors(scores)
                logger.info(
                    "Sector analysis: top=%s",
                    [(s.name, s.strength_score) for s in top],
                )
        except Exception as e:
            logger.error("Sector analysis failed: %s", e)

    async def _cleanup_watchlist(
        existing: list, candidate_symbols: list[str],
    ) -> list[str]:
        """Auto-remove scanner-added stocks that no longer qualify.

        Rules:
        - Only remove stocks added by scanner (source='scanner')
        - Never remove manually added stocks
        - Never remove stocks with open positions
        - Remove if: scanner-added + not in current candidates + added >7 days ago
        - Cap watchlist at 60 active symbols
        """
        from db.trade_repository import TradeRepository
        from datetime import timedelta

        now = datetime.utcnow()
        max_watchlist = 100
        min_age_days = 7  # Don't remove recently added

        # Get current positions to protect
        try:
            positions = await market_data.get_positions()
            held_symbols = {p.symbol for p in positions if p.quantity > 0}
        except Exception as e:
            logger.warning("Position fetch failed during watchlist cleanup: %s", e)
            held_symbols = set()

        removable = []
        for item in existing:
            # Never remove manual adds or held positions
            if item.source != "scanner":
                continue
            if item.symbol in held_symbols:
                continue
            # Don't remove if recently added
            if item.added_at and (now - item.added_at) < timedelta(days=min_age_days):
                continue
            # Remove if not in current top candidates
            if item.symbol not in candidate_symbols:
                removable.append(item)

        # If still under limit, only remove the weakest
        active_count = sum(1 for w in existing if w.is_active)
        if active_count <= max_watchlist and not removable:
            return []

        # If over limit, remove enough to get back under
        if active_count > max_watchlist:
            # Sort by oldest first (stale scanner picks)
            removable.sort(key=lambda w: w.added_at or now)
            to_remove = removable[:active_count - max_watchlist + 5]  # Remove 5 extra buffer
        else:
            # Under limit but have removable: trim stale scanner picks
            removable.sort(key=lambda w: w.added_at or now)
            to_remove = removable[:max(len(removable) // 2, 1)]  # Remove half of stale

        removed = []
        async with session_factory() as session:
            repo = TradeRepository(session)
            for item in to_remove:
                await repo.remove_from_watchlist(item.symbol)
                removed.append(item.symbol)

        return removed

    async def task_after_hours_scan():
        """T4a: Post-market stock analysis & watchlist update.

        Runs 3-layer pipeline on a dynamically discovered universe.
        Uses yfinance screeners + sector rotation to find candidates.
        Top candidates are auto-added to watchlist.
        Stale scanner-added stocks are auto-removed.
        """
        from db.trade_repository import TradeRepository

        try:
            # Get existing US watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                existing = await repo.get_watchlist(active_only=True, market="US")
                existing_syms = [w.symbol for w in existing]

            # Dynamic universe expansion
            sector_data = await external_data.get_sector_performance()
            universe_result = await universe_expander.expand(
                existing_watchlist=existing_syms,
                sector_data=sector_data,
            )
            universe = universe_result.symbols
            logger.info(
                "After-hours scan: %d symbols in universe (discovered=%d)",
                len(universe), universe_result.total_discovered,
            )

            # Fetch fresh news if available (for Layer 2.5)
            news_summary = None
            if news_service.available and news_sentiment_agent:
                try:
                    # Only fetch for non-ETF universe symbols
                    news_syms = [s for s in universe if not any(
                        s.endswith(x) for x in ("QQQ", "SPY", "XL", "SO")
                    )][:25]
                    if news_syms:
                        batch = await news_service.fetch_batch(
                            symbols=news_syms, days_back=3, max_per_symbol=3,
                        )
                        if batch.articles:
                            news_summary = await news_sentiment_agent.analyze_batch(
                                batch.articles,
                            )
                            logger.info(
                                "After-hours news: %d articles analyzed",
                                len(batch.articles),
                            )
                except Exception as e:
                    logger.warning("After-hours news fetch failed: %s", e)

            # Run pipeline (grade C to cast a wider net)
            candidates = await scanner_pipeline.run_full_scan(
                symbols=universe, min_grade="C", max_candidates=15,
                news_summary=news_summary,
            )

            if not candidates:
                logger.info("After-hours scan: no candidates found")
                # Still run cleanup even if no new candidates
                removed = await _cleanup_watchlist(existing, [])
                if removed:
                    logger.info("Watchlist cleanup: removed %s", removed)
                return

            # Auto-add top candidates + ETF universe to watchlist
            top_symbols = [c["symbol"] for c in candidates[:10]]
            etf_symbols = universe_result.etf_symbols
            async with session_factory() as session:
                repo = TradeRepository(session)
                added = []
                for sym in top_symbols:
                    if sym not in existing_syms:
                        await repo.add_to_watchlist(
                            symbol=sym, source="scanner",
                        )
                        added.append(sym)
                # Ensure ETF universe is in watchlist
                for sym in etf_symbols:
                    if sym not in existing_syms:
                        await repo.add_to_watchlist(
                            symbol=sym, source="etf_universe",
                        )
                        added.append(sym)

            # Auto-remove stale scanner-added stocks
            removed = await _cleanup_watchlist(existing, top_symbols)

            # Update evaluation loop watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True, market="US")
                evaluation_loop.set_watchlist([w.symbol for w in wl])

            # Send scan results via Discord
            msg = f"After-Hours Scan Complete\nCandidates: {len(candidates)}\n"
            for c in candidates[:10]:
                ai_note = ""
                if c.get("ai_recommendation"):
                    ai_note = f" | AI: {c['ai_recommendation']}"
                news_note = ""
                if c.get("news_sentiment") and c["news_sentiment"] != 0:
                    news_note = f" | news={c['news_sentiment']:+.2f}"
                msg += (
                    f"  {c['symbol']}: score={c['combined_score']:.0f} "
                    f"grade={c.get('grade', '?')}{ai_note}{news_note}\n"
                )
            if added:
                msg += f"\nNew watchlist adds: {', '.join(added)}"
            if removed:
                msg += f"\nAuto-removed (stale): {', '.join(removed)}"

            await notification.notify_system_event("after_hours_scan", msg)
            logger.info(
                "After-hours scan: %d candidates, +%d added, -%d removed",
                len(candidates), len(added), len(removed),
            )
        except Exception as e:
            logger.error("After-hours scan failed: %s", e)

    async def task_daily_briefing():
        """T4b: Post-market daily briefing via notification."""
        try:
            positions = await market_data.get_positions()
            balance = await market_data.get_balance()
            daily_pnl = sum(
                (p.current_price - p.avg_price) * p.quantity
                for p in positions
            )
            await notification.notify_daily_summary(
                equity=balance.total,
                daily_pnl=daily_pnl,
                positions=len(positions),
            )
            logger.info("Daily briefing sent")
        except Exception as e:
            logger.error("Daily briefing failed: %s", e)

    scheduler.add_task(
        "market_state_update", task_market_state_update,
        interval_sec=900, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "etf_evaluation", task_etf_evaluation,
        interval_sec=900, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "portfolio_snapshot", task_portfolio_snapshot,
        interval_sec=3600, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "order_reconciliation", task_order_reconciliation,
        interval_sec=120, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "intraday_hot_scan", task_intraday_hot_scan,
        interval_sec=1800, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "sector_analysis", task_sector_analysis,
        interval_sec=3600, phases=[MarketPhase.REGULAR],
    )
    scheduler.add_task(
        "after_hours_scan", task_after_hours_scan,
        interval_sec=86400, phases=[MarketPhase.AFTER_HOURS],
    )
    scheduler.add_task(
        "daily_briefing", task_daily_briefing,
        interval_sec=86400, phases=[MarketPhase.AFTER_HOURS],
    )

    async def task_macro_update():
        """T5: Fetch FRED macro indicators (daily, pre-market)."""
        if not fred_service.available:
            return
        try:
            indicators = await fred_service.fetch_macro_indicators()
            if indicators.yield_curve_inverted:
                logger.warning("Yield curve inverted — recession signal")
            logger.info(
                "Macro update: FFR=%.2f, 10Y=%.2f, spread=%.2f, CPI=%.1f%%",
                indicators.fed_funds_rate or 0,
                indicators.treasury_10y or 0,
                indicators.yield_spread or 0,
                indicators.cpi_yoy or 0,
            )
        except Exception as e:
            logger.error("Macro update failed: %s", e)

    scheduler.add_task(
        "macro_update", task_macro_update,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET],
    )

    # WebSocket lifecycle management (market hours only)
    async def task_ws_lifecycle():
        """Manage KIS WebSocket connection lifecycle.

        - Connect at market open, disconnect at close
        - Refresh session if it exceeds max duration
        - Never keep connections alive indefinitely (KIS policy)
        """
        if not kis_ws:
            return
        try:
            from engine.scheduler import get_market_phase
            phase = get_market_phase()

            if phase == MarketPhase.REGULAR:
                if not kis_ws.is_connected:
                    logger.info("Market open: connecting WebSocket")
                    await kis_ws.connect()
                elif kis_ws.session_age_sec > kis_ws._max_session_sec:
                    logger.info("WS session expired, refreshing")
                    await kis_ws.refresh_session()
            else:
                # Non-market hours: disconnect if connected
                if kis_ws.is_connected:
                    logger.info("Market closed: disconnecting WebSocket")
                    await kis_ws.close()
        except Exception as e:
            logger.error("WS lifecycle error: %s", e)

    scheduler.add_task(
        "ws_lifecycle", task_ws_lifecycle,
        interval_sec=300, phases=None,  # always — manages its own phase logic
    )

    # Trade review: review completed trades after hours
    async def task_trade_review():
        """Review recent completed trades using AI trade review agent."""
        if not trade_review_agent:
            return
        try:
            from db.trade_repository import TradeRepository
            async with session_factory() as session:
                repo = TradeRepository(session)
                # Get trades completed in the last 24h
                recent_trades = await repo.get_recent_trades(hours=24)

            if not recent_trades:
                return

            trade_dicts = [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "strategy": t.strategy_name,
                    "pnl": getattr(t, "pnl", 0),
                    "filled_at": str(t.filled_at),
                }
                for t in recent_trades
            ]

            # Daily summary review
            balance = await market_data.get_balance()
            summary = await trade_review_agent.review_daily_trades(
                trades=trade_dicts,
                portfolio_summary={
                    "total_value": balance.total,
                    "cash": balance.available,
                },
            )
            logger.info(
                "Daily trade review: grade=%s score=%d trades=%d",
                summary.get("overall_grade", "?"),
                summary.get("overall_score", 0),
                summary.get("total_trades", 0),
            )

            # Send review via notification
            if summary.get("summary"):
                msg = (
                    f"Daily Trade Review: {summary['overall_grade']} "
                    f"(score={summary['overall_score']})\n"
                    f"Trades: {summary['total_trades']}\n"
                    f"{summary['summary']}"
                )
                await notification.notify_system_event("trade_review", msg)

        except Exception as e:
            logger.error("Trade review task failed: %s", e)

    scheduler.add_task(
        "trade_review", task_trade_review,
        interval_sec=86400, phases=[MarketPhase.AFTER_HOURS],
    )

    # Agent memory cleanup: remove expired + enforce limits
    async def task_agent_memory_cleanup():
        """Clean up expired agent memories and enforce per-agent limits."""
        if not agent_ctx:
            return
        try:
            expired = await agent_ctx.cleanup_expired()
            total_trimmed = 0
            for agent_type in ("market_analyst", "risk", "trade_review", "news_sentiment"):
                trimmed = await agent_ctx.enforce_limits(agent_type)
                total_trimmed += trimmed
            if expired or total_trimmed:
                logger.info(
                    "Agent memory cleanup: %d expired, %d trimmed",
                    expired, total_trimmed,
                )
        except Exception as e:
            logger.error("Agent memory cleanup failed: %s", e)

    scheduler.add_task(
        "agent_memory_cleanup", task_agent_memory_cleanup,
        interval_sec=86400, phases=[MarketPhase.CLOSED],
    )

    # News sentiment analysis: fetch + analyze news for watchlist symbols
    async def task_news_analysis():
        """Fetch Finnhub news and run LLM sentiment analysis.

        Runs pre-market (once) + every 30min during regular hours.
        Results cached on scanner pipeline + API endpoint.
        """
        if not news_service.available:
            return
        try:
            from db.trade_repository import TradeRepository
            from api.news import update_sentiment_cache

            # Get current watchlist symbols
            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True, market="US")
                # Only fetch news for non-ETF stocks (ETFs rarely have useful news)
                symbols = [
                    w.symbol for w in wl
                    if w.source != "etf_universe"
                ][:15]  # Limit to save LLM costs + avoid rate limits

            if not symbols:
                return

            # Fetch news batch from Finnhub
            batch = await news_service.fetch_batch(
                symbols=symbols, days_back=3, max_per_symbol=5,
            )

            if not batch.articles:
                logger.debug("News analysis: no articles found")
                return

            # Run LLM sentiment analysis
            if news_sentiment_agent:
                summary = await news_sentiment_agent.analyze_batch(batch.articles)
                scanner_pipeline.set_news_summary(summary)

                actionable = len(summary.actionable_signals)
                logger.info(
                    "News analysis: %d articles -> %d symbols, "
                    "market_sentiment=%.2f, actionable=%d",
                    len(batch.articles), len(summary.symbol_sentiments),
                    summary.market_sentiment, actionable,
                )

                # Cache for API endpoint
                update_sentiment_cache(
                    summary.to_dict(),
                    [s.to_dict() for s in summary.actionable_signals],
                )

                # Feed sentiment to evaluation loop for protective sells
                if summary.symbol_sentiments:
                    evaluation_loop.update_news_sentiment(summary.symbol_sentiments)

                # Discord alert for high-impact news
                if actionable > 0:
                    msg = "News Sentiment Alert\n"
                    for sig in summary.actionable_signals[:5]:
                        emoji = "+" if sig.sentiment > 0 else "-"
                        msg += (
                            f"  {sig.symbol}: [{sig.trading_signal}] "
                            f"{sig.key_event} "
                            f"(sentiment={sig.sentiment:{emoji}.2f}, "
                            f"impact={sig.impact})\n"
                        )
                    msg += f"\nMarket sentiment: {summary.market_sentiment:+.2f}"
                    await notification.notify_system_event("news_sentiment", msg)

        except Exception as e:
            logger.error("News analysis failed: %s", e)

    scheduler.add_task(
        "news_analysis", task_news_analysis,
        interval_sec=3600, phases=[MarketPhase.PRE_MARKET, MarketPhase.REGULAR],
    )

    # Event calendar refresh (earnings, insider transactions)
    async def task_event_calendar_refresh():
        """Daily: refresh earnings calendar + insider transactions."""
        if not earnings_svc.available:
            return
        try:
            from db.trade_repository import TradeRepository
            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True, market="US")
                symbols = [
                    w.symbol for w in wl
                    if w.source != "etf_universe"
                ][:40]

            if not symbols:
                return

            await event_calendar.refresh_all(symbols)

            # Discord alert for held positions with upcoming earnings
            for symbol in position_tracker.tracked_symbols:
                events = earnings_svc.get_upcoming(symbol, days_ahead=3)
                if events:
                    dates = ", ".join(e.date for e in events)
                    await notification.notify_system_event(
                        "earnings_alert",
                        f"Earnings Alert: {symbol} reports on {dates}. SL widened.",
                    )

            logger.info("Event calendar refreshed: %d symbols", len(symbols))
        except Exception as e:
            logger.error("Event calendar refresh failed: %s", e)

    scheduler.add_task(
        "event_calendar_refresh", task_event_calendar_refresh,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET],
    )

    # KR news sentiment analysis (Naver Finance)
    from data.naver_news_service import NaverNewsService
    naver_news_service = NaverNewsService()
    app.state.naver_news_service = naver_news_service

    async def task_kr_news_analysis():
        """Fetch Naver Finance news and run LLM sentiment analysis for KR stocks.

        Runs pre-market (once) + every 30min during KR regular hours.
        Results cached on API endpoint + fed to KR evaluation loop.
        """
        try:
            from db.trade_repository import TradeRepository
            from api.news import update_kr_sentiment_cache
            from data.stock_name_service import get_name as get_stock_name

            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True, market="KR")
                symbols = [
                    w.symbol for w in wl
                    if w.source != "etf_universe"
                ][:15]  # Limit to save LLM costs

            if not symbols:
                return

            # Build symbol->name mapping for LLM context
            kr_names = {}
            for sym in symbols:
                name = get_stock_name(sym, "KR")
                if name:
                    kr_names[sym] = name

            batch = await naver_news_service.fetch_batch(
                symbols=symbols, max_per_symbol=5,
            )

            if not batch.articles:
                logger.debug("KR news analysis: no articles found")
                return

            if kr_news_sentiment_agent:
                summary = await kr_news_sentiment_agent.analyze_batch(
                    batch.articles, symbol_names=kr_names,
                )

                actionable = len(summary.actionable_signals)
                logger.info(
                    "KR news analysis: %d articles -> %d symbols, "
                    "market_sentiment=%.2f, actionable=%d",
                    len(batch.articles), len(summary.symbol_sentiments),
                    summary.market_sentiment, actionable,
                )

                # Cache for API endpoint
                update_kr_sentiment_cache(
                    summary.to_dict(),
                    [s.to_dict() for s in summary.actionable_signals],
                )

                # Feed sentiment to KR evaluation loop for protective sells
                if summary.symbol_sentiments:
                    kr_evaluation_loop.update_news_sentiment(summary.symbol_sentiments)

                # Discord alert for high-impact KR news
                if actionable > 0:
                    msg = "KR News Sentiment Alert\n"
                    for sig in summary.actionable_signals[:5]:
                        emoji = "+" if sig.sentiment > 0 else "-"
                        msg += (
                            f"  {sig.symbol}: [{sig.trading_signal}] "
                            f"{sig.key_event} "
                            f"(sentiment={sig.sentiment:{emoji}.2f}, "
                            f"impact={sig.impact})\n"
                        )
                    msg += f"\nKR market sentiment: {summary.market_sentiment:+.2f}"
                    await notification.notify_system_event("kr_news_sentiment", msg)

        except Exception as e:
            logger.error("KR news analysis failed: %s", e)

    scheduler.add_task(
        "kr_news_analysis", task_kr_news_analysis,
        interval_sec=3600, phases=[MarketPhase.PRE_MARKET, MarketPhase.REGULAR],
        market="KR",
    )

    # ── KR market tasks ──────────────────────────────────────────────

    # KR evaluation loop (same strategies, KR market data + order manager)
    kr_evaluation_loop = EvaluationLoop(
        adapter=kr_adapter,
        market_data=kr_market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=SignalCombiner(consensus_config=consensus_cfg),
        order_manager=kr_order_manager,
        risk_manager=kr_risk_manager,
        adaptive_weights=AdaptiveWeightManager(
            alpha=adaptive_cfg.get("alpha", 0.6),
            ema_decay=adaptive_cfg.get("ema_decay", 0.1),
            min_signals_for_adaptation=adaptive_cfg.get("min_signals", 5),
        ),
        position_tracker=kr_position_tracker,
        market="KR",
    )
    kr_evaluation_loop._daily_buy_limit = config.trading.daily_buy_limit
    # STOCK-43: Apply config cooldown + Redis persistence + PositionTracker callback
    kr_evaluation_loop._sell_cooldown_secs = config.trading.cooldown_after_sell_sec
    kr_evaluation_loop.set_cache(cache)
    kr_position_tracker.register_on_sell(kr_evaluation_loop.register_sell_cooldown)
    app.state.kr_evaluation_loop = kr_evaluation_loop

    # Cross-link market data for combined portfolio allocation (통합증거금)
    evaluation_loop.set_other_market_data(kr_market_data)
    kr_evaluation_loop.set_other_market_data(market_data)

    async def task_kr_position_check():
        await kr_position_tracker.check_all()

    async def task_kr_position_db_sync():
        """Periodic reconciliation: sync KR in-memory positions to DB."""
        await kr_position_tracker.sync_to_db()

    async def task_kr_order_reconciliation():
        changes = await kr_order_manager.reconcile_all()
        if changes:
            logger.info("KR order reconciliation: %d status changes", len(changes))
            from api.trades import update_order_in_db, record_trade
            for change in changes:
                # Update existing DB row status (also updates in-memory log entry)
                await update_order_in_db(
                    kis_order_id=change.get("order_id", ""),
                    status=change["new_status"],
                    filled_price=change.get("filled_price"),
                    filled_quantity=change.get("filled_quantity"),
                )
                # STOCK-33: If newly filled, ensure it appears in trade log.
                # record_trade() is now idempotent: if the order_id already
                # exists in _trade_log (from place_sell), it merges without
                # creating a duplicate. PnL is preserved during merge.
                if change["new_status"] == "filled" and change["old_status"] != "filled":
                    record_trade({
                        "order_id": change.get("order_id", ""),
                        "symbol": change["symbol"],
                        "side": change["side"],
                        "quantity": change.get("quantity", 0),
                        "price": change.get("price"),
                        "filled_price": change.get("filled_price"),
                        "filled_quantity": change.get("filled_quantity", 0),
                        "strategy": change.get("strategy", ""),
                        "status": "filled",
                        "market": "KR",
                        "created_at": "",
                    })
        # Cancel stale unfilled orders
        stale = await kr_order_manager.cancel_stale_orders(
            config.trading.pending_order_ttl_min,
        )
        if stale:
            from api.trades import update_order_in_db
            for s in stale:
                await update_order_in_db(
                    kis_order_id=s["order_id"], status="cancelled",
                )
            if notification:
                symbols = ", ".join(f"{s['side']} {s['symbol']}" for s in stale)
                await notification.notify_system_event(
                    "kr_stale_order_cancel",
                    f"KR 미체결 주문 자동 취소 ({len(stale)}건, "
                    f"TTL={config.trading.pending_order_ttl_min}분): {symbols}",
                )

    async def task_kr_portfolio_snapshot():
        await kr_portfolio_manager.save_snapshot()
        logger.debug("KR portfolio snapshot saved")

    async def task_kr_evaluation_loop():
        """KR evaluation: run strategies on KR watchlist symbols."""
        await kr_evaluation_loop._evaluate_all()

    async def task_kr_daily_scan():
        """KR daily scan: discover stocks via KRScreener, update KR watchlist."""
        from scanner.kr_screener import KRScreener, get_kr_exchange
        from db.trade_repository import TradeRepository

        try:
            screener = KRScreener()
            result = screener.screen()
            logger.info(
                "KR scan: %d symbols discovered from %d sources",
                result.total_discovered, len(result.sources),
            )

            if not result.symbols:
                return

            # Add top candidates to KR watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                existing = await repo.get_watchlist(active_only=True, market="KR")
                existing_syms = {w.symbol for w in existing}
                added = []
                for sym in result.symbols[:40]:
                    if sym not in existing_syms:
                        await repo.add_to_watchlist(
                            symbol=sym, exchange=get_kr_exchange(sym),
                            source="scanner", market="KR",
                        )
                        added.append(sym)

            # Update KR evaluation loop watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True, market="KR")
                kr_evaluation_loop.set_watchlist([w.symbol for w in wl])

            if added:
                logger.info("KR watchlist: +%d added (%s...)", len(added), added[:5])
                await notification.notify_system_event(
                    "kr_daily_scan",
                    f"KR Daily Scan: {result.total_discovered} discovered, "
                    f"+{len(added)} added to watchlist",
                )
        except Exception as e:
            logger.error("KR daily scan failed: %s", e)

    scheduler.add_task(
        "kr_position_check", task_kr_position_check,
        interval_sec=60, phases=[MarketPhase.REGULAR], market="KR",
    )
    scheduler.add_task(
        "kr_position_db_sync", task_kr_position_db_sync,
        interval_sec=300, phases=[MarketPhase.REGULAR], market="KR",
    )
    scheduler.add_task(
        "kr_order_reconciliation", task_kr_order_reconciliation,
        interval_sec=120, phases=[MarketPhase.REGULAR], market="KR",
    )
    scheduler.add_task(
        "kr_portfolio_snapshot", task_kr_portfolio_snapshot,
        interval_sec=3600, phases=[MarketPhase.REGULAR], market="KR",
    )
    scheduler.add_task(
        "kr_evaluation_loop", task_kr_evaluation_loop,
        interval_sec=300, phases=[MarketPhase.REGULAR], market="KR",
    )
    scheduler.add_task(
        "kr_daily_scan", task_kr_daily_scan,
        interval_sec=86400, phases=[MarketPhase.PRE_MARKET], market="KR",
    )

    async def task_kr_market_state_update():
        """Update KR market regime from KODEX 200 data."""
        try:
            kospi_df = await kr_market_data.get_ohlcv("069500", limit=250)
            if not kospi_df.empty:
                state = kr_market_state_detector.detect(kospi_df)
                kr_evaluation_loop.set_market_state(state.regime.value)
                kr_risk_manager.set_market_regime("KR", state.regime.value)
                logger.info(
                    "KR market state: %s (KODEX200=%.0f, SMA200=%.0f)",
                    state.regime.value, state.spy_price, state.spy_sma200,
                )
        except Exception as e:
            logger.error("KR market state update failed: %s", e)

    scheduler.add_task(
        "kr_market_state_update", task_kr_market_state_update,
        interval_sec=900, phases=[MarketPhase.REGULAR], market="KR",
    )

    async def task_kr_etf_evaluation():
        """KR ETF Engine: regime-based leveraged pair + sector ETF rotation."""
        try:
            # Use KODEX 200 (069500) as KOSPI proxy for regime detection
            kospi_df = await kr_market_data.get_ohlcv("069500", limit=250)
            if kospi_df.empty:
                return
            state = kr_market_state_detector.detect(kospi_df)
            kr_risk_manager.set_market_regime("KR", state.regime.value)

            # Get KR sector data for sector rotation
            kr_sector_data = await external_data.get_kr_sector_performance()

            # Run KR ETF engine evaluation
            actions = await kr_etf_engine.evaluate(
                market_state=state,
                sector_data=kr_sector_data,
            )

            total = sum(len(v) for v in actions.values())
            if total > 0:
                logger.info(
                    "KR ETF Engine: %d actions (regime=%d, sector=%d, risk=%d)",
                    total, len(actions["regime"]),
                    len(actions["sector"]), len(actions["risk"]),
                )
        except Exception as e:
            logger.error("KR ETF evaluation failed: %s", e)

    scheduler.add_task(
        "kr_etf_evaluation", task_kr_etf_evaluation,
        interval_sec=900, phases=[MarketPhase.REGULAR], market="KR",
    )

    # ── Extended Hours Tasks ─────────────────────────────────────────
    # SL/TP monitoring during pre-market/after-hours (US + KR)
    # Gated by config.extended_hours.enabled kill switch

    async def task_us_extended_position_check():
        """US extended hours: SL/TP monitoring with limit-order sells."""
        if not config.extended_hours.enabled or not config.extended_hours.us_enabled:
            return
        from engine.scheduler import get_market_phase
        phase = get_market_phase()
        session = "pre_market" if phase == MarketPhase.PRE_MARKET else "after_hours"
        try:
            triggered = await position_tracker.check_all(session=session)
            if triggered:
                logger.info(
                    "US extended hours [%s]: %d SL/TP triggered",
                    session, len(triggered),
                )
        except Exception as e:
            logger.error("US extended position check failed: %s", e)

    async def task_kr_extended_position_check():
        """KR extended hours: SL/TP monitoring with limit-order sells."""
        if not config.extended_hours.enabled or not config.extended_hours.kr_enabled:
            return
        from engine.scheduler import get_kr_market_phase
        phase = get_kr_market_phase()
        if phase == MarketPhase.PRE_MARKET:
            session = "pre_market"
        elif phase == MarketPhase.AFTER_HOURS:
            session = "after_hours"
        else:
            return
        try:
            triggered = await kr_position_tracker.check_all(session=session)
            if triggered:
                logger.info(
                    "KR extended hours [%s]: %d SL/TP triggered",
                    session, len(triggered),
                )
        except Exception as e:
            logger.error("KR extended position check failed: %s", e)

    scheduler.add_task(
        "us_extended_position_check", task_us_extended_position_check,
        interval_sec=120, phases=[MarketPhase.PRE_MARKET, MarketPhase.AFTER_HOURS],
    )
    scheduler.add_task(
        "kr_extended_position_check", task_kr_extended_position_check,
        interval_sec=120, phases=[MarketPhase.PRE_MARKET, MarketPhase.AFTER_HOURS],
        market="KR",
    )

    app.state.scheduler = scheduler

    # Load watchlists into evaluation loops from DB
    try:
        from db.trade_repository import TradeRepository
        async with session_factory() as session:
            repo = TradeRepository(session)
            # US watchlist
            wl = await repo.get_watchlist(active_only=True, market="US")
            symbols = [w.symbol for w in wl]
            if symbols:
                evaluation_loop.set_watchlist(symbols)
                logger.info("US watchlist loaded: %d symbols", len(symbols))
            # KR watchlist
            kr_wl = await repo.get_watchlist(active_only=True, market="KR")
            kr_symbols = [w.symbol for w in kr_wl]
            if kr_symbols:
                kr_evaluation_loop.set_watchlist(kr_symbols)
                logger.info("KR watchlist loaded: %d symbols", len(kr_symbols))
    except Exception as e:
        logger.warning("Failed to load watchlist: %s", e)

    # STOCK-43: Restore sell cooldowns from Redis so they survive restarts
    try:
        us_cooldowns = await evaluation_loop.load_sell_cooldowns()
        kr_cooldowns = await kr_evaluation_loop.load_sell_cooldowns()
        if us_cooldowns or kr_cooldowns:
            logger.info(
                "Sell cooldowns restored: US=%d, KR=%d", us_cooldowns, kr_cooldowns
            )
    except Exception as e:
        logger.warning("Failed to load sell cooldowns: %s", e)

    # Startup position reconciliation — restore tracking from exchange
    # Falls back to DB restore if exchange API fails (STOCK-7)
    import asyncio
    try:
        us_restored = await position_tracker.restore_from_exchange(session_factory)
        if not us_restored:
            us_restored = await position_tracker.restore_from_db(session_factory)
            if us_restored:
                logger.info("US positions restored from DB fallback: %d", len(us_restored))
        kr_restored = await kr_position_tracker.restore_from_exchange(session_factory)
        if not kr_restored:
            kr_restored = await kr_position_tracker.restore_from_db(session_factory)
            if kr_restored:
                logger.info("KR positions restored from DB fallback: %d", len(kr_restored))

        # Notify via Discord on startup
        startup_lines = ["System restarted — position reconciliation complete."]
        if us_restored:
            startup_lines.append(f"\n**US Positions ({len(us_restored)}):**")
            for p in us_restored:
                sign = "+" if p["pnl_pct"] >= 0 else ""
                startup_lines.append(
                    f"  • {p['symbol']}: {p['quantity']}주 @ ${p['entry_price']:.2f} "
                    f"→ ${p['current_price']:.2f} ({sign}{p['pnl_pct']:.1f}%)"
                )
        if kr_restored:
            startup_lines.append(f"\n**KR Positions ({len(kr_restored)}):**")
            for p in kr_restored:
                sign = "+" if p["pnl_pct"] >= 0 else ""
                startup_lines.append(
                    f"  • {p['symbol']}: {p['quantity']}주 @ ₩{p['entry_price']:,.0f} "
                    f"→ ₩{p['current_price']:,.0f} ({sign}{p['pnl_pct']:.1f}%)"
                )
        if not us_restored and not kr_restored:
            startup_lines.append("No open positions found.")

        await notification.notify_system_event(
            "startup_reconciliation", "\n".join(startup_lines),
        )
        logger.info(
            "Startup reconciliation: US=%d, KR=%d positions restored",
            len(us_restored), len(kr_restored),
        )
    except Exception as e:
        logger.error("Startup position reconciliation failed: %s", e)

    # Restore ETF Engine managed_positions from broker + DB (STOCK-23)
    try:
        us_etf_restored = await etf_engine.restore_managed_positions(session_factory)
        kr_etf_restored = await kr_etf_engine.restore_managed_positions(session_factory)
        etf_total = len(us_etf_restored) + len(kr_etf_restored)
        if etf_total:
            etf_lines = ["ETF Engine positions restored:"]
            if us_etf_restored:
                etf_lines.append(f"  US: {', '.join(r['symbol'] for r in us_etf_restored)}")
            if kr_etf_restored:
                etf_lines.append(f"  KR: {', '.join(r['symbol'] for r in kr_etf_restored)}")
            await notification.notify_system_event(
                "etf_restore", "\n".join(etf_lines),
            )
            logger.info(
                "ETF Engine restore: US=%d, KR=%d positions",
                len(us_etf_restored), len(kr_etf_restored),
            )
    except Exception as e:
        logger.error("ETF Engine position restore failed: %s", e)

    # Cancel orphaned pending orders on KIS (from previous server instance)
    try:
        for label, adapter_inst in [("US", adapter), ("KR", kr_adapter)]:
            pending = await adapter_inst.fetch_pending_orders()
            if pending:
                cancelled = 0
                for order in pending:
                    try:
                        ok = await adapter_inst.cancel_order(order.order_id, order.symbol)
                        if ok:
                            cancelled += 1
                            logger.info(
                                "Cancelled orphaned %s order: %s %s %s %d @ %.0f",
                                label, order.order_id, order.side,
                                order.symbol, int(order.quantity), order.price or 0,
                            )
                    except Exception as cancel_err:
                        logger.warning("Failed to cancel order %s: %s", order.order_id, cancel_err)
                if cancelled:
                    await notification.notify_system_event(
                        "orphaned_orders_cancelled",
                        f"{label}: {cancelled}/{len(pending)} orphaned pending orders cancelled on startup.",
                    )
    except Exception as e:
        logger.warning("Orphaned order cleanup failed: %s", e)

    # Reconcile pending DB orders + restore trade log
    try:
        from api.trades import (
            reconcile_pending_orders,
            restore_trade_log,
            recover_not_found_orders,
        )

        # STOCK-38: Recover orders stuck in 'not_found' (from before the fix)
        recovered = await recover_not_found_orders()
        if recovered:
            logger.info(
                "STOCK-38: Recovered %d not_found orders on startup", recovered
            )

        # Collect all currently held symbols (from position restore above)
        held_symbols: set[str] = set()
        for p in (us_restored or []):
            held_symbols.add(p["symbol"])
        for p in (kr_restored or []):
            held_symbols.add(p["symbol"])

        # Update pending DB orders based on what we actually hold
        reconciled = await reconcile_pending_orders(held_symbols)
        if reconciled:
            logger.info("Reconciled %d pending DB orders on startup", reconciled)

        # Restore in-memory trade log from DB (includes just-reconciled orders)
        count = await restore_trade_log()
        logger.info("Trade log restored: %d entries from DB", count)
    except Exception as e:
        logger.warning("DB order reconciliation/restore failed: %s", e)

    # Run initial data fetches in background (non-blocking startup)
    async def _initial_data_fetch():
        errors = []
        try:
            await task_event_calendar_refresh()
            logger.info("Event calendar loaded")
        except Exception as e:
            logger.warning("Event calendar fetch failed: %s", e)
            errors.append(f"Event calendar: {e}")
        try:
            await task_news_analysis()
            logger.info("US news sentiment loaded")
        except Exception as e:
            logger.warning("US news analysis failed: %s", e)
            errors.append(f"US news: {e}")
        try:
            await task_kr_news_analysis()
            logger.info("KR news sentiment loaded")
        except Exception as e:
            logger.warning("KR news analysis failed: %s", e)
            errors.append(f"KR news: {e}")

        if errors and notification:
            try:
                msg = "⚠️ **Startup Background Tasks Failed**\n" + "\n".join(f"- {e}" for e in errors)
                await notification.send_message(msg)
            except Exception as e:
                logger.error("Failed to send startup failure notification: %s", e)

    asyncio.create_task(_initial_data_fetch(), name="initial-data-fetch")

    # Auto-start scheduler (store task ref to detect crashes)
    _scheduler_task = asyncio.create_task(scheduler.start(), name="scheduler")

    def _on_scheduler_done(task: asyncio.Task):
        if task.cancelled():
            logger.warning("Scheduler task was cancelled")
        elif task.exception():
            logger.error("Scheduler task crashed: %s", task.exception())
        else:
            logger.info("Scheduler task finished normally")

    _scheduler_task.add_done_callback(_on_scheduler_done)
    app.state.scheduler_task = _scheduler_task
    logger.info("Scheduler auto-started")

    # Install WebSocket log handler for real-time log streaming
    install_log_handler()

    logger.info(
        "Services initialized: %d strategies loaded",
        len(registry.get_names()),
    )

    yield

    # Shutdown
    if scheduler.running:
        await scheduler.stop()
    if kis_ws and kis_ws.is_connected:
        await kis_ws.close()
    await news_service.close()
    await naver_news_service.close()
    await cache.close()
    await adapter.close()
    await kr_adapter.close()
    await engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="US Stock Auto-Trading Engine",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health_check():
    if hasattr(app.state, "health"):
        result = await app.state.health.check_all()
        result["version"] = "0.9.0"
        return result
    return {"status": "ok", "version": "0.9.0"}


app.include_router(api_router, prefix="/api/v1")
