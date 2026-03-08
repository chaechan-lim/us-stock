"""US Stock Auto-Trading Engine - FastAPI Application."""

import logging
from contextlib import asynccontextmanager

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
from engine.order_manager import OrderManager, set_trade_recorder
from engine.position_tracker import PositionTracker
from engine.scheduler import TradingScheduler, MarketPhase
from engine.evaluation_loop import EvaluationLoop
from engine.portfolio_manager import PortfolioManager
from engine.recovery import RecoveryManager
from scanner.pipeline import ScannerPipeline
from scanner.fundamental_enricher import FundamentalEnricher
from scanner.stock_scanner import StockScanner
from scanner.sector_analyzer import SectorAnalyzer
from data.market_state import MarketStateDetector
from data.external_data_service import ExternalDataService
from data.fred_service import FREDService
from db.session import get_session_factory
from services.cache import CacheService
from services.health import HealthMonitor
from services.notification import NotificationService

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

    # Create database tables
    engine = get_engine(config.database)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

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

    # Engine components
    risk_manager = RiskManager()
    order_manager = OrderManager(adapter=adapter, risk_manager=risk_manager)
    app.state.risk_manager = risk_manager
    app.state.order_manager = order_manager
    app.state.combiner = SignalCombiner()

    # Wire trade recording
    from api.trades import record_trade
    set_trade_recorder(record_trade)

    # Health monitor
    health = HealthMonitor()

    async def check_adapter():
        await adapter.fetch_balance()
        return {"mode": config.trading.mode}

    health.register_check("adapter", check_adapter)
    app.state.health = health

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

    # Position tracker
    position_tracker = PositionTracker(
        adapter=adapter,
        risk_manager=risk_manager,
        order_manager=order_manager,
        notification=notification,
        market_data=market_data,
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

    # Evaluation loop
    indicator_svc = app.state.indicator_svc
    combiner = app.state.combiner
    evaluation_loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=combiner,
        order_manager=order_manager,
        risk_manager=risk_manager,
        adaptive_weights=adaptive_weights,
    )
    app.state.evaluation_loop = evaluation_loop

    # Portfolio manager
    session_factory = get_session_factory(config.database)
    portfolio_manager = PortfolioManager(
        market_data=market_data, session_factory=session_factory,
    )
    app.state.portfolio_manager = portfolio_manager

    # Scanner pipeline
    enricher = FundamentalEnricher()
    scanner_pipeline = ScannerPipeline(
        market_data=market_data,
        indicator_svc=indicator_svc,
        enricher=enricher,
    )
    app.state.scanner_pipeline = scanner_pipeline

    # Stock scanner & sector analyzer
    stock_scanner = StockScanner(adapter=adapter, market_data=market_data)
    sector_analyzer = SectorAnalyzer()
    external_data = ExternalDataService()
    app.state.stock_scanner = stock_scanner
    app.state.sector_analyzer = sector_analyzer
    app.state.external_data = external_data

    # Market state detector
    market_state_detector = MarketStateDetector()
    app.state.market_state_detector = market_state_detector

    # FRED macro data service
    fred_service = FREDService(api_key=config.external.fred_api_key)
    app.state.fred_service = fred_service

    # Recovery manager for circuit breakers
    recovery_mgr = RecoveryManager(notification=notification)
    app.state.recovery = recovery_mgr

    # Trading scheduler with tasks
    scheduler = TradingScheduler(recovery_manager=recovery_mgr)

    async def task_health_check():
        await health.check_all()

    async def task_position_check():
        await position_tracker.check_all()

    async def task_daily_reset():
        risk_manager.reset_daily()
        logger.info("Daily risk counters reset")

    async def task_evaluation_loop():
        await evaluation_loop._evaluate_all()

    async def task_daily_scan():
        from db.trade_repository import TradeRepository
        try:
            async with session_factory() as session:
                repo = TradeRepository(session)
                watchlist = await repo.get_watchlist(active_only=True)
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
        "position_check", task_position_check,
        interval_sec=60, phases=[MarketPhase.REGULAR],
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

    async def task_portfolio_snapshot():
        await portfolio_manager.save_snapshot()
        logger.debug("Portfolio snapshot saved")

    async def task_market_state_update():
        """T0: Update market regime from SPY data."""
        try:
            spy_df = await market_data.get_ohlcv("SPY", limit=250)
            if not spy_df.empty:
                state = market_state_detector.detect(spy_df)
                evaluation_loop.set_market_state(state.regime.value)
                logger.info(
                    "Market state: %s (SPY=%.2f, SMA200=%.2f, VIX=%.1f)",
                    state.regime.value, state.spy_price,
                    state.spy_sma200, state.vix_level,
                )
        except Exception as e:
            logger.error("Market state update failed: %s", e)

    async def task_intraday_hot_scan():
        """T2: Intraday hot scan — find active stocks during session."""
        from db.trade_repository import TradeRepository
        try:
            async with session_factory() as session:
                repo = TradeRepository(session)
                watchlist = await repo.get_watchlist(active_only=True)
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

    async def task_after_hours_scan():
        """T4a: Post-market stock analysis & watchlist update.

        Runs 3-layer pipeline on a broad universe to find tomorrow's candidates.
        Top candidates are auto-added to watchlist.
        """
        from db.trade_repository import TradeRepository

        # Broad universe: current watchlist + popular large-caps
        base_universe = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META",
            "JPM", "JNJ", "XOM", "UNH", "V", "MA", "HD", "PG",
            "COST", "ABBV", "CRM", "AMD", "NFLX", "ADBE", "AVGO",
            "LLY", "MRK", "PEP", "KO", "WMT", "DIS", "INTC", "CSCO",
        ]
        try:
            # Merge with existing watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                existing = await repo.get_watchlist(active_only=True)
                existing_syms = {w.symbol for w in existing}

            universe = list(existing_syms | set(base_universe))
            logger.info("After-hours scan: %d symbols in universe", len(universe))

            # Run 3-layer pipeline (grade C to cast a wider net)
            candidates = await scanner_pipeline.run_full_scan(
                symbols=universe, min_grade="C", max_candidates=15,
            )

            if not candidates:
                logger.info("After-hours scan: no candidates found")
                return

            # Auto-add top candidates to watchlist
            top_symbols = [c["symbol"] for c in candidates[:10]]
            async with session_factory() as session:
                repo = TradeRepository(session)
                added = []
                for sym in top_symbols:
                    if sym not in existing_syms:
                        await repo.add_to_watchlist(
                            symbol=sym, source="scanner",
                        )
                        added.append(sym)

            # Update evaluation loop watchlist
            async with session_factory() as session:
                repo = TradeRepository(session)
                wl = await repo.get_watchlist(active_only=True)
                evaluation_loop.set_watchlist([w.symbol for w in wl])

            # Send scan results via Discord
            msg = f"After-Hours Scan Complete\nCandidates: {len(candidates)}\n"
            for c in candidates[:10]:
                ai_note = ""
                if c.get("ai_recommendation"):
                    ai_note = f" | AI: {c['ai_recommendation']}"
                msg += (
                    f"  {c['symbol']}: score={c['combined_score']:.0f} "
                    f"grade={c.get('grade', '?')}{ai_note}\n"
                )
            if added:
                msg += f"\nNew watchlist adds: {', '.join(added)}"

            await notification.notify_system_event("after_hours_scan", msg)
            logger.info(
                "After-hours scan: %d candidates, %d new adds",
                len(candidates), len(added),
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
        "portfolio_snapshot", task_portfolio_snapshot,
        interval_sec=3600, phases=[MarketPhase.REGULAR],
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
            indicators = fred_service.fetch_macro_indicators()
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

    app.state.scheduler = scheduler

    # Load watchlist into evaluation loop from DB
    try:
        from db.trade_repository import TradeRepository
        async with session_factory() as session:
            repo = TradeRepository(session)
            wl = await repo.get_watchlist(active_only=True)
            symbols = [w.symbol for w in wl]
            if symbols:
                evaluation_loop.set_watchlist(symbols)
                logger.info("Watchlist loaded: %d symbols", len(symbols))
    except Exception as e:
        logger.warning("Failed to load watchlist: %s", e)

    # Auto-start scheduler
    import asyncio
    asyncio.create_task(scheduler.start())
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
    await cache.close()
    await adapter.close()
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
