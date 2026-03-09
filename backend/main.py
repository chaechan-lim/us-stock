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

    # KR adapter (shares same auth for live, separate paper instance)
    if config.is_paper:
        kr_adapter = PaperAdapter(config.trading.initial_balance_usd * 1_300)  # ~KRW
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
    risk_manager = RiskManager()
    order_manager = OrderManager(adapter=adapter, risk_manager=risk_manager, notification=notification, market_data=market_data)
    app.state.risk_manager = risk_manager
    app.state.order_manager = order_manager
    consensus_cfg = registry._config_loader.get_consensus_config()
    app.state.combiner = SignalCombiner(consensus_config=consensus_cfg)

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

    # Portfolio manager
    session_factory = get_session_factory(config.database)
    portfolio_manager = PortfolioManager(
        market_data=market_data, session_factory=session_factory,
    )
    app.state.portfolio_manager = portfolio_manager

    # Wire trade DB persistence
    from api.trades import init_trades
    init_trades(session_factory)

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

    # AI agents (market analyst, risk assessment, trade review)
    ai_agent = None
    risk_agent = None
    trade_review_agent = None
    if llm_client:
        from agents.market_analyst import MarketAnalystAgent
        from agents.risk_assessment import RiskAssessmentAgent
        from agents.trade_review import TradeReviewAgent
        ai_agent = MarketAnalystAgent(llm_client=llm_client, context_service=agent_ctx)
        risk_agent = RiskAssessmentAgent(llm_client=llm_client, context_service=agent_ctx)
        trade_review_agent = TradeReviewAgent(llm_client=llm_client, context_service=agent_ctx)
        logger.info("AI agents enabled (analyst, risk, trade_review)")
    app.state.risk_agent = risk_agent
    app.state.trade_review_agent = trade_review_agent

    # Resolve services from app.state
    indicator_svc = app.state.indicator_svc
    combiner = app.state.combiner

    # Scanner pipeline (with AI agent if LLM enabled)
    enricher = FundamentalEnricher()
    scanner_pipeline = ScannerPipeline(
        market_data=market_data,
        indicator_svc=indicator_svc,
        enricher=enricher,
        ai_agent=ai_agent,
    )
    app.state.scanner_pipeline = scanner_pipeline

    # Evaluation loop (after agents — risk_agent used for pre-trade check)
    evaluation_loop = EvaluationLoop(
        adapter=adapter,
        market_data=market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=combiner,
        order_manager=order_manager,
        risk_manager=risk_manager,
        adaptive_weights=adaptive_weights,
        risk_agent=risk_agent,
    )
    app.state.evaluation_loop = evaluation_loop

    # Stock scanner & sector analyzer
    stock_scanner = StockScanner(adapter=adapter, market_data=market_data)
    sector_analyzer = SectorAnalyzer()
    external_data = ExternalDataService()
    etf_universe = ETFUniverse()
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
    # Share rate limiter with US — same KIS credentials, same rate limit
    kr_market_data = MarketDataService(
        adapter=kr_adapter, rate_limiter=rate_limiter,
        yf_symbol_mapper=lambda s: kr_to_yfinance(s, "KRX"),
    )
    kr_order_manager = OrderManager(
        adapter=kr_adapter, risk_manager=risk_manager, notification=notification,
        market_data=kr_market_data,
    )
    kr_position_tracker = PositionTracker(
        adapter=kr_adapter,
        risk_manager=risk_manager,
        order_manager=kr_order_manager,
        notification=notification,
        market_data=kr_market_data,
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

    async def task_order_reconciliation():
        changes = await order_manager.reconcile_all()
        if changes:
            logger.info("Order reconciliation: %d status changes", len(changes))

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
        max_watchlist = 60
        min_age_days = 7  # Don't remove recently added

        # Get current positions to protect
        try:
            positions = await market_data.get_positions()
            held_symbols = {p.symbol for p in positions if p.quantity > 0}
        except Exception:
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

            # Run 3-layer pipeline (grade C to cast a wider net)
            candidates = await scanner_pipeline.run_full_scan(
                symbols=universe, min_grade="C", max_candidates=15,
            )

            if not candidates:
                logger.info("After-hours scan: no candidates found")
                # Still run cleanup even if no new candidates
                removed = await _cleanup_watchlist(existing, [])
                if removed:
                    logger.info("Watchlist cleanup: removed %s", removed)
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
                msg += (
                    f"  {c['symbol']}: score={c['combined_score']:.0f} "
                    f"grade={c.get('grade', '?')}{ai_note}\n"
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
                    "executed_at": str(t.executed_at),
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
            for agent_type in ("market_analyst", "risk", "trade_review"):
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

    # ── KR market tasks ──────────────────────────────────────────────

    # KR evaluation loop (same strategies, KR market data + order manager)
    kr_evaluation_loop = EvaluationLoop(
        adapter=kr_adapter,
        market_data=kr_market_data,
        indicator_svc=indicator_svc,
        registry=registry,
        combiner=SignalCombiner(consensus_config=consensus_cfg),
        order_manager=kr_order_manager,
        risk_manager=risk_manager,
        adaptive_weights=AdaptiveWeightManager(
            alpha=adaptive_cfg.get("alpha", 0.6),
            ema_decay=adaptive_cfg.get("ema_decay", 0.1),
            min_signals_for_adaptation=adaptive_cfg.get("min_signals", 5),
        ),
    )
    app.state.kr_evaluation_loop = kr_evaluation_loop

    async def task_kr_position_check():
        await kr_position_tracker.check_all()

    async def task_kr_order_reconciliation():
        changes = await kr_order_manager.reconcile_all()
        if changes:
            logger.info("KR order reconciliation: %d status changes", len(changes))

    async def task_kr_portfolio_snapshot():
        await kr_portfolio_manager.save_snapshot()
        logger.debug("KR portfolio snapshot saved")

    async def task_kr_evaluation_loop():
        """KR evaluation: run strategies on KR watchlist symbols."""
        await kr_evaluation_loop._evaluate_all()

    async def task_kr_daily_scan():
        """KR daily scan: discover stocks via KRScreener, update KR watchlist."""
        from scanner.kr_screener import KRScreener
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
                for sym in result.symbols[:30]:
                    if sym not in existing_syms:
                        await repo.add_to_watchlist(
                            symbol=sym, exchange="KRX", source="scanner", market="KR",
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

    async def task_kr_etf_evaluation():
        """KR ETF Engine: regime-based leveraged pair + sector ETF rotation."""
        try:
            # Use KODEX 200 (069500) as KOSPI proxy for regime detection
            kospi_df = await kr_market_data.get_ohlcv("069500", limit=250)
            if kospi_df.empty:
                return
            state = kr_market_state_detector.detect(kospi_df)

            # Run KR ETF engine evaluation
            actions = await kr_etf_engine.evaluate(
                market_state=state,
                sector_data=None,  # KR sector data from ETF universe config
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

    # Auto-start scheduler (store task ref to detect crashes)
    import asyncio
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


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Bearer token authentication for API endpoints.

    Skips auth for: /health, /docs, /openapi.json, /redoc, WebSocket.
    If AUTH_API_TOKEN is not set, auth is disabled (backward compatible).
    """
    from fastapi.responses import JSONResponse

    config = getattr(getattr(request, "app", None), "state", None)
    token = getattr(config, "config", None)
    api_token = token.auth.api_token if token else ""

    if not api_token:
        return await call_next(request)

    path = request.url.path
    skip_paths = ("/health", "/docs", "/openapi.json", "/redoc")
    if path in skip_paths or path.startswith("/api/v1/ws"):
        return await call_next(request)

    if path.startswith("/api/"):
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {api_token}":
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

    return await call_next(request)


@app.get("/health")
async def health_check():
    if hasattr(app.state, "health"):
        result = await app.state.health.check_all()
        result["version"] = "0.9.0"
        return result
    return {"status": "ok", "version": "0.9.0"}


app.include_router(api_router, prefix="/api/v1")
