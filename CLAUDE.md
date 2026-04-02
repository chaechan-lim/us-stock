# US Stock Auto-Trading Engine

## Project Overview
Dual-market (US + KR) automated stock trading system using Korea Investment & Securities (KIS) Open API.
Architecture inherited from ~/coin project (crypto trading bot). **Live trading** on real accounts (US + KR).

## Tech Stack
- Backend: Python 3.12+, FastAPI (port 8001), SQLAlchemy 2.0 (asyncpg), PostgreSQL, Redis
- Frontend: React 18, TypeScript, Vite, TailwindCSS (port 3001)
- Strategy config: config/strategies.yaml (YAML-based parameter management, runtime hot-reload)
- Testing: pytest + pytest-asyncio + pytest-cov (1276+ tests)
- Deploy: Raspberry Pi ARM64, systemd services, nginx HTTPS reverse proxy (port 8443)

## Core Rules

### Code
- All code must have unit tests (no untested code commits)
- Async functions use async/await pattern (asyncpg, aiohttp)
- External API calls must have error handling + retry logic
- Type hints required (mypy strict compatible)
- Pydantic models for data validation
- Never hardcode strategy parameters in code — use config/strategies.yaml

### Strategy Rules
- New strategies must pass backtest before activation (CAGR>12%, Sharpe>1.0, MDD<25%)
- Inherit BaseStrategy, implement analyze() -> Signal
- Strategy weights defined per market state in profiles section of strategies.yaml
- Start new strategies at weight 0.05, increase gradually after paper validation
- US/KR combo selection (which strategies are enabled per market): threshold is measurable improvement over prior enabled combo on all four dimensions (Ret, Sharpe, MDD, PF) — not the absolute new-strategy gate above. Document baseline vs new combo metrics in strategies.yaml markets section.

### Testing Requirements
- Unit tests: pytest + pytest-asyncio, coverage target 90%+
- Scenario tests: tests/scenarios/ (core trading flows)
- Backtest: required before any strategy goes live
- DB tests: aiosqlite in-memory (test isolation)
- External APIs: always mock (KIS, yfinance, Claude, Gemini, Finnhub)

### Commits & PRs
- Conventional Commits: feat/fix/refactor/test/docs/config/ci
- PR requires all tests passing before merge
- Strategy change PRs must include backtest results

### Directory Layout
- backend/exchange/: Exchange adapters (KIS US REST, KIS KR REST, KIS WebSocket, Paper)
- backend/strategies/: Trading strategies (1 file per strategy, 14 active strategies)
- backend/engine/: Trading engine (order, position, risk, portfolio managers, evaluation loop)
- backend/scanner/: Stock scanning (3-Layer pipeline: Indicator -> yfinance -> AI, + news Layer 2.5)
- backend/data/: Data services (market data, indicators, FRED, news, earnings, events)
- backend/agents/: AI agents (market analyst, risk assessment, trade review, news sentiment)
- backend/services/llm/: Multi-provider LLM client (Anthropic + Gemini fallback)
- backend/services/: Cache, notification, rate limiter, exchange resolver, health monitor
- backend/analytics/: Quant analytics (factor model, Kelly sizing, signal quality tracker)
- backend/api/: REST + WebSocket endpoints (13 modules)
- backend/backtest/: Backtesting engine (single-strategy + full-pipeline)
- config/: Strategy & ETF YAML config files (US + KR)
- deploy/: systemd services, DB setup/backup scripts
- frontend/src/components/: 20 React components

### Key Architecture Decisions
- **Dual-market**: US + KR in same process, separate adapter/MarketDataService/OrderManager/PositionTracker per market
- KIS API rate limit: 20 req/sec (real), 5 req/sec (paper) — use RateLimiter
- KIS WebSocket: max 41 subscriptions per session, market-hours only, 4h max session
- yfinance-first for bulk data (no rate limits), KIS reserved for orders/balance only
- KR OHLCV: yfinance with symbol mapper (005930 → 005930.KS)
- 3-Layer screening: IndicatorScreener (tech) -> FundamentalEnricher (yfinance) -> AI (LLMClient)
  - Layer 2.5: News sentiment enricher (Finnhub US / Naver KR, ±15 point adjustment, 15% weight)
- LLM multi-provider: services/llm/ — Anthropic primary -> fallback -> Gemini (auto-failover with retry)
  - Fallback chain: claude-haiku -> claude-sonnet -> gemini-2.5-flash
  - Retry: 3x per model with exponential backoff (2s, 4s, 8s)
  - All 4 agents (market_analyst, risk_assessment, trade_review, news_sentiment) use LLMClient
- Dual engine: US/KR Stock Engine (individual stocks) + ETF Engine (leveraged/inverse + sector ETFs)
  - ETF Engine: engine/etf_engine.py — regime-based leveraged pair switching + sector ETF rotation
  - US: TQQQ/SQQQ, SOXL/SOXS, sector ETFs (XLK/XLF/XLE) — SPY regime detection
  - KR: KODEX leveraged/inverse, KOSDAQ150 leveraged, 7 sector ETFs — KODEX 200 regime detection
  - ETF risk rules: max 10-day hold for leveraged, max 30% portfolio, max 15% single ETF
  - Mutual exclusivity: sell conflicting 1x/2x sibling before buying target
- Dynamic universe: scanner/universe_expander.py — yfinance screeners + sector weighting + KIS ranking APIs
- All strategy params in config/strategies.yaml with runtime hot-reload
- 14 strategies total: 10 original + 3 ported from coin (cis_momentum, larry_williams, bnf_deviation) + volume_surge
- SignalCombiner: Mode B (HOLD excluded from denominator, 15% min active ratio), group consensus (trend/mean_reversion)
- Port separation: us-stock 8001/3001, coin 8000/3000
- Shared infra: PostgreSQL (coin's container, separate DB), Redis db 1 (coin uses db 0)
- Notification: adapter pattern (Discord primary, Telegram/Slack supported)
- FRED integration: data/fred_service.py — macro indicators (async via asyncio.to_thread)
- Order safety: dedup check, reconciliation task (2min), slippage/partial fill tracking, fetch_executed_orders fallback
- Event calendar: earnings (buy block D-3, SL widen 1.5x), macro (FOMC block, CPI/Jobs 50%), insider (confidence ±0.10)
- ATR-based dynamic SL/TP: per-stock volatility-adjusted (US 3-15%, KR 5-20%)
- Market allocation: 50:50 US/KR, regime-based dynamic (bull +20%, bear -20%, clamp 20%-70%)
- DST auto-detection: zoneinfo("America/New_York") for US market phase
- MCP server: backend/mcp_server.py (FastMCP, 28 tools for Claude Desktop/Code)
- DB backup: local daily (7-day retention) + GitHub weekly (4-week), systemd timers
- API auth: Bearer token middleware (AUTH_API_TOKEN env, empty=disabled)
- Scheduler tasks: 29 total (20 US + 7 KR + system tasks)
- Donchian breakout: uses previous bar's channel (pandas-ta donchian includes current bar)
- Bollinger squeeze: squeeze_min_bars=3 (daily timeframe; 6 was too strict)
- AI agent integration: RiskAssessmentAgent pre-trade check in evaluation_loop (non-blocking), TradeReviewAgent daily review (after-hours), agent memory cleanup (daily)
- Agent persistent memory: AgentContextService (DB-backed, token budget, auto-expiry, importance-based eviction)

### Reference
- ~/coin: Crypto trading bot (architecture reference)
- SYSTEM_DESIGN.md: Full system design document
- config/strategies.yaml: Strategy parameters and weights
