# US Stock Auto-Trading Engine

## Project Overview
Automated US stock trading system using Korea Investment & Securities (KIS) Open API.
Architecture inherited from ~/coin project (crypto trading bot).

## Tech Stack
- Backend: Python 3.12+, FastAPI, SQLAlchemy 2.0 (asyncpg), PostgreSQL, Redis
- Frontend: React 18, TypeScript, Vite, TailwindCSS
- Strategy config: config/strategies.yaml (YAML-based parameter management)
- Testing: pytest + pytest-asyncio + pytest-cov

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

### Testing Requirements
- Unit tests: pytest + pytest-asyncio, coverage target 90%+
- Scenario tests: tests/scenarios/ (core trading flows)
- Backtest: required before any strategy goes live
- DB tests: aiosqlite in-memory (test isolation)
- External APIs: always mock (KIS, yfinance, Claude, Gemini)

### Commits & PRs
- Conventional Commits: feat/fix/refactor/test/docs/config/ci
- PR requires all tests passing before merge
- Strategy change PRs must include backtest results

### Directory Layout
- backend/exchange/: Exchange adapters (KIS REST, KIS WebSocket, Paper)
- backend/strategies/: Trading strategies (1 file per strategy)
- backend/engine/: Trading engine (order, position, risk management)
- backend/scanner/: Stock scanning (3-Layer pipeline: Indicator -> yfinance -> AI)
- backend/data/: Data services (market data, indicators, external data)
- backend/agents/: AI agents (market analysis, risk, trade review) — use LLMClient
- backend/services/llm/: Multi-provider LLM client (Anthropic + Gemini fallback)
- backend/api/: REST + WebSocket endpoints
- backend/backtest/: Backtesting engine
- config/: Strategy & ETF YAML config files
- docs/: Architecture, API reference, guides

### Key Architecture Decisions
- KIS API rate limit: 20 req/sec (real), 5 req/sec (paper) — use RateLimiter
- KIS WebSocket: max 41 subscriptions per session, market-hours only, 4h max session
- yfinance-first for bulk data (no rate limits), KIS reserved for orders/balance only
- 3-Layer screening: IndicatorScreener (tech only) -> FundamentalEnricher (yfinance) -> AI (LLMClient)
- LLM multi-provider: services/llm/ — Anthropic primary -> fallback -> Gemini (auto-failover with retry)
  - Fallback chain: claude-haiku -> claude-sonnet -> gemini-3-flash-preview
  - Retry: 3x per model with exponential backoff (2s, 4s, 8s)
  - All 3 agents (market_analyst, risk_assessment, trade_review) use LLMClient
- Dual engine: US Stock Engine (individual stocks) + ETF Engine (leveraged/inverse + sector ETFs)
  - ETF Engine: engine/etf_engine.py — regime-based leveraged pair switching + sector ETF rotation
  - Leveraged pairs: TQQQ/SQQQ, SOXL/SOXS, UPRO/SPXU, TECL/TECS (auto-switch on regime change)
  - Sector ETFs: XLK, XLF, XLE, etc. (buy top sectors, sell bottom sectors)
  - ETF risk rules: max 10-day hold for leveraged, max 30% portfolio, max 15% single ETF
- Dynamic universe: scanner/universe_expander.py — yfinance screeners + sector-weighted discovery
- All strategy params in config/strategies.yaml with runtime hot-reload
- 13 strategies total: 10 original + 3 ported from coin project (cis_momentum, larry_williams, bnf_deviation)
- Coin strategy adaptations: 4h crypto → 1D stocks, thresholds adjusted (e.g. BNF deviation -10%→-5%)
- Port separation: us-stock 8001/3001, coin 8000/3000
- Shared infra: PostgreSQL (coin's container, separate DB), Redis db 1 (coin uses db 0)
- Notification: adapter pattern (Discord/Telegram/Slack)
- FRED integration: data/fred_service.py — macro indicators (FFR, 10Y/2Y yields, CPI, unemployment)
- Donchian breakout: uses previous bar's channel (pandas-ta donchian includes current bar)
- Bollinger squeeze: squeeze_min_bars=3 (daily timeframe; 6 was too strict)
- Backtest verification: backtest/verify_strategies.py — 13 strategies × 8 stocks × 3yr
- Scheduler tasks: 16 total (health_check, position_check, daily_reset, evaluation_loop, daily_scan, market_state_update, etf_evaluation, portfolio_snapshot, intraday_hot_scan, sector_analysis, after_hours_scan, daily_briefing, macro_update, ws_lifecycle, order_reconciliation)

### Reference
- ~/coin: Crypto trading bot (architecture reference)
- SYSTEM_DESIGN.md: Full system design document
- config/strategies.yaml: Strategy parameters and weights
