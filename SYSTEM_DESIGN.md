# US Stock Auto-Trading Engine - System Design
## 한국투자증권 API 기반 미국주식 자동매매 시스템

---

## 1. 프로젝트 개요

한국투자증권(KIS) Open API를 활용한 **US + KR 듀얼마켓** 자동매매 시스템.
기존 ~/coin 프로젝트의 아키텍처를 계승하면서 주식시장 특성에 맞게 확장.
**현재 실계좌 라이브 운용 중** (US + KR 동시 운영).

### 핵심 목표
- US/KR 주식 실시간 시세 감시 및 자동 매매
- 추세 매매 전략 중심 + 14개 다중 전략 조합
- 레버리지/인버스 ETF 적극 활용 (US: TQQQ/SQQQ, KR: KODEX 레버리지/인버스)
- 종목 스캐닝 및 섹터 분석 기반 동적 종목 선정
- 뉴스 감성 분석 + 이벤트 캘린더 통합
- 충분한 백테스트 후 라이브 전환
- 외부 데이터 소스 통합 (yfinance, FRED, Finnhub, Naver)

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React 18 + TypeScript)             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬────────┐ │
│  │ Overview │ Trades   │ Signals  │ Strategy │ Scanner  │ System │ │
│  │ Dashboard│ History  │ Log      │ Perf     │ Sectors  │ Log    │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴────────┘ │
│           ▲ REST API (axios)          ▲ WebSocket (real-time)       │
└───────────┼───────────────────────────┼─────────────────────────────┘
            │                           │
┌───────────┼───────────────────────────┼─────────────────────────────┐
│           ▼           FastAPI         ▼                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  REST API   │  │  WebSocket   │  │  Scheduler   │              │
│  │  Router     │  │  Manager     │  │  (APScheduler)│              │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                │                  │                       │
│  ┌──────▼──────────────────────────────────▼───────────────────┐   │
│  │                    Trading Engine                            │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │   │
│  │  │  US Stock   │ │  KR Stock  │ │  ETF        │              │   │
│  │  │  Engine     │ │  Engine    │ │  Engine     │              │   │
│  │  └────────────┘ └────────────┘ └────────────┘              │   │
│  │       │               │               │                     │   │
│  │  ┌────▼───────────────▼───────────────▼──────────────────┐  │   │
│  │  │              Strategy Layer                            │  │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐  │  │   │
│  │  │  │ Trend   │ │Momentum │ │ Mean    │ │ Breakout   │  │  │   │
│  │  │  │Following│ │         │ │Reversion│ │            │  │  │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └────────────┘  │  │   │
│  │  │  ┌─────────────────────────────────────────────────┐  │  │   │
│  │  │  │        Signal Combiner (Weighted Voting)        │  │  │   │
│  │  │  └─────────────────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │   │
│  │  │ Order    │ │Portfolio │ │ Risk     │ │ Position     │   │   │
│  │  │ Manager  │ │ Manager  │ │ Manager  │ │ Tracker      │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Data Layer                                 │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐               │   │
│  │  │ KIS API    │ │ Market     │ │ External   │               │   │
│  │  │ Adapter    │ │ Data Svc   │ │ Data Svc   │               │   │
│  │  │ (REST+WS)  │ │ (Cache)    │ │(yfinance+) │               │   │
│  │  └────────────┘ └────────────┘ └────────────┘               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Intelligence Layer                         │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐               │   │
│  │  │ Scanner    │ │ Sector     │ │ AI Agent   │               │   │
│  │  │ Service    │ │ Analyzer   │ │(LLMClient) │               │   │
│  │  └────────────┘ └────────────┘ └────────────┘               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │ PostgreSQL │  │   Redis    │  │ Notification│                   │
│  │ (persist)  │  │  (cache)   │  │ (Discord)   │                   │
│  └────────────┘  └────────────┘  └────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 기술 스택

### Backend
| 구분 | 기술 | 비고 |
|------|------|------|
| Language | Python 3.12+ | 기존 coin 프로젝트와 동일 |
| Framework | FastAPI + Uvicorn | async 기반 |
| ORM | SQLAlchemy 2.0+ (asyncpg) | PostgreSQL async |
| Scheduler | APScheduler 3.10+ | 주기적 평가/스캔 |
| KIS API | python-kis (Soju06) + 자체 래퍼 | WebSocket + REST |
| 기술분석 | pandas-ta, ta-lib | 지표 계산 |
| 외부데이터 | yfinance, fredapi | 주가, 경제지표 |
| AI Agent | anthropic + google-genai | 시장분석, 매매 리뷰 (멀티 프로바이더) |
| Cache | Redis 7+ | 시세 캐싱, Rate limit |
| DB | PostgreSQL 16 | 주문/포지션/백테스트 |
| Logging | structlog | 구조화 로깅 |
| Serialization | orjson | 고속 JSON |
| News | finnhub-python | 미국 뉴스 (Finnhub API) |
| KR News | Naver Finance API | 한국 뉴스 (API key 불필요) |
| MCP | mcp==1.26.0 | Claude Desktop/Code 통합 |

### Frontend (coin 프로젝트 계승)
| 구분 | 기술 |
|------|------|
| Framework | React 18 + TypeScript |
| Build | Vite 5 |
| Styling | TailwindCSS 3 |
| Charts | Recharts + lightweight-charts (캔들차트) |
| State | @tanstack/react-query |
| HTTP | axios |
| WebSocket | native + custom hook |

### Infra
| 구분 | 기술 |
|------|------|
| Server | Raspberry Pi ARM64 (192.168.50.244) |
| DB | PostgreSQL 16 (coin 프로젝트 Docker 컨테이너 공유, DB: us_stock_trading) |
| Cache | Redis 7 (공유, db 1 — coin은 db 0) |
| Deploy | systemd (usstock-backend.service, usstock-frontend.service) |
| HTTPS | nginx reverse proxy (port 8443, coin과 인증서 공유) |
| Backup | local daily (7-day) + GitHub weekly (4-week), systemd timers |
| Ports | Backend 8001, Frontend 3001 (coin: 8000/3000) |

---

## 4. 디렉토리 구조

```
~/us-stock/
├── CLAUDE.md                     # AI Agent 지침서
├── SYSTEM_DESIGN.md              # 이 문서
├── docker-compose.yml            # DB/Redis 참조용 (실서비스는 systemd)
├── .env                          # 환경 설정 (gitignored)
├── .env.example                  # 환경 설정 예시
├── .mcp.json                     # MCP 서버 설정 (gitignored)
├── .mcp.json.example             # MCP 설정 예시
│
├── .github/workflows/
│   ├── ci.yml                    # PR 테스트 자동화
│   └── backtest.yml              # 전략 변경 시 백테스트
│
├── config/
│   ├── strategies.yaml           # 전략 파라미터 + 가중치 (메인, 핫 리로드)
│   ├── etf_universe.yaml         # US ETF 유니버스 정의
│   └── kr_etf_universe.yaml      # KR ETF 유니버스 정의
│
├── deploy/
│   ├── usstock-backend.service   # Backend systemd
│   ├── usstock-frontend.service  # Frontend systemd
│   ├── setup-db.sh               # DB 초기화
│   ├── backup-db.sh              # DB 백업
│   ├── db-backup.service/timer   # 로컬 백업 systemd
│   ├── db-backup-remote.*        # GitHub 원격 백업 systemd
│   └── install.sh                # 설치 스크립트
│
├── backend/
│   ├── main.py                   # FastAPI 앱 + lifespan (1600+ lines)
│   ├── config.py                 # Pydantic Settings
│   ├── mcp_server.py             # MCP 서버 (FastMCP, 28 tools)
│   ├── requirements.txt
│   │
│   ├── core/
│   │   ├── models.py             # SQLAlchemy ORM (10+ 테이블)
│   │   ├── enums.py              # SignalType, OrderStatus, MarketState, Market 등
│   │   └── events.py             # EventBus (내부 이벤트)
│   │
│   ├── db/
│   │   ├── session.py            # async engine, session factory
│   │   └── trade_repository.py   # 거래 CRUD
│   │
│   ├── exchange/
│   │   ├── base.py               # ExchangeAdapter Protocol
│   │   ├── kis_adapter.py        # KIS US REST API (주문/잔고/체결내역)
│   │   ├── kis_kr_adapter.py     # KIS KR REST API (한국주식)
│   │   ├── kis_websocket.py      # KIS WebSocket (실시간 시세, 41종목)
│   │   ├── kis_auth.py           # OAuth2 토큰 관리 (Redis 캐싱, 자동 갱신)
│   │   └── paper_adapter.py      # 모의투자 어댑터
│   │
│   ├── engine/
│   │   ├── evaluation_loop.py    # 메인 평가 루프 (US/KR)
│   │   ├── etf_engine.py         # 레버리지/인버스 ETF 전용 엔진
│   │   ├── order_manager.py      # 주문 생명주기 (dedup, partial fill)
│   │   ├── portfolio_manager.py  # 잔고/포지션/PnL/스냅샷
│   │   ├── risk_manager.py       # 리스크 관리 (US/KR 별도 파라미터)
│   │   ├── position_tracker.py   # ATR-based 동적 SL/TP, 보유 추적
│   │   ├── scheduler.py          # APScheduler 작업 등록 (29 tasks)
│   │   ├── adaptive_weights.py   # 적응형 가중치
│   │   ├── stock_classifier.py   # 종목 분류
│   │   └── recovery.py           # 에러 복구
│   │
│   ├── strategies/               # 16 active strategies
│   │   ├── base.py               # BaseStrategy (abstract)
│   │   ├── registry.py           # StrategyRegistry (동적 등록)
│   │   ├── combiner.py           # SignalCombiner (Mode B, 그룹 합의)
│   │   ├── config_loader.py      # YAML 핫 리로드
│   │   │  # 추세 매매 (Core)
│   │   ├── trend_following.py    # EMA Cross + ADX
│   │   ├── dual_momentum.py      # 절대+상대 모멘텀
│   │   ├── donchian_breakout.py  # 돈치안 채널 돌파
│   │   ├── supertrend_strategy.py # ATR 기반 슈퍼트렌드
│   │   │  # coin 포팅 전략
│   │   ├── cis_momentum.py       # CIS 모멘텀 (ROC + 거래량)
│   │   ├── larry_williams.py     # 래리 윌리엄스 (변동성 돌파 + %R)
│   │   ├── bnf_deviation.py      # BNF 이격도 (평균 회귀)
│   │   │  # 보조 전략
│   │   ├── macd_histogram.py     # MACD 히스토그램 다이버전스
│   │   ├── rsi_divergence.py     # RSI 다이버전스
│   │   ├── bollinger_squeeze.py  # 볼린저 밴드 스퀴즈
│   │   ├── volume_profile.py     # 거래량 프로파일
│   │   ├── volume_surge_strategy.py # 거래량 급증 확인
│   │   │  # ETF 전용 전략
│   │   ├── regime_switch.py      # 시장 레짐 전환
│   │   └── sector_rotation.py    # 섹터 로테이션
│   │
│   ├── scanner/
│   │   ├── stock_scanner.py      # 종목 스캔 인터페이스
│   │   ├── indicator_screener.py # Layer 1: 기술적 지표 스크리닝
│   │   ├── fundamental_enricher.py # Layer 2: yfinance 데이터 보강 (US)
│   │   ├── kr_fundamental_enricher.py # Layer 2: KR 펀더멘털
│   │   ├── news_enricher.py      # Layer 2.5: 뉴스 감성 분석 (±15pt)
│   │   ├── pipeline.py           # 풀 스캐닝 파이프라인
│   │   ├── kr_screener.py        # KR 종목 스크리닝
│   │   ├── sector_analyzer.py    # 섹터별 강도 분석
│   │   ├── universe_expander.py  # 동적 유니버스 확장 (yfinance+KIS ranking)
│   │   └── etf_universe.py       # ETF 유니버스 (레버리지/인버스 매핑)
│   │
│   ├── data/
│   │   ├── market_data_service.py    # yfinance-first OHLCV + 캐싱
│   │   ├── external_data_service.py  # yfinance 기본 데이터
│   │   ├── fred_service.py           # FRED 매크로 (async via to_thread)
│   │   ├── market_state.py           # SPY/VIX 시장 레짐 감지
│   │   ├── indicator_service.py      # 기술적 지표 계산
│   │   ├── news_service.py           # US 뉴스 (Finnhub API)
│   │   ├── naver_news_service.py     # KR 뉴스 (Naver Finance)
│   │   ├── earnings_service.py       # 실적 발표 일정 (Finnhub)
│   │   ├── insider_service.py        # 내부자 매매 (Finnhub)
│   │   ├── macro_calendar.py         # US 매크로 캘린더 (FOMC, CPI 등)
│   │   ├── kr_macro_calendar.py      # KR 매크로 캘린더
│   │   ├── event_calendar.py         # 통합 이벤트 캘린더 (파사드)
│   │   ├── stock_name_service.py     # 종목명 캐싱 (DB batch)
│   │   ├── kr_symbol_mapper.py       # KR 심볼 매핑 (005930→005930.KS)
│   │   └── kr_tick_size.py           # KR 호가 단위
│   │
│   ├── services/
│   │   ├── llm/
│   │   │   ├── __init__.py          # LLMClient, LLMResponse
│   │   │   ├── providers.py         # AnthropicProvider + GeminiProvider
│   │   │   └── client.py            # 멀티 프로바이더 (폴백 + 재시도)
│   │   ├── cache.py                 # Redis CacheService
│   │   ├── notification.py          # Discord (primary) / Telegram / Slack
│   │   ├── rate_limiter.py          # KIS API Rate Limit
│   │   ├── exchange_resolver.py     # Exchange 코드 자동 감지
│   │   ├── health.py                # 헬스체크 + Discord 알림
│   │   └── log_manager.py           # 로그 로테이션
│   │
│   ├── agents/
│   │   ├── market_analyst.py        # 시장 분석 (LLMClient + AgentContext)
│   │   ├── risk_assessment.py       # 매매 전 리스크 체크
│   │   ├── trade_review.py          # 일일 매매 리뷰
│   │   └── news_sentiment_agent.py  # 뉴스 감성 분석 (US+KR)
│   │
│   ├── analytics/
│   │   ├── factor_model.py          # MultiFactorModel (z-scores)
│   │   ├── position_sizing.py       # KellyPositionSizer
│   │   └── signal_quality.py        # SignalQualityTracker
│   │
│   ├── backtest/
│   │   ├── engine.py                # 백테스트 엔진
│   │   ├── full_pipeline.py         # 풀 파이프라인 백테스트
│   │   ├── data_loader.py           # yfinance 과거 데이터
│   │   ├── simulator.py             # 주문 시뮬레이션
│   │   ├── metrics.py               # 성과 지표 (Sharpe, MDD 등)
│   │   ├── optimizer.py             # 파라미터 최적화
│   │   ├── result_store.py          # 결과 DB 저장
│   │   └── verify_strategies.py     # 전략 검증 스크립트
│   │
│   ├── api/                          # 13 모듈
│   │   ├── router.py                # 메인 라우터
│   │   ├── portfolio.py             # 포트폴리오
│   │   ├── trades.py                # 거래 이력
│   │   ├── strategies.py            # 전략 관리
│   │   ├── scanner_api.py           # 스캐너 결과
│   │   ├── backtest_api.py          # 백테스트 실행/결과
│   │   ├── engine_api.py            # 엔진 제어
│   │   ├── market.py                # 시장 데이터 + 이벤트
│   │   ├── watchlist.py             # 감시 목록
│   │   ├── news.py                  # 뉴스 감성 분석
│   │   ├── ws.py                    # WebSocket
│   │   └── dependencies.py          # DI
│   │
│   └── tests/                        # 103 test files, 1276+ tests
│       ├── conftest.py
│       ├── test_exchange/ (4)
│       ├── test_strategies/ (15)
│       ├── test_engine/ (10)
│       ├── test_scanner/ (11)
│       ├── test_data/ (11)
│       ├── test_backtest/ (7)
│       ├── test_api/ (3)
│       ├── test_agents/ (4)
│       ├── test_analytics/ (3)
│       ├── test_services/ (6)
│       └── scenarios/ (8)
│
└── frontend/
    ├── package.json
    ├── vite.config.ts
    │
    └── src/
        ├── main.tsx
        ├── components/              # 20 components
        │   ├── App.tsx
        │   ├── Dashboard.tsx        # 메인 레이아웃 (탭 네비게이션)
        │   ├── PortfolioChart.tsx    # 자산 추이 차트
        │   ├── PositionList.tsx     # 보유 종목 (SL/TP 표시)
        │   ├── TradeHistory.tsx     # 거래 이력 (US/KR 필터)
        │   ├── SignalPanel.tsx      # 전략 신호 로그
        │   ├── StrategyPanel.tsx    # 전략 설정
        │   ├── StrategyPerformance.tsx # 전략별 성과
        │   ├── ScannerPanel.tsx     # 스캐너 결과
        │   ├── SectorHeatmap.tsx    # 섹터 히트맵 (US/KR)
        │   ├── WatchlistPanel.tsx   # 감시 목록
        │   ├── StockChart.tsx       # 캔들차트
        │   ├── BacktestPanel.tsx    # 백테스트 실행/결과
        │   ├── ETFPanel.tsx         # ETF 모니터
        │   ├── EngineControl.tsx    # 엔진 제어
        │   ├── LogPanel.tsx         # 시스템 로그
        │   ├── MarketToggle.tsx     # US/KR 마켓 전환
        │   ├── NewsSentiment.tsx    # 뉴스 감성 분석
        │   ├── EventsCalendar.tsx   # 이벤트 캘린더
        │   └── OptimizePanel.tsx    # 최적화
        │
        ├── contexts/
        │   └── MarketContext.tsx     # US/KR 마켓 상태 관리
        ├── hooks/
        │   ├── useApi.ts
        │   └── usePriceStream.ts
        ├── api/
        │   └── client.ts
        ├── utils/
        │   └── format.ts            # 통화 포맷 (USD/KRW)
        └── types/
            └── index.ts
```

---

## 5. KIS API 통합 설계

### 5.1 인증 흐름

```
┌──────────────┐     POST /oauth2/tokenP      ┌───────────┐
│  KIS Auth    │ ──────────────────────────►   │  KIS API  │
│  Manager     │ ◄──────────────────────────   │  Server   │
│              │     access_token (24h TTL)    │           │
│  - 자동 갱신  │                               │           │
│  - 토큰 캐싱  │     POST /oauth2/Approval    │           │
│  - 1일 1회   │ ──────────────────────────►   │           │
│    토큰 발급  │     approval_key (WebSocket)  │           │
└──────────────┘                               └───────────┘
```

**토큰 관리 전략:**
- 시작 시 토큰 발급, Redis에 캐싱 (TTL 23시간)
- 앱 재시작 시 Redis에서 기존 토큰 복원 (1일 1회 제한 대응)
- 만료 1시간 전 자동 갱신
- WebSocket approval_key 별도 관리

### 5.2 REST API 어댑터

```python
class KISAdapter(BaseExchangeAdapter):
    """KIS REST API 통합 어댑터"""

    # 시세 조회
    async def get_price(symbol, exchange) -> Price
    async def get_ohlcv(symbol, timeframe, count) -> DataFrame
    async def get_orderbook(symbol, exchange) -> Orderbook
    async def get_minute_candles(symbol, minutes) -> DataFrame

    # 주문
    async def buy(symbol, qty, price, order_type) -> Order
    async def sell(symbol, qty, price, order_type) -> Order
    async def cancel_order(order_id) -> bool
    async def modify_order(order_id, new_price, new_qty) -> Order

    # 계좌
    async def get_balance() -> Balance
    async def get_positions() -> list[Position]
    async def get_buying_power() -> float
    async def get_order_history(start, end) -> list[Order]
    async def get_pending_orders() -> list[Order]

    # 스캐너
    async def get_volume_surges() -> list[StockInfo]
    async def get_price_movers(direction) -> list[StockInfo]
    async def get_market_cap_ranking() -> list[StockInfo]
    async def get_sector_prices(sector_code) -> list[StockInfo]
    async def get_new_highs_lows() -> list[StockInfo]
    async def get_news(symbol?) -> list[News]
```

### 5.3 WebSocket 실시간 시세

```python
class KISWebSocket:
    """KIS WebSocket 실시간 데이터 스트림"""

    # 제한: 세션당 41종목, 계정당 1연결
    # US 데이터는 지연시세 (약 15분)

    async def connect(approval_key)
    async def subscribe(symbol, data_type)   # 체결가 or 호가
    async def unsubscribe(symbol, data_type)
    async def on_price(callback)             # 체결가 수신
    async def on_orderbook(callback)         # 호가 수신
    async def on_execution(callback)         # 체결 통보

    # PINGPONG 자동 응답 (연결 유지)
    # 재연결 로직 (exponential backoff)
    # 동적 구독 관리 (41종목 제한 내 로테이션)
```

**41종목 제한 대응 전략:**
1. 우선순위 기반 구독: 보유 종목 > 감시 종목 > 스캔 종목
2. 비보유 종목은 REST 폴링 (30초 간격)으로 보완
3. 멀티 계정 전략 (선택): 2계정 = 82종목 구독 가능

### 5.4 Rate Limiter

```python
class KISRateLimiter:
    """Redis 기반 Token Bucket Rate Limiter"""

    # 실계좌: 20 req/sec
    # 모의투자: 5 req/sec

    async def acquire(weight=1) -> bool
    async def wait_for_slot() -> None

    # 요청 큐잉: 초과 시 대기열에 넣고 순차 처리
    # 우선순위: 주문 > 잔고조회 > 시세조회 > 스캔
```

---

## 6. 외부 데이터 소스 통합 + AI 종목 추천

KIS API에는 해외주식 종목 추천/애널리스트 의견 API가 없음.
yfinance 컨센서스 데이터 + Claude AI 분석을 결합하여 자체 종목 추천 시스템 구축.

### 6.1 데이터 소스 매트릭스

| 데이터 | Primary Source | Secondary Source | 용도 |
|--------|---------------|-----------------|------|
| 실시간 시세 | KIS WebSocket (지연) | yfinance (무료, 지연) | 가격 감시 |
| 일봉/주봉 OHLCV | KIS REST API | yfinance | 전략 계산 |
| 분봉 데이터 | KIS 분봉조회 API | - | 단기 전략 |
| 섹터 데이터 | KIS 업종별시세 | yfinance sector | 섹터 분석 |
| 종목 스캐닝 | KIS 시세분석 APIs | - | 종목 발굴 |
| **애널리스트 의견** | **yfinance recommendations** | - | **종목 평가** |
| **목표가 컨센서스** | **yfinance analyst_price_targets** | - | **업사이드 판단** |
| **재무 데이터** | **yfinance financials** | - | **펀더멘털 분석** |
| **기관 보유** | **yfinance institutional_holders** | - | **스마트머니 추적** |
| 경제 지표 | FRED API | - | 매크로 분석 |
| 뉴스 | KIS 해외뉴스 | yfinance news | 이벤트 감지 |
| 과거 데이터 (백테스트) | yfinance | - | 백테스트 |
| ETF 구성종목 | yfinance | - | ETF 분석 |

### 6.2 yfinance 데이터 활용 상세

```python
class YFinanceDataService:
    """yfinance 기반 펀더멘털 + 컨센서스 데이터"""

    # === 애널리스트 컨센서스 ===
    async def get_recommendations(symbol) -> DataFrame:
        """
        애널리스트 투자의견 이력
        Returns: period, strongBuy, buy, hold, sell, strongSell
        예: AAPL → strongBuy: 12, buy: 20, hold: 8, sell: 1, strongSell: 0
        """

    async def get_analyst_price_targets(symbol) -> dict:
        """
        목표가 컨센서스
        Returns: current, low, high, mean, median
        예: AAPL → current: 185, mean: 210, high: 250, low: 170
        → 업사이드: (210 - 185) / 185 = +13.5%
        """

    async def get_upgrades_downgrades(symbol) -> DataFrame:
        """
        최근 투자의견 변경 이력
        Returns: firm, toGrade, fromGrade, action (upgrade/downgrade/init)
        """

    # === 펀더멘털 ===
    async def get_financials(symbol) -> dict:
        """
        재무제표 요약
        - revenue_growth: 매출 성장률
        - earnings_growth: 이익 성장률
        - profit_margin: 순이익률
        - roe: 자기자본이익률
        - debt_to_equity: 부채비율
        - free_cash_flow: 잉여현금흐름
        """

    async def get_valuation(symbol) -> dict:
        """
        밸류에이션 지표
        - pe_ratio: PER (trailing / forward)
        - peg_ratio: PEG
        - ps_ratio: PSR
        - pb_ratio: PBR
        - ev_ebitda: EV/EBITDA
        """

    # === 기관/내부자 ===
    async def get_institutional_holders(symbol) -> DataFrame:
        """기관 보유 현황 (Vanguard, BlackRock 등)"""

    async def get_insider_transactions(symbol) -> DataFrame:
        """내부자 매매 내역 (CEO 매수/매도 등)"""

    # === 섹터/산업 ===
    async def get_stock_info(symbol) -> StockInfo:
        """
        종목 기본 정보
        - sector, industry, market_cap, beta
        - 52week_high, 52week_low
        - avg_volume, short_ratio
        """

    async def get_sector_performance() -> dict[str, float]:
        """11개 GICS 섹터 ETF 수익률 (1D/1W/1M/3M)"""
```

### 6.3 3단계 종목 추천 파이프라인

```
┌──────────────────────────────────────────────────────────────────────┐
│                    종목 추천 파이프라인 (3-Layer)                      │
│                                                                      │
│  Layer 1: 기술적 지표 스크리닝 (Technical Indicator Screener)         │
│  ──────────────────────────────────────────────────────              │
│  입력: KIS API 시세 데이터만 사용 (외부 의존 없음)                     │
│  목적: 순수 차트 기반으로 추세 매매에 적합한 종목 필터링                │
│  출력: 기술적 추천 종목 리스트 + Indicator Score                      │
│                                                                      │
│                     ↓ (상위 50개)                                     │
│                                                                      │
│  Layer 2: yfinance 데이터 보강 (Fundamental Enrichment)              │
│  ──────────────────────────────────────────────────────              │
│  입력: Layer 1 통과 종목에 대해 yfinance 데이터 수집                   │
│  목적: 컨센서스, 재무, 스마트머니 데이터로 종합 프로파일 구성           │
│  출력: StockProfile (기술적 + 펀더멘털 + 컨센서스 결합)               │
│                                                                      │
│                     ↓ (50개 + 풍부한 데이터)                          │
│                                                                      │
│  Layer 3: AI 종합 분석 (Claude AI Final Analysis)                    │
│  ──────────────────────────────────────────────────────              │
│  입력: Layer 1 점수 + Layer 2 데이터 + 시장 컨텍스트 + 포트폴리오     │
│  목적: 정량 데이터로 잡지 못하는 정성적 판단                           │
│  출력: 최종 추천 (action, conviction, thesis, risks, strategy)        │
│                                                                      │
│                     ↓ (최종 15~20개 유니버스)                          │
└──────────────────────────────────────────────────────────────────────┘
```

#### Layer 1: 기술적 지표 스크리닝 (IndicatorScreener)

KIS API 시세 데이터만으로 순수 기술적 분석. yfinance 없이 독립 동작 가능.

```python
class IndicatorScreener:
    """KIS 시세 기반 기술적 지표 스크리닝 (외부 의존 없음)"""

    # === 추세 지표군 (Trend Indicators) — 40% ===

    def score_ema_alignment(ohlcv: DataFrame) -> float:
        """
        EMA 배열 정렬도 (가장 중요한 추세 지표)

        EMA(10), EMA(20), EMA(50), EMA(200) 계산 후:
        - 완전 정배열 (Price > 10 > 20 > 50 > 200) → 100
        - 단기 정배열 (Price > 10 > 20 > 50, 50 < 200) → 70
        - 부분 정배열 (Price > 20, 20 > 50) → 50
        - 혼합 (일부만 충족) → 30
        - 역배열 (Price < 10 < 20 < 50 < 200) → 0

        추가 보너스:
        - 골든크로스 발생 10일 이내 → +15
        - 데드크로스 발생 10일 이내 → -15
        """

    def score_adx_strength(ohlcv: DataFrame) -> float:
        """
        ADX(14) 추세 강도

        - ADX > 50 → 100 (극강 추세)
        - ADX 40~50 → 90
        - ADX 30~40 → 75
        - ADX 25~30 → 60
        - ADX 20~25 → 40
        - ADX < 20 → 10 (추세 없음, 횡보)

        방향 판별:
        - +DI > -DI → 상승 추세 (양수 유지)
        - +DI < -DI → 하락 추세 (점수 반감)
        """

    def score_price_vs_ma(ohlcv: DataFrame) -> float:
        """
        현재가 vs 주요 이동평균 이격도

        - Price > SMA(200) by 10%+ → 100
        - Price > SMA(200) by 5~10% → 80
        - Price > SMA(200) by 0~5% → 60
        - Price < SMA(200) by 0~5% → 30
        - Price < SMA(200) by 5%+ → 10

        SMA(50) 대비도 동일 가중:
        - 두 점수의 평균
        """

    # === 모멘텀 지표군 (Momentum Indicators) — 25% ===

    def score_rsi_momentum(ohlcv: DataFrame) -> float:
        """
        RSI(14) 모멘텀 상태

        - RSI 55~70 → 100 (건강한 상승 모멘텀)
        - RSI 50~55 → 80 (상승 전환)
        - RSI 70~80 → 60 (과열 접근, 주의)
        - RSI > 80 → 30 (과매수, 조정 임박)
        - RSI 40~50 → 40 (약세 접근)
        - RSI 30~40 → 20 (약세)
        - RSI < 30 → 10 (과매도)

        RSI 다이버전스 감지:
        - 가격 신고가 + RSI 하락 → 베어리시 다이버전스 → -20
        - 가격 신저가 + RSI 상승 → 불리시 다이버전스 → +20
        """

    def score_macd(ohlcv: DataFrame) -> float:
        """
        MACD(12,26,9) 모멘텀

        히스토그램 방향:
        - 히스토그램 양수 + 증가 중 → 100 (강한 상승)
        - 히스토그램 양수 + 감소 중 → 60 (상승 둔화)
        - 히스토그램 음수 + 증가 중 (절대값 감소) → 50 (하락 둔화)
        - 히스토그램 음수 + 감소 중 → 10 (강한 하락)

        시그널 크로스:
        - MACD선 > 시그널선 전환 5일 이내 → +20
        - MACD선 < 시그널선 전환 5일 이내 → -20
        """

    def score_rate_of_change(ohlcv: DataFrame) -> float:
        """
        ROC (Rate of Change) 다중 기간

        ROC(5):  5일 수익률
        ROC(10): 10일 수익률
        ROC(20): 20일 수익률
        ROC(60): 60일 수익률

        모든 ROC 양수 → 100 (다중 기간 상승)
        ROC(5,10,20) 양수, ROC(60) 음수 → 70 (최근 반등)
        ROC(5) 양수, 나머지 음수 → 40 (단기 반등)
        모든 ROC 음수 → 10
        """

    # === 변동성/거래량 지표군 (Volatility & Volume) — 20% ===

    def score_volume_surge(ohlcv: DataFrame) -> float:
        """
        거래량 모멘텀

        volume_ratio = 최근5일평균 / 20일평균:
        - ratio > 3.0 → 100 (거래량 폭증)
        - ratio 2.0~3.0 → 80
        - ratio 1.5~2.0 → 65
        - ratio 1.0~1.5 → 45
        - ratio < 1.0 → 20 (거래량 감소)

        가격 상승 + 거래량 증가 → 보너스 +15 (건강한 상승)
        가격 하락 + 거래량 증가 → 페널티 -15 (투매)
        """

    def score_atr_position(ohlcv: DataFrame) -> float:
        """
        ATR 기반 변동성 위치

        ATR(14)/Price (변동성 비율):
        - 1~2% → 100 (적정 변동성, 추세 매매 최적)
        - 2~3% → 80
        - 3~5% → 60
        - < 1% → 40 (변동성 부족)
        - > 5% → 30 (과도한 변동성)

        볼린저 밴드 스퀴즈:
        - BB 폭 20일 최저 근처 → +20 (돌파 임박)
        """

    def score_52w_position(ohlcv: DataFrame) -> float:
        """
        52주 범위 내 위치

        position = (현재가 - 52주저가) / (52주고가 - 52주저가):
        - > 0.95 → 100 (신고가 근접/돌파)
        - 0.85~0.95 → 85
        - 0.70~0.85 → 70
        - 0.50~0.70 → 45
        - 0.30~0.50 → 25
        - < 0.30 → 10

        신고가 돌파 → 보너스 +10
        """

    # === 지지/저항 지표군 (Support/Resistance) — 15% ===

    def score_breakout_proximity(ohlcv: DataFrame) -> float:
        """
        돌파 근접도

        20일 돈치안 채널:
        - 현재가 > 20일 고가 (돌파!) → 100
        - 20일 고가의 98% 이내 → 80 (돌파 임박)
        - 20일 고가의 95% 이내 → 60

        피봇 포인트:
        - 현재가 > R1 → 80
        - S1 < 현재가 < R1 → 50
        - 현재가 < S1 → 20
        """

    def score_support_strength(ohlcv: DataFrame) -> float:
        """
        지지선 강도 (하방 리스크 판단)

        EMA(50) 이격도:
        - 현재가가 EMA(50) 위 3% 이내 → 100 (지지선 근접 매수 기회)
        - 현재가가 EMA(50) 위 3~8% → 80
        - 현재가가 EMA(50) 위 8%+ → 60 (지지선 멀어짐)
        - 현재가가 EMA(50) 아래 → 20

        연속 양봉 수 (최근 10일):
        - 7일+ 양봉 → +15
        - 5~6일 양봉 → +10
        """

    # === 종합 Indicator Score ===
    def calc_indicator_score(
        symbol: str, ohlcv: DataFrame
    ) -> IndicatorScore:
        """
        total = (trend * 0.40) + (momentum * 0.25) +
                (volatility_volume * 0.20) + (support_resistance * 0.15)

        Returns: IndicatorScore {
            total: float,                # 0~100
            trend: float,                # EMA배열 + ADX + MA이격
            momentum: float,             # RSI + MACD + ROC
            volatility_volume: float,    # 거래량 + ATR + 52주위치
            support_resistance: float,   # 돌파근접 + 지지강도
            grade: str,                  # A+(90~), A(80~), B+(70~), B(60~), C(50~), D(<50)
            signals: list[str],          # ["골든크로스 5일전", "거래량 2.5배", ...]
            best_strategy: str           # 가장 적합한 전략 ID
        }
        """
```

#### Layer 2: yfinance 데이터 보강 (FundamentalEnricher)

Layer 1 통과 종목에 대해서만 yfinance 호출 (API 효율).

```python
class FundamentalEnricher:
    """Layer 1 통과 종목에 yfinance 데이터 결합"""

    async def enrich(
        candidates: list[IndicatorScore],
        max_candidates: int = 50
    ) -> list[EnrichedCandidate]:
        """
        Layer 1 상위 50개에 대해 yfinance 데이터 수집 + 결합:

        수집 데이터:
        1. 애널리스트 컨센서스
           - recommendations: strongBuy/buy/hold/sell/strongSell
           - analyst_price_targets: mean/high/low → 업사이드 계산
           - upgrades_downgrades: 최근 30일 변경 이력

        2. 펀더멘털
           - revenue_growth, earnings_growth
           - profit_margin, roe, debt_to_equity
           - pe_ratio, forward_pe, peg_ratio
           - free_cash_flow

        3. 스마트머니
           - institutional_holders: 기관 보유 비율
           - insider_transactions: 최근 90일 내부자 매매
           - short_ratio: 공매도 비율

        4. 기본 정보
           - sector, industry, market_cap, beta
           - earnings_date (실적 발표 일정)

        출력: EnrichedCandidate {
            indicator_score: IndicatorScore,   # Layer 1 결과
            consensus: ConsensusData,
            fundamentals: FundamentalData,
            smart_money: SmartMoneyData,
            stock_info: StockInfo,
            enrichment_score: float            # yfinance 기반 보조 점수 (0~100)
        }

        enrichment_score 계산:
          consensus_score (35%):
            - buy_ratio, target_upside, recent_upgrades
          fundamental_score (35%):
            - revenue_growth, peg_ratio, profit_margin, roe
          smart_money_score (30%):
            - institutional_pct, insider_net_buy, short_ratio
        """
```

### 6.4 AI 종목 추천 엔진 (정량 + 정성 하이브리드)

```
┌──────────────────────────────────────────────────────────────────┐
│        AI Stock Recommender (Quantitative + Qualitative)         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Phase 1: 정량 프리필터 (QuantScorer)                       │  │
│  │                                                            │  │
│  │  150개 후보 → 5개 지표 정량 스코어링                         │  │
│  │  ├─ trend_score (25%)                                     │  │
│  │  ├─ momentum_score (20%)                                  │  │
│  │  ├─ consensus_score (20%)                                 │  │
│  │  ├─ fundamental_score (20%)                               │  │
│  │  └─ risk_score (15%)                                      │  │
│  │  → Quant Score 상위 40개 통과 (grade B 이상)                │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Phase 2: AI 정성 분석 (Claude)                             │  │
│  │                                                            │  │
│  │  입력 데이터 (종목당):                                       │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ {                                                    │  │  │
│  │  │   "symbol": "NVDA",                                  │  │  │
│  │  │   "price": 875.28,                                   │  │  │
│  │  │   "quant_score": {                                   │  │  │
│  │  │     "total": 82, "grade": "A",                       │  │  │
│  │  │     "trend": 95, "momentum": 85,                     │  │  │
│  │  │     "consensus": 78, "fundamental": 72, "risk": 65   │  │  │
│  │  │   },                                                 │  │  │
│  │  │   "technical": {                                     │  │  │
│  │  │     "ema_alignment": "PERFECT_BULL",                 │  │  │
│  │  │     "adx": 38.5, "rsi": 62.3,                       │  │  │
│  │  │     "macd_histogram": 2.45,                          │  │  │
│  │  │     "volume_ratio": 1.8,                             │  │  │
│  │  │     "near_52w_high": true,                           │  │  │
│  │  │     "support_level": 820, "resistance_level": 900    │  │  │
│  │  │   },                                                 │  │  │
│  │  │   "consensus": {                                     │  │  │
│  │  │     "analyst_count": 44,                             │  │  │
│  │  │     "strong_buy": 18, "buy": 21, "hold": 4,         │  │  │
│  │  │     "sell": 1, "strong_sell": 0,                     │  │  │
│  │  │     "target_mean": 1050, "target_upside": "+20%",    │  │  │
│  │  │     "recent_upgrades": 3, "recent_downgrades": 0     │  │  │
│  │  │   },                                                 │  │  │
│  │  │   "fundamentals": {                                  │  │  │
│  │  │     "revenue_growth": "+122%",                       │  │  │
│  │  │     "earnings_growth": "+168%",                      │  │  │
│  │  │     "profit_margin": "55.8%",                        │  │  │
│  │  │     "roe": "115%",                                   │  │  │
│  │  │     "pe_ratio": 65.2, "forward_pe": 35.8,           │  │  │
│  │  │     "peg_ratio": 0.85,                               │  │  │
│  │  │     "debt_to_equity": 0.41,                          │  │  │
│  │  │     "free_cash_flow": "$27.5B"                       │  │  │
│  │  │   },                                                 │  │  │
│  │  │   "smart_money": {                                   │  │  │
│  │  │     "institutional_pct": "65.8%",                    │  │  │
│  │  │     "insider_buys_90d": 2, "insider_sells_90d": 5,   │  │  │
│  │  │     "short_ratio": "3.2%"                            │  │  │
│  │  │   },                                                 │  │  │
│  │  │   "sector": {                                        │  │  │
│  │  │     "name": "Technology",                            │  │  │
│  │  │     "sector_strength": 82,                           │  │  │
│  │  │     "sector_rank": "1/11"                            │  │  │
│  │  │   }                                                  │  │  │
│  │  │ }                                                    │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  시스템 프롬프트:                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ 당신은 미국 주식시장 전문 퀀트 애널리스트입니다.         │  │  │
│  │  │                                                      │  │  │
│  │  │ ## 역할                                               │  │  │
│  │  │ 정량 스코어링을 통과한 종목들을 정성적으로 심층 분석하여  │  │  │
│  │  │ 최종 매매 추천을 생성합니다. 당신의 분석은 실제 자동     │  │  │
│  │  │ 매매 시스템에 직접 반영되므로, 보수적이고 근거 기반의     │  │  │
│  │  │ 판단을 해야 합니다.                                    │  │  │
│  │  │                                                      │  │  │
│  │  │ ## 현재 시장 컨텍스트                                   │  │  │
│  │  │ - 매크로 레짐: {regime} (BULL/BEAR/NEUTRAL)            │  │  │
│  │  │ - VIX: {vix}                                         │  │  │
│  │  │ - S&P 500 추세: {spy_trend}                          │  │  │
│  │  │ - 금리 환경: {rate_env}                               │  │  │
│  │  │ - 강세 섹터: {strong_sectors}                         │  │  │
│  │  │ - 약세 섹터: {weak_sectors}                           │  │  │
│  │  │                                                      │  │  │
│  │  │ ## 현재 포트폴리오                                     │  │  │
│  │  │ - 보유 종목: {positions}                               │  │  │
│  │  │ - 섹터 비중: {sector_weights}                          │  │  │
│  │  │ - 현금 비율: {cash_pct}%                              │  │  │
│  │  │                                                      │  │  │
│  │  │ ## 분석 기준                                           │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 1. 정량 지표 검증 (Quant Score 교차 검증)           │  │  │
│  │  │ - 정량 점수가 높지만 함정이 있는 종목을 식별             │  │  │
│  │  │   (예: 일시적 거래량 급증, 뉴스 기반 단기 급등)          │  │  │
│  │  │ - 정량 점수가 중간이지만 질적으로 우수한 종목 발굴       │  │  │
│  │  │   (예: 실적 턴어라운드 초기, 신사업 모멘텀)              │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 2. 추세 지속 가능성 분석                            │  │  │
│  │  │ - EMA 배열 + ADX + 거래량 패턴의 조합 판단             │  │  │
│  │  │ - 단순 기술적 신호가 아닌 "왜 이 추세가 계속될 것인가"  │  │  │
│  │  │   에 대한 내러티브 (실적, 산업 트렌드, 정책 등)          │  │  │
│  │  │ - 저항선/지지선 근접 여부와 돌파 가능성                  │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 3. 컨센서스 심층 해석                               │  │  │
│  │  │ - 단순 Buy/Sell 비율이 아닌 의견 변경 추이 판단          │  │  │
│  │  │ - 목표가 분산도 (high-low 범위가 넓으면 불확실성 높음)   │  │  │
│  │  │ - 최근 upgrade가 있다면 그 이유는 무엇인지              │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 4. 포트폴리오 적합성                                │  │  │
│  │  │ - 기존 보유종목과의 섹터/산업 중복 최소화               │  │  │
│  │  │ - 상관관계 분산 (동일 테마 종목 과다 보유 방지)          │  │  │
│  │  │ - 현재 시장 레짐에서의 적합성                           │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 5. 리스크 팩터 식별                                 │  │  │
│  │  │ - 실적 발표 임박 여부 (실적 전 신규 진입 주의)           │  │  │
│  │  │ - 내부자 매도 패턴 (경영진 대량 매도 시 경고)            │  │  │
│  │  │ - 공매도 급증 추이                                     │  │  │
│  │  │ - 밸류에이션 과열 수준                                  │  │  │
│  │  │ - 매크로 리스크 노출도 (금리 민감, 경기 민감 등)         │  │  │
│  │  │                                                      │  │  │
│  │  │ ### 6. 매매 전략 매핑                                   │  │  │
│  │  │ - 이 종목에 가장 적합한 전략 식별:                       │  │  │
│  │  │   trend_following / dual_momentum / donchian_breakout  │  │  │
│  │  │   / supertrend / bollinger_squeeze                     │  │  │
│  │  │ - 진입 타이밍: 즉시 / 풀백 대기 / 돌파 대기             │  │  │
│  │  │ - 추천 포지션 크기: small(5%) / medium(10%) / large(15%)│  │  │
│  │  │ - 추천 SL/TP 레벨                                     │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  응답 스키마 (Structured JSON):                              │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ {                                                    │  │  │
│  │  │   "market_summary": "...",                           │  │  │
│  │  │   "recommendations": [                               │  │  │
│  │  │     {                                                │  │  │
│  │  │       "symbol": "NVDA",                              │  │  │
│  │  │       "action": "BUY",                               │  │  │
│  │  │       "conviction": "HIGH",    // HIGH/MEDIUM/LOW    │  │  │
│  │  │       "ai_score": 92,          // 0~100              │  │  │
│  │  │       "final_score": 87,       // (quant*0.5 + ai*0.5)│ │  │
│  │  │       "thesis": "AI 인프라 수요 지속 확대...",         │  │  │
│  │  │       "trend_narrative": "데이터센터 GPU 독점...",     │  │  │
│  │  │       "bull_case": "AI capex 사이클 3년차 진입...",    │  │  │
│  │  │       "bear_case": "경쟁 심화, 중국 규제 리스크...",    │  │  │
│  │  │       "catalyst": "다음 실적발표 (4/22), GTC...",     │  │  │
│  │  │       "risks": [                                     │  │  │
│  │  │         {"factor": "밸류에이션", "severity": "MEDIUM"},│  │  │
│  │  │         {"factor": "내부자 매도", "severity": "LOW"}   │  │  │
│  │  │       ],                                             │  │  │
│  │  │       "strategy": "trend_following",                  │  │  │
│  │  │       "entry_timing": "pullback_to_ema20",           │  │  │
│  │  │       "position_size": "medium",                     │  │  │
│  │  │       "stop_loss": "$820 (-6.3%)",                   │  │  │
│  │  │       "take_profit": "$1000 (+14.2%)",               │  │  │
│  │  │       "time_horizon": "2~4 weeks",                   │  │  │
│  │  │       "portfolio_fit": "섹터 중복 없음, 분산 양호"     │  │  │
│  │  │     },                                               │  │  │
│  │  │     ...                                              │  │  │
│  │  │   ],                                                 │  │  │
│  │  │   "avoid_list": [                                    │  │  │
│  │  │     {                                                │  │  │
│  │  │       "symbol": "XYZ",                               │  │  │
│  │  │       "reason": "실적 발표 임박 + 내부자 대량 매도"     │  │  │
│  │  │     }                                                │  │  │
│  │  │   ],                                                 │  │  │
│  │  │   "sector_view": "Technology > Healthcare > ..."      │  │  │
│  │  │ }                                                    │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Phase 3: 최종 스코어 결합                                  │  │
│  │                                                            │  │
│  │  final_score = (quant_score * 0.5) + (ai_score * 0.5)      │  │
│  │                                                            │  │
│  │  등급:                                                     │  │
│  │  - STRONG_BUY: final >= 85 AND conviction == HIGH          │  │
│  │  - BUY:        final >= 70 AND conviction >= MEDIUM        │  │
│  │  - WATCH:      final >= 55 (감시 목록 유지)                 │  │
│  │  - PASS:       final < 55 (유니버스 제외)                   │  │
│  │                                                            │  │
│  │  유니버스 진입: BUY 이상 → 활성 감시 + 전략 평가 대상        │  │
│  │  유니버스 퇴출: 3일 연속 PASS → 감시 해제                    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

```python
class AIStockRecommender:
    """정량 + AI 정성 하이브리드 종목 추천 엔진"""

    def __init__(self, quant_scorer: QuantScorer, llm_client):
        self.quant = quant_scorer
        self.llm = llm_client

    async def analyze_candidates(
        candidates: list[StockProfile],
        market_context: MarketContext,
        portfolio: PortfolioState
    ) -> list[StockRecommendation]:
        """
        전체 파이프라인:
        1. QuantScorer로 150개 → 40개 (grade B 이상)
        2. Claude에 40개 StockProfile + QuantScore + MarketContext 전달
        3. AI 분석 결과 수신 (2배치 x 20개)
        4. final_score = quant(50%) + ai(50%) 결합
        5. BUY 이상만 반환
        """

    async def generate_daily_briefing(
        market_data: MarketSnapshot,
        portfolio: PortfolioSummary,
        sector_analysis: dict,
        today_trades: list[Trade]
    ) -> DailyBriefing:
        """
        일일 시장 브리핑 (장 마감 후):
        - 시장 상황 요약 (지수, 섹터, VIX, 뉴스)
        - 당일 매매 리뷰 (진입/청산 판단의 적절성)
        - 보유 종목 점검 (유지/청산/비중조절)
        - 내일 신규 매수 추천 (최대 5개 + 상세 사유)
        - 리스크 경고 (이벤트, 실적 발표 등)
        - 전략 파라미터 조정 제안
        """

    async def evaluate_exit(
        position: Position,
        current_data: StockProfile,
        market_context: MarketContext
    ) -> ExitRecommendation:
        """
        보유 종목 청산 여부 AI 판단:
        - 추세 유지 여부 재평가
        - 펀더멘털 변화 감지
        - 매크로 환경 변화 영향
        - 목표가 도달 여부
        → HOLD / TRIM(일부 매도) / EXIT(전량 매도)
        """
```

### 6.4 External Data Service (통합)

```python
class ExternalDataService:
    """외부 데이터 소스 통합 서비스"""

    def __init__(self):
        self.yfinance = YFinanceDataService()
        self.fred = FREDDataService()
        self.ai_recommender = AIStockRecommender()

    # yfinance (주가 + 컨센서스 + 재무)
    async def get_history(symbol, period, interval) -> DataFrame
    async def get_stock_profile(symbol) -> StockProfile  # 통합 프로파일
    async def get_sector_performance() -> dict

    # FRED (매크로 경제지표) - data/fred_service.py
    # FREDService: fredapi 기반 매크로 경제지표 수집
    # 주요 시리즈: FEDFUNDS, DGS10, DGS2, UNRATE, CPIAUCSL, ICSA
    # MacroIndicators: 스냅샷 데이터클래스 (금리, 수익률 곡선, 실업률, CPI)
    # 수익률 곡선 역전 감지, 금리 환경 분류 (low/moderate/high)
    def fetch_macro_indicators() -> MacroIndicators
    def fetch_series(series_id, ...) -> pd.Series
    def get_yield_curve_history(months) -> pd.DataFrame

    # 매크로 레짐 판단
    async def get_macro_regime() -> MacroRegime
    # BULL: VIX < 20, 금리 안정, 고용 양호
    # BEAR: VIX > 30, 금리 상승, 경기 둔화
    # NEUTRAL: 그 외

    # AI 추천
    async def get_ai_recommendations(candidates) -> list[StockRecommendation]
    async def get_daily_briefing() -> DailyBriefing
```

### 6.5 StockProfile (종목 통합 프로파일)

```python
@dataclass
class StockProfile:
    """yfinance + KIS + 기술지표를 합친 종목 종합 데이터"""

    # 기본 정보
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    exchange: str                    # NASD, NYSE, AMEX

    # 가격 데이터
    current_price: float
    change_pct_1d: float
    change_pct_1w: float
    change_pct_1m: float
    change_pct_3m: float
    high_52w: float
    low_52w: float
    avg_volume: int

    # 기술적 지표
    ema_20: float
    ema_50: float
    ema_200: float
    rsi_14: float
    adx_14: float
    macd_histogram: float
    atr_14: float
    volume_ratio: float              # 오늘 거래량 / 20일 평균

    # 애널리스트 컨센서스 (yfinance)
    analyst_count: int
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    target_price_mean: float
    target_price_high: float
    target_price_low: float
    target_upside_pct: float         # (목표가 - 현재가) / 현재가

    # 펀더멘털 (yfinance)
    pe_ratio: float
    forward_pe: float
    peg_ratio: float
    revenue_growth: float
    earnings_growth: float
    profit_margin: float
    roe: float
    debt_to_equity: float
    free_cash_flow: float

    # 기관/내부자
    institutional_pct: float         # 기관 보유 비율
    insider_buy_recent: bool         # 최근 내부자 매수 여부
```

---

## 7. 매매 전략 설계

### 7.1 전략 체계

```
┌─────────────────────────────────────────────────┐
│              Strategy Layer                       │
│                                                   │
│  ┌─── 추세 매매 (Core) ──────────────────────┐   │
│  │  1. Trend Following (EMA Cross + ADX)     │   │
│  │  2. Dual Momentum (절대 + 상대 모멘텀)      │   │
│  │  3. Donchian Breakout (터틀 트레이딩)       │   │
│  │  4. Supertrend (ATR 기반 추세)             │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌─── coin 프로젝트 포팅 전략 ──────────────┐   │
│  │  5. CIS Momentum (ROC + 거래량 확인)      │   │
│  │  6. Larry Williams (변동성 돌파 + %R)      │   │
│  │  7. BNF Deviation (SMA 이격도 평균 회귀)    │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌─── 보조 전략 ─────────────────────────────┐   │
│  │  8. MACD Histogram Divergence             │   │
│  │  9. RSI Divergence                        │   │
│  │ 10. Bollinger Squeeze (변동성 확장)         │   │
│  │ 11. Volume Profile (거래량 분석)            │   │
│  │ 14. Volume Surge (거래량 급증 확인)          │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌─── ETF 전용 전략 ─────────────────────────┐   │
│  │ 12. Regime Switch (Bull/Bear ETF 전환)     │   │
│  │ 13. Sector Rotation (섹터 강도 기반 회전)    │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  ┌─── Signal Combiner ──────────────────────┐   │
│  │  가중 투표 + 시장 상태별 적응형 가중치        │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 7.2 핵심 전략 상세

#### Strategy 1: Trend Following (EMA Crossover + ADX Filter)
```
매수 조건:
  - EMA(20) > EMA(50) (골든크로스)
  - ADX(14) > 25 (추세 강도 확인)
  - 종가 > EMA(200) (장기 상승 추세)
  - 거래량 > 20일 평균 거래량

매도 조건:
  - EMA(20) < EMA(50) (데드크로스)
  - OR ADX(14) < 20 (추세 약화)
  - OR Trailing Stop 도달

타임프레임: 일봉 (1D)
적합 시장: 강한 추세장
가중치: 0.30 (상승추세) / 0.15 (횡보)
```

#### Strategy 2: Dual Momentum
```
월간 리밸런싱:
  1. 절대 모멘텀: 12개월 수익률 > 0 (상승 추세)
  2. 상대 모멘텀: 유니버스 내 상위 N개 종목 선정
  3. 두 조건 모두 충족 시 매수, 아니면 현금/채권 ETF

매수: 절대+상대 모멘텀 상위 종목
매도: 절대 모멘텀 음전환 OR 상대 순위 하락

타임프레임: 월봉 기준, 일봉 모니터링
적합 시장: 중장기 추세
가중치: 0.25
```

#### Strategy 3: Donchian Breakout (Turtle)
```
매수 조건:
  - 종가 > 20일 최고가 돌파
  - ATR(20) 기반 포지션 사이징
  - 이전 매매 수익 시 스킵 (피라미딩 방지)

매도 조건:
  - 종가 < 10일 최저가 이탈
  - OR 2 * ATR(20) 손절

타임프레임: 일봉
가중치: 0.20
```

#### Strategy 4: Supertrend
```
매수: 가격이 Supertrend 밴드 위로 전환
매도: 가격이 Supertrend 밴드 아래로 전환

파라미터: ATR Period=10, Multiplier=3.0
타임프레임: 일봉 + 4시간봉 확인
가중치: 0.15
```

#### Strategy 5: CIS Momentum (ROC + Volume, coin 포팅)
```
매수 조건:
  - ROC(5) > 3% AND ROC(10) > 5%
  - 거래량 비율 > 1.2 (20일 평균 대비)
  - 거래량 2.0x 이상 시 confidence 보너스

매도 조건:
  - ROC(5) < -3% AND ROC(10) < -5% (모멘텀 반전)

타임프레임: 일봉 (1D)
적합 시장: 추세장 (trending)
원본: coin/strategies/cis_momentum.py (4h → 1D 적응, ROC 임계값 상향)
```

#### Strategy 6: Larry Williams (변동성 돌파 + Williams %R, coin 포팅)
```
매수 조건:
  - close > open + k × (전일 고가 - 전일 저가), k=0.5
  - Williams %R 과매도(-80) 탈출
  - close > SMA(20)

매도 조건:
  - close < open - k × (전일 변동폭)
  - Williams %R 과매수(-20) 진입

타임프레임: 일봉 (1D)
적합 시장: 추세장 (trending)
원본: coin/strategies/larry_williams.py (4h → 1D 적응)
```

#### Strategy 7: BNF Deviation (SMA 이격도 평균 회귀, coin 포팅)
```
매수 조건:
  - SMA(25) 대비 이격도 <= -5% (과매도)
  - RSI < 35 시 confidence 보너스
  - 이격도 -10% 이하: confidence 0.85

매도 조건:
  - SMA(25) 대비 이격도 >= +3% (과매수)
  - 이격도 +8% 이상: confidence 0.80

타임프레임: 일봉 (1D)
적합 시장: 전체 (all) - 횡보/하락장에서도 유효
원본: coin/strategies/bnf_deviation.py (4h → 1D, 임계값 -10%→-5%, +5%→+3%)
```

#### Strategy 12: Regime Switch (ETF 전용)
```
시장 레짐 판단:
  - BULL: SPY > SMA(200), VIX < 20, 금리 안정
  - BEAR: SPY < SMA(200), VIX > 25, 금리 상승
  - NEUTRAL: 그 외

레짐별 ETF 선택:
  BULL  → TQQQ, SOXL, UPRO (3x 레버리지)
  BEAR  → SQQQ, SOXS, SPXU (3x 인버스)
  NEUTRAL → QQQ, SPY, SOXX (1x 일반)

전환 조건: 레짐 변경 + 2일 확인 (whipsaw 방지)
가중치: 0.30 (ETF 엔진 전용)
```

#### Strategy 13: Sector Rotation
```
매월 섹터 강도 평가:
  1. 11개 GICS 섹터 ETF 상대강도 계산
  2. 상위 3개 섹터 ETF 매수
  3. 하위 섹터에서 상위로 이동한 섹터 신규 진입

섹터 ETF 매핑:
  XLK (기술), XLF (금융), XLE (에너지), XLV (헬스케어),
  XLY (경기소비재), XLP (필수소비재), XLI (산업),
  XLB (소재), XLU (유틸리티), XLRE (부동산), XLC (통신)

가중치: 0.25 (ETF 엔진 전용)
```

### 7.3 Signal Combiner (적응형 가중 투표)

```
시장 상태별 가중치 프로파일 (14전략):

STRONG_UPTREND:
  TrendFollowing: 0.25, DualMomentum: 0.15, Donchian: 0.15,
  Supertrend: 0.10, CISMomentum: 0.15, LarryWilliams: 0.15,
  BNFDeviation: 0.05

UPTREND:
  TrendFollowing: 0.20, DualMomentum: 0.15, Donchian: 0.15,
  Supertrend: 0.10, MACD: 0.05, CISMomentum: 0.15,
  LarryWilliams: 0.10, BNFDeviation: 0.10

SIDEWAYS:
  TrendFollowing: 0.05, DualMomentum: 0.05, Donchian: 0.05,
  Supertrend: 0.05, MACD: 0.10, RSI_Div: 0.20,
  Bollinger: 0.15, Volume: 0.10, CISMomentum: 0.05,
  LarryWilliams: 0.05, BNFDeviation: 0.15

WEAK_DOWNTREND:
  DualMomentum: 0.15, RSI_Div: 0.25, Bollinger: 0.15,
  BNFDeviation: 0.25, VolumeSurge: 0.10, RegimeSwitch: 0.10
  (초기 하락 감지, 노출 축소, 방어적 전략 위주)

DOWNTREND:
  RSI_Div: 0.30, BNFDeviation: 0.40,
  VolumeSurge: 0.10, RegimeSwitch: 0.20
  (개별 종목 매수 억제, ETF 인버스 전략으로 전환)

최소 신뢰도 임계값: 0.50
HOLD 신호: 기권 처리 (가중치 재분배)
```

---

## 8. 엔진 설계

### 8.1 US Stock Engine (메인 엔진)

```
평가 루프 (매 5분 또는 시장 시간 기반):

1. 시장 상태 판단 (Market State Detector)
   - SPY/QQQ 추세 방향
   - VIX 수준
   - 거래량 트렌드
   - ADX 강도
   → STRONG_UPTREND / UPTREND / SIDEWAYS / WEAK_DOWNTREND / DOWNTREND

2. 유니버스 갱신 (Universe Manager)
   - KIS 스캐너 결과 반영
   - 보유 종목 우선
   - 감시 목록 종목

3. 종목별 전략 평가 (Strategy Evaluation)
   - 등록된 전략 병렬 실행
   - OHLCV + 지표 데이터 전달
   - Signal(type, confidence) 반환

4. 신호 결합 (Signal Combiner)
   - 시장 상태별 가중치 적용
   - 가중 투표 → 최종 BUY/SELL/HOLD

5. 리스크 검증 (Risk Manager)
   - 포지션 집중도 체크 (단일종목 < 20%)
   - 총 투자비율 체크 (< 80%)
   - 일일 손실 한도 (< 3%)
   - 최대 낙폭 (< 15%)

6. 주문 실행 (Order Manager)
   - 포지션 사이징 (ATR 기반 or 고정 비율)
   - KIS API 주문 전송
   - 체결 대기 및 확인
   - SL/TP 설정

7. 포지션 관리 (Position Tracker)
   - Trailing Stop 갱신
   - SL/TP 도달 모니터링
   - 시간 기반 청산 (보유기간 초과)

8. 상태 브로드캐스트 (WebSocket)
   - 포트폴리오 업데이트
   - 매매 실행 알림
   - 전략 신호 로그
```

### 8.2 ETF Engine (레버리지/인버스 전용)

```
US Stock Engine과 독립적으로 운영.
전용 전략: RegimeSwitch, SectorRotation

ETF 유니버스:
┌────────────────────────────────────────────────────┐
│  Bull (3x Long)    │  Bear (3x Short)  │ Neutral  │
│  TQQQ (NASDAQ)     │  SQQQ             │ QQQ      │
│  UPRO (S&P 500)    │  SPXU             │ SPY      │
│  SOXL (반도체)      │  SOXS             │ SOXX     │
│  TECL (기술)        │  TECS             │ XLK      │
│  FAS (금융)         │  FAZ              │ XLF      │
│  LABU (바이오)      │  LABD             │ XBI      │
│  TNA (러셀2000)     │  TZA              │ IWM      │
│  UDOW (다우)        │  SDOW             │ DIA      │
│  FNGU (FANG+)      │  FNGD             │ -        │
└────────────────────────────────────────────────────┘

특수 로직:
- 레버리지 ETF는 일중 변동성 감쇠(decay) 고려
- 장기 보유 지양 (최대 5-10일)
- 레짐 전환 시 2일 확인 후 스위칭
- 인버스 전환 시 기존 롱 포지션 선 청산
- VIX 급등 시 포지션 크기 자동 축소
```

### 8.3 스케줄러 (29 tasks)

```
시장 시간 (KST 기준, DST 자동 감지):
  US: PRE_MARKET → REGULAR → AFTER_HOURS → CLOSED
  KR: PRE_MARKET(08:30-09:00) → REGULAR(09:00-15:30) → CLOSED

US 태스크 (20개):
  health_check           │ 2분    │ always      │ 시스템 + Discord 알림
  position_check         │ 1분    │ regular     │ SL/TP/ATR 동적 관리
  daily_reset            │ daily  │ pre-market  │ 일일 리셋
  evaluation_loop        │ 5분    │ regular     │ 14전략 평가 + 주문
  daily_scan (T1)        │ daily  │ pre-market  │ 3-Layer 풀스캔
  intraday_hot_scan (T2) │ 30분   │ regular     │ 장중 핫종목
  sector_analysis (T3)   │ 1시간  │ regular     │ 섹터 강도
  after_hours_scan (T4a) │ daily  │ after-hours │ 장후 스캔
  daily_briefing (T4b)   │ daily  │ after-hours │ AI 일일 브리핑
  macro_update           │ daily  │ pre-market  │ FRED 매크로
  market_state_update    │ 15분   │ regular     │ SPY/VIX 레짐
  etf_evaluation         │ 15분   │ regular     │ ETF 엔진
  portfolio_snapshot     │ 1시간  │ regular     │ 자산 스냅샷
  order_reconciliation   │ 2분    │ regular     │ 주문 대사 + 체결 확인
  ws_lifecycle           │ 5분    │ always      │ WebSocket 세션 관리
  trade_review           │ daily  │ after-hours │ AI 매매 리뷰
  agent_memory_cleanup   │ daily  │ closed      │ 에이전트 메모리 정리
  update_watchlist_names │ daily  │ pre-market  │ 종목명 DB 캐싱
  news_analysis          │ 30분   │ pre+regular │ Finnhub 뉴스 감성
  event_calendar_refresh │ daily  │ pre-market  │ 실적/매크로/내부자

KR 태스크 (7개):
  kr_position_check      │ 1분    │ regular     │ KR SL/TP 관리
  kr_order_reconciliation│ 2분    │ regular     │ KR 주문 대사
  kr_portfolio_snapshot  │ 1시간  │ regular     │ KR 자산 스냅샷
  kr_evaluation_loop     │ 5분    │ regular     │ KR 14전략 평가
  kr_daily_scan          │ daily  │ pre-market  │ KR 종목 스캔
  kr_etf_evaluation      │ 15분   │ regular     │ KR ETF 엔진
  kr_news_analysis       │ 30분   │ pre+regular │ Naver 뉴스 감성
```

---

## 9. 리스크 관리

### 9.1 포지션 레벨 (현재 라이브 설정)

| 항목 | US | KR | 설명 |
|------|-----|-----|------|
| 종목당 최대 비중 | 8% | 8% | 분산 투자 (20 포지션) |
| 동적 손절 | ATR 3-15% | ATR 5-20% | ATR 기반 동적 SL |
| 이익실현 | 50% | 25% | 와이드 TP (winners run) |
| Trailing Stop | 비활성 | 비활성 | 수익 조기 차단 방지 |
| Kelly fraction | 0.40 | 0.40 | 공격적 사이징 |

### 9.2 포트폴리오 레벨 (현재 라이브 설정)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| 최대 동시 보유 | 20종목 | 분산 투자 (백테스트 최적) |
| 마켓 배분 | 50:50 US/KR | 레짐 기반 동적 ±20% (clamp 20-70%) |
| 일일 손실 한도 | -3% | 초과 시 신규 매수 중단 |
| 최대 낙폭 (MDD) | -15% | 초과 시 전량 청산 |
| 레버리지 ETF 비중 | 30% | 전체 포트폴리오 대비 |
| 섹터 집중 한도 | 40% | 단일 섹터 |
| 레짐 적응 | 강상승 15%/95% | 하락장 5%/40% |

### 9.3 Exit Management (청산 관리)

| 항목 | 값 | 설명 |
|------|-----|------|
| Hard Stop Loss | -15% | 무조건 손절 (min hold bypass) |
| Profit Protection | ≥25% | HOLD + 고수익 → 자동 매도 |
| High Profit Auto-Sell | ≥10% PnL | HOLD시 기술적 약세 없이도 매도 |
| Trailing Stop 활성화 | +8% | 활성화 후 -4% 추적 |
| Profit Protection % | 25% | 고수익 포지션 보호 비율 |

- **Winners 보호**: 높은 threshold (25%)로 설정하여 수익 포지션이 조기 청산되지 않도록 함
- **Losers 빠른 제거**: Hard SL -15%로 큰 손실 방지, ATR 기반 동적 SL로 변동성 적응

### 9.4 시스템 레벨

```
- API 에러 연속 5회 → 엔진 일시 중지, 알림 발송
- 주문 실패 3회 → 해당 종목 쿨다운 (30분)
- 잔고 불일치 감지 → 거래소 동기화 후 재확인
- 네트워크 단절 → 재연결 시도 (exponential backoff)
- VIX > 35 → 포지션 크기 50% 축소, 레버리지 ETF 매매 중단
```

---

## 10. 종목 스캐닝 시스템 + 스크리닝 주기

### 10.1 스크리닝 타임라인 (한국 시간 기준)

```
미국 시장 시간 (한국 시간):
  프리마켓:   18:00 ~ 23:30 (서머타임: 17:00 ~ 22:30)
  정규장:     23:30 ~ 06:00 (서머타임: 22:30 ~ 05:00)
  애프터마켓: 06:00 ~ 10:00 (서머타임: 05:00 ~ 09:00)
  장 마감:    10:00 ~ 18:00 (서머타임: 09:00 ~ 17:00)

┌─────────────────────────────────────────────────────────────────────┐
│                    스크리닝 주기 (4 Tier)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: 일일 풀스캔 (Daily Full Scan)                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 실행: 매일 1회 — 장 시작 2시간 전 (21:30 KST / 20:30 DST)   │   │
│  │ 소요: ~10분 (yfinance bulk + KIS API)                       │   │
│  │                                                             │   │
│  │ 수행 내용:                                                   │   │
│  │ 1. KIS 시세분석 API 전체 스캔                                 │   │
│  │    ├─ 거래량 급증 종목 (상위 50)                               │   │
│  │    ├─ 가격 급등/급락 (상위 50)                                │   │
│  │    ├─ 신고가/신저가 (각 30)                                   │   │
│  │    ├─ 거래대금 상위 (상위 50)                                 │   │
│  │    └─ 시총 상위 (상위 100)                                    │   │
│  │    → 중복 제거 후 후보 ~150개                                 │   │
│  │                                                             │   │
│  │ 2. yfinance 펀더멘털 + 컨센서스 수집                          │   │
│  │    ├─ 후보 150개 → batch download (5개씩 병렬)               │   │
│  │    ├─ recommendations (애널리스트 의견)                       │   │
│  │    ├─ analyst_price_targets (목표가)                         │   │
│  │    ├─ financials (매출, 이익, 성장률)                         │   │
│  │    ├─ info (섹터, 시총, PER, 베타)                           │   │
│  │    └─ insider_transactions (내부자 매매)                      │   │
│  │    → StockProfile 150개 생성                                 │   │
│  │                                                             │   │
│  │ 3. 기술적 필터 (자동)                                        │   │
│  │    ├─ 시총 > $2B                                            │   │
│  │    ├─ 일평균 거래량 > 50만주                                  │   │
│  │    ├─ 가격 > SMA(200) (장기 상승추세)                         │   │
│  │    ├─ ADX > 20 (추세 존재)                                   │   │
│  │    └─ 스프레드 < 0.5%                                        │   │
│  │    → 후보 ~40개로 축소                                       │   │
│  │                                                             │   │
│  │ 4. AI 종목 분석 (Claude)                                     │   │
│  │    ├─ 40개 StockProfile → Claude API 전송 (2배치 x 20개)     │   │
│  │    ├─ 종합 점수 (0~100) + 등급 + 사유                        │   │
│  │    └─ 추천 전략 매핑                                         │   │
│  │    → 최종 상위 15~20개 유니버스 확정                           │   │
│  │                                                             │   │
│  │ 5. Universe Manager 갱신                                     │   │
│  │    ├─ 감시 목록 업데이트 (신규 진입/퇴출)                      │   │
│  │    ├─ WebSocket 구독 종목 갱신 (보유 + 감시 = 최대 41개)       │   │
│  │    └─ DB 저장 (scanner_results, watchlist)                   │   │
│  │                                                             │   │
│  │ Rate Limit: KIS ~300 req (15초), yfinance ~150 req (3분)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  TIER 2: 장중 핫스캔 (Intraday Hot Scan)                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 실행: 정규장 중 30분 간격                                     │   │
│  │ 소요: ~2분                                                   │   │
│  │                                                             │   │
│  │ 수행 내용:                                                   │   │
│  │ 1. KIS 실시간 스캐너                                         │   │
│  │    ├─ 거래량 급증 (전일 대비 3배 이상)                         │   │
│  │    ├─ 가격 급등 (장중 +3% 이상)                               │   │
│  │    └─ 신고가 돌파 종목                                       │   │
│  │                                                             │   │
│  │ 2. 기존 유니버스와 교차 확인                                   │   │
│  │    ├─ 이미 감시 중인 종목 → 점수 가산                          │   │
│  │    └─ 신규 발견 → 기본 필터만 적용 후 임시 감시 추가            │   │
│  │                                                             │   │
│  │ 3. 긴급 진입 판단                                            │   │
│  │    └─ 기존 유니버스 종목이 돌파 신호 → 전략 평가 우선 실행       │   │
│  │                                                             │   │
│  │ Rate Limit: KIS ~30 req (2초)                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  TIER 3: 섹터 분석 (Sector Analysis)                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 실행: 1시간 간격 (장중) + 장 마감 후 1회                       │   │
│  │ 소요: ~1분                                                   │   │
│  │                                                             │   │
│  │ 수행 내용:                                                   │   │
│  │ 1. 11개 GICS 섹터 ETF 시세 조회                               │   │
│  │    └─ XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC│   │
│  │                                                             │   │
│  │ 2. 섹터별 상대강도 계산                                       │   │
│  │    ├─ SPY 대비 수익률 (1D/1W/1M/3M)                          │   │
│  │    ├─ 거래량 변화율 (자금 유입/유출 프록시)                     │   │
│  │    └─ 기술적 추세 (EMA 배열, RSI)                             │   │
│  │                                                             │   │
│  │ 3. 섹터 로테이션 신호                                         │   │
│  │    ├─ 강세 섹터 상위 3개 (매수 대상)                           │   │
│  │    ├─ 약세 섹터 하위 3개 (회피 대상)                           │   │
│  │    └─ 전환 감지 (순위 2단계 이상 변동)                         │   │
│  │                                                             │   │
│  │ 4. ETF 엔진 연동                                             │   │
│  │    └─ 강세 섹터 레버리지 ETF 우선순위 조정                      │   │
│  │                                                             │   │
│  │ Rate Limit: KIS ~15 req (1초), yfinance ~11 req              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  TIER 4: AI 일일 브리핑 (Daily AI Briefing)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 실행: 매일 1회 — 장 마감 후 (07:00 KST / 06:00 DST)          │   │
│  │ 소요: ~3분                                                   │   │
│  │                                                             │   │
│  │ 수행 내용:                                                   │   │
│  │ 1. 당일 매매 결과 집계                                        │   │
│  │                                                             │   │
│  │ 2. Claude AI 종합 분석                                       │   │
│  │    ├─ 시장 상황 요약 (지수, 섹터, VIX, 뉴스)                   │   │
│  │    ├─ 당일 매매 리뷰 (진입/청산 판단 평가)                     │   │
│  │    ├─ 보유 종목 점검 (유지/청산/비중조절 의견)                  │   │
│  │    ├─ 내일 신규 매수 추천 (최대 5개 + 사유)                    │   │
│  │    ├─ 섹터 전환 제안                                         │   │
│  │    └─ 리스크 경고 (이벤트, 실적 발표 등)                       │   │
│  │                                                             │   │
│  │ 3. Discord 알림 발송 (요약)                                   │   │
│  │ 4. DB 저장 (agent_logs)                                      │   │
│  │                                                             │   │
│  │ Rate Limit: Claude API 1 req (8K~15K tokens)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 스크리닝 주기 요약표

```
┌──────────────────────────────────────────────────────────────────┐
│  Tier  │ 이름           │ 주기     │ 시점        │ 데이터소스    │
├────────┼────────────────┼──────────┼─────────────┼──────────────┤
│  T1    │ 일일 풀스캔     │ 1일 1회  │ 장전 2시간   │ KIS+yf+AI   │
│  T2    │ 장중 핫스캔     │ 30분     │ 정규장 중    │ KIS only    │
│  T3    │ 섹터 분석      │ 1시간    │ 정규장 중    │ KIS+yf      │
│  T4    │ AI 브리핑      │ 1일 1회  │ 장 마감 후   │ All+Claude  │
├────────┼────────────────┼──────────┼─────────────┼──────────────┤
│  -     │ 전략 평가      │ 5분      │ 정규장 중    │ KIS+캐시    │
│  -     │ SL/TP 체크     │ 1분      │ 정규장 중    │ KIS WS/REST │
│  -     │ 시장상태 판단   │ 15분     │ 정규장 중    │ SPY/VIX     │
│  -     │ 매크로 레짐    │ 6시간    │ 상시        │ FRED        │
│  -     │ 헬스체크       │ 2분      │ 상시        │ 내부        │
└──────────────────────────────────────────────────────────────────┘
```

### 10.3 Rate Limit 예산 배분

```
KIS API: 20 req/sec (실계좌), 5 req/sec (모의투자)

일일 API 호출 예산 (정규장 6.5시간 기준):

  주문/잔고 (최우선):       예약 5 req/sec
  ├─ 주문 실행:             ~2 req/sec (피크)
  ├─ 잔고 조회:             ~1 req/sec
  └─ 체결 확인:             ~2 req/sec

  시세 조회 (높음):          예약 8 req/sec
  ├─ 보유종목 시세:          ~3 req/sec (10종목 x 1분 간격)
  ├─ 감시종목 시세:          ~3 req/sec (15종목 x 1분 간격)
  └─ 지표 계산용 OHLCV:     ~2 req/sec

  스캐너 (보통):            예약 5 req/sec
  ├─ T2 핫스캔 (30분):      ~30 req/burst → 1 req/sec 평균
  ├─ T3 섹터 (1시간):       ~15 req/burst → 0.3 req/sec 평균
  └─ 여유분:                ~3.7 req/sec

  T1 풀스캔 (장전):         전체 20 req/sec 사용 가능
  ├─ 장 시작 전이므로 다른 작업 없음
  └─ ~300 req → 15초 완료

yfinance: 비공식 제한 ~2,000 req/hour
  T1 풀스캔:               ~150 req (후보 종목 프로파일)
  T3 섹터:                 ~11 req/hour (섹터 ETF)
  기타:                    ~50 req/hour (보유종목 갱신)
  → 시간당 ~200 req (충분한 여유)

LLM API (Anthropic + Gemini 폴백):
  T1 AI 분석:              2 req/day (20종목 x 2배치)
  T4 AI 브리핑:            1 req/day
  보유종목 점검:            ~5 req/day
  폴백 체인:               Anthropic 실패 시 Gemini 자동 전환
  → 일일 ~8 req (비용: ~$0.5/day with Haiku)
```

### 10.4 스캐너 파이프라인 (확장)

```
┌─────────────────────────────────────────────────────────────┐
│                Stock Scanner Pipeline (T1 Full Scan)         │
│                                                             │
│  Stage 1: KIS API 스크리닝 (자동)                            │
│  ├─ 거래량 급증 종목 (상위 50)                                │
│  ├─ 가격 급등/급락 종목 (상위 50)                             │
│  ├─ 신고가/신저가 종목 (각 30)                                │
│  ├─ 거래대금 상위 (상위 50)                                   │
│  └─ 시가총액 상위 (상위 100)                                  │
│           ↓ 중복 제거 (~150개)                               │
│                                                             │
│  Stage 2: yfinance 데이터 수집 (병렬 5개씩)                   │
│  ├─ 기본 정보 (섹터, 시총, 베타)                              │
│  ├─ 애널리스트 의견 (strongBuy ~ strongSell)                  │
│  ├─ 목표가 (mean, high, low)                                │
│  ├─ 재무 (매출성장률, PER, PEG, ROE)                         │
│  └─ 내부자/기관 매매                                         │
│           ↓ StockProfile 생성 (~150개)                      │
│                                                             │
│  Stage 3: 기술적 필터 (자동)                                 │
│  ├─ 시총 > $2B                                              │
│  ├─ 일평균 거래량 > 50만주                                    │
│  ├─ 가격 > SMA(200) (장기 상승추세)                           │
│  ├─ ADX > 20 (추세 존재)                                     │
│  └─ 업사이드 > 5% (목표가 기준)                               │
│           ↓ (~40개)                                         │
│                                                             │
│  Stage 4: AI 종합 분석 (Claude)                              │
│  ├─ 40개 StockProfile → JSON 정리                           │
│  ├─ Claude API 호출 (2배치 x 20개)                           │
│  ├─ 평가 기준:                                               │
│  │   ├─ 기술적 추세 강도 (30%)                                │
│  │   ├─ 애널리스트 컨센서스 (20%)                              │
│  │   ├─ 펀더멘털 건전성 (20%)                                 │
│  │   ├─ 모멘텀/거래량 (15%)                                   │
│  │   └─ 리스크 요인 (15%)                                    │
│  └─ 출력: score(0~100), grade, reasons, risks               │
│           ↓ 상위 15~20개                                    │
│                                                             │
│  Stage 5: Universe Manager 갱신                              │
│  ├─ 기존 유니버스와 diff 계산                                 │
│  │   ├─ 신규 진입: 이전에 없던 종목 → 알림                     │
│  │   ├─ 유지: score 변동 추적                                │
│  │   └─ 퇴출: 3일 연속 하위 → 감시 해제                       │
│  ├─ WebSocket 구독 갱신                                      │
│  │   └─ 보유종목(필수) + 상위 감시종목 = 최대 41개              │
│  ├─ DB 저장 (watchlist, scanner_results)                     │
│  └─ 대시보드 브로드캐스트 (신규 종목 알림)                      │
└─────────────────────────────────────────────────────────────┘
```

### 10.5 Sector Analyzer

```python
class SectorAnalyzer:
    """11개 GICS 섹터 강도 분석"""

    SECTORS = {
        "XLK": "Technology",     "XLF": "Financials",
        "XLE": "Energy",         "XLV": "Healthcare",
        "XLY": "Consumer Disc.", "XLP": "Consumer Staples",
        "XLI": "Industrials",    "XLB": "Materials",
        "XLU": "Utilities",      "XLRE": "Real Estate",
        "XLC": "Communications"
    }

    async def get_sector_strength() -> dict[str, SectorScore]:
        """
        각 섹터 ETF 대비 SPY 상대강도 계산:
        - 1주/1개월/3개월 수익률
        - 자금 유입/유출 (거래량 변화)
        - 기술적 추세 (SMA 배열)
        → 종합 점수 0~100
        """

    async def get_sector_rotation_signal() -> list[SectorSignal]:
        """상위 3개 섹터 매수, 하위 3개 회피"""

    async def get_sector_etf_mapping() -> dict[str, list[str]]:
        """
        강세 섹터 → 해당 섹터 레버리지 ETF 매핑
        예: Technology 강세 → TECL (3x), XLK (1x) 추천
        """
```

### 10.6 장 마감 중 스크리닝 (Off-Hours)

```
장이 닫혀있는 시간 (10:00~18:00 KST)에도 유용한 작업:

1. yfinance 데이터 프리로딩 (12:00 KST)
   - 전일 종가 기준 기술지표 재계산
   - 애널리스트 의견 변경 감지
   - 내부자 매매 신규 공시 확인
   - 실적 발표 일정 확인 (다음 1주)

2. FRED 매크로 데이터 갱신 (14:00 KST)
   - 금리, VIX, 실업률 등

3. 백테스트 자동 실행 (15:00 KST, 선택)
   - 전략 파라미터 자동 최적화 (walk-forward)
   - 신규 유니버스 종목 백테스트 검증

4. AI 프리마켓 분석 (17:00 KST)
   - 프리마켓 시세 확인
   - 주요 뉴스/이벤트 기반 당일 전략 조정 제안
```

---

## 11. 백테스트 시스템

### 11.1 아키텍처

```
┌────────────────────────────────────────────────────┐
│                 Backtest Engine                      │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   │
│  │  Data    │──►│ Strategy │──►│  Simulator   │   │
│  │  Loader  │   │ Runner   │   │  (Matching)  │   │
│  │(yfinance)│   │          │   │              │   │
│  └──────────┘   └──────────┘   └──────┬───────┘   │
│                                        │            │
│                                 ┌──────▼───────┐   │
│                                 │   Metrics    │   │
│                                 │   Calculator │   │
│                                 └──────┬───────┘   │
│                                        │            │
│                                 ┌──────▼───────┐   │
│                                 │   Report     │   │
│                                 │   Generator  │   │
│                                 └──────────────┘   │
└────────────────────────────────────────────────────┘
```

### 11.2 기능

```
실행 모드:
  1. 단일 종목: python backtest.py --symbol AAPL --days 365
  2. 포트폴리오: python backtest.py --portfolio --days 730
  3. ETF 전환: python backtest.py --etf-regime --days 365
  4. 섹터 로테이션: python backtest.py --sector-rotation --days 1080
  5. 워크포워드: python backtest.py --walk-forward --symbol AAPL --train 180 --test 60

시뮬레이션:
  - 슬리피지: 0.05% (대형주) ~ 0.2% (중소형)
  - 수수료: 주문당 $0 (KIS 미국주식 무료), 환전 스프레드 0.25%
  - 환율 영향: USD/KRW 변동 반영 (선택)
  - 레버리지 ETF: 일일 리밸런싱 시뮬레이션 (decay 반영)

성과 지표:
  - 총 수익률, 연환산 수익률 (CAGR)
  - Sharpe Ratio, Sortino Ratio
  - 최대 낙폭 (MDD), MDD 기간
  - 승률, 손익비 (Profit Factor)
  - 평균 보유기간
  - 월별/연도별 수익률 히트맵
  - 벤치마크 대비 (SPY) 초과수익

파라미터 최적화:
  - Grid Search: 주요 파라미터 범위 탐색
  - Walk-Forward: 과적합 방지 (학습/테스트 분리)
  - Sensitivity Analysis: 파라미터 민감도 분석
```

### 11.3 백테스트 → 라이브 전환 기준

```
최소 통과 기준 (3년 백테스트):
  ✓ CAGR > 12% (SPY 장기 평균 이상)
  ✓ Sharpe Ratio > 1.0
  ✓ MDD < 25%
  ✓ 승률 > 45% (추세 전략 기준)
  ✓ Profit Factor > 1.5
  ✓ 월별 수익 분포 정규성 (특정 월 과의존 없음)

전환 프로세스:
  1. 3년 백테스트 통과
  2. 모의투자 2주 실행 (KIS 모의투자 계좌)
  3. 소액 실거래 2주 ($1,000)
  4. 단계적 자본 확대
```

---

## 12. 데이터베이스 스키마

```sql
-- 주문
CREATE TABLE orders (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(10) NOT NULL,    -- NASD, NYSE, AMEX
    side            VARCHAR(4) NOT NULL,     -- BUY, SELL
    order_type      VARCHAR(10) NOT NULL,    -- MARKET, LIMIT
    quantity        DECIMAL(18,8) NOT NULL,
    price           DECIMAL(18,8),
    filled_quantity DECIMAL(18,8) DEFAULT 0,
    filled_price    DECIMAL(18,8),
    status          VARCHAR(20) NOT NULL,    -- PENDING, FILLED, CANCELLED
    strategy_name   VARCHAR(50),
    kis_order_id    VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    filled_at       TIMESTAMPTZ,
    pnl             DECIMAL(18,8)
);

-- 포지션
CREATE TABLE positions (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    exchange        VARCHAR(10) NOT NULL,
    quantity        DECIMAL(18,8) NOT NULL,
    avg_price       DECIMAL(18,8) NOT NULL,
    current_price   DECIMAL(18,8),
    unrealized_pnl  DECIMAL(18,8),
    stop_loss       DECIMAL(18,8),
    take_profit     DECIMAL(18,8),
    trailing_stop   DECIMAL(18,8),
    strategy_name   VARCHAR(50),
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 포트폴리오 스냅샷 (시계열)
CREATE TABLE portfolio_snapshots (
    id              SERIAL PRIMARY KEY,
    total_value_usd DECIMAL(18,2) NOT NULL,
    cash_usd        DECIMAL(18,2) NOT NULL,
    invested_usd    DECIMAL(18,2) NOT NULL,
    realized_pnl    DECIMAL(18,2),
    unrealized_pnl  DECIMAL(18,2),
    daily_pnl       DECIMAL(18,2),
    drawdown_pct    DECIMAL(8,4),
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 전략 실행 로그
CREATE TABLE strategy_logs (
    id              SERIAL PRIMARY KEY,
    strategy_name   VARCHAR(50) NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    signal_type     VARCHAR(10) NOT NULL,    -- BUY, SELL, HOLD
    confidence      DECIMAL(5,4) NOT NULL,
    indicators      JSONB,                   -- 지표 스냅샷
    market_state    VARCHAR(20),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 스캐너 결과
CREATE TABLE scanner_results (
    id              SERIAL PRIMARY KEY,
    scan_type       VARCHAR(30) NOT NULL,    -- VOLUME_SURGE, PRICE_MOVER 등
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(10),
    score           DECIMAL(8,4),
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 섹터 분석
CREATE TABLE sector_analysis (
    id              SERIAL PRIMARY KEY,
    sector_code     VARCHAR(10) NOT NULL,
    sector_name     VARCHAR(30) NOT NULL,
    strength_score  DECIMAL(8,4),
    return_1w       DECIMAL(8,4),
    return_1m       DECIMAL(8,4),
    return_3m       DECIMAL(8,4),
    trend           VARCHAR(10),             -- UP, DOWN, FLAT
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- AI 에이전트 로그
CREATE TABLE agent_logs (
    id              SERIAL PRIMARY KEY,
    agent_type      VARCHAR(30) NOT NULL,    -- MARKET, RISK, REVIEW
    content         TEXT NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 시스템 이벤트
CREATE TABLE events (
    id              SERIAL PRIMARY KEY,
    event_type      VARCHAR(30) NOT NULL,
    severity        VARCHAR(10) NOT NULL,    -- INFO, WARNING, ERROR
    message         TEXT NOT NULL,
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 백테스트 결과
CREATE TABLE backtest_results (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    config          JSONB NOT NULL,          -- 전략, 파라미터, 기간
    metrics         JSONB NOT NULL,          -- CAGR, Sharpe, MDD 등
    trades          JSONB,                   -- 개별 거래 내역
    equity_curve    JSONB,                   -- 자산 곡선
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 감시 목록 (유니버스)
CREATE TABLE watchlist (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    exchange        VARCHAR(10) NOT NULL,
    name            VARCHAR(100),
    sector          VARCHAR(30),
    market_cap      BIGINT,
    source          VARCHAR(20),             -- SCANNER, MANUAL, ETF
    score           DECIMAL(8,4),
    is_active       BOOLEAN DEFAULT TRUE,
    added_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_snapshots_recorded ON portfolio_snapshots(recorded_at);
CREATE INDEX idx_strategy_logs_created ON strategy_logs(created_at);
CREATE INDEX idx_scanner_created ON scanner_results(created_at);
CREATE INDEX idx_watchlist_active ON watchlist(is_active);
```

---

## 13. API 엔드포인트

### REST API

```
/api/v1/
├── portfolio/
│   ├── GET  /summary              # 포트폴리오 요약
│   ├── GET  /positions            # 보유 종목 목록
│   ├── GET  /history              # 자산 추이 (스냅샷)
│   ├── GET  /daily-pnl            # 일별 PnL
│   └── GET  /performance          # 성과 지표 (Sharpe, MDD 등)
│
├── trades/
│   ├── GET  /                     # 거래 이력 (페이지네이션)
│   ├── GET  /:id                  # 개별 거래 상세
│   └── GET  /summary              # 거래 통계
│
├── strategies/
│   ├── GET  /                     # 전략 목록 + 상태
│   ├── GET  /:name/performance    # 전략별 성과
│   ├── GET  /signals              # 최근 신호 로그
│   └── GET  /comparison           # 전략 비교
│
├── scanner/
│   ├── GET  /results              # 최근 스캔 결과
│   ├── POST /run                  # 수동 스캔 실행
│   ├── GET  /sectors              # 섹터 분석 결과
│   └── GET  /sectors/heatmap      # 섹터 히트맵 데이터
│
├── watchlist/
│   ├── GET  /                     # 감시 목록
│   ├── POST /                     # 종목 추가
│   ├── DELETE /:symbol            # 종목 제거
│   └── GET  /universe             # 전체 유니버스 (ETF 포함)
│
├── backtest/
│   ├── POST /run                  # 백테스트 실행
│   ├── GET  /results              # 결과 목록
│   ├── GET  /results/:id          # 개별 결과 상세
│   └── GET  /results/:id/chart    # 자산 곡선 데이터
│
├── engine/
│   ├── GET  /status               # 엔진 상태 (US Stock + ETF)
│   ├── POST /start                # 엔진 시작
│   ├── POST /stop                 # 엔진 정지
│   └── GET  /market-state         # 현재 시장 상태
│
├── agents/
│   ├── GET  /market-analysis      # 시장 분석 결과
│   ├── GET  /risk-assessment      # 리스크 평가
│   └── GET  /trade-review         # 매매 리뷰
│
├── events/
│   └── GET  /                     # 시스템 이벤트 로그
│
└── market/
    ├── GET  /price/:symbol        # 현재가 조회
    ├── GET  /chart/:symbol        # 차트 데이터 (OHLCV)
    └── GET  /etf/universe         # ETF 유니버스 현황

WebSocket:
  WS /ws/dashboard                 # 실시간 대시보드 업데이트
```

---

## 14. 프론트엔드 설계 (coin 프로젝트 기반)

### 14.1 탭 구성 (10개)

| # | 탭 이름 | 설명 | coin 프로젝트 대응 |
|---|---------|------|-------------------|
| 1 | Overview | 포트폴리오 요약, 자산 추이 차트, 엔진 제어 | 개요 탭 계승 |
| 2 | Trades | 거래 이력 테이블 (필터, 페이지네이션) | 거래 이력 탭 계승 |
| 3 | Signals | 전략 신호 로그, 지표 스냅샷 | 신호 로그 탭 계승 |
| 4 | Strategy | 전략별 성과, 승률, 비교 차트 | 전략 성과 탭 계승 |
| 5 | Scanner | 종목 스캔 결과, 감시 목록 관리 | 로테이션 탭 확장 |
| 6 | Sectors | 섹터 히트맵, 섹터별 강도 | **신규** |
| 7 | ETF Monitor | 레버리지/인버스 ETF 현황, 레짐 상태 | **신규** |
| 8 | Backtest | 백테스트 실행/결과/자산곡선 | **신규** |
| 9 | Agents | AI 시장분석, 리스크, 매매 리뷰 | 에이전트 탭 계승 |
| 10 | System | 시스템 로그, 일일 PnL 통계 | 시스템 로그 탭 계승 |

### 14.2 주요 컴포넌트

```
Dashboard.tsx
├── PortfolioSummary.tsx    # 총 자산, 현금, 투자금, PnL, 보유종목 테이블
│   └── PortfolioChart.tsx  # 자산 추이 라인차트 (Recharts)
├── TradeHistory.tsx        # 거래 이력 테이블, 종목/전략 필터
├── SignalLog.tsx           # 전략 신호 + 지표 데이터
├── StrategyPerformance.tsx # 전략별 승률, 수익, 비교 바차트
├── StockScanner.tsx        # 스캔 결과 테이블, 수동 스캔 버튼
│   └── WatchlistManager   # 감시 목록 추가/제거
├── SectorHeatmap.tsx       # 11개 섹터 히트맵 (색상 = 강도)
├── ETFMonitor.tsx          # ETF 유니버스, 현재 레짐, Bull/Bear 포지션
├── BacktestPanel.tsx       # 전략/기간 선택, 실행, 결과 차트
│   └── EquityCurve.tsx     # 자산곡선 + 벤치마크 비교
├── AgentStatus.tsx         # AI 분석 결과 카드
├── EngineControl.tsx       # 시작/정지 버튼, 상태 표시
├── CandlestickChart.tsx    # lightweight-charts 캔들차트
├── DailyPnLStats.tsx       # 일일 PnL 통계
└── SystemLog.tsx           # 이벤트 로그 (severity별 색상)
```

---

## 15. 설정 (.env)

```bash
# ===== KIS API =====
KIS_APP_KEY=your_app_key_here           # 36자리
KIS_APP_SECRET=your_app_secret_here     # 180자리
KIS_ACCOUNT_NO=12345678                 # 계좌번호 (8자리)
KIS_ACCOUNT_PRODUCT=01                  # 계좌상품코드
KIS_BASE_URL=https://openapi.koreainvestment.com:9443       # 실투자
# KIS_BASE_URL=https://openapivts.koreainvestment.com:29443 # 모의투자
KIS_WS_URL=ws://ops.koreainvestment.com:21000

# ===== Trading =====
TRADING_MODE=paper                      # paper | live
TRADING_EVALUATION_INTERVAL_SEC=300     # 전략 평가 간격 (초)
TRADING_INITIAL_BALANCE_USD=10000       # 초기 자본 (USD)
TRADING_MIN_CONFIDENCE=0.50             # 최소 신뢰도
TRADING_MAX_POSITIONS=10                # 최대 동시 보유

# ===== ETF Engine =====
ETF_ENGINE_ENABLED=true
ETF_MAX_PORTFOLIO_PCT=0.30              # 전체 대비 ETF 최대 비중
ETF_MAX_HOLD_DAYS=10                    # 레버리지 ETF 최대 보유일

# ===== Risk =====
RISK_MAX_SINGLE_STOCK_PCT=0.20          # 종목당 최대 비중
RISK_MAX_INVESTED_PCT=0.80              # 최대 투자비율
RISK_MAX_DRAWDOWN_PCT=0.15              # MDD 한도
RISK_DAILY_LOSS_LIMIT_PCT=0.03          # 일일 손실 한도
RISK_VIX_THRESHOLD=35                   # VIX 방어 임계값

# ===== Database =====
DB_URL=postgresql+asyncpg://usstock:usstock@localhost:5432/us_stock_trading

# ===== Redis =====
REDIS_URL=redis://localhost:6379/0

# ===== Notifications =====
NOTIFY_ENABLED=true
NOTIFY_PROVIDER=discord
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# ===== LLM (AI Agent — Multi-Provider) =====
LLM_ENABLED=true
LLM_API_KEY=sk-ant-...                         # Anthropic API key
LLM_MODEL=claude-haiku-4-5-20251001            # Primary model
LLM_FALLBACK_MODEL=claude-sonnet-4-6           # Anthropic fallback
LLM_GEMINI_API_KEY=AIza...                     # Google Gemini API key
LLM_GEMINI_FALLBACK_MODEL=gemini-2.5-flash  # Gemini fallback
LLM_MAX_TOKENS=4096

# ===== External Data =====
FRED_API_KEY=your_fred_api_key          # 경제지표 (무료)
EXTERNAL_FINNHUB_API_KEY=your_key       # Finnhub 뉴스 (무료 tier)

# ===== Auth =====
AUTH_API_TOKEN=                          # Bearer token (empty=disabled)

# ===== KR Market =====
KIS_KR_APP_KEY=                         # KR 계좌 앱키
KIS_KR_APP_SECRET=                      # KR 계좌 시크릿
KIS_KR_ACCOUNT_NO=                      # KR 계좌번호

# ===== Logging =====
APP_LOG_LEVEL=INFO
```

---

## 16. 배포 구조

**참고**: 실제 운영 환경에서는 Docker Compose가 아닌 systemd 서비스로 배포.
PostgreSQL/Redis는 coin 프로젝트의 Docker 컨테이너를 공유.

```
실서비스 구성:
- PostgreSQL: coin-postgres-1 컨테이너 공유 (DB: us_stock_trading)
- Redis: 공유 (db 1, coin은 db 0)
- Backend: systemd (usstock-backend.service) — port 8001
- Frontend: systemd (usstock-frontend.service) — port 3001
- HTTPS: nginx reverse proxy (port 8443)
- Backup: systemd timers (local daily + GitHub weekly)
```

---

## 17. 구현 현황

모든 Phase 완료. 현재 US + KR 실계좌 라이브 운용 중.

- Phase 1: 기반 구축 — ✅ KIS API, Paper adapter, DB, FastAPI
- Phase 2: 데이터 파이프라인 — ✅ yfinance, FRED, KIS WebSocket, Redis cache
- Phase 3: 백테스트 엔진 — ✅ 단일 전략 + 풀 파이프라인 백테스트
- Phase 4: 전략 구현 — ✅ 14개 전략 + Signal Combiner (Mode B)
- Phase 5: 매매 엔진 — ✅ 평가 루프, ATR-based SL/TP, Kelly sizing
- Phase 6: ETF + 스캐너 — ✅ US/KR ETF 엔진, 3-Layer scanner
- Phase 7: AI + 알림 — ✅ 4 agents (market/risk/review/news), Discord notification
- Phase 8: 프론트엔드 — ✅ 20 components, US/KR 토글
- Phase 9: 검증 + 라이브 — ✅ 1276+ tests, 실계좌 운용
- Phase 10: KR 주식 — ✅ KIS KR adapter, 듀얼마켓, KR 스크리너
- Phase 11: 뉴스/이벤트 — ✅ Finnhub/Naver 뉴스, 실적/매크로/내부자 캘린더
- Phase 12: 시스템 고도화 — ✅ MCP 서버, DB 백업, 주문 안전장치

---

## 18. coin 프로젝트 대비 주요 차이점

| 항목 | coin 프로젝트 | US Stock 프로젝트 |
|------|--------------|------------------|
| 거래소 | Bithumb, Binance | KIS (한국투자증권) |
| 시장 | 암호화폐 (24/7) | 미국주식 (정규장 + 시간외) |
| API | ccxt 라이브러리 | python-kis + 자체 래퍼 |
| 실시간 데이터 | ccxt.pro WebSocket | KIS WebSocket (지연시세) |
| 레버리지 | Binance Futures (1-10x) | 레버리지 ETF (TQQQ 3x 등) |
| 숏 포지션 | Binance Futures 공매도 | 인버스 ETF (SQQQ 등) |
| 종목 수 | ~20 코인 | ~3000+ (스캐닝 필요) |
| 섹터 | 없음 | 11개 GICS 섹터 |
| 외부 데이터 | 없음 | yfinance, FRED, 뉴스 |
| 백테스트 | 단일 파일 (backtest.py) | 모듈화된 백테스트 엔진 |
| 거래 시간 | 24/7 | 정규장 + 시간외 (시간 관리 필요) |
| 통화 | KRW, USDT | USD (원화 환전 필요) |
| 수수료 | 0.25% | $0 (환전 스프레드 0.25%) |
| Rate Limit | 거래소별 상이 | 20 req/sec (실), 5 req/sec (모의) |

---

## 19. 핵심 설계 원칙

1. **coin 프로젝트 아키텍처 계승**: 검증된 구조(엔진/전략/어댑터 패턴)를 재사용
2. **추세 매매 중심**: 미국주식 특성에 맞는 추세 추종 전략 우선
3. **ETF 적극 활용**: 시장 레짐에 따라 레버리지/인버스 자동 스위칭
4. **충분한 백테스트**: 라이브 전 3년 이상 검증 필수
5. **외부 데이터 통합**: KIS API만으로 부족한 정보를 yfinance, FRED로 보완
6. **시장 시간 인식**: 미국장 거래 시간에 맞춘 스케줄링
7. **점진적 전환**: 백테스트 → 모의투자 → 소액 → 본격 운용
8. **전략 가변 적용**: 모든 전략 파라미터는 config로 교체 가능
9. **테스트 완비**: 모든 코드에 유닛 테스트 + 시나리오 테스트, 전략은 백테스트 필수

---

## 20. 전략 가변 적용 시스템 (Pluggable Strategy Config)

### 20.1 전략 Config 구조

모든 전략의 파라미터, 가중치, 활성화 여부를 단일 YAML 파일로 관리.
코드 수정 없이 전략 교체/튜닝 가능.

```yaml
# config/strategies.yaml

# === 전역 설정 ===
global:
  min_confidence: 0.50              # 최소 신뢰도 (이하 HOLD)
  evaluation_interval_sec: 300      # 전략 평가 주기
  max_active_strategies: 6          # 동시 활성 전략 수
  signal_combination: weighted_vote # weighted_vote | majority | highest_confidence

# === 시장 상태별 전략 가중치 프로파일 ===
profiles:
  strong_uptrend:
    trend_following: 0.35
    dual_momentum: 0.25
    donchian_breakout: 0.20
    supertrend: 0.20
    macd_histogram: 0.00
    rsi_divergence: 0.00
    bollinger_squeeze: 0.00
    volume_profile: 0.00

  uptrend:
    trend_following: 0.30
    dual_momentum: 0.25
    donchian_breakout: 0.20
    supertrend: 0.15
    macd_histogram: 0.10
    rsi_divergence: 0.00
    bollinger_squeeze: 0.00
    volume_profile: 0.00

  sideways:
    trend_following: 0.10
    dual_momentum: 0.10
    donchian_breakout: 0.05
    supertrend: 0.05
    macd_histogram: 0.15
    rsi_divergence: 0.25
    bollinger_squeeze: 0.20
    volume_profile: 0.10

  downtrend:
    # 개별 종목 매수 억제, 점수 전체 하향
    trend_following: 0.05
    rsi_divergence: 0.20
    bollinger_squeeze: 0.10
    # ETF 엔진으로 전환
    _note: "개별종목 매수 최소화, ETF 인버스 활용"

  etf_engine:
    regime_switch: 0.55
    sector_rotation: 0.45

# === 개별 전략 파라미터 ===
strategies:
  trend_following:
    enabled: true
    timeframe: "1D"
    params:
      ema_fast: 20
      ema_slow: 50
      ema_long: 200
      adx_period: 14
      adx_threshold: 25
      volume_ma_period: 20
      volume_min_ratio: 1.0
    stop_loss:
      type: atr              # atr | fixed_pct | trailing
      atr_multiplier: 2.0
      max_pct: 0.08
    take_profit:
      type: atr
      atr_multiplier: 4.0
      max_pct: 0.20
    trailing_stop:
      enabled: true
      activation_pct: 0.05   # 5% 수익 이후 활성화
      trail_pct: 0.03        # 고점 대비 3% 하락 시 청산

  dual_momentum:
    enabled: true
    timeframe: "1M"           # 월간 리밸런싱
    params:
      lookback_months: 12
      top_n: 5                # 상대 모멘텀 상위 N개
      rebalance_day: 1        # 매월 1일
      min_absolute_return: 0.0
      cash_etf: "SHY"        # 모멘텀 음수 시 대체 ETF
    stop_loss:
      type: fixed_pct
      max_pct: 0.10

  donchian_breakout:
    enabled: true
    timeframe: "1D"
    params:
      entry_period: 20        # 20일 고가 돌파 진입
      exit_period: 10         # 10일 저가 이탈 청산
      atr_period: 20
      position_sizing: atr    # atr | equal_weight
      atr_risk_pct: 0.01     # 자본의 1% 리스크
    stop_loss:
      type: atr
      atr_multiplier: 2.0

  supertrend:
    enabled: true
    timeframe: "1D"
    params:
      atr_period: 10
      multiplier: 3.0
      confirmation_bars: 2    # 전환 후 2봉 확인
    stop_loss:
      type: supertrend        # Supertrend 밴드 자체가 SL

  macd_histogram:
    enabled: true
    timeframe: "1D"
    params:
      fast_period: 12
      slow_period: 26
      signal_period: 9
      divergence_lookback: 14
      min_histogram_change: 0.5
    stop_loss:
      type: fixed_pct
      max_pct: 0.05

  rsi_divergence:
    enabled: true
    timeframe: "1D"
    params:
      rsi_period: 14
      overbought: 70
      oversold: 30
      divergence_lookback: 14
      min_price_move_pct: 3.0
    stop_loss:
      type: fixed_pct
      max_pct: 0.04

  bollinger_squeeze:
    enabled: true
    timeframe: "1D"
    params:
      bb_period: 20
      bb_std: 2.0
      keltner_period: 20
      keltner_atr_mult: 1.5
      squeeze_min_bars: 3     # 최소 3봉 스퀴즈 후 돌파 (일봉 기준)
    stop_loss:
      type: atr
      atr_multiplier: 1.5

  volume_profile:
    enabled: true
    timeframe: "1D"
    params:
      lookback_days: 60
      volume_surge_threshold: 2.0
      obv_ma_period: 20
    stop_loss:
      type: fixed_pct
      max_pct: 0.05

  regime_switch:
    enabled: true
    timeframe: "1D"
    params:
      spy_sma_period: 200
      vix_bull_threshold: 20
      vix_bear_threshold: 25
      confirmation_days: 2    # 레짐 전환 2일 확인
      max_leverage_etf_hold_days: 10
    etf_mapping:
      bull: [TQQQ, SOXL, UPRO, TECL, FNGU]
      bear: [SQQQ, SOXS, SPXU, TECS, FNGD]
      neutral: [QQQ, SOXX, SPY, XLK]

  sector_rotation:
    enabled: true
    timeframe: "1M"
    params:
      top_sectors: 3
      lookback_weeks: 12
      rebalance_day: 1
      min_strength_score: 60
    sector_etfs:
      Technology: [XLK, TECL]
      Financials: [XLF, FAS]
      Energy: [XLE, ERX]
      Healthcare: [XLV, LABU]
      Consumer_Disc: [XLY]
      Consumer_Staples: [XLP]
      Industrials: [XLI]
      Materials: [XLB]
      Utilities: [XLU]
      Real_Estate: [XLRE]
      Communications: [XLC]

# === 스크리닝 설정 ===
screening:
  indicator_screener:
    weights:
      trend: 0.40
      momentum: 0.25
      volatility_volume: 0.20
      support_resistance: 0.15
    min_grade: "B"            # B 이상만 Layer 2 진행
    max_candidates: 50

  fundamental_enricher:
    weights:
      consensus: 0.35
      fundamental: 0.35
      smart_money: 0.30

  ai_recommender:
    score_blend:
      indicator_weight: 0.35  # Layer 1 기술적 점수
      enrichment_weight: 0.25 # Layer 2 yfinance 점수
      ai_weight: 0.40         # Layer 3 AI 점수
    min_final_score: 70       # BUY 최소 점수
    min_conviction: "MEDIUM"
```

### 20.2 전략 런타임 관리

```python
class StrategyRegistry:
    """전략 동적 등록/활성화/비활성화"""

    def load_config(path: str) -> StrategyConfig
    def reload_config() -> None          # 런타임 핫 리로드
    def get_active_strategies() -> list[BaseStrategy]
    def get_weights(market_state) -> dict[str, float]
    def enable_strategy(name) -> None
    def disable_strategy(name) -> None
    def update_params(name, params) -> None  # 런타임 파라미터 변경

# 전략 추가: strategies/ 디렉토리에 파일 추가 + yaml에 설정 추가만 하면 됨
# BaseStrategy를 상속하고 evaluate() 구현 → 자동 등록
```

### 20.3 전략 적용 규칙

```
새로운 전략 적용 프로세스:

1. 전략 코드 작성 (strategies/new_strategy.py)
   - BaseStrategy 상속, evaluate() 구현
   - 유닛 테스트 작성 (100% 커버리지)

2. 백테스트 필수 선행
   - 3년 이상 백테스트 실행
   - 통과 기준: CAGR>12%, Sharpe>1.0, MDD<25%
   - 기존 전략 대비 개선 확인

3. config/strategies.yaml에 등록
   - params, stop_loss, take_profit 설정
   - 초기 가중치 0.05 (낮게 시작)

4. 모의투자 검증 (2주)
   - 기존 전략과 병행 운영
   - 독립 성과 추적

5. 가중치 점진 증가
   - 모의투자 결과 양호 → 0.05 → 0.10 → 목표 가중치
```

---

## 21. 테스트 전략

### 21.1 테스트 피라미드

```
                    ┌───────────┐
                    │ E2E 시나리오│  ← 적음 (핵심 플로우만)
                    │   테스트    │
                   ┌┴───────────┴┐
                   │  통합 테스트  │  ← 모듈 간 연동
                  ┌┴─────────────┴┐
                  │   유닛 테스트   │  ← 대부분 (모든 함수/클래스)
                 ┌┴───────────────┴┐
                 │    백테스트      │  ← 전략 검증 (필수 선행)
                 └─────────────────┘
```

### 21.2 유닛 테스트

```
모든 코드에 유닛 테스트 필수. 테스트 없는 코드는 머지 불가.

coverage 목표: 90% 이상 (전략/엔진 코드는 95% 이상)
프레임워크: pytest + pytest-asyncio + pytest-cov
DB: aiosqlite (인메모리, 테스트 격리)

테스트 구조:
tests/
├── conftest.py                  # 공통 fixture (DB, mock adapter)
├── test_exchange/
│   ├── test_kis_adapter.py      # KIS API 모킹 테스트
│   ├── test_kis_auth.py         # OAuth 토큰 관리
│   ├── test_kis_websocket.py    # WebSocket 연결/구독
│   ├── test_paper_adapter.py    # 모의투자 어댑터
│   └── test_rate_limiter.py     # Rate Limit 동작
│
├── test_strategies/
│   ├── test_trend_following.py  # 각 전략별 테스트
│   ├── test_dual_momentum.py
│   ├── test_donchian_breakout.py
│   ├── test_supertrend.py
│   ├── test_macd_histogram.py
│   ├── test_rsi_divergence.py
│   ├── test_bollinger_squeeze.py
│   ├── test_volume_profile.py
│   ├── test_regime_switch.py
│   ├── test_sector_rotation.py
│   ├── test_combiner.py        # 신호 결합 로직
│   └── test_registry.py        # 전략 등록/가중치
│
├── test_engine/
│   ├── test_trading_engine.py   # 평가 루프, 시장상태 판단
│   ├── test_etf_engine.py       # ETF 전환 로직
│   ├── test_order_manager.py    # 주문 생명주기
│   ├── test_portfolio_manager.py
│   ├── test_position_tracker.py # SL/TP/Trailing
│   ├── test_risk_manager.py     # 리스크 한도 체크
│   └── test_health_monitor.py
│
├── test_scanner/
│   ├── test_indicator_screener.py  # Layer 1 지표 스코어링
│   ├── test_fundamental_enricher.py # Layer 2 yfinance 보강
│   ├── test_ai_recommender.py      # Layer 3 AI 추천 (모킹)
│   ├── test_sector_analyzer.py
│   └── test_universe_manager.py
│
├── test_data/
│   ├── test_market_data_service.py
│   ├── test_external_data_service.py
│   ├── test_fred_service.py
│   ├── test_market_state.py
│   └── test_indicator_service.py
│
├── test_backtest/
│   ├── test_backtest_engine.py
│   ├── test_simulator.py
│   ├── test_metrics.py
│   └── test_data_loader.py
│
├── test_api/
│   ├── test_portfolio_api.py
│   ├── test_trades_api.py
│   ├── test_scanner_api.py
│   └── test_websocket.py
│
└── test_agents/
    ├── test_market_analysis.py
    ├── test_risk_assessment.py
    └── test_trade_review.py

유닛 테스트 작성 규칙:
- 함수/메서드당 최소 3개 테스트 (정상, 경계값, 에러)
- 외부 API는 반드시 모킹 (KIS, yfinance, Claude)
- 전략 테스트는 알려진 OHLCV 데이터로 기대값 검증
- 비동기 함수는 pytest-asyncio 사용
- DB 테스트는 트랜잭션 롤백으로 격리
```

### 21.3 시나리오 테스트

```
실제 매매 흐름을 시뮬레이션하는 End-to-End 시나리오 테스트.

tests/scenarios/
├── test_scenario_normal_buy_sell.py
│   """정상 매수-보유-매도 플로우"""
│   1. 스캐너가 AAPL 발견 (IndicatorScore 85)
│   2. yfinance 보강 (EnrichmentScore 78)
│   3. AI 추천 (BUY, conviction HIGH)
│   4. 전략 평가 → BUY 신호 (confidence 0.72)
│   5. 리스크 검증 통과
│   6. 주문 실행 → 체결
│   7. 포지션 생성, SL/TP 설정
│   8. 가격 상승 → Trailing Stop 갱신
│   9. TP 도달 → 매도 실행
│   10. PnL 계산, DB 기록, 알림 발송
│
├── test_scenario_stop_loss_triggered.py
│   """손절 시나리오"""
│   1. 매수 후 가격 하락
│   2. SL 가격 도달 → 자동 매도
│   3. 쿨다운 적용 확인
│   4. PnL 음수 기록
│
├── test_scenario_regime_switch.py
│   """시장 레짐 전환 시나리오"""
│   1. BULL 레짐 → TQQQ 보유 중
│   2. SPY < SMA(200), VIX > 25
│   3. 2일 확인 후 BEAR 전환
│   4. TQQQ 청산 → SQQQ 매수
│   5. 가중치 프로파일 변경 확인
│
├── test_scenario_daily_full_scan.py
│   """일일 풀스캔 파이프라인"""
│   1. KIS API 스캔 (모킹)
│   2. Layer 1 지표 스크리닝
│   3. Layer 2 yfinance 보강
│   4. Layer 3 AI 분석
│   5. 유니버스 갱신
│   6. WebSocket 구독 변경
│
├── test_scenario_risk_limit_breach.py
│   """리스크 한도 초과 시나리오"""
│   1. 일일 손실 3% 도달
│   2. 신규 매수 차단 확인
│   3. MDD 15% 도달 → 전량 청산
│   4. 알림 발송 확인
│
├── test_scenario_api_failure_recovery.py
│   """API 장애 복구 시나리오"""
│   1. KIS API 연속 5회 에러
│   2. 엔진 일시 중지
│   3. 재연결 시도 (exponential backoff)
│   4. 복구 후 포지션 동기화
│
├── test_scenario_market_hours.py
│   """시장 시간대 동작 시나리오"""
│   1. 장전: T1 풀스캔 실행
│   2. 정규장: 전략 평가 루프 + T2 핫스캔
│   3. 장 마감: T4 AI 브리핑
│   4. 장외: 스케줄러 비활성 확인
│
├── test_scenario_strategy_hot_reload.py
│   """전략 설정 런타임 변경"""
│   1. strategies.yaml 파라미터 변경
│   2. reload_config() 호출
│   3. 다음 평가 루프에서 새 파라미터 적용 확인
│   4. 가중치 변경 반영 확인
│
└── test_scenario_new_strategy_onboarding.py
    """새 전략 추가 플로우"""
    1. 전략 코드 + 유닛 테스트 존재 확인
    2. 백테스트 실행 → 통과 기준 확인
    3. yaml에 등록 (가중치 0.05)
    4. 전략 평가 루프에서 신호 생성 확인
    5. 신호 결합에 가중치 반영 확인
```

### 21.4 백테스트 = 전략의 필수 테스트

```
모든 전략은 백테스트 통과 없이 라이브 적용 불가.

백테스트 자동 실행 조건:
  - 전략 코드 변경 시 (CI에서 자동 실행)
  - 전략 파라미터 변경 시
  - 주간 정기 실행 (파라미터 드리프트 감지)

백테스트 통과 기준 (3년 데이터):
  ✓ CAGR > 12%
  ✓ Sharpe Ratio > 1.0
  ✓ Max Drawdown < 25%
  ✓ Win Rate > 45%
  ✓ Profit Factor > 1.5
  ✓ 월별 수익 표준편차 < 15% (안정성)

CI에서 백테스트 실패 시:
  - PR 머지 차단
  - 실패 지표 상세 리포트 생성
  - 기존 통과 버전과 비교 diff 제공
```

---

## 22. 개발 프로세스 + Agent 문서 체계

### 22.1 프로젝트 문서 구조

```
~/us-stock/
├── CLAUDE.md                    # AI Agent 핵심 지침서 (코드 규칙, 아키텍처)
├── SYSTEM_DESIGN.md             # 이 문서 (시스템 설계 상세)
├── .github/workflows/
│   ├── ci.yml                   # PR 시 자동 테스트
│   └── backtest.yml             # 전략 변경 시 백테스트
├── config/
│   ├── strategies.yaml          # 전략 파라미터 (메인, 핫 리로드)
│   ├── etf_universe.yaml        # US ETF 유니버스
│   └── kr_etf_universe.yaml     # KR ETF 유니버스
├── deploy/                      # systemd 서비스, DB 설정/백업
├── backend/                     # Python 3.12+ FastAPI
└── frontend/                    # React 18 + Vite
```

### 22.2 CLAUDE.md (AI Agent 지침서)

CLAUDE.md는 프로젝트 루트에 위치. 최신 내용은 CLAUDE.md 파일 직접 참조.
주요 포함 내용: 프로젝트 개요, 기술 스택, 코드/전략/테스트 규칙, 디렉토리 구조, 아키텍처 결정사항.

### 22.3 개발 워크플로우 (GitHub Flow)

```
브랜치 전략: GitHub Flow (main + feature branches)

main ──────────────────────────────────────────►
       \         \         \         \
        feat/    feat/     fix/      feat/
        kis-auth scanner   sl-tp     backtest
        \___PR__/ \___PR__/ \___PR__/ \___PR__/

브랜치 규칙:
- main: 항상 배포 가능 상태
- feature/*: 기능 개발
- fix/*: 버그 수정
- refactor/*: 리팩토링
- test/*: 테스트 추가

PR 프로세스:
1. feature 브랜치 생성
2. 코드 작성 + 유닛 테스트 + 시나리오 테스트 (해당 시)
3. 전략 변경인 경우 백테스트 실행 결과 첨부
4. PR 생성
5. CI 자동 실행:
   - lint (ruff)
   - type check (mypy)
   - unit tests (pytest --cov)
   - scenario tests
   - 전략 변경 시 backtest
6. coverage 90% 미만 → 머지 차단
7. 테스트 전체 통과 → 머지 가능

커밋 컨벤션 (Conventional Commits):
  feat: 새로운 기능
  fix: 버그 수정
  refactor: 코드 리팩토링 (동작 변경 없음)
  test: 테스트 추가/수정
  docs: 문서 변경
  config: 설정 변경 (strategies.yaml 등)
  ci: CI/CD 변경

예시:
  feat(strategy): add bollinger squeeze strategy
  fix(engine): fix trailing stop not updating on gap up
  test(scanner): add scenario test for daily full scan
  config(strategy): tune trend_following ADX threshold to 25
```

### 22.4 CI/CD 파이프라인

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  lint:
    # ruff check + ruff format --check
    # mypy --strict

  test:
    # pytest --cov=backend --cov-report=xml
    # coverage 90% 미만 시 실패

  scenario-test:
    # pytest tests/scenarios/ -v

  backtest:
    # 전략 파일 변경 감지 시에만 실행
    if: contains(changed_files, 'strategies/') or contains(changed_files, 'strategies.yaml')
    # python -m backtest --ci --min-cagr=0.12 --min-sharpe=1.0 --max-mdd=0.25
    # 기준 미달 시 실패

  frontend:
    # npm run build
    # npm run lint
```

### 22.5 배포 프로세스

```
환경:
  - dev:  로컬 Docker Compose (개발/테스트)
  - staging: 모의투자 계좌 연결 (KIS 모의투자 서버)
  - prod: 실계좌 연결 (KIS 실투자 서버)

배포 흐름:
  main 머지 → staging 자동 배포 → 2주 모의투자 검증 → prod 수동 배포

┌─────────────────────────────────────────────────────────┐
│  Stage        │ 트리거          │ 대상      │ 자동/수동  │
├───────────────┼─────────────────┼───────────┼───────────┤
│  dev          │ 로컬 실행       │ localhost │ 수동      │
│  staging      │ main 머지       │ 서버      │ 자동      │
│  prod         │ GitHub Release  │ 서버      │ 수동      │
└─────────────────────────────────────────────────────────┘

배포 체크리스트 (prod):
  ✓ 모든 테스트 통과 (1276+ tests)
  ✓ 백테스트 통과 기준 충족
  ✓ 환경 변수 확인 (KIS 실투자 URL, 실계좌)

배포 방법:
  - systemd 서비스 (Raspberry Pi): sudo systemctl restart usstock-backend
  - 정규장 외 시간에만 배포 (미체결 주문 없는지 확인)
  - 배포 후 자동 헬스체크 (API, DB, Redis, KIS 연결)

무중단 배포:
  - 정규장 외 시간에만 배포 (10:00~18:00 KST)
  - 배포 전 보유 포지션 확인 (미체결 주문 없는지)
  - 배포 후 자동 헬스체크 (API, DB, Redis, KIS 연결)
```

### 22.6 현재 상태

시스템은 v1.0 이상으로 US + KR 실계좌 라이브 운용 중.
변경 이력은 git log로 관리. 커밋 메시지에 Conventional Commits 적용.
