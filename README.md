# US Stock Auto-Trading Engine

한국투자증권(KIS) Open API 기반 **US + KR 듀얼마켓 자동매매 시스템**.
~/coin (암호화폐 봇) 아키텍처를 계승하여 주식 시장 특성에 맞게 확장.

> ⚠️ **이 레포는 실계좌 라이브 트레이딩 코드입니다.**
> 모든 변경은 실제 자금에 영향을 미칩니다. 페이퍼/시뮬레이션이 아닙니다.
> 자신의 KIS 계좌에 연결해 운용하려면 반드시 백테스트와 페이퍼 검증을 거치세요.

---

## 핵심 정보

| 항목 | 값 |
|---|---|
| **Backend** | Python 3.12+, FastAPI, SQLAlchemy 2.0 (asyncpg), PostgreSQL 16, Redis 7 |
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS |
| **Brokerage** | 한국투자증권 Open API (REST + WebSocket) |
| **Markets** | US (NYSE/NASDAQ), KR (KOSPI/KOSDAQ) — 동시 운영 |
| **Strategies** | 17개 등록, market별 disabled_strategies로 활성/비활성 제어 |
| **Backtesting** | 단일 전략 + full pipeline (`backend/backtest/`) |
| **Tests** | pytest + pytest-asyncio (1276+ tests, coverage 90%+ 목표) |
| **Deploy** | Raspberry Pi ARM64, systemd, nginx HTTPS reverse proxy |
| **Ports** | Backend `8001`, Frontend `3001`, HTTPS `8443` |

---

## 주요 기능

### Trading Engine
- **Dual market**: US/KR 별도 evaluation_loop, 별도 risk/order/position manager
- **Stock + ETF 이중 엔진**: 개별주는 stock engine, 레버리지/인버스/섹터 ETF는 etf_engine
- **Dynamic universe**: yfinance 스크리너 + KIS 랭킹 API + 섹터 가중치
- **3-Layer screening**: IndicatorScreener → FundamentalEnricher → AI Recommender (+ Layer 2.5 News sentiment)
- **Multi-strategy combiner**: 17 strategies → SignalCombiner (group consensus, quadratic conviction weighting)
- **Adaptive weights**: 종목 분류 + 시장 regime 기반 동적 가중치
- **Kelly position sizing**: factor score + signal quality 누적 학습
- **ATR 기반 dynamic SL/TP**: 종목별 변동성 조정

### Market Intelligence
- **Market regime detection**: SPY/KODEX 200 + SMA200 + VIX → 5 regimes (STRONG_UPTREND ~ DOWNTREND, + WEAK_DOWNTREND)
- **Asymmetric regime confirmation**: risk-off 1일, risk-on 2일
- **FRED macro indicators** (async via `asyncio.to_thread`)
- **Event calendar**: earnings, FOMC, CPI, Jobs, insider trades
- **News sentiment**: Finnhub (US) + Naver Finance (KR), ±15pt 가중치

### AI Layer
- **Multi-provider LLM**: Anthropic primary → fallback → Gemini (3x retry, exponential backoff)
- **AI agents**: market_analyst, risk_assessment, trade_review, news_sentiment
- **Persistent agent memory**: DB-backed AgentContextService (token budget, importance-based eviction)

### Operations
- **YAML hot-reload**: `config/strategies.yaml` 변경 시 무중단 적용
- **Order safety**: 중복 검사, 2분 reconciliation, partial fill 추적, fetch_executed_orders 폴백
- **Notification adapter**: Discord 기본, Telegram/Slack 지원
- **Backups**: 로컬 일일 (7일) + GitHub 주간 (4주) systemd 타이머
- **MCP server**: `backend/mcp_server.py` (Claude Desktop/Code 통합용 28 tools)

---

## 디렉토리 구조

```
us-stock/
├── README.md                  ← 이 파일
├── CLAUDE.md                  ← AI agent 지침서 (코드베이스 규칙)
├── SYSTEM_DESIGN.md           ← 전체 시스템 설계 문서
├── WORKFLOW.md                ← 개발 워크플로
├── docker-compose.yml         ← DB/Redis 참조 (실서비스는 systemd)
│
├── config/
│   ├── strategies.yaml        ← 전략 파라미터 + market disabled list (메인)
│   ├── etf_universe.yaml      ← US ETF 매핑
│   └── kr_etf_universe.yaml   ← KR ETF 매핑
│
├── backend/
│   ├── main.py                ← FastAPI lifespan + startup wiring
│   ├── mcp_server.py          ← MCP server (28 tools)
│   ├── exchange/              ← KIS US/KR REST + WebSocket adapter
│   ├── engine/                ← evaluation_loop, etf_engine, order/risk/portfolio managers
│   ├── strategies/            ← 17개 전략 + base/registry/combiner/config_loader
│   ├── scanner/               ← 3-Layer 스크리닝 파이프라인
│   ├── data/                  ← market_data, indicators, FRED, news, events
│   ├── agents/                ← 4개 AI agent
│   ├── analytics/             ← factor_model, kelly sizing, signal_quality, ML factor
│   ├── services/              ← LLM client, cache, notifications, rate limiter
│   ├── backtest/              ← single + full_pipeline backtest
│   ├── api/                   ← 13 REST + WebSocket 모듈
│   └── tests/                 ← 1276+ tests (unit, integration, scenario)
│
├── frontend/                  ← React 18 + TypeScript dashboard
│   └── src/components/        ← 20개 컴포넌트
│
├── deploy/
│   ├── usstock-backend.service / usstock-frontend.service  ← systemd
│   ├── setup-db.sh / backup-db.sh                          ← DB 운영
│   ├── nginx/                                              ← HTTPS reverse proxy
│   └── install.sh                                          ← 설치 스크립트
│
└── data/
    └── backtest_results/      ← 백테스트 결과 JSON 저장소
```

---

## 빠른 시작

### 1. 사전 요건
- Python 3.12+
- PostgreSQL 16, Redis 7 (`docker-compose.yml` 참조)
- 한국투자증권 OpenAPI 계정 (US + KR 거래 권한)
- (선택) Anthropic API key, Gemini API key, Finnhub API key, FRED API key

### 2. 설치
```bash
git clone https://github.com/<owner>/us-stock.git
cd us-stock

# Python venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# 환경 설정
cp .env.example .env
# .env 편집: KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, DB/Redis 접속정보 등

# DB 초기화
bash deploy/setup-db.sh
```

### 3. 실행 (개발 모드)
```bash
# Backend
cd backend
../venv/bin/python -m uvicorn main:app --reload --port 8001

# Frontend (별도 터미널)
cd frontend
npm run dev   # → http://localhost:3001
```

### 4. 라이브 운용 (Raspberry Pi systemd)
```bash
sudo cp deploy/usstock-backend.service /etc/systemd/system/
sudo cp deploy/usstock-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now usstock-backend usstock-frontend
sudo journalctl -u usstock-backend -f
```

---

## 백테스트

### 단일 전략
```bash
cd backend
../venv/bin/python -c "
import asyncio
from backtest.engine import BacktestEngine
asyncio.run(BacktestEngine().run('supertrend', 'AAPL', period='3y'))
"
```

### 풀 파이프라인 (전체 시스템 시뮬레이션)
```bash
cd backend
../venv/bin/python scripts/validate_new_strategy.py
```

> **주의**: 백테스트 universe(`DEFAULT_UNIVERSE` 55개 대형주)는 라이브 dynamic_universe와 다릅니다.
> 백테스트 결과는 같은 universe 내 *상대 비교*에는 신뢰할 수 있지만,
> 라이브 절대 알파 추정치로는 보정이 필요합니다 (universe alignment 작업 진행 중).

### 새 전략 활성화 게이트
- **신규 전략**: walk-forward backtest 통과 — CAGR > 12%, Sharpe > 1.0, MDD < 25%
- **전략 조합 변경**: 직전 enabled 조합 대비 4지표(Ret/Sharpe/MDD/PF) 모두 개선
- **US 활성화 floor** (relaxed): Sharpe > 0, MDD < 15%, PF > 1.0
- 위 조건 미달 시 `strategies.yaml`에 PROVISIONAL 마크 + 후속 최적화 티켓 필수

---

## 전략 시스템

전략은 `BaseStrategy`를 상속하고 `analyze() -> Signal`을 구현합니다. 모든 파라미터는 `config/strategies.yaml`에서 관리되며 런타임 핫 리로드됩니다.

### 활성 전략 (17개 등록)
**Core trend**: trend_following, dual_momentum, donchian_breakout, supertrend
**Coin port**: cis_momentum, larry_williams, bnf_deviation
**보조**: macd_histogram, rsi_divergence, bollinger_squeeze, volume_profile, volume_surge
**ETF 전용**: regime_switch, sector_rotation
**최근 추가** (2026-04-07): cross_sectional_momentum, quality_factor, pead_drift

### Signal Combiner
- **Group consensus**: trend / mean_reversion 그룹 내 합의 시 가중치 부스트, 불일치 시 패널티
- **Quadratic conviction weighting**: 강한 신호 (conf 0.9)가 약한 신호 (0.45) 대비 4배 영향
- **Mode B**: HOLD를 분모에서 제외, min_active_ratio 0.15
- **Held-position bias**: 보유 종목엔 SELL 임계 완화 (held_sell_bias 0.05)

### Market regime별 weight 프로파일
`profiles:` 섹션에서 `strong_uptrend / uptrend / sideways / weak_downtrend / downtrend / etf_engine` 별로 다른 weight 분포 정의. `adaptive_weights.py`가 종목 카테고리(growth_momentum / stable_large_cap / cyclical_value / high_volatility)와 블렌딩.

### Market별 disabled_strategies
`markets.US.disabled_strategies` / `markets.KR.disabled_strategies` 로 시장 단위 강제 비활성. profile weight와 무관하게 필터링됩니다.

---

## 운영 정보

### 스케줄러 (총 31 tasks)
- US: 22 (evaluation_loop, daily_scan, intraday_hot_scan, sector_analysis, news_analysis, after_hours_scan, ml_factor_update, ...)
- KR: 7 (kr_evaluation_loop, kr_daily_scan, kr_news_analysis, kr_etf_evaluation, ...)
- 시스템: 2 (health_check, ws_lifecycle)

### KIS API 제약
- Rate limit: 20 req/sec (real), 5 req/sec (paper) — `services/rate_limiter.py`로 관리
- WebSocket: 세션당 41개 종목 구독 한도, market hours only, 최대 4시간 세션
- 토큰: OAuth2 자동 갱신 (Redis 캐싱)

### Risk 관리
- ATR 기반 종목별 동적 SL/TP (US 3-15%, KR 5-20%)
- Hard SL `-15%` (min_hold 4시간 우회)
- Profit protection `+25%` (전 strategy HOLD인데 큰 평가익 시 강제 청산)
- High-profit auto-sell `+25%`
- Trailing stop activation `+8%` / trail `-4%` + tiered tightening
- Whipsaw 가드: 7일 내 손실 매도 2회 시 재매수 차단
- Sell cooldown: 1일

### 알림
- Discord webhook (기본)
- 매수/매도/SL/TP, 일일 brief, 에러, 회복 이벤트 모두 전송
- 어댑터 패턴 — Telegram/Slack 추가 가능

---

## 데이터베이스

- **PostgreSQL** (운영) — coin 프로젝트 컨테이너 공유, DB명 `us_stock_trading`
- **aiosqlite in-memory** (테스트) — 격리
- **Alembic** 마이그레이션 (`backend/alembic/versions/`)
- **백업**: `deploy/db-backup.timer` 일일 로컬 + `deploy/db-backup-remote.timer` 주간 GitHub

테이블 (10+):
`accounts, trades, positions, portfolio_snapshots, strategy_states, signals, factor_scores, signal_quality, agent_contexts, watchlists, ...`

---

## 테스트

```bash
cd backend

# 전체 (병렬)
../venv/bin/python -m pytest -n auto

# 특정 모듈
../venv/bin/python -m pytest tests/test_strategies/

# 시나리오 테스트
../venv/bin/python -m pytest tests/scenarios/

# Coverage
../venv/bin/python -m pytest --cov=. --cov-report=term-missing
```

테스트 규칙 (CLAUDE.md):
- 모든 코드는 unit test 동반 (커버리지 90%+)
- 외부 API는 항상 mock (KIS, yfinance, Claude, Gemini, Finnhub)
- DB 테스트는 aiosqlite in-memory로 격리
- 시나리오 테스트는 `tests/scenarios/`

---

## MCP Server

Claude Desktop / Claude Code에서 시스템 상태를 직접 조회·제어할 수 있는 MCP server를 제공합니다 (`backend/mcp_server.py`, FastMCP, 28 tools).

```jsonc
// .mcp.json (gitignored)
{
  "mcpServers": {
    "us-stock": {
      "command": "/home/chans/us-stock/venv/bin/python",
      "args": ["/home/chans/us-stock/backend/mcp_server.py"]
    }
  }
}
```

주요 tools:
- `get_portfolio_summary`, `get_positions`, `get_trade_history`, `get_market_state`
- `run_backtest`, `run_pipeline_backtest`, `get_backtest_results`
- `start_engine`, `stop_engine`, `run_evaluation`, `reload_strategies`
- `scan_stocks`, `discover_universe`, `get_factor_scores`, `get_signal_quality`

`reload_strategies`로 `strategies.yaml` 변경을 무중단 반영할 수 있습니다.

---

## 추가 문서

- **[CLAUDE.md](CLAUDE.md)** — 코드베이스 규칙 + AI agent 지침
- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** — 상세 아키텍처, 모듈 책임, 설계 결정
- **[WORKFLOW.md](WORKFLOW.md)** — 개발 워크플로 (브랜치 전략, PR 리뷰 등)

---

## 알려진 이슈 / TODO

- [ ] **백테스트 KR currency 미스매치** — `backend/backtest/full_pipeline.py:1464`. KR 시장 백테스트가 default `initial_equity=100_000` (USD) 단위를 KRW 가격에 그대로 적용해 quantity=0 → 0 trades. 회피책: KR 백테스트는 `initial_equity=100_000_000` 명시
- [ ] **백테스트 universe alignment** — `DEFAULT_UNIVERSE` (대형주 55개)와 라이브 dynamic_universe (small-cap 포함) 불일치
- [ ] **Stale backtest results** — `data/backtest_results/*.json`의 4-07 09:37 (commit `ff6279f`) 이전 결과는 look-ahead bias / Kelly param / asymmetric slippage 픽스 이전이므로 무효
- [ ] **백테스트 → 라이브 state 주입** — signal_quality 누적, gating 등 라이브 상태가 백테스트 시작 시 빈 상태
- [ ] **Strategy slim down** — alpha 음수 구조 문제 (자세한 내용은 SYSTEM_DESIGN.md 또는 commit 로그 참조)

---

## 라이선스

Private project. Not licensed for redistribution.

---

## 참고

- 본 프로젝트는 **~/coin** (BTC/ETH 등 암호화폐 자동매매 봇) 코드베이스의 아키텍처를 그대로 계승했습니다.
- DB/Redis 인프라를 coin 프로젝트와 공유하며 (PostgreSQL 컨테이너 / Redis db 1) 포트는 분리되어 있습니다 (us-stock 8001/3001 ↔ coin 8000/3000).
