# Strategy Layer Improvement Plan

**작성일**: 2026-04-09  
**상태**: Phase 1 진행 중  
**근거**: 9일치 라이브 거래 데이터 패턴 분석 + 백테스트 신뢰도 회복 작업 결과

---

## 0. 배경

사용자 thesis: **"개별 종목 burst를 잡아 올라타서 잘 먹는다"**.  
2026-04-09 사용자 피드백 — "1% 수익은 조ㅇ은 장에 너무 적다, 시스템 자체를 재정의해야 하지 않나".

기존 17-strategy 시스템은 thesis와 misaligned:
- 전략 메뉴는 long-horizon momentum + mean reversion + sector rotation + quality factor 등 잡탕
- Combiner가 burst signal을 다른 13개의 HOLD로 희석
- Global exit thresholds (6% profit_taking, 8% trailing activation)는 burst-style과 long-trend를 구분 못함
- min_hold 4시간이 burst의 빠른 reversal에 노출
- Strategy yaml의 `stop_loss: type: ...` config가 dead code (호출되는 곳 없음, ATR/default만 사용)

---

## 1. 진단 — 9일치 라이브 거래 패턴 (39 round-trips)

**스크립트**: `backend/scripts/analyze_trade_patterns.py`  
**기간**: 2026-03-31 ~ 2026-04-08

### 패턴 분포
| 패턴 | 건수 | 비중 | 합산 PnL% | 평균 |
|---|---:|---:|---:|---:|
| **CAUGHT_BURST_OK** ✓ | 5 | 12.8% | **+31.63%** | **+6.33%** |
| **EARLY_EXIT** (a) | 4 | 10.3% | +9.84% | +2.46% |
| **GAVE_BACK** (b) | 8 | 20.5% | -0.25% | -0.03% |
| **SMALL_WIN_BIG_LOSS** (c) | 5 | 12.8% | **−39.68%** | **−7.94%** |
| FLAT_NEVER_MOVED | 7 | 17.9% | -3.36% | -0.48% |
| OTHER | 9 | 23.1% | +4.19% | +0.47% |
| CLEAN_LOSS | 1 | 2.6% | -4.17% | -4.17% |

→ 17건이 손실/누락 패턴 (a/b/c). 5건만 깨끗한 익절. **5건의 SMALL_WIN_BIG_LOSS가 합산 −39.68%**로 전체 출혈의 핵심.

### 전략별 기여
| 전략 | 거래 | W/L | 평균 | Best | Worst |
|---|---:|---|---:|---:|---:|
| **supertrend** | 11 | **8/3** | **+1.41%** | **+10.43%** (BW) | −7.19% (펄어비스, SL 버그) |
| etf_engine_sector | 18 | 10/8 | +0.31% | +6.66% | -0.47% |
| **dual_momentum** | 10 | **3/7** | **−2.28%** | +3.75% | **−10.22%** (FANG) |

### 핵심 발견
1. **(c) 원인 = SL 버그 + dual_momentum SL 광범위**
   - **SL 버그** (중대): `evaluation_loop.py:1612~1619`가 ATR 또는 `RiskParams.default_stop_loss_pct`만 사용. strategy yaml의 `stop_loss: type: supertrend / fixed_pct / atr` 블록은 절대 안 읽힘. `config_loader.get_stop_loss_config()`는 정의돼있지만 호출 0건.
   - 펄어비스/삼성전기 (둘 다 supertrend 매수, position_cleanup −7%/−6.7% 청산) — supertrend 라인 기반 tight SL이 작동했어야 함
   - dual_momentum SL은 발동했지만 −7~−10%로 너무 늦음 (FANG/CVE/XOM 케이스)
2. **(b) 원인 = trailing activation 8%가 너무 높음**
   - 5~7% 무빙 8건이 trailing 발동 못 하고 그대로 회귀
3. **(a) 원인 = trailing trail 4%가 너무 좁음**
   - 발동 후 4% 풀백이면 즉시 청산. MFE의 절반만 실현
4. **dual_momentum = 명백한 bleeder** — 라이브 10/7 손실, backtest −$3,865, avg −2.28%
5. **supertrend = 라이브 best performer** — 끄지 말 것, SL 버그 픽스 필요
6. **etf_engine_sector = noise** — 18건 대부분 flat 또는 give-back, trailing 타이트닝 또는 disable 검토

---

## 2. 실행 계획

### Phase 1 — 즉시 (yaml + 한 코드 픽스)

**1.1 SL config 버그 픽스 (코드)** ← **가장 임팩트 큼**
- `evaluation_loop.py:1612` 분기 추가:
  - strategy의 `stop_loss.type` config 우선 적용
  - `fixed_pct` → `max_pct` 사용
  - `atr` → 현재 ATR-based 로직
  - `supertrend` → supertrend 라인 기반 (현재 indicator에서 가져오기)
  - 명시 없으면 ATR fallback
- 회귀 위험: 일부 전략의 SL이 좁아짐 → 단기적으로 SMALL_WIN_BIG_LOSS 줄지만 EARLY_STOP 늘 수 있음
- 검증: 펄어비스 케이스 재현 unit test, 기존 SL/TP 테스트 회귀 확인
- 라이브 영향: 다음 evaluation 사이클부터 적용 (백엔드 재시작 필요 — 코드 변경)

**1.2 dual_momentum US disable (yaml)** ← **즉시 출혈 멈춤**
- `config/strategies.yaml` `markets.US.disabled_strategies`에 `dual_momentum` 추가
- 근거: 라이브 10건 7손실 평균 -2.28% + backtest -$3,865 + 3건의 SMALL_WIN_BIG_LOSS (FANG, CVE, XOM)
- 라이브 영향: 다음 evaluation 사이클부터 (hot-reload 가능)
- 위험: 라이브 거래량 큰 폭 감소 (dual_momentum이 trade volume의 큰 비중)

### Phase 2 — 1주일 모니터링 후 (yaml)

**2.1 trailing stop 재조정**
- 현재 글로벌 `activation_pct: 0.08, trail_pct: 0.04`
- 안 1: `0.05 / 0.025` (GAVE_BACK 줄이기)
- 안 2: per-strategy YAML 추가 (etf_engine_sector는 1.5%/1%, supertrend는 8%/4% 유지)
- 검증 기준: 1주일 후 GAVE_BACK 비중 20.5% → 10% 미만 + EARLY_EXIT 비중 10.3% → 15% 이내 유지

**2.2 etf_engine_sector trailing 별도 또는 disable 검토**
- 18건 대부분 noise. trailing 타이트하게 1.5%/1% 또는 strategy 자체 disable
- 결정 근거: Phase 1 적용 후 1주일 데이터 재분석

### Phase 3 — 2~4주 (코드, 별도 PR)

**3.1 Per-strategy lifecycle 인프라**
- 각 전략이 own SL/TP/trailing/time-stop config 갖음
- yaml 스키마 확장:
  ```yaml
  strategies:
    supertrend:
      lifecycle:
        sl: { type: supertrend, fallback_pct: 0.05 }
        tp: { type: ratio, ratio: 3.0 }      # 3x risk
        trailing: { activation: 0.05, trail: 0.025 }
        time_stop: { days: 5, min_pnl: 0.02 }
  ```
- evaluation_loop와 position_tracker가 lifecycle 객체를 직접 사용
- min_hold 4시간 → strategy별 (burst 1시간, 추세 4시간)

**3.2 Time-based exit**
- 매 evaluation 사이클마다 "보유 N일 + 평가익 < threshold면 sell" 룰
- GAVE_BACK 패턴 감소 효과 기대

**3.3 Combiner: best-signal-wins 모드 (옵션)**
- 17 weighted vote 대신 "가장 conviction 높은 1개가 결정"
- 4-5개로 슬림다운한 후엔 weighted vote로도 충분할 수 있음 — 슬림다운 후 재평가

### Phase 4 — 선택 (1개월~)

**4.1 Discovery 강화**
- `intraday_hot_scan` interval 30분 → 5분 (KIS rate limit 검증 필요)
- Real-time relative volume metric 추가
- 섹터 강세 시 그 섹터 종목 우선 watchlist

**4.2 새 burst-specific 전략 작성** (별도 PR씩)
- `gap_and_go.py`: 갭 상승 + 첫 봉 돌파 + 거래량 서지
- `volume_breakout.py` (volume_surge fork): 평균 거래량 3x + 5d high
- 각 전략 자체 SL/TP/trailing 설정

---

## 3. 검증 방법

### 라이브 모니터링 지표 (매주)
1. **패턴 분포**: `analyze_trade_patterns.py` 주 1회 실행
   - SMALL_WIN_BIG_LOSS 비중 12.8% → 5% 미만 목표
   - GAVE_BACK 비중 20.5% → 10% 미만 목표
   - CAUGHT_BURST_OK 비중 12.8% → 25% 이상 목표
2. **전략별 PnL**: dual_momentum 음수 잔존 여부, supertrend +1.41% → +3% 이상 향상 여부
3. **펄어비스류 재발생**: 매수 전략의 SL이 발동하지 않고 position_cleanup이 −5% 이하에서 fire되는 케이스 0건

### 백테스트 (보조)
- 백테스트는 universe alignment 한계로 절대값 신뢰 어려움
- **상대 비교만** 의미: "Phase 1 변경 적용 시 baseline 대비 Sharpe 향상"
- `validate_*.py` 스크립트들은 코드 변경 회귀 체크 용도

### 절대 하지 말 것
- **백테스트 단독으로 burst-catcher 전략 결정** — daily-bar 백테스트는 dynamic discovery 못 잡음
- **새 strategy 추가 (이미 17개 너무 많음)** — Phase 4의 새 burst 전략은 기존 무용 전략 disable과 동시에만
- **인프라 (KIS API, scheduler, MCP, DB) 손대기** — 멀쩡함, 시간 낭비

---

## 4. 이전에 적용된 변경 (Phase 0)

이미 끝났고 라이브 반영 완료 — **그대로 유지**:

| commit | 내용 | 상태 |
|---|---|---|
| `d1b6cc3` | 백테스트 신뢰도 5종 픽스 (KR currency, stale tagging, universe alignment, signal_quality seed, strategies.yaml stale 주석 정리) | merged |
| `90b3897` | Cash parking US 활성화 (validate_cash_parking.py V1: +13.3pp / +0.97 sharpe) | merged, **재시작 시 라이브 적용** |
| (yaml only) | US donchian_breakout disable | merged + reload_strategies로 라이브 적용 |
| (yaml only) | KR cross_sectional_momentum / pead_drift / quality_factor disable | merged + reload_strategies |
| `8ee3b50` | 프로젝트 README.md | merged |

---

### Phase 2.5 — 수수료 반영 (Commission-aware trading)

**2026-04-13 추가**: 시스템 전체가 거래 수수료를 **완전히 무시**하고 있음.
- `backtest/full_pipeline.py`: `commission_per_order: 0.0` (기본값)
- `engine/evaluation_loop.py`: 수수료 체크 코드 0건
- `engine/order_manager.py`: 수수료 언급 0건
- `engine/risk_manager.py`: 수수료 언급 0건
- `engine/position_tracker.py` PnL 계산: 수수료 미차감

KIS US 수수료 0.25%/건 → round trip 0.50%. **수익 0.50% 이하 거래는 전부 실질 손해**.
이 맹점이 cash_parking churn 93회 (~860k KRW 수수료)의 근본 원인 중 하나.

**필요 작업:**
1. **매수 판단 시 minimum profit threshold**: expected PnL > round-trip commission (0.50%) 이상일 때만 BUY
2. **PnL 실현 시 수수료 차감**: position_tracker/order_manager에서 체결가 × 수수료율 차감
3. **백테스트 commission 반영**: `PipelineConfig.commission_per_order`를 실 수수료로 세팅 (US $5~7/order, KR 매도세 포함 ~0.3%)
4. **cash_parking 등 시스템 거래에 수수료 guard**: 예상 수익 < 수수료면 거래 차단

**배포 규칙 추가 (2026-04-13):**
- **라이브 배포 전 최소 1일 페이퍼 운용** 또는 **라이브에서 첫 1시간 수동 모니터링** 필수
- 백테스트 OK만으로 라이브 즉시 배포 금지 (cash_parking 사건 재발 방지)

---

## 5. 알려진 한계 / 미해결

1. **백테스트 universe ≠ 라이브 universe** — `WIDE_UNIVERSE` 추가했지만 동적 discovery 못 replicate. 절대 alpha 추정 부정확
2. **데이터 9일치만** — 작은 표본. Phase 1 적용 후 4주 누적 후 재분석 필요
3. **SL 버그 픽스 회귀 위험** — 일부 strategy의 SL이 갑자기 좁아져서 EARLY_STOP 패턴 새로 생길 수 있음
4. **dual_momentum disable 후 거래량 급감** — 의도된 변화지만 사용자 perception 주의
5. **etf_engine_sector** — 18건 noise지만 KR ETF rotation 자체는 의미 있음. trailing 조정으로 살릴지 disable할지 1주일 모니터링 후 결정
6. **min_hold 4시간** — burst trading에 너무 김. Phase 3 per-strategy lifecycle에서 해소
7. **수수료 무시** — 시스템 전체가 commission-blind. Phase 2.5에서 해결 예정 (위 참조)
8. **cash_parking churn** (2026-04-09~10) — 93회 SPY round-trip으로 ~860k KRW 수수료 발생. 4-11 rewrite로 park-once-hold로 전환 + 1시간 cooldown 추가. 추가 재발 시 cash_parking 자체를 삭제

---

## 6. 결정 history (왜 이렇게 했는지)

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-04-08 | A (sector ETF rotation thesis) 폐기 | 사용자가 "개별 종목 burst 잡는 게 thesis"라고 명시 |
| 2026-04-08 | B (quality momentum 3개) 폐기 | 백테스트 Sharpe 0.22로 worst |
| 2026-04-08 | C (mean reversion) 부분 채택 | C 자체는 거의 작동 안 함 (단일 trade), cash parking 효과만 큼 |
| 2026-04-09 | dual_momentum disable 결정 | 백테스트 + 라이브 데이터 모두 음수 |
| 2026-04-09 | supertrend 유지 결정 | 라이브 8/3 win, best +10.43%, 최악 케이스는 SL 버그 |
| 2026-04-09 | SL 버그 픽스가 1번 우선순위 | 5건 SMALL_WIN_BIG_LOSS 중 2건 직접 원인 (펄어비스, 삼성전기), 나머지 3건도 비슷한 메커니즘 (SL 너무 넓음) |
| 2026-04-09 | trailing 조정은 Phase 2로 미룸 | EARLY_EXIT vs GAVE_BACK trade-off — Phase 1 적용 후 데이터 재수집 후 결정 |

---

## 7. 다음 액션 (체크리스트)

### Phase 1 (오늘)
- [ ] **1.1** SL config 버그 픽스 (코드 + 테스트)
  - `evaluation_loop.py` SL 분기 추가
  - `position_tracker.py` 필요 시 supertrend type 지원
  - 단위 테스트: type별 SL 적용 검증
  - 회귀 테스트: 기존 SL/TP 동작 보존
- [ ] **1.2** dual_momentum US disable (yaml)
  - `markets.US.disabled_strategies`에 추가
  - 변경 사유 주석 (라이브 데이터 인용)
- [ ] **1.3** commit + push
- [ ] **1.4** `reload_strategies` MCP 호출 (yaml 부분만 즉시 라이브 반영)
- [ ] **1.5** 백엔드 재시작 (코드 변경 적용 위해) — **사용자 승인 필요**

### Phase 2 (Phase 1 후 1주일)
- [ ] 패턴 분석 재실행 (`analyze_trade_patterns.py`)
- [ ] SMALL_WIN_BIG_LOSS / GAVE_BACK / EARLY_EXIT 비중 변화 확인
- [ ] trailing stop 조정 안 1 vs 안 2 백테스트 비교
- [ ] yaml 적용 + reload

### Phase 3 (2~4주 후)
- [ ] per-strategy lifecycle YAML 스키마 설계
- [ ] 코드 작성 (PR 별도)
- [ ] time-based exit 룰 추가

---

## 8. 참조

- **분석 스크립트**: `backend/scripts/analyze_trade_patterns.py`
- **백테스트 검증 스크립트**: `backend/scripts/validate_cash_parking.py`, `validate_new_strategy.py`, `verify_donchian_disable.py`, `verify_kr_no_trend_following.py`, `compare_strategy_count.py`, `compare_thesis.py`
- **거래 이력 데이터**: MCP `get_trade_history` (US + KR, 150 trades 한도)
- **백테스트 인프라 픽스**: commit `d1b6cc3` (참조 필수, 그 이전 백테스트 결과는 stale)
- **이 문서**: `docs/IMPROVEMENT_PLAN.md`
