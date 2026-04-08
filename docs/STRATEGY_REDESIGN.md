# Strategy Layer Redesign — 전략 자체 재정의

**작성일**: 2026-04-09  
**상태**: 검토 중 (실행은 IMPROVEMENT_PLAN.md Phase 3+ 참조)  
**목적**: 현재 17-strategy 시스템이 사용자 thesis와 misaligned 됐다는 진단과, "burst-catcher native" 시스템으로 가는 비전을 기록

> 이 문서는 *왜 재정의가 필요한가* 와 *어떤 모양으로 가야 하는가* 에 대한 것.  
> *어떻게 단계적으로 가는가* 는 [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) 참조.

---

## 1. 사용자 thesis (what we're actually trying to do)

> "개별 종목 잘 튈때 올라타서 잘 먹는다"

이게 핵심이고, 다음 함의들이 따라옴:

| 함의 | 의미 |
|---|---|
| **개별 종목** | sector ETF rotation 아님. small/mid cap 포함 |
| **잘 튈때** | volatility burst, breakout, gap-up, post-earnings drift, 거래량 서지 |
| **올라타서** | 진입 시 이미 모멘텀 시작됨. trend가 형성되기 전 진입 아님 |
| **잘 먹는다** | 단순 +1~2% 익절 아님. burst의 큰 무빙(+5~15%)을 noticeable하게 가져감 |

이 thesis는 잘 알려진 retail/discretionary momentum trader 스타일과 가까움. 학술적으론:
- Jegadeesh-Titman 단기 모멘텀 (1-3개월)
- Post-earnings announcement drift (PEAD)
- Range expansion / volatility breakout (Larry Williams 스타일)
- Gap-and-go (intraday)
- Relative strength (Stan Weinstein)

**Long-horizon momentum** (12-1, dual_momentum 식)이나 **mean reversion** (BNF 이격도, RSI divergence)과는 시간 horizon과 risk profile이 다르다.

---

## 2. 현재 17-strategy의 문제 — Thesis Mismatch Map

### 17개 전략을 thesis 적합도로 분류

| 카테고리 | 전략 | thesis 적합도 | 비고 |
|---|---|---|---|
| **Burst catcher** | supertrend | ★★★★ | 라이브 best performer (8/3 win, +1.41% avg) |
| **Burst catcher** | volume_surge | ★★★ | 거래량 서지 감지 — 정확히 thesis의 그 burst |
| **Burst catcher** | pead_drift | ★★★ | post-earnings 모멘텀 — 정확히 적합 |
| **Burst catcher** | donchian_breakout | ★★ | 채널 돌파 — 적합하지만 backtest에선 손실 (US disabled) |
| Long-horizon momentum | dual_momentum | ★ | 12-18개월 lookback. thesis와 시간 horizon 안 맞음. **라이브 bleeder**, disable 예정 |
| Long-horizon momentum | trend_following | ★★ | EMA20/50 cross. burst보다 느림 |
| Long-horizon momentum | cis_momentum | ★ | 이미 disabled |
| Cross-sectional | cross_sectional_momentum | ★ | 12-1 ranking. thesis와 horizon 안 맞음. backtest에서 거의 안 fire |
| Mean reversion | rsi_divergence | ★ | 평균 회귀, burst와 정반대 |
| Mean reversion | bollinger_squeeze | ★★ | squeeze breakout — 부분적으로 적합 |
| Mean reversion | bnf_deviation | ★ | 이미 disabled |
| Mean reversion | larry_williams | ★ | Williams %R, 이미 disabled |
| Mean reversion | macd_histogram | ★★ | divergence 기반, burst와는 결이 다름 |
| Volume / Other | volume_profile | ★★ | 거래량 프로파일, 보조적 |
| Quality factor | quality_factor | ★ | ROE/margin top decile — burst와 무관, 이미 disabled (KR) |
| Sector | sector_rotation | – | sector ETF용. 개별주 thesis와 무관 |
| Regime | regime_switch | – | 레버리지 ETF용 |

→ **17개 중 burst catcher 4개. 13개는 다른 thesis용 또는 보조.**

### Combiner가 burst signal을 희석하는 메커니즘

`backend/strategies/combiner.py`의 weighted vote:
- buy_score = Σ (confidence² × weight) — only BUY signals
- buy_norm = buy_score / active_weight
- min_confidence = 0.50 → 미달이면 HOLD

문제: supertrend 1개가 0.85 conviction BUY 내도, **trend_following이 0.50 SELL 내면** active_weight 분모가 2배가 되어 buy_norm이 0.36으로 떨어짐. min_confidence 미달 → HOLD. (이게 KR debug에서 봤던 SK텔레콤 케이스)

**근본 원인**: 4개 burst catcher가 13개 다른 전략의 노이즈에 묻힌다. 슬림다운이 답.

### Global exit thresholds가 strategy-style을 무시함

현재 글로벌 (config/strategies.yaml `global` block):
- `profit_taking.threshold_pct: 0.06` — 6% 도달 시 부분 익절
- `trailing_stop_defaults.activation_pct: 0.08, trail_pct: 0.04` — 8% 후 발동, 4% 풀백 청산
- `hard_sl_pct: -0.15` — 15% 손실 hard stop
- `min_hold_secs: 4시간` — 매수 후 4시간 강제 보유
- `stale_pnl_threshold: -0.05` — -5% + 모든 전략 HOLD면 position_cleanup

이 thresholds는 모든 전략에 동일 적용된다. 그러나:
- **burst catcher**는 5-7% 빠른 무빙을 잡는 게 목표 → 8% activation은 너무 늦음
- **long trend**는 20-30% 장기 무빙 → 8% activation OK
- **mean reversion**은 평균 회귀 시점 익절 → 6% threshold OK이지만 trail 필요 없음

→ 한 set의 thresholds로 다른 thesis를 동시에 다루려는 게 근본 한계.

### Strategy yaml `stop_loss: type: ...` 가 dead code

각 전략 yaml에 `stop_loss.type` config가 있음 (e.g. supertrend는 `type: supertrend`, fixed_pct는 `type: fixed_pct max_pct: 0.05`). 그런데:
- `config_loader.get_stop_loss_config()`는 정의돼 있음
- **호출되는 곳: 0건** (grep 결과)
- 실제 라이브 SL: `evaluation_loop.py:1612`에서 ATR 또는 `RiskParams.default_stop_loss_pct`만 사용

→ 17개 전략 모두 의도된 SL이 아닌 generic SL로 운영 중. **이게 펄어비스 -7.19%, 삼성전기 -6.72%의 직접 원인**. 둘 다 supertrend가 매수했고 supertrend의 tight line-based SL이 작동했어야 했는데 ATR-based SL이 너무 넓어서 안 발동, 결국 position_cleanup -5% 안전망에 잡힘.

---

## 3. Burst Catcher Native 시스템 — 비전

### 3.1 핵심 원칙

| 원칙 | 현재 vs 목표 |
|---|---|
| **전략 수** | 17 → **4-5개** (burst-specific) |
| **Combiner** | weighted vote (희석) → **best-signal-wins** 또는 weighted vote on 4개 (희석 적음) |
| **Exit lifecycle** | global thresholds → **per-strategy** |
| **min_hold** | 4h global → **strategy별** (burst 1h, swing 4h) |
| **Discovery** | UniverseExpander 매일 1회 → **intraday hot scan 5-min interval** |
| **Position sizing** | Kelly (장기 누적) → **risk-based** (entry-stop = risk per share, account×1.5% per trade) |
| **시간 관리** | 보유 무한 → **time-based exit** (3-5일 무빙 없으면 청산) |

### 3.2 새 전략 메뉴 (예시)

**Active core (4개)**:

1. **`supertrend`** (개량): 현재 유지 + per-strategy lifecycle 적용
   - SL: supertrend 라인 기반 (line cross 시 즉시 청산)
   - TP: 3R (risk × 3) 또는 supertrend 라인 반대 cross
   - Trailing: ATR 1.5x trailing
   - Time stop: 5일 무빙 < 2%면 청산
   - Min hold: 1시간

2. **`gap_and_go`** (신규):
   - Entry: 갭 상승 +3~8% + 첫 봉 high 돌파 + 거래량 평균 2x↑
   - SL: 갭 메우기 (entry low 또는 -3%)
   - TP: 1차 +5% (1/2), 2차 +10% (1/2 trail)
   - Time stop: 1일 (intraday only)
   - Min hold: 0 (intraday)

3. **`volume_breakout`** (volume_surge fork):
   - Entry: 거래량 3x + 5일 high 돌파 + EMA20 위
   - SL: -3% 또는 EMA20 하향 cross
   - TP: trailing +5% activation, 2% trail
   - Time stop: 3일
   - Min hold: 1시간

4. **`pead_drift`** (현재, trigger 완화):
   - Entry: earnings 발표 후 +3% gap + 거래량 2x + 5일 drift window
   - SL: -4%
   - TP: trailing +6% activation, 3% trail
   - Time stop: 5일
   - Min hold: 2시간 (overnight 가능)

**Sector strategies (보조, ETF Engine 전용 — 개별주와 분리)**:
- `sector_rotation`, `regime_switch` — ETF Engine으로 운영, 개별주 portfolio와 capital 분리

**Disabled / 삭제**:
- 13개 (dual_momentum 포함). long-horizon momentum + mean reversion + quality factor 전부

### 3.3 Combiner 변경

**Best signal wins** 모드 추가:
```python
def combine_best_wins(signals: list[Signal], weights: dict) -> Signal:
    """가장 높은 (weighted_conf) BUY가 결정. SELL은 합산."""
    buy_signals = [s for s in signals if s.signal_type == BUY]
    if buy_signals:
        best = max(buy_signals, key=lambda s: s.confidence ** 2 * weights[s.strategy_name])
        if best.confidence >= min_confidence:
            return Signal(BUY, best.confidence, best.strategy_name)
    # SELL은 어느 strategy든 발동되면 청산 (weighted vote)
    ...
```

→ 4개로 슬림다운 + best wins 조합이면 burst signal 희석 거의 없음

### 3.4 Per-strategy lifecycle YAML 스키마

```yaml
strategies:
  supertrend:
    enabled: true
    timeframe: "1D"
    params: { ... }
    lifecycle:
      sl:
        type: supertrend         # supertrend 라인 cross 시 즉시
        fallback_pct: 0.05       # supertrend 라인 못 가져오면 5%
      tp:
        type: ratio
        risk_multiple: 3.0       # 3R
      trailing:
        activation_pct: 0.05
        trail_pct: 0.025
      time_stop:
        days: 5
        min_pnl_pct: 0.02
      min_hold_secs: 3600        # 1h (burst tolerant)
      sizing:
        type: risk_based
        account_risk: 0.015      # 1.5% account per trade
```

이렇게 되면:
- 글로벌 threshold (`profit_taking`, `trailing_stop_defaults`, `min_hold_secs`)가 fallback만 됨
- 전략별로 자신에 맞는 exit/sizing
- yaml hot-reload로 운영 중 조정 가능

### 3.5 Discovery pipeline 변경

현재:
- `daily_scan` (1일 1회): UniverseExpander → 80 symbols
- `intraday_hot_scan` (30분 1회): yfinance day_gainers → watchlist 보강

목표:
- `daily_scan` 유지 (전체 universe baseline)
- `intraday_hot_scan` 30분 → **5분** (KIS rate limit 검증 필요. 라이브 1일 분량 = 78개 호출, 가능 범위)
- 새 metric: **relative volume** (현재 거래량 / 평균 거래량) > 2x인 종목만 watchlist 즉시 추가
- 새 metric: **opening range breakout** (개장 30분 high 돌파)

### 3.6 Position sizing 변경

현재 Kelly-based:
- win_rate, avg_win, avg_loss 누적 → Kelly 분수 → max position pct
- 누적 데이터 없으면 fallback default
- 문제: burst catcher는 large winner / small loser 분포라 Kelly가 underestimate

목표 Risk-based:
- 1.5% account risk per trade
- entry price - stop price = risk per share
- quantity = (account × 0.015) / risk_per_share
- 예: $100k 계좌, entry $50, stop $48 (risk $2/share) → 750 shares = $37.5k position
- max position pct cap (e.g. 25%) 유지

이렇게 되면:
- 작은 stop = 큰 position
- 큰 stop = 작은 position
- 모든 trade의 max loss가 동일 (1.5%)
- burst catcher의 large winner을 활용 가능

---

## 4. 절대 안 할 것 (rejected options)

| 옵션 | 거부 이유 |
|---|---|
| **Sector ETF rotation 만 (Option A)** | 사용자 명시: "개별 종목 잘 튈때 올라타는게 thesis". sector ETF는 thesis 정반대 |
| **Quality momentum 3개 (Option B)** | 백테스트 Sharpe 0.22, dual_momentum/supertrend/trend_following 조합인데 dual_momentum이 bleeder. supertrend 단독이 더 나음 |
| **Mean reversion 3개 (Option C)** | 백테스트 Sharpe 1.27이지만 거의 거래 안 함 (45 trades), cash parking이 96% PnL. 사실상 passive |
| **인프라 (KIS/scheduler/MCP) 손대기** | 멀쩡함, 전략 레이어가 문제 |
| **백테스트로만 burst catcher 검증** | daily-bar 백테스트는 동적 discovery 못 replicate. 라이브 forward test 필수 |
| **17개 전략 그대로 두고 weight만 조정** | 잡탕이 근본 문제. 슬림다운 안 하면 항상 노이즈에 묻힘 |
| **새 전략 추가 (먼저)** | 17개도 너무 많음. 슬림다운 + lifecycle 확립 후에만 추가 |

---

## 5. 검증 — Burst catcher가 돈 되는지 어떻게 아나

**문제**: 백테스트로는 검증 불가 (universe alignment 한계). 그럼 어떻게?

### 5.1 Live forward test (정식)
- Phase 1 적용 후 1주일 → 패턴 분석 재실행
- Phase 2 적용 후 2주일 → 패턴 분석 재실행
- Phase 3 적용 후 4주일 → 패턴 분석 재실행
- 매번:
  - SMALL_WIN_BIG_LOSS 비중 12.8% → 5% 미만?
  - GAVE_BACK 비중 20.5% → 10% 미만?
  - CAUGHT_BURST_OK 비중 12.8% → 25% 이상?
  - 평균 PnL/trade 향상?

### 5.2 Trade journal (보조)
- 매주 거래 이력 다운로드 → 수동 라벨링 ("이 burst를 놓쳤나? 너무 일찍 익절했나?")
- 누적 데이터로 burst 패턴 그라운드 트루스 구축
- 6개월쯤 쌓이면 ML factor selector에 학습 데이터로 활용 가능

### 5.3 Intraday backtest (제한적)
- yfinance 1분/5분 봉 60일 한정 → 2개월 backtest
- gap_and_go 같은 intraday 전략 검증 가능
- 단점: 60일은 너무 짧음, 시장 regime 1개만 커버

### 5.4 Paper trade fork (Phase 4 후 검토)
- 새 전략을 paper account에서 30일 운용
- 라이브 vs paper 결과 비교
- 라이브 적용 결정

---

## 6. 시간 horizon

| Phase | 기간 | 무엇 | 위험 |
|---|---|---|---|
| Phase 0 | 완료 | 백테스트 신뢰도 회복, donchian US disable, KR clean, cash parking | 완료 |
| **Phase 1** | **이번 주 (1-2일)** | **SL 버그 픽스 + dual_momentum disable** | **낮음 (yaml + 한 코드 픽스)** |
| Phase 2 | 1-2주 | trailing 재조정, etf_engine_sector 조정 | 낮음 (yaml only) |
| Phase 3 | 2-4주 | per-strategy lifecycle 인프라, time-based exit, combiner best-wins 모드 | 중간 (코드 다수) |
| Phase 4 | 1-2개월 | gap_and_go, volume_breakout 새 전략 작성 + paper test | 큼 (새 코드, 검증) |
| Phase 5 | 3개월+ | discovery 강화 (5분 hot scan, real-time relative volume) | 큼 (KIS rate limit, real-time infra) |

---

## 7. 결정 framework (미래 자기 자신/사용자에게)

새 변경 검토 시 매번 물을 질문:

1. **Thesis와 align 되는가?** "개별 종목 burst 잡기"에 도움 되나, 방해 되나, 무관한가?
2. **검증 가능한가?** 백테스트로 OK인가, 라이브 forward test 필요한가?
3. **회귀 위험은?** 라이브에 어떤 영향, 어떻게 롤백 가능한가?
4. **단순화 효과는?** 시스템을 복잡하게 만드나, 단순하게 만드나?
5. **데이터로 뒷받침되나?** 추측인가, 실제 거래/백테스트 결과 인용 가능한가?

이 5개 모두 통과 못 하면 **기각** 또는 **데이터 수집 후 재검토**.

---

## 8. 참조

- **이 문서**: `docs/STRATEGY_REDESIGN.md`
- **실행 계획**: `docs/IMPROVEMENT_PLAN.md`
- **분석 스크립트**: `backend/scripts/analyze_trade_patterns.py`
- **거부된 옵션 비교**: `backend/scripts/compare_thesis.py` (출력은 commit 메시지에 있음)
- **시스템 설계 (general)**: `SYSTEM_DESIGN.md`
- **AI 지침**: `CLAUDE.md`
