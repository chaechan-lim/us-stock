# DART API 키 발급 + 활성화 가이드

KR DART (전자공시시스템) insider 신호를 활성화하려면 무료 API 키가 필요합니다.

## 1. 회원 가입 + API 키 신청

1. https://opendart.fss.or.kr 접속
2. 우상단 **로그인 → 회원가입** (개인)
   - 본인인증 (휴대폰 / 아이핀)
   - 이메일 + 비밀번호 + 소속/직책 입력
3. 로그인 후 **인증키 신청/관리** 메뉴 (좌측)
4. **인증키 신청** 클릭
   - 사용 목적: "개인 투자/연구"
   - 사용 시스템: "KR stock auto-trading system (개인 사용)"
   - 일일 호출 한도: 10,000회 기본 (충분)
5. **신청** → 보통 즉시 발급 (이메일 확인 필요할 수 있음)

발급된 키는 **인증키 신청/관리** 페이지에서 확인 가능. 32자 hex 문자열.

## 2. .env 파일 설정

```bash
# /home/chans/us-stock/.env 에 추가
DART_API_KEY=발급받은_32자_hex_키
DART_ENABLED=true
```

## 3. 회사코드 맵 1회 다운로드

DART는 종목코드(6자리)가 아닌 corp_code(8자리)로 색인됨. 매핑 파일 생성:

```bash
cd /home/chans/us-stock
source .env && set -a && export $(cat .env | xargs) && set +a
cd backend
python scripts/fetch_dart_corp_codes.py
```

성공 시 `data/dart_corp_map.json` 생성 (~3,000개 매핑).

분기마다 1회 재실행 권장 (IPO/상폐 반영).

## 4. main.py 통합 (코드 수정 1회)

`backend/main.py`에서 `EventCalendarService` 인스턴스화 부분 찾아 수정:

```python
# Before
from data.insider_service import InsiderTradingService
insider = InsiderTradingService(api_key=config.finnhub_api_key)
event_calendar = EventCalendarService(earnings, macro, insider)

# After
from data.insider_service import InsiderTradingService
from data.dart_service import DARTInsiderService

insider = InsiderTradingService(api_key=config.finnhub_api_key)
kr_insider = DARTInsiderService(
    api_key=os.getenv("DART_API_KEY"),
    enabled=os.getenv("DART_ENABLED", "false").lower() == "true",
)
event_calendar = EventCalendarService(
    earnings, macro, insider, kr_insider=kr_insider,
)
```

(`config/__init__.py`에 `dart_api_key` 필드 추가 + AppConfig에 매핑하면 더 깔끔. 현재는 직접 env 호출도 OK.)

## 5. 새로고침 task 추가 (scheduler)

`main.py` scheduler 부분에 daily refresh 추가:

```python
async def task_dart_refresh():
    if not event_calendar.kr_insider or not event_calendar.kr_insider.enabled:
        return
    # Get KR watchlist symbols
    syms = [w.symbol for w in app.state.kr_watchlist if w.symbol.isdigit()]
    await event_calendar.kr_insider.refresh(syms)
    logger.info("DART refresh: %d KR symbols", len(syms))

scheduler.add_task("dart_insider_refresh", task_dart_refresh,
                   interval_sec=86400, phases=[PRE_MARKET, REGULAR])
```

## 6. 백엔드 재시작 + 확인

```bash
sudo systemctl restart usstock-backend.service
sleep 5
sudo journalctl -u usstock-backend.service --since "30s ago" | grep -i dart
```

기대 로그:
```
DART corp map loaded: 3142 entries
DART refresh: 41 KR symbols
```

## 7. 효과 측정 (1-2주)

활성화 후:
- 인사이더 매수 filing이 발생하면 해당 종목 BUY confidence +0.05~+0.10
- 매도 filing은 -0.05~-0.10
- 결과적으로 BUY signal 통과율 ↑ 또는 ↓ (filing에 따라)

대시보드 `/api/v1/events` 응답에 `kr_insider` 섹션 추가됨.

---

## 백테스트 활성화 (선택)

DART 키 받고 corp_map 다운로드 후, 12개월치 filing 일괄 다운로드해서 backtest에 주입:

```python
# scripts/backfill_dart_history.py (별도 구현 필요)
# for each (symbol, corp_code) in corp_map:
#     fetch filings 12개월
#     save to data/dart_history/{symbol}_{YYYYMMDD}.json
# 그 후 backtest 엔진의 confidence 계산에 historical filings 주입
```

별도 1-2h 작업. 키 발급 후 진행.

---

## 신청 → 발급 → 활성화 총 소요시간

- 등록 + 신청: 10분
- 키 발급: 즉시~수시간 (보통 즉시)
- corp_map 다운로드: 30초
- main.py 수정 + 재시작: 5분
- 효과 측정 대기: 1-2주

**현재 코드는 키만 추가하면 동작 가능 상태**. 인프라 다 완성됨.
