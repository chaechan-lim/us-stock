# KR Alpha — Phase B Roadmap (Infrastructure Track)

**Context (2026-06-02)**: All yaml-only / daily-bar-strategy / universe
levers exhausted on KR. PR #183 captured the +6pp Ret cumulative wins.
Three independent falsifications of "deploy more cash → more alpha":
  - sizing-up forced top-up (-4 to -7pp)
  - max ETF cap V3 30/15 (MDD breach)
  - universe expansion 41→79 (-18pp Ret)

Next-phase alpha requires structural infrastructure work. This doc
captures three candidate tracks with effort estimates and concrete
implementation outlines so a future session can pick up cleanly.

---

## Track A — Intraday 1-min OHLCV pipeline (12–16h)

**Motivation**: KR daily-bar close-buy strategies fail because by
close, the alpha is already faded (retail profit-take, circuit-breaker
rebound, chaebol sell-on-strength). Both `gap_and_go` and
`eod_momentum` produced identical -₩14K avg/trade. The fix is
*entering at 09:00–09:30*, not at close.

**Architecture**:

```
KIS Domestic Stock 1-min API
   ↓
1-min OHLCV cache (Redis or Postgres)
   ↓
IntradayBacktestEngine (new, separate from FullPipelineBacktest)
   ↓
strategies/orb.py + strategies/intraday_gap_continuation.py
```

**Concrete steps**:

1. **KIS 1-min historical fetcher** (3–4h)
   - Endpoint: `inquire-time-itemchartprice` (KIS 일별·분봉 차트 조회 — 30일치 1-min)
   - Endpoint: `inquire-daily-itemchartprice` 분봉=Y (~90d 1-min)
   - Bulk-fetch 90d × 79 symbols on startup; nightly delta refresh
   - Cache: `data/kr_intraday/{symbol}_{YYYYMMDD}.parquet`
   - Rate limiter: KIS 20 req/sec real, 5/sec paper

2. **Intraday data loader for backtest** (3–4h)
   - `backtest/intraday_loader.py` mirroring `backtest/data_loader.py`
   - Aligns 1-min bars across symbols by minute-of-day
   - Constructs intraday OHLCV view that strategies analyse minute-by-minute

3. **Intraday strategies** (3–4h)
   - `OpeningRangeBreakout`: first 30-min range establishes high/low,
     break above range_high in 09:30–12:00 = BUY, SL=range_low
   - `IntradayGapContinuation`: 09:00 gap up >2% + 09:00–09:15 volume
     2x avg → BUY at 09:15, hold to close or trail
   - Both extend `BaseStrategy` but `analyze()` consumes 1-min DF

4. **IntradayBacktestEngine** (3–4h)
   - Loops minute-by-minute through trading session
   - Calls strategies on each 1-min bar slice (`df.iloc[:current_minute+1]`)
   - Exit logic: SL/TP intraday, time-stop at 15:20 KST, hold-to-close
   - Reuses existing PositionSizer, RiskManager, Trade recording

5. **Live integration** (2–3h)
   - KIS WebSocket already subscribes to held + watchlist
   - Wire intraday strategies into eval loop's intraday tick callback
   - Backtest-vs-live parity test on 1 week of data

**Risks**:
- KIS historical 1-min may only go back 30–90d (smaller backtest window)
- Survivorship bias if universe changes over 90d window
- Tick-size effects more pronounced intraday
- Compute load: 79 symbols × 390 1-min bars/day × 90 days = ~2.7M data points/run

**Acceptance gate** (per CLAUDE.md, KR PROVISIONAL):
- Sharpe > 0, MDD < 15%, PF > 1.0 on the 90d window
- Improves 4/4 dims over current 14d-window non-intraday baseline

---

## Track B — KR DART insider signal (6–10h)

**Motivation**: Existing `data/insider_service.py` covers US via Finnhub
+ feeds `event_calendar.get_confidence_adjustment()` for ±0.10 conf
adjustment per CLAUDE.md. KR has no insider input — the
`get_confidence_adjustment()` call returns 0 for all KR symbols.

DART (전자공시시스템) is KR's official insider/major-shareholder filing
system. Open API: opendart.fss.or.kr/api. Free with API key.

**Endpoints**:
- 임원·주요주주 특정증권 등 소유상황보고서: `list.json` with `pblntf_detail_ty=H1` (5%+) and `pblntf_ty=H` (insider)
- Provides: filing date, reporter, shares before/after, change_qty,
  transaction_type (취득/처분), price, relation_to_corp

**Concrete steps**:

1. **DART client** (2h)
   - `data/dart_service.py` analog to insider_service.py
   - aiohttp + rate-limit (DART allows 10,000 req/day; not strict per-second)
   - Models: `KRInsiderTransaction` mirroring `InsiderTransaction`

2. **Daily refresh task** (1h)
   - Add to scheduler.add_task("dart_insider_refresh", interval=86400)
   - Fetch last 30d filings for watchlist symbols
   - Persist to Postgres (new table `kr_insider_transactions`)

3. **Signal adjustment logic** (1h)
   - Bullish: 5%+ holder buys >0.5% of float in last 14d → +0.10 conf
   - Bullish: executive (대표이사 / 등기임원) buys → +0.05 conf
   - Bearish: large insider sale → -0.10 conf
   - Window: 14 days post-filing
   - Returns float -0.10 to +0.10, plugged into eval_loop:2152

4. **Backtest support** (2h)
   - Snapshot DART filings into `data/kr_insider_history.json` (~6 months)
   - Backtest reads snapshot, applies adjustment per (symbol, date)
   - Compare 2y backtest with/without DART signal

5. **Live wiring** (1h)
   - event_calendar.refresh() includes dart_service.refresh()
   - get_confidence_adjustment() routes US→Finnhub, KR→DART

**Risks**:
- Filing latency (T+5 business days typical) — alpha may be priced in
- DART data quality: 환산주식수, 보고이유 fields are free-text
- Backtest: only recent ~6 months filings available; older needs separate crawl

**Acceptance gate**:
- Backtest improves Ret without MDD regression
- Live: track 5%-holder filings for top 5 KR positions for 2 weeks

---

## Track C — Sector-RS standalone strategy (3–5h)

**Motivation**: Existing `sector_boost_weight: 0.3` only *amplifies*
signals from other strategies. A standalone Sector-RS would generate
its own BUY signals on top-sector top-momentum names, capturing alpha
that no other strategy currently emits.

`sector_rotation` strategy exists but is ETF-engine focused (rotates
XLK/XLF). New strategy: pick top individual stocks within strong sectors.

**Concrete steps**:

1. **Strategy class** (2h)
   - `strategies/sector_top_pick.py` (avoiding name conflict)
   - Uses existing `sector_history` data
   - Weekly rebalance (configurable)
   - BUY top-2 momentum names in top-3 sectors

2. **Backtest** (1h)
   - Use existing compare_kr_*.py pattern
   - Standalone V_only test to measure raw alpha contribution

3. **Integration** (1h)
   - yaml: enable + profile weights
   - Backend restart

**Risks**: high overlap with dual_momentum + sector_boost combined.
Result likely small or negative. Lowest priority of three tracks.

---

## Recommendation priority

1. **Track A (intraday)** — highest expected alpha, highest cost
2. **Track B (DART)** — orthogonal alpha source, medium cost
3. **Track C (Sector-RS)** — lowest expected delta, lowest cost
4. **Or: stop adding alpha** and let current PR-#183 yaml run for 2-4 weeks before more work
