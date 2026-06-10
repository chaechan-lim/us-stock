# KR Strategy Candidates — Untapped Alpha Sources

**Context (2026-06-01)**: All existing KR strategies (dual_momentum,
supertrend) are exhausted as deployment levers. compare_kr_strategy_density.py
showed bnf_deviation, volume_surge, bollinger_squeeze, macd_histogram,
rsi_divergence produce zero KR signals; trend_following activates but
hurts -1.5pp Ret. The remaining alpha must come from **strategies that
don't exist yet**.

This doc lists candidate strategies ordered by (1) expected KR alpha
potential, (2) implementation effort, (3) signal-density contribution
to chronic-low-deployment problem.

---

## Tier 1 — High signal density, KR-favorable structure

### 1. Opening Range Breakout (ORB)
**Hypothesis**: First 30 min of KR session (09:00-09:30 KST) establishes
a price range. Symbol breaking above range high in 09:30-12:00 window
continues to close. Range-bound continuation pattern, well-documented
on KOSPI mid-caps with daily liquidity > ₩1B.

**Signal density estimate**: 1-3 BUYs/day per watchlist of 40-50 names
(20-50× current sparse dual_momentum). Directly addresses live-low-deploy.

**Implementation**:
- New strategy class `OpeningRangeBreakout(BaseStrategy)` (~200 LOC)
- Reads first 30 1-min bars (need intraday OHLCV — KIS WebSocket already
  subscribed for held names; need pre-market scan-subscribe for watchlist)
- Trigger: close > range_high * 1.005 (5bp buffer) AND volume_30min > vol_avg_5d × 1.5
- SL: range_low. TP: range_high + (range_high - range_low) × 1.5
- Hold until close or SL

**Backtest plan**: 2y KR, 1-min OHLCV from yfinance (or KIS day-bar
  resampled if 1-min unavailable beyond 7d). compare_kr_orb.py with
  range_minutes ∈ {15, 30, 45} × volume_filter ∈ {1.0, 1.5, 2.0}.

**Risk**: 1-min data unavailable yfinance for >7d; might need KIS pull.

**Effort**: 6-8h (strategy + backtest harness adapter for intraday data).

---

### 2. Gap-and-Go
**Hypothesis**: Stock gapping up >3% at open (vs prev close) with
above-average pre-market volume continues for 30-90 min. Works on
news-driven KOSPI names. Korean market has structural overnight news
catalyst (US session close → KR pre-market reaction).

**Signal density**: 2-5 BUYs/day across watchlist on news days; 0 on
quiet days. Bursty.

**Implementation**:
- Strategy class `GapAndGo(BaseStrategy)` (~150 LOC)
- Trigger: open / prev_close >= 1.03 AND first_5min_volume / avg_5min_vol > 2.0
- Exit: trailing 2% after +5%, or 30-min stagnation (price within
  ±0.5% of entry for 6 consecutive bars)

**Backtest**: 2y, daily OHLCV with open-close logic. Simulate at
  open-price entry, conservative exit at first-30min close.

**Effort**: 4-6h (lower complexity than ORB; daily-bar approximation OK).

---

## Tier 2 — Medium signal density, KR-specific

### 3. End-of-Day Momentum (EOD-Mom)
**Hypothesis**: Stocks closing in upper 20% of daily range with above-
avg volume continue overnight + into next-day open. Capture overnight
drift premium.

**Signal density**: 3-8 BUYs/day (very high), but most fail.

**Issue**: Overnight gap risk (KR equities close 15:30, US session
overnight). Acceptable in trending market, dangerous in selloff.

**Effort**: 3-4h. Lower priority than Tier 1.

---

### 4. Sector Relative Strength Rotation (Sector-RS)
**Hypothesis**: Buy top-3 KOSPI sectors' top-2 names weekly; sell
underperforming sectors. Already partially covered by sector_boost,
but as a STRATEGY (not a confidence multiplier) it would generate
its own BUY signals.

**Signal density**: 5-10 BUYs/week (weekly rebalance). Steady, not bursty.

**Effort**: 5-7h. Already have sector_history infrastructure.

---

## Tier 3 — Specialty / advanced

### 5. KOSDAQ Theme Momentum (Theme-Mom)
**Hypothesis**: KOSDAQ has thematic clusters (2차전지, 바이오, 게임,
엔터). When a theme is hot, all its names move together. Detect via
correlation breakout of cluster-average.

**Implementation**: Cluster definition file (~50 names tagged by theme),
relative-strength computation per cluster, BUY top mover in hot cluster.

**Signal density**: 1-3 BUYs/week.

**Effort**: 8-10h (theme tagging + cluster RS engine).

---

### 6. Insider Buying (DART)
**Hypothesis**: 5% holders + executives buying their own stock in open
market signals 2-6 week alpha. KR DART system provides daily filings.

**Issue**: External data dependency (DART API), filing latency 1-2 days.

**Effort**: 10-12h (DART parser + buy detector + backtest).

---

## Priority Recommendation

If only one is built first: **#1 Opening Range Breakout**.
- Highest signal density per effort hour
- Directly addresses chronic-low-deploy (1-3 BUYs/day = 4-12× current)
- Backtestable cleanly with 1-min data
- Hard SL via range_low = natural risk control

Second: **#2 Gap-and-Go**. Simple, complements ORB (different signal pattern,
different time-of-day).

After both: re-measure deploy + alpha contribution; if still gap, move to Tier 2.

---

## Acceptance criteria for any new strategy

Per CLAUDE.md gates:
- Backtest CAGR > 12% (or measurably improves Sharpe/MDD/PF vs current 4-strategy combo)
- Sharpe > 1.0
- MDD < 25%
- Paper-validated 2 weeks live before yaml weight > 0.05

For KR combo (relaxed floor):
- Sharpe > 0, MDD < 15%, PF > 1.0 (PROVISIONAL allowed)
- All four metrics improve vs current KR live combo (post-PR #183 baseline:
  Ret +14.5% Sharpe +0.71 MDD -11.2% PF 1.30)
