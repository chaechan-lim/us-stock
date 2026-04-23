# Backtest Contract

**What the backtest can and cannot tell you.** Written 2026-04-23 after
the C1 reliability pass; keep updated as the limits change.

---

## Use cases the backtest answers well

### 1. Relative comparison between config variants
"Does variant C (dual_momentum `volatility_filter: false`) produce more
return than the baseline (`volatility_filter: true`)?"

The **direction** of the delta (ΔRet, ΔSharpe, ΔMDD, ΔPF) is reliable
because both sides run on the same universe, same slippage model, same
period. This is what `scripts/compare_kr_strategy_mix_2026_04.py` and
the `ci_backtest_gate.py` gate exist for.

### 2. Detecting silent regressions
Trade count collapse, Sharpe flipping negative, MDD doubling — these
show up immediately in a fresh run against `backtest_baselines.json`.
The gate catches them before merge.

### 3. Ranking strategies against each other
"Is `supertrend` adding alpha on top of `dual_momentum` or just
overlapping it?" — per-strategy PnL breakdown in `SignalQualityTracker`
is trustworthy because every strategy sees the same bars.

---

## Use cases the backtest answers poorly

### 1. Absolute return / Sharpe projection
Live slippage is ~15-18× the backtest's fixed `slippage_pct` (C1.4,
`scripts/compare_slippage.py`):

| Market | Backtest slippage_pct | Live median \|slippage\| | Live p75 \|slippage\| |
|---|---|---|---|
| US | 0.05% | 0.31% | 0.76% |
| KR | 0.08% | 0.78% | 1.42% |

Effect: backtest **overstates** Return and Sharpe by the missing
frictional cost. The absolute number you see is a best-case; real live
performance is lower. Treat backtest Ret as "direction-valid, magnitude
not."

### 2. Performance on symbols outside the universe
~52% of recent US live trades happen on symbols the backtest's
`WIDE_UNIVERSE` doesn't even contain (C1.3, `scripts/compare_universe.py`).
Backtest says nothing about `ALM`, `AMPX`, `DELL`, `LION`, etc.
performance.

### 3. Burst / intraday / gap-and-go strategies
The pipeline uses **daily bars** for regime + signal generation. A
gap-and-go or volume-breakout strategy that relies on minute-level
follow-through is unverifiable here — the simulator never sees the
intraday move. From `IMPROVEMENT_PLAN.md §3` "절대 하지 말 것":
*"daily-bar 백테스트로 burst-catcher 전략 결정하지 말것"*.

### 4. Paper-vs-live alpha prediction
System-level alpha has been roughly −24% vs SPY in live over 2y
(memory: `project_alpha_deficit.md`). Backtest reported better
numbers before the C1 calibration because of understated slippage.
Even after calibration, backtest cannot predict live alpha tightly —
regime timing, KIS execution quality, and signal-quality divergence
(see §5 below) all add slack.

### 5. Cold-start signal quality
Live `SignalQualityTracker` has 6+ months of Kelly inputs + gating
state. Backtest without `signal_quality_seed_path` starts cold, so
early trades use different (uninformed) sizing and the early
trajectory diverges. Fix: seed from
`data/signal_quality_snapshot.json`
(C1.2, weekly `signal-quality-snapshot.timer`).

---

## Fixed assumptions baked into every run

Documented in `PipelineConfig` + `full_pipeline.py`. Edit carefully —
changing any of these invalidates all committed baselines.

- **`initial_equity`**: 100K USD (US), 100M KRW (KR) — auto-scaled by
  market. 2026-04-07 currency bug invalidated every KR baseline pre-that-date.
- **`slippage_pct`**: single fixed percentage per market (see §1 of
  "answers poorly"). `volume_adjusted_slippage=True` scales by
  participation rate, but the base constant is still understated.
- **Commission**: 0 by default. Live KR has 0.015% + tax; US has per-
  share commission. Treat backtest as commission-free — subtract
  ~0.3%/roundtrip for KR live, negligible US.
- **Universe**: static from `DEFAULT_UNIVERSE` / `WIDE_UNIVERSE` /
  `DEFAULT_KR_UNIVERSE`. Does **not** pull from live DB watchlist or
  KIS rankings, so new movers live sees never enter the backtest.
- **Dividends**: `auto_adjust=False` in yfinance calls — prices are
  raw (not dividend-adjusted). Return figures exclude dividends.
- **Taxes**: none.
- **Fills**: every order fills at the next bar's price (regular orders)
  or current bar's open (extended hours) with the fixed slippage applied.
  Partial fills not simulated. Limit orders use a fill-probability model.

---

## Calibration multipliers (rule of thumb)

When reading a backtest result, apply these mental corrections before
comparing to what live might deliver:

| Metric | Backtest shows | Expect live ≈ |
|---|---|---|
| Total Return (2y) | +X% | X% − (1% × round-trips/yr × years) |
| Sharpe | s | s − ~0.3 (slippage drag) |
| MDD | −d% | ~−d% (slippage doesn't change drawdown shape much) |
| Profit Factor | p | p × 0.80–0.85 |
| Win Rate | w% | w% − a few pp (edge fills worse than simulator) |

These are back-of-envelope. The authoritative way to refine is to run
`scripts/compare_slippage.py` quarterly as live data grows and re-tune
`slippage_pct`.

---

## When in doubt

- **Decision boundary is which config is better**: backtest is
  trustworthy. Use `ci_backtest_gate.py`.
- **Decision boundary is "will this make money in live"**: backtest
  is one weak input. Run it as a smoke check, then paper-trade + live
  1-2 weeks before scaling up.
- **Decision is about a brand-new strategy type** (burst, HF, intraday):
  backtest can't tell you. Paper-trade from day 1.
