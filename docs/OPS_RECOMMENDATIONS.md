# Closed-loop recommendations (daily + weekly)

End state the operator asked for (#60):

> "분석을 바탕으로 제안도 해야할거같다 — 제안 → 내가 승인 → 라이브 적용 → 추이 보고 아니다 싶으면 빼는 하네스"

This doc is the single source of truth for how that loop runs.

## Flow

```
06:00 KST   systemd: daily-post-market-analysis.timer
            └── scripts/daily_post_market_analysis.py
                ├── deterministic SQL aggregates (PnL, cleanup count, top strategies, SPY)
                ├── write data/daily_analyses/{date}.json  (Dashboard reads)
                ├── Discord embed
                └── subprocess.Popen for LLM recs (non-blocking)
                    ├── --mode daily        (always)
                    └── --mode weekly       (only when weekday() == 0, KST)

                          scripts/generate_recommendations.py
                          ├── gather: orders 24h/7d, funnel, positions, summary, current yaml, pending recs
                          ├── build single prompt with services.yaml_mutator.ALLOWED_PARAM_PREFIXES inline
                          ├── invoke `claude --print` + `codex exec` in parallel via asyncio.to_thread
                          ├── balanced-brace JSON parse (robust to codex preamble + tokens summary trailer)
                          ├── filter: whitelist + dedupe vs pending
                          ├── merge: same path + same value → one row (notes=both); diff → two rows
                          ├── insert AgentRecommendation (status=pending)
                          ├── kick off services.recommendation_validator
                          │   └── 2y full pipeline backtest baseline vs proposed,
                          │       writes {baseline, proposed, delta, passes_floor} onto the row
                          └── Discord summary

Dashboard /recommendations panel
            └── operator clicks Accept
                └── POST /api/v1/recommendations/{id}/accept
                    ├── services.yaml_mutator.apply_yaml_change
                    │   ├── whitelist check
                    │   ├── type match check
                    │   ├── strategies.yaml.bak backup
                    │   └── atomic temp+rename write
                    └── apply_us/kr_eval_overrides (hot-reload)
```

No separate timer for the LLM run. The deterministic daily script is
the only scheduled entrypoint; everything else chains off it.

## Why this pattern, not separate systemd units

The first iteration of #60 shipped four extra unit files
(`llm-recommendations-{daily,weekly}.{service,timer}`) plus a manual
sudo install step. The operator pushed back — "기존 서비스의 훅으로
돌리는게 더 낫지 않나 싶긴 하다. 뭔가 설정법이 번거로워보여서". The
chained-subprocess version (this design) is what landed. Reasons:

- One schedule to think about (06:00 KST). New cadence later = one
  systemd edit, not two.
- Failure isolation isn't worth a separate unit: the deterministic
  daily script is itself the system-health canary. If it dies, the
  LLM recs would be meaningless anyway.
- Weekly schedule is `now.weekday() == 0` in Python — one line, no
  cron expression to misread.
- `subprocess.Popen` with `start_new_session=True` survives the
  parent exiting; the LLM call (~60s per CLI) runs detached so the
  systemd unit completes promptly.

## What lives where

| Path | Role |
|---|---|
| `scripts/daily_post_market_analysis.py` | Deterministic daily report + LLM trigger |
| `scripts/generate_recommendations.py` | LLM call orchestrator (daily + weekly modes) |
| `scripts/run_weekly_claude_analysis.sh` | Legacy one-shot narrative dump (kept for manual use) |
| `backend/services/yaml_mutator.py` | Whitelist + atomic apply |
| `backend/services/recommendation_validator.py` | Auto-backtest 2y per recommendation |
| `backend/api/recommendations.py` | List / accept / reject endpoints |
| `backend/core/models.py::AgentRecommendation` | DB model |
| `frontend/src/components/RecommendationsPanel.tsx` | Operator UI |
| `data/llm_recommendations/{stamp}-{source}-{mode}.txt` | Raw CLI stdout audit log |
| `data/daily_analyses/{date}.json` | Deterministic daily artifact |
| `deploy/daily-post-market-analysis.{service,timer}` | The one systemd unit |

## Whitelist (yaml paths the LLM is allowed to propose)

Source: `services.yaml_mutator.ALLOWED_PARAM_PREFIXES`.

- `markets.{KR,US}.disabled_strategies`
- `markets.{KR,US}.evaluation_loop.*` (sector_boost_weight, opening_avoidance_minutes, daily_buy_limit, daily_buy_escalation_*, sizing_up.*, stale_time_*, …)
- `markets.KR.risk.{max_positions, max_position_pct, min_position_pct, default_stop_loss_pct, default_take_profit_pct, default_trailing_activation_pct, default_trailing_stop_pct}`
- `markets.{KR,US}.cash_parking.*`
- `tiered_trailing_stop.{enabled, tiers}`
- `breakeven_stop.{enabled, activation_ratio, lock_ratio, lock_pct}`

Anything outside is silently dropped during parsing. Operator-driven
yaml changes that need a backend restart (`RiskParams`-bound: Kelly,
allow_one_share_round_up, enforce_min_position_pct_floor) live
**outside** the whitelist on purpose — those need a PR + deploy, not
a one-click dashboard accept.

## Acceptance contract (what the operator sees)

A pending row in the dashboard surfaces:

- `param_path` + `current_value` → `proposed_value`
- `rationale` (≤240 chars, must cite numbers)
- `expected_effect` (≤160 chars)
- `confidence` (low / medium / high)
- `risk` (low / medium / high)
- `backtest_result.delta` (Ret pp, Sharpe, MDD pp, PF — proposed minus baseline)
- `backtest_result.passes_floor` (proposed didn't regress any axis past tolerance)

`accept` writes the yaml + hot-reloads. `reject` records the reason
and leaves the live config alone. Both are reversible — the previous
yaml is stashed at `config/strategies.yaml.bak` before each apply.

## What to do when the loop misbehaves

- LLM returns `{"recommendations": []}` for days: design intent. The
  prompt explicitly allows it ("If you see no high-confidence
  move…"). Inspect `data/llm_recommendations/` for raw output.
- LLM proposes the same path repeatedly: dedupe against pending
  already runs. If you accepted-then-the-metric-degraded, the
  validator's baseline will reflect the new yaml on the next run, so
  the LLM should naturally pull back. If it doesn't, manually reject
  + add a note.
- Validator fails to fill `backtest_result`: check the row — `skip`
  reason means the path isn't in `_BACKTEST_PARAM_MAP`
  (`recommendation_validator.py`); `error` means the backtest itself
  crashed (yfinance cache miss, etc.). Operator can still
  accept/reject by judgment; backtest is advisory.
- Codex CLI missing / failing: parser logs "no JSON object" and
  skips. Claude side keeps running. Same when claude CLI is missing.

## Test plan when changing this

- `backend/tests/test_scripts/test_generate_recommendations.py` (26
  tests) covers JSON parsing variants, whitelist filter, dedupe,
  merge. Add tests there for any new prompt-output handling.
- For end-to-end smoke: `python scripts/generate_recommendations.py
  --mode daily --dry-run` prints would-be inserts as JSON without
  touching the DB.
