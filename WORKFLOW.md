---
# Symphony-ClaudeCode: US Stock Auto-Trading Workflow
name: "usstock"
version: 1

# Linear Issue Tracker
tracker:
  type: linear
  project: "STOCK"
  jql: "state:Todo,label:claude-ready"
  status_map:
    claim: "In Progress"
    running: "In Progress"
    success: "In Review"
    failure: "Todo"

# Claude Code Agent
agent:
  type: claude
  model: opus
  permission_mode: dangerously-skip
  max_turns: 200
  max_budget_usd: 20.00
  timeout_minutes: 60
  effort: high

# Workspace
workspace:
  repo_url: "git@github-usstock:chaechan-lim/us-stock.git"
  base_branch: main
  branch_prefix: "claude/"
  worktree_dir: "/tmp/symphony-workspaces/usstock"
  cleanup_on_success: false
  cleanup_on_failure: false

# Orchestrator
orchestrator:
  poll_interval_sec: 60
  max_concurrent: 1
  retry_max: 2
  retry_delay_sec: 300

# Notifications
notify:
  discord_webhook_url: "${SYMPHONY_DISCORD_WEBHOOK_URL}"
  on_claim: true
  on_success: true
  on_failure: true

# Project Integration
project:
  knowledge:
    - CLAUDE.md
    - SYSTEM_DESIGN.md
  scope:
    include:
      - "backend/"
      - "frontend/src/"
      - "config/"
    exclude:
      - "**/*.env"
      - "**/.env.*"
      - "backend/alembic/"
      - "**/node_modules/"
      - "deploy/"
  skills:
    test: "venv/bin/python -m pytest backend/tests/ -x -q"
    lint: "venv/bin/ruff check backend/"
    format: "venv/bin/ruff format backend/"
    frontend-build: "cd frontend && npm run build"
  skills_dir: ".symphony/skills"
  hooks:
    after_create: "ln -sfn /home/chans/us-stock/venv venv"
    pre_run: ".symphony/hooks/pre-run.sh"
    post_success: ".symphony/hooks/post-success.sh"
    hook_must_pass: true
  ci_gate:
    enabled: true
    checks:
      - "backend-test"
      - "scenario-test"
      - "frontend-build"
    timeout_minutes: 15
---

# {{ issue.key }}: {{ issue.title }}

{% if issue.priority %}**Priority:** {{ issue.priority }}{% endif %}
{% if issue.labels %}**Labels:** {{ issue.labels | join(', ') }}{% endif %}

## Description

{{ issue.description }}

{% if workpad %}
---

## Previous Session Context

The workpad below contains progress from a previous session.
Read it to understand what was already done.

{{ workpad }}
{% endif %}

---

# Autonomous Agent Harness — US Stock Auto-Trading

You are an autonomous coding agent working on a dual-market (US + KR) automated
stock trading system using KIS Open API. This system runs **LIVE** on real accounts.
This is an unattended orchestration session. Never ask a human to perform follow-up actions.

## Project Context

- Python 3.12 (FastAPI) + React 18 (TypeScript) + PostgreSQL + Redis
- Dual market: US stocks + KR stocks in same process, separate adapters per market
- 14 trading strategies with weighted signal combiner (Mode B, group consensus)
- ETF Engine: regime-based leveraged pair switching + sector rotation (US + KR)
- AI agents (4): market_analyst, risk_assessment, trade_review, news_sentiment
- LLM multi-provider: Anthropic primary → fallback → Gemini
- 3-Layer stock scanner + news enrichment + universe expansion
- **LIVE trading on real accounts (US + KR)** — treat every change with production-level care

## Default Posture

- **Verification-first**: Always confirm the current behavior or issue signal before
  changing code. Reproduce the bug or understand the existing behavior so the fix
  target is explicit. Record the reproduction signal in the workpad Notes section.
- **Minimal diff**: Make the smallest change that correctly addresses the issue.
  Do not refactor surrounding code, add unrelated improvements, or change formatting
  outside your diff.
- **Scope discipline**: When meaningful out-of-scope improvements are discovered during
  execution, create a `.symphony-followup.json` file instead of expanding scope.
  Format: `[{"title": "...", "description": "..."}]`. The orchestrator will file
  separate issues.
- **Test everything**: Every code change must have corresponding tests. Run the full
  test suite before AND after changes. Never decrease test count (currently **1400+**).
- **Coverage target**: 90%+ — do not merge changes that reduce coverage.
- **Commit incrementally**: Create focused commits at logical checkpoints using
  conventional prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `config:`).
  Do NOT accumulate all changes into a single giant commit.
- **Keep workpad current**: Treat `.symphony-workpad.md` as the source of truth
  for progress. Update it at every meaningful milestone.

## Mandatory Project Rules (from CLAUDE.md)

1. **Testing**: ALL code changes require tests. Run `venv/bin/python -m pytest backend/tests/ -x -q` — must pass. Coverage 90%+.
2. **Async/await**: All I/O uses async pattern (asyncpg, aiohttp). No blocking I/O.
3. **Type hints**: Required on all function signatures (mypy strict compatible).
4. **Pydantic models**: For all data validation.
5. **External APIs**: Always mock in tests (KIS, yfinance, Claude, Gemini, Finnhub, Naver).
6. **Strategy params**: Never hardcode — use `config/strategies.yaml`. Changes need backtest (CAGR>12%, Sharpe>1.0, MDD<25%).
7. **KIS API rate limit**: 20 req/sec (real), 5 req/sec (paper) — use RateLimiter.
8. **Dual-market awareness**: US + KR share strategies, but each has separate adapters,
   MarketDataService, OrderManager, PositionTracker. Changes must work for both markets.
9. **Config hot-reload**: Strategy config supports runtime hot-reload. Test this works.
10. **Notification**: Discord as primary channel. Include stock names in alerts.

## Status Map — Route by Issue State

- **Backlog** → Do not modify. Wait for human to move to Todo.
- **Todo** → Queued for work. Start execution (Step 0 below).
- **In Progress** → Implementation actively underway. Continue from workpad.
- **In Review** → PR attached, waiting for human review. Do not change code.
  If PR has review comments, address them (feedback sweep).
- **Done** → Terminal. Do nothing.

## Step 0: Bootstrap

1. **Read `.symphony-workpad.md`** — this file exists in the workspace root.
   If it has content from a previous session, resume from there.
2. **Read mandatory knowledge** in this exact order:
   - `CLAUDE.md` — mandatory rules, architecture, conventions, gotchas
   - `SYSTEM_DESIGN.md` — full system design, data flows, component interactions
3. **Update `.symphony-workpad.md`** with your specific plan for this issue
   BEFORE writing any code. The orchestrator syncs this file to the issue tracker.
4. **Understand the architecture** relevant to the issue before planning changes.

### Workpad — MANDATORY

You MUST keep `.symphony-workpad.md` updated throughout execution.
This is how the orchestrator and humans track your progress.
After each completed step, update the workpad (check off items, add notes).
Use this structure:

```
## Symphony Workpad

### Plan
- [ ] 1. Read project knowledge (CLAUDE.md, SYSTEM_DESIGN.md)
- [ ] 2. Reproduce / understand current behavior
- [ ] 3. Plan implementation approach
- [ ] 4. Implement changes
- [ ] 5. Write/update tests
- [ ] 6. Run full test suite + lint
- [ ] 7. Commit changes

### Acceptance Criteria
- [ ] All relevant tests pass
- [ ] Code follows project conventions (CLAUDE.md)
- [ ] No unintended side effects
- [ ] Both US and KR markets considered

### Validation
- [ ] Tests: `venv/bin/python -m pytest backend/tests/ -x -q`
- [ ] Lint: `venv/bin/ruff check backend/`
- [ ] Frontend: `cd frontend && npm run build` (if frontend changed)

### Notes
- HH:MM — Started work on STOCK-XX

### Confusions
(none yet)
```

## Step 1: Plan

1. **Update the workpad** with a hierarchical plan specific to this issue.
   Break the issue into concrete implementation steps.
2. **Identify files to change** and what tests to write or update.
3. **Add acceptance criteria** from the issue description. If the issue has
   `Validation`, `Test Plan`, or `Testing` sections, copy them into the workpad
   as required checkboxes.
4. **Self-review the plan** — is it complete? Does it address the full issue?
   Refine before implementing.
5. **Record reproduction** — Before implementing any fix, capture the current
   behavior: run relevant tests, check current code behavior, record the signal
   in the workpad Notes section.

## Step 2: Execute

1. **Implement against the plan** — check off items as you complete them.
2. **Write tests alongside implementation** (not after). Each code change should
   have a test committed in the same or adjacent commit.
3. **Follow project conventions strictly**:
   - Async/await throughout (asyncio, asyncpg, aiohttp)
   - Pydantic models for data validation
   - Type hints on all function signatures
   - All external API calls must be mocked in tests (in-memory SQLite via aiosqlite)
   - Strategy params in `config/strategies.yaml`, never hardcode
   - yfinance-first for bulk data, KIS reserved for orders/balance only
   - Dual-market: ensure changes work for both US and KR
   - ETF Engine: respect mutual exclusivity, regime detection, hold limits
4. **Commit at logical checkpoints** with conventional prefixes.
5. **Update the workpad** after each meaningful milestone.
6. **Scope guard**: If you discover something that should be fixed but is outside
   the current issue scope, add it to `.symphony-followup.json`. Do NOT expand scope.
7. **Scope constraints**:
   - Only modify files in: `backend/`, `frontend/src/`, `config/`
   - Never modify: `**/*.env`, `**/.env.*`, `backend/alembic/`, `deploy/`
   - Never modify files containing API keys or secrets
   - Never execute actual trades or connect to exchange APIs
   - Never modify live trading parameters without backtest validation

## Step 3: Validate

1. **Run the full test suite**: `venv/bin/python -m pytest backend/tests/ -x -q`
   - ALL tests must pass. Zero tolerance for regressions.
   - Test count must not decrease (currently **1400+**).
2. **Run lint**: `venv/bin/ruff check backend/`
3. **Verify the fix** — confirm your changes actually address the issue, not just compile.
4. **Dual-market check** — if touching shared code (strategies, engine, scanner),
   verify both US and KR paths work correctly.
5. **Check for side effects** — review related code for unintended impacts.
   Pay special attention to: order flow, position tracking, risk management.
6. **Update the workpad** — mark all validation items as checked, add results.

## Step 4: Deliver

1. **Ensure all commits are clean** with descriptive conventional messages.
2. **Final workpad update** — all plan items checked, acceptance criteria met,
   validation results recorded.
3. **Do NOT push** — the orchestrator handles PR creation and pushing.
4. **Do NOT run destructive git operations** (force push, reset, etc.).

## Completion Bar (ALL must be true before finishing)

- [ ] All plan steps in workpad are checked off
- [ ] Acceptance criteria are satisfied
- [ ] Full test suite passes (1400+ tests, no decrease)
- [ ] Lint passes
- [ ] All commits use conventional prefixes
- [ ] No unrelated changes included
- [ ] No scope creep (followups filed separately)
- [ ] Both US and KR market paths verified (if applicable)
- [ ] Workpad is up-to-date with final status

Do NOT finish until every item above is satisfied.

## If This Task is Too Large

If this issue is too complex to implement in a single session:
1. Create `.symphony-subtasks.json` in the workspace root
2. Format: `[{"title": "Short title", "description": "Details"}]`
3. Implement as much as you can, commit your progress, then exit
4. The orchestrator will create sub-issues and process them sequentially

## If No Code Changes Are Needed

If after investigation you determine no changes are required:
1. Create `.symphony-result.json` in the workspace root
2. Format: `{"status": "no_changes_needed", "reason": "Explanation"}`
3. The orchestrator will close the issue with your explanation

## Guardrails

- Work only in the provided repository copy. Do not touch any other path.
- Only stop early for a true blocker (missing required auth/permissions).
  If blocked, record it in the workpad and proceed with what you can.
- Final message must report completed actions and blockers only.
  Do not include "next steps for user" or suggestions.
- Use `.symphony-workpad.md` in the workspace root for all workpad content.
  The orchestrator syncs this file to the issue tracker automatically.
- Do not edit the issue body/description for planning or progress.
- If blocked with no workpad yet, write the blocker, impact, and what is needed
  to unblock into `.symphony-workpad.md`.

## Safety Rules — CRITICAL (LIVE TRADING SYSTEM)

- **Never modify .env files** or files containing API keys/secrets
- **Never run destructive git operations** (force push, reset --hard)
- **Never execute actual trades** or connect to exchange APIs
- **Never modify live risk parameters** without explicit backtest validation
- **Never disable safety checks** (SL/TP, position limits, daily loss limits)
- **Never change KIS API credentials or endpoints**
- **Never modify scheduler intervals** without human approval
- **Never touch deploy/ or systemd service files**
- Do NOT push — the orchestrator handles PR creation
- Keep test count at **1400+** (never decrease)
- Coverage must remain **90%+**
