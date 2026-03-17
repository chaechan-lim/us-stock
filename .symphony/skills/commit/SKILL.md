---
name: commit
description: Create clean git commits with conventional messages for us-stock.
---

# Commit

## Steps

1. Inspect changes: `git status`, `git diff`
2. Stage intended changes (no .env, no build artifacts, no deploy/)
3. Choose prefix: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `config:`, `ci:`
4. Write subject: imperative mood, ≤72 chars
5. Write body: summary + rationale + test status
6. Append: `Co-authored-by: Claude <noreply@anthropic.com>`

## Rules

- One logical change per commit
- Code + tests in the same commit
- Never commit .env files, API keys, or credentials
- Never commit deploy/ changes without explicit approval
- Strategy parameter changes must reference backtest results
