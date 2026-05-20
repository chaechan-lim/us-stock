#!/usr/bin/env bash
# Weekly Claude analysis runner. Triggered by cron on Sun 21:00 KST.
# See docs/CLAUDE_WEEKLY_ANALYSIS.md for design + prompt template.
#
# Output:
#   /tmp/claude-weekly-YYYY-MM-DD.md  — raw markdown
#   Discord webhook  — title + first 1500 chars

set -euo pipefail

REPO="/home/chans/us-stock"
OUT="/tmp/claude-weekly-$(date +%F).md"
cd "$REPO"

# Source .env to get DISCORD_WEBHOOK_URL
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PROMPT='You are reviewing the most recent week of this auto-trading bot.

Pull data:
  1. Run `cd backend && ../venv/bin/python scripts/strategy_contribution_research.py 2>&1 | tail -40` to get live PnL contributions by strategy.
  2. Read the last 7 days of git log in config/strategies.yaml (`git log -p --since="7 days ago" config/strategies.yaml`).
  3. Query the orders DB for daily cleanup count + PnL trend over the last 14 days (use scripts/daily_post_market_analysis.py for the SQL pattern).

Answer three questions, max 400 words total:

  Q1. Which single strategy contribution moved the most this week (vs the prior 7-day window)? Cite numbers from the strategy_contribution_research output.

  Q2. Did any cleanup pattern emerge that the daily reports missed? (specific symbols, intra-day times, market regime, etc.)

  Q3. Recommend ONE small change for next week. Must follow project memory `feedback_backtest_first` — state the backtest script name (existing or new) needed to validate it.

Output as plain markdown. No preamble.'

# Run Claude headless. Requires `claude` CLI in PATH + an authenticated session.
if ! command -v claude &>/dev/null; then
    echo "claude CLI not found in PATH — skip" >&2
    exit 0
fi

echo "$PROMPT" | claude --print > "$OUT" 2>&1 || {
    echo "claude CLI failed; raw output:" >&2
    cat "$OUT" >&2
    exit 1
}

# Push to Discord (truncate to 1500 chars to fit a single embed body).
if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
    TITLE="📈 Weekly Claude analysis — $(date +%F)"
    BODY=$(head -c 1500 "$OUT")
    JSON=$(python3 -c "
import json, sys
body = sys.argv[1]
title = sys.argv[2]
payload = {
    'embeds': [{
        'title': title,
        'description': body,
        'color': 0x3B82F6,
    }]
}
print(json.dumps(payload))
" "$BODY" "$TITLE")
    curl -s -H 'Content-Type: application/json' -d "$JSON" "$DISCORD_WEBHOOK_URL" > /dev/null || true
fi

echo "Weekly analysis written to $OUT"
