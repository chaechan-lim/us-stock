# Weekly Claude Analysis (cron pattern)

Pairs with the deterministic `scripts/daily_post_market_analysis.py` (runs
every day at 06:00 KST). The daily script handles the 80% case
(numeric metrics → Discord). The weekly Claude run handles the 20% —
qualitative pattern-finding that benefits from a model in the loop.

## Why this is separate from the daily script

Daily metrics are stable and cheap. We want a robot that says
"yesterday cleanup count = 9, baseline 1.2/day, ⚠️" without spending a
dollar on inference. Claude shines for "look across these five
strategies' weekly P&L curves and tell me which one's edge is
deteriorating" — open-ended, requires synthesis.

## Recommended cadence

- **Sunday 21:00 KST** (after the KR market is closed and operator has
  time before the Monday open). Once a week is enough; the operator
  reviews the output before the new week's trading starts.

## Cron entry

Append to your `crontab -e`:

```cron
# Weekly Claude analysis — Sunday 21:00 KST. Output goes to a
# timestamped markdown file and is shipped to the Discord webhook
# via the same notification adapter the bot uses.
0 21 * * 0 cd /home/chans/us-stock && /usr/bin/env bash scripts/run_weekly_claude_analysis.sh >> /tmp/claude-weekly.log 2>&1
```

The helper script handles: claude CLI invocation, file output, and
the Discord push.

## Prompt template (claude --print)

The helper script feeds Claude a prompt like:

```
You're reviewing this trading system's most recent week.

Read:
  1. backend/scripts/strategy_contribution_research.py output (live DB)
  2. The last 7 daily post-market reports (Discord screenshots or
     /tmp/daily_*.md if you saved them)
  3. config/strategies.yaml recent diffs (git log --since="1 week")

Answer THREE specific questions, max 400 words total:

  Q1. Which single strategy contribution moved the most this week
      (better or worse vs the prior 7-day window)? Cite numbers.

  Q2. Did any cleanup pattern emerge that wasn't visible in the
      daily reports? (e.g. specific symbols, specific intra-day
      times, specific market regime.)

  Q3. Recommend ONE small change for next week. Must be backtest-
      first per project memory feedback_backtest_first. State the
      backtest script name even if it doesn't exist yet.

Output as markdown. No preamble. Skip anything you can't ground in
the data you pulled.
```

## What the operator does with the output

- Skim Monday morning before market open.
- If Q3 recommends a change, it goes into the regular PR workflow
  (branch → backtest → CI → review → merge → reload).
- The weekly report itself is not auto-applied — operator gate.

## When to skip / disable

- Cost: monthly Claude spend trending high → drop to bi-weekly.
- Signal: three weeks in a row with no actionable recommendation →
  reduce frequency or rotate prompts.
- The deterministic daily script stays on regardless — that's the
  floor.
