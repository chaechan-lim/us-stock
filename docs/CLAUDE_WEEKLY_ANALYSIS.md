# Weekly Claude Analysis (legacy — superseded 2026-05-22)

The original weekly-only cron pattern documented here was replaced
when `scripts/generate_recommendations.py` (#60) absorbed both the
daily and weekly LLM passes and the chained Python trigger
(`daily_post_market_analysis.py` → `subprocess.Popen`, see
[OPS_RECOMMENDATIONS.md](OPS_RECOMMENDATIONS.md)) removed the need
for a separate timer entirely.

The legacy helper `scripts/run_weekly_claude_analysis.sh` still
exists for one-shot manual runs (qualitative narrative dump for
operator reading) and `scripts/generate_recommendations.py` is the
machine-readable recommendation path that feeds the dashboard's
`/recommendations` queue.

See [OPS_RECOMMENDATIONS.md](OPS_RECOMMENDATIONS.md) for the live
design.
