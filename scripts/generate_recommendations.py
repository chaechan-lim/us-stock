"""LLM-driven AgentRecommendation generator (#60).

Pulls live trading context (orders, funnel, positions, equity, yaml),
asks Claude CLI and Codex CLI for structured proposals, validates each
against the yaml_mutator whitelist + existing pending recommendations,
and inserts surviving rows into agent_recommendations. The existing
recommendation_validator backtest worker then fills baseline / proposed
metrics on each row.

Usage:
    python scripts/generate_recommendations.py --mode daily
    python scripts/generate_recommendations.py --mode weekly
    python scripts/generate_recommendations.py --mode daily --dry-run

Trigger: chained from scripts/daily_post_market_analysis.py once the
deterministic report finishes. Daily fires every run; weekly fires
additionally on Monday-KST runs. There is no separate systemd timer
for this script — the goal is one schedule (daily-post-market) so the
operator only has to think about one moving part.

Each surviving row gets `agent_type` of `llm_<source>_<mode>` (e.g.
`llm_claude_weekly`). Same param_path from two sources is collapsed
into one row (`notes` records the dual sign-off). If they disagree on
proposed_value, both rows are kept and the operator picks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import desc, select, text  # noqa: E402

from core.models import AgentRecommendation  # noqa: E402
from db.session import get_session_factory  # noqa: E402
from services.yaml_mutator import ALLOWED_PARAM_PREFIXES, _is_path_allowed  # noqa: E402


logger = logging.getLogger("generate_recommendations")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KST = timezone(timedelta(hours=9))
REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "config" / "strategies.yaml"

# LLM CLI invocation — each gets ~3 min, plenty for a 5-10kB prompt.
CLI_TIMEOUT_SECS = 240
KRW_USD = 1370.0  # display only — no live trades affected


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _usd_eq(market: str, pnl: float | None) -> float:
    if pnl is None:
        return 0.0
    return float(pnl) / KRW_USD if market == "KR" else float(pnl)


async def _gather_orders(days_back: int) -> dict[str, Any]:
    """Aggregate live orders (non-paper) over a window."""
    f = get_session_factory()
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now_utc - timedelta(days=days_back)

    async with f() as s:
        r = await s.execute(text("""
            SELECT market, side, status, strategy_name, symbol,
                   count(*) AS n, sum(pnl) AS pnl
            FROM orders
            WHERE is_paper = FALSE AND filled_at >= :s
            GROUP BY market, side, status, strategy_name, symbol
        """), {"s": start})
        rows = list(r)

    by_strategy: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    by_market = defaultdict(lambda: {"buys": 0, "sells": 0, "pnl": 0.0})
    losers: list[tuple[str, float]] = []
    for row in rows:
        n = int(row.n or 0)
        pnl = _usd_eq(row.market, row.pnl)
        if row.side == "BUY" and row.status == "filled":
            by_market[row.market]["buys"] += n
        if row.side == "SELL" and row.status == "filled":
            by_market[row.market]["sells"] += n
            by_market[row.market]["pnl"] += pnl
            strat = row.strategy_name or "?"
            by_strategy[strat]["n"] += n
            by_strategy[strat]["pnl"] += pnl
            losers.append((f"{row.symbol}/{strat}", pnl))

    losers.sort(key=lambda kv: kv[1])
    return {
        "window_days": days_back,
        "by_market": {k: dict(v) for k, v in by_market.items()},
        "by_strategy": dict(sorted(
            by_strategy.items(), key=lambda kv: kv[1]["pnl"],
        )),
        "worst_5": losers[:5],
        "best_5": list(reversed(losers[-5:])),
    }


async def _gather_live_snapshot() -> dict[str, Any]:
    """Hit the local API for funnel + positions + portfolio summary.

    These endpoints already exist (rejection-funnel, portfolio/positions)
    and the daily script uses the same pattern; we just compose the
    payload as plain dicts for the prompt.
    """
    import aiohttp

    async def _get(path: str) -> Any:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:8001{path}",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
        except Exception as e:
            logger.warning("API fetch %s failed: %s", path, e)
            return None

    funnel = await _get("/api/v1/engine/rejection-funnel")
    pos_us = await _get("/api/v1/portfolio/positions?market=US")
    pos_kr = await _get("/api/v1/portfolio/positions?market=KR")
    sum_us = await _get("/api/v1/portfolio/summary?market=US")
    sum_kr = await _get("/api/v1/portfolio/summary?market=KR")
    return {
        "funnel": funnel or {},
        "positions": {
            "US": pos_us or [],
            "KR": pos_kr or [],
        },
        "summary": {
            "US": sum_us or {},
            "KR": sum_kr or {},
        },
    }


def _read_yaml_text() -> str:
    """Return strategies.yaml content for the prompt context.

    Truncated to <=12 kB to keep the prompt bounded — markets sections
    are what the LLM should propose against, and that fits comfortably.
    """
    try:
        raw = YAML_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("yaml read failed: %s", e)
        return ""
    if len(raw) <= 12_000:
        return raw
    return raw[:12_000] + "\n# … (truncated) …\n"


def _load_today_exposure() -> dict[str, Any] | None:
    """Read today's exposure snapshot written earlier in the daily chain
    by scripts/daily_post_market_analysis.py → exposure_tracker.

    Returns None if the file is missing or unreadable. The LLM prompt
    will then omit the exposure section instead of erroring.

    Falls back to yesterday's snapshot if today's hasn't been written
    yet (e.g., generator invoked standalone before the daily chain).
    """
    today = datetime.now(KST).date()
    history_dir = REPO_ROOT / "data" / "exposure_history"
    for back in (0, 1):
        candidate = history_dir / f"{today - timedelta(days=back)}.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("exposure read %s failed: %s", candidate, e)
                return None
    return None


async def _gather_pending_recommendations() -> list[dict]:
    """Existing pending rows — feed to the LLM to discourage duplicates
    and to enforce the dedupe rule we apply post-hoc."""
    f = get_session_factory()
    async with f() as s:
        stmt = (
            select(AgentRecommendation)
            .where(AgentRecommendation.status == "pending")
            .order_by(desc(AgentRecommendation.created_at))
            .limit(50)
        )
        rows = (await s.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "agent_type": r.agent_type,
            "param_path": r.param_path,
            "current_value": r.current_value,
            "proposed_value": r.proposed_value,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _format_exposure_section(exposure: dict[str, Any] | None) -> str:
    """Render today's exposure snapshot into the prompt.

    Empty string when no snapshot is available (e.g., file missing on
    first run). When sentinel flags exist, the section ends with an
    explicit ask so the LLM prioritizes exposure-targeting proposals.
    """
    if not exposure:
        return ""
    lines = ["\n== Exposure snapshot (Hermes sentinel) =="]
    for market, exp in (exposure.get("markets") or {}).items():
        funnel = exp.get("funnel") or {}
        funnel_summary = ""
        if funnel.get("top_reasons"):
            parts = [
                f"{r[0]} {r[1]} ({r[2]*100:.0f}%)"
                for r in funnel["top_reasons"]
            ]
            funnel_summary = (
                f" | funnel: {funnel.get('total_signals', 0)}→"
                f"{funnel.get('buys_placed', 0)} placed; "
                f"{', '.join(parts)}"
            )
        lines.append(
            f"  {market}: deployed {exp.get('deployed_pct', 0)*100:.1f}% "
            f"(slot_fill {exp.get('slot_fill_ratio', 0)*100:.0f}%, "
            f"placeholders {exp.get('placeholder_count', 0)}/"
            f"{exp.get('position_count', 0)}, "
            f"idle_days {exp.get('cash_idle_days', 0)})"
            f"{funnel_summary}"
        )
    flags = exposure.get("flags") or []
    if flags:
        lines.append("")
        lines.append("Sentinel flags raised today:")
        for f in flags:
            lines.append(
                f"  - [{f.get('severity', 'warning')}] {f.get('market')} "
                f"{f.get('flag')}: {f.get('detail')}"
            )
        lines.append("")
        lines.append(
            "If you propose any change today, **prioritize** lifting the "
            "binding rejection reason above (or expanding the universe so "
            "fewer signals hit it). A small param tweak that reduces the "
            "dominant funnel reason is worth more than a Sharpe-optimizing "
            "tweak elsewhere."
        )
    return "\n".join(lines) + "\n"


def _build_prompt(mode: str, ctx: dict[str, Any]) -> str:
    whitelist_text = "\n".join(f"  - {p.rstrip('.')}*" for p in ALLOWED_PARAM_PREFIXES)
    pending_summary = (
        "\n".join(
            f"  - #{r['id']} {r['param_path']} {r['current_value']!r} → {r['proposed_value']!r}"
            for r in ctx["pending"]
        ) or "  (none)"
    )
    exposure_section = _format_exposure_section(ctx.get("exposure"))

    return f"""You are a senior quant assistant reviewing a live auto-trading system (US + KR).
You will propose parameter changes that the operator can accept from a dashboard.

== Mode ==
{mode}

== Hard rules ==
1. Output ONLY a single JSON object — no markdown, no commentary, no \"```json\" fence.
2. Keys: {{ "recommendations": [ ... ] }}. Each recommendation has exactly these fields:
     param_path: dotted path inside strategies.yaml
     current_value: existing value (must match the live yaml; null if unknown)
     proposed_value: new value (same primitive type as current_value)
     rationale: ≤ 240 chars, cite specific numbers from the context
     expected_effect: ≤ 160 chars, what should change in metrics
     confidence: "low" | "medium" | "high"
     risk: "low" | "medium" | "high"
3. param_path MUST start with one of these allowed prefixes (others are silently dropped):
{whitelist_text}
4. Do NOT propose anything that duplicates a pending row (listed below).
5. Max 4 recommendations total. Lean toward fewer, high-confidence proposals.
6. If you see no high-confidence move, return {{"recommendations": []}}.

== Pending recommendations (do not duplicate param_path) ==
{pending_summary}
{exposure_section}
== Live snapshot ==
Portfolio summary:
{json.dumps(ctx["snapshot"]["summary"], ensure_ascii=False, indent=2)}

Funnel today:
{json.dumps(ctx["snapshot"]["funnel"], ensure_ascii=False, indent=2)}

Positions (US):
{json.dumps(ctx["snapshot"]["positions"]["US"][:10], ensure_ascii=False, indent=2)}

Positions (KR):
{json.dumps(ctx["snapshot"]["positions"]["KR"][:10], ensure_ascii=False, indent=2)}

== Order history (window={ctx['orders']['window_days']}d) ==
By market: {json.dumps(ctx['orders']['by_market'], ensure_ascii=False)}
Worst 5 (symbol/strategy, USD pnl): {json.dumps(ctx['orders']['worst_5'], ensure_ascii=False)}
Best  5 (symbol/strategy, USD pnl): {json.dumps(ctx['orders']['best_5'], ensure_ascii=False)}
By strategy (sorted asc by pnl):
{json.dumps(ctx['orders']['by_strategy'], ensure_ascii=False, indent=2)}

== Current strategies.yaml (head) ==
```yaml
{ctx['yaml_text']}
```

Now produce the JSON object."""


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run_cli(cmd: list[str], prompt: str) -> str | None:
    """Run a CLI with the prompt on stdin, return stdout text. None on
    failure (timeout, non-zero exit, missing binary)."""
    if _which(cmd[0]) is None:
        logger.warning("%s CLI not in PATH — skip", cmd[0])
        return None
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ds", cmd[0], CLI_TIMEOUT_SECS)
        return None
    if proc.returncode != 0:
        logger.warning(
            "%s exit=%d stderr=%s", cmd[0], proc.returncode, proc.stderr[-500:],
        )
        return None
    return proc.stdout


def _extract_json_blocks(raw: str) -> list[str]:
    """Find every balanced top-level {...} block in `raw`.

    Codex CLI prints preamble + `codex` heading + JSON + token usage
    summary, so a greedy single regex would splice unrelated text into
    one invalid JSON span. We scan for opening braces and walk a depth
    counter to extract each closed block independently.
    """
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(raw[start:i + 1])
                start = -1
            elif depth < 0:
                depth = 0
                start = -1
    return blocks


def _parse_llm_output(raw: str | None) -> list[dict]:
    """Extract recommendations[] from an LLM CLI's stdout. Tries every
    balanced JSON block and accepts the first one that contains a list
    under the `recommendations` key — robust to preambles, code fences,
    and trailing token-usage blocks."""
    if not raw:
        return []
    candidates = _extract_json_blocks(raw)
    if not candidates:
        logger.warning("no JSON object in LLM output: %s", raw[:300])
        return []
    for block in candidates:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        recs = payload.get("recommendations")
        if not isinstance(recs, list):
            continue
        out: list[dict] = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            if not all(k in r for k in ("param_path", "proposed_value")):
                continue
            out.append(r)
        return out
    logger.warning("no JSON block exposed recommendations[] (n_blocks=%d)", len(candidates))
    return []


# ---------------------------------------------------------------------------
# Validation + merge + insert
# ---------------------------------------------------------------------------


def _filter_whitelist(recs: list[dict]) -> list[dict]:
    kept = []
    for r in recs:
        if not _is_path_allowed(r["param_path"]):
            logger.info("drop (not in whitelist): %s", r["param_path"])
            continue
        kept.append(r)
    return kept


def _filter_against_pending(
    recs: list[dict], pending_paths: set[str],
) -> list[dict]:
    kept = []
    for r in recs:
        if r["param_path"] in pending_paths:
            logger.info("drop (pending exists): %s", r["param_path"])
            continue
        kept.append(r)
    return kept


def _merge_sources(
    claude_recs: list[dict], codex_recs: list[dict],
) -> list[dict]:
    """Cross-LLM merge.

    Same param_path + same proposed_value → 1 row, notes records dual.
    Same param_path + different proposed_value → keep both, notes
    records which LLM. No path collision → pass through.
    """
    by_path_claude = {r["param_path"]: r for r in claude_recs}
    by_path_codex = {r["param_path"]: r for r in codex_recs}
    all_paths = set(by_path_claude) | set(by_path_codex)
    merged: list[dict] = []
    for path in all_paths:
        c = by_path_claude.get(path)
        x = by_path_codex.get(path)
        if c and x and c.get("proposed_value") == x.get("proposed_value"):
            # Agreement — single row.
            row = dict(c)
            row["_source"] = "both"
            row["_notes_extra"] = (
                f"Claude + Codex agreed. Codex rationale: {x.get('rationale')}"
            )
            merged.append(row)
            continue
        if c:
            row = dict(c)
            row["_source"] = "claude"
            merged.append(row)
        if x:
            row = dict(x)
            row["_source"] = "codex"
            merged.append(row)
    return merged


async def _insert_recommendations(
    rows: list[dict], mode: str,
) -> list[int]:
    """Insert merged rows. Returns the new IDs."""
    if not rows:
        return []
    f = get_session_factory()
    new_ids: list[int] = []
    async with f() as session:
        for r in rows:
            agent_type = f"llm_{r.get('_source', 'unknown')}_{mode}"
            extras: list[str] = []
            if r.get("_notes_extra"):
                extras.append(r["_notes_extra"])
            notes = " | ".join(extras) if extras else None
            rec = AgentRecommendation(
                agent_type=agent_type,
                param_path=r["param_path"],
                current_value=r.get("current_value"),
                proposed_value=r["proposed_value"],
                rationale=(r.get("rationale") or "")[:480],
                expected_effect=(r.get("expected_effect") or "")[:320],
                confidence=(r.get("confidence") or "medium")[:10],
                risk=(r.get("risk") or "medium")[:10],
                status="pending",
                notes=notes,
            )
            session.add(rec)
            await session.flush()
            new_ids.append(rec.id)
        await session.commit()
    logger.info("Inserted %d recommendations: ids=%s", len(new_ids), new_ids)
    return new_ids


async def _kickoff_validation(ids: list[int]) -> None:
    """Schedule the auto-backtest worker (sequential via its own lock)."""
    if not ids:
        return
    from services.recommendation_validator import validate_recommendation
    f = get_session_factory()
    for rid in ids:
        asyncio.create_task(validate_recommendation(rid, f))
    logger.info("Spawned %d validator tasks", len(ids))


# ---------------------------------------------------------------------------
# Discord push
# ---------------------------------------------------------------------------


async def _discord_summary(
    mode: str, claude_n: int, codex_n: int, kept_ids: list[int],
) -> None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        from services.notification import DiscordAdapter, AlertLevel
        from config import NotificationConfig
    except Exception as e:
        logger.warning("notification import failed: %s", e)
        return
    webhook = NotificationConfig().discord_webhook_url or os.environ.get(
        "DISCORD_WEBHOOK_URL", "",
    )
    if not webhook:
        return
    today = datetime.now(KST).date()
    title = f"🤖 LLM {mode} recommendations — {today}"
    body_lines = [
        f"Claude proposed: **{claude_n}**",
        f"Codex proposed: **{codex_n}**",
        f"Inserted (after whitelist + dedupe): **{len(kept_ids)}**",
    ]
    if kept_ids:
        body_lines.append(f"IDs: {kept_ids}")
        body_lines.append("Open the dashboard → '🤖 에이전트 권고' to review.")
    body = "\n".join(body_lines)
    level = AlertLevel.INFO if kept_ids else AlertLevel.WARNING
    adapter = DiscordAdapter(webhook_url=webhook)
    await adapter.send_rich(title=title, body=body, level=level, fields={"Mode": mode})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(mode: str, dry_run: bool) -> int:
    if mode not in ("daily", "weekly"):
        raise SystemExit(f"--mode must be daily or weekly, got {mode!r}")
    window_days = 1 if mode == "daily" else 7

    orders = await _gather_orders(window_days)
    snapshot = await _gather_live_snapshot()
    pending = await _gather_pending_recommendations()
    pending_paths = {r["param_path"] for r in pending}
    yaml_text = _read_yaml_text()
    exposure = _load_today_exposure()

    ctx = {
        "orders": orders,
        "snapshot": snapshot,
        "pending": pending,
        "yaml_text": yaml_text,
        "exposure": exposure,
    }
    prompt = _build_prompt(mode, ctx)
    logger.info("Built prompt: %d chars, %d pending rows", len(prompt), len(pending))

    # Both CLIs in parallel. Each LLM gets the same prompt.
    claude_task = asyncio.to_thread(_run_cli, ["claude", "--print"], prompt)
    codex_task = asyncio.to_thread(_run_cli, ["codex", "exec"], prompt)
    claude_raw, codex_raw = await asyncio.gather(claude_task, codex_task)

    # Loud diagnostic when both CLIs returned None — this looks identical
    # to "no recs to propose" but is almost always a PATH / env issue
    # under systemd (see deploy/daily-post-market-analysis.service).
    if claude_raw is None and codex_raw is None:
        logger.error(
            "Both Claude and Codex CLIs returned None — likely missing "
            "from PATH (PATH=%s). Check systemd service Environment=PATH.",
            os.environ.get("PATH", "<unset>"),
        )

    # Persist raw outputs so a failed parse is debuggable (and so the
    # operator can audit what the LLM actually said).
    debug_dir = REPO_ROOT / "data" / "llm_recommendations"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).strftime("%Y-%m-%dT%H-%M-%S")
    if claude_raw is not None:
        (debug_dir / f"{stamp}-claude-{mode}.txt").write_text(claude_raw, encoding="utf-8")
    if codex_raw is not None:
        (debug_dir / f"{stamp}-codex-{mode}.txt").write_text(codex_raw, encoding="utf-8")

    claude_recs = _parse_llm_output(claude_raw)
    codex_recs = _parse_llm_output(codex_raw)
    logger.info(
        "LLM parse: claude=%d codex=%d", len(claude_recs), len(codex_recs),
    )

    claude_recs = _filter_whitelist(claude_recs)
    codex_recs = _filter_whitelist(codex_recs)
    claude_recs = _filter_against_pending(claude_recs, pending_paths)
    codex_recs = _filter_against_pending(codex_recs, pending_paths)

    merged = _merge_sources(claude_recs, codex_recs)
    logger.info("After whitelist + dedupe + merge: %d rows", len(merged))

    if dry_run:
        print(json.dumps(
            {"would_insert": merged}, ensure_ascii=False, default=str, indent=2,
        ))
        return 0

    new_ids = await _insert_recommendations(merged, mode)
    await _kickoff_validation(new_ids)
    await _discord_summary(mode, len(claude_recs), len(codex_recs), new_ids)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-driven AgentRecommendation generator")
    p.add_argument("--mode", choices=["daily", "weekly"], required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be inserts as JSON; do not touch the DB.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(args.mode, args.dry_run)))
