"""Daily post-market analysis — deterministic SQL + Discord push.

Replaces the LLM-based task_trade_review (disabled 2026-05-20). LLM
recommendations were inconsistent; deterministic metrics are sufficient
for daily ops. Weekly LLM analysis runs separately (see
docs/CLAUDE_WEEKLY_ANALYSIS.md).

Runs daily after US close (06:00 KST = US 17:00 EST):
  - Yesterday's daily PnL by market
  - Cleanup count + PnL (P1 health proxy)
  - Top winners / losers (round-trip)
  - SPY benchmark same-day move
  - Open positions snapshot (gain/flat/loss buckets)
  - F1 funnel today summary
  - Trend vs prior 5-day baseline

Posts a rich Discord embed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import text

from db.session import get_session_factory
from services.notification import DiscordAdapter, AlertLevel
from services.exposure_tracker import collect_snapshot, write_snapshot
from config import NotificationConfig


KST = timezone(timedelta(hours=9))
KRW_USD = 1370.0


def _usd_eq(market: str, pnl: float | None) -> float:
    if pnl is None:
        return 0.0
    return float(pnl) / KRW_USD if market == "KR" else float(pnl)


async def _daily_aggregates(target: date) -> dict[str, Any]:
    """Aggregate yesterday's trades + 5-day baseline for trend comparison."""
    f = get_session_factory()
    end_utc = datetime.combine(target + timedelta(days=1), datetime.min.time()).astimezone(timezone.utc).replace(tzinfo=None)
    start_utc = datetime.combine(target, datetime.min.time()).astimezone(timezone.utc).replace(tzinfo=None)
    baseline_start = (target - timedelta(days=5))
    base_start_utc = datetime.combine(baseline_start, datetime.min.time()).astimezone(timezone.utc).replace(tzinfo=None)

    async with f() as s:
        # Target day stats
        r = await s.execute(text("""
            SELECT market, side, status, strategy_name,
                   count(*) AS n,
                   sum(pnl) AS pnl
            FROM orders
            WHERE is_paper=FALSE AND filled_at >= :s AND filled_at < :e
            GROUP BY market, side, status, strategy_name
        """), {"s": start_utc, "e": end_utc})
        rows = list(r)

        # 5-day baseline (excluding target) — average daily metrics
        r2 = await s.execute(text("""
            SELECT date(filled_at) AS d, market, side, status, strategy_name,
                   sum(pnl) AS pnl
            FROM orders
            WHERE is_paper=FALSE
              AND filled_at >= :bs AND filled_at < :s
              AND side='SELL' AND status='filled'
            GROUP BY date(filled_at), market, side, status, strategy_name
        """), {"bs": base_start_utc, "s": start_utc})
        base_rows = list(r2)

    # Aggregate target day
    sells = cleanups = buys = 0
    pnl_total_usd = 0.0
    cleanup_pnl_usd = 0.0
    by_strat: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for row in rows:
        n = row.n or 0
        pnl_usd = _usd_eq(row.market, row.pnl)
        if row.side == "BUY" and row.status == "filled":
            buys += n
        if row.side == "SELL" and row.status == "filled":
            sells += n
            pnl_total_usd += pnl_usd
            strat = row.strategy_name or "?"
            by_strat[strat]["n"] += n
            by_strat[strat]["pnl"] += pnl_usd
            if strat == "position_cleanup":
                cleanups += n
                cleanup_pnl_usd += pnl_usd

    # 5-day baseline daily averages
    base_daily = defaultdict(lambda: {"sells": 0, "cleanups": 0, "pnl": 0.0})
    for row in base_rows:
        d = str(row.d)
        pnl_usd = _usd_eq(row.market, row.pnl)
        base_daily[d]["sells"] += 1
        base_daily[d]["pnl"] += pnl_usd
        if row.strategy_name == "position_cleanup":
            base_daily[d]["cleanups"] += 1
    n_days = max(1, len(base_daily))
    baseline_avg = {
        "sells_per_day": sum(v["sells"] for v in base_daily.values()) / n_days,
        "cleanups_per_day": sum(v["cleanups"] for v in base_daily.values()) / n_days,
        "pnl_per_day": sum(v["pnl"] for v in base_daily.values()) / n_days,
    }

    return {
        "date": target,
        "buys": buys,
        "sells": sells,
        "cleanups": cleanups,
        "pnl_usd": round(pnl_total_usd, 2),
        "cleanup_pnl_usd": round(cleanup_pnl_usd, 2),
        "by_strat": dict(by_strat),
        "baseline_5d_avg": baseline_avg,
    }


def _spy_same_day(target: date) -> float | None:
    """SPY close-to-close return for the target trading day."""
    try:
        df = yf.download(
            "SPY", start=target - timedelta(days=5),
            end=target + timedelta(days=2),
            progress=False, auto_adjust=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        # find target date row
        idx_dates = [d.date() for d in closes.index]
        if target not in idx_dates:
            return None
        i = idx_dates.index(target)
        if i < 1:
            return None
        prev = float(closes.iloc[i - 1])
        cur = float(closes.iloc[i])
        return (cur - prev) / prev * 100 if prev > 0 else None
    except Exception:
        return None


async def _open_positions() -> dict[str, int]:
    """US open positions classified by unrealized PnL%."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8001/api/v1/portfolio/positions?market=US",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return {}
                positions = await resp.json()
        gain = sum(1 for p in positions if (p.get("unrealized_pnl_pct") or 0) > 0)
        loss = sum(1 for p in positions if (p.get("unrealized_pnl_pct") or 0) < -2)
        total = len(positions)
        flat = total - gain - loss
        return {"total": total, "gain": gain, "flat": flat, "loss": loss}
    except Exception:
        return {}


async def _funnel_today() -> dict[str, Any]:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8001/api/v1/engine/rejection-funnel",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return {}
                return await resp.json()
    except Exception:
        return {}


def _format_report(daily: dict, spy_pct: float | None,
                   positions: dict, funnel: dict) -> tuple[str, str, dict]:
    d = daily["date"]
    spy_str = f"{spy_pct:+.2f}%" if spy_pct is not None else "—"
    base = daily["baseline_5d_avg"]

    # Verdict: comparing today vs 5-day baseline
    pnl_diff = daily["pnl_usd"] - base["pnl_per_day"]
    cleanups_diff = daily["cleanups"] - base["cleanups_per_day"]

    title = f"📊 Daily post-market — {d}"
    body_lines = [
        f"**Daily PnL**: `${daily['pnl_usd']:+.2f}` "
        f"(vs 5d avg `${base['pnl_per_day']:+.2f}` → "
        f"`{'↑' if pnl_diff > 0 else '↓'} ${abs(pnl_diff):.0f}`)",
        f"**Cleanups**: `{daily['cleanups']}` "
        f"(vs 5d avg `{base['cleanups_per_day']:.1f}/day` → "
        f"`{'↑' if cleanups_diff > 0 else '↓'} {abs(cleanups_diff):.1f}`)",
        f"**Cleanup PnL**: `${daily['cleanup_pnl_usd']:+.2f}`",
        f"**Buys / Sells**: `{daily['buys']}` / `{daily['sells']}`",
        f"**SPY same day**: `{spy_str}`",
    ]
    if positions:
        body_lines.append(
            f"**US positions**: {positions['total']} "
            f"({positions['gain']}↑ / {positions['flat']}~ / {positions['loss']}↓<-2%)"
        )
    if funnel:
        us = funnel.get("US") or {}
        if us:
            body_lines.append(
                f"**US funnel today**: {us.get('buy_signals_total', 0)} signals → "
                f"{us.get('buys_placed', 0)} placed "
                f"(fill {(us.get('fill_rate') or 0) * 100:.1f}%)"
            )

    # Top 3 strategies by abs PnL
    by_strat = daily["by_strat"]
    if by_strat:
        sorted_s = sorted(by_strat.items(), key=lambda kv: -abs(kv[1]["pnl"]))[:3]
        body_lines.append("")
        body_lines.append("**Top strategies (by |PnL|):**")
        for name, v in sorted_s:
            sign = "+" if v["pnl"] >= 0 else ""
            body_lines.append(f"  `{name}` × {v['n']}: `{sign}${v['pnl']:.0f}`")

    # Verdict line
    if daily["cleanups"] >= base["cleanups_per_day"] + 3:
        verdict = "⚠️ cleanup count elevated — check P1 / market state"
        level = AlertLevel.WARNING
    elif daily["pnl_usd"] < base["pnl_per_day"] - 50:
        verdict = "⚠️ PnL below 5-day baseline"
        level = AlertLevel.WARNING
    else:
        verdict = "✓ within normal range"
        level = AlertLevel.INFO

    body_lines.append("")
    body_lines.append(verdict)

    body = "\n".join(body_lines)
    fields = {"Date": str(d)}
    return title, body, fields, level


def _save_artifact(target: date, daily: dict, spy_pct: float | None,
                   positions: dict, funnel: dict,
                   title: str, body: str, level: AlertLevel) -> str:
    """Persist a JSON artifact for the dashboard to read.

    Writes to data/daily_analyses/{date}.json. Atomic write via tmp+rename
    so a partial write never corrupts what the API serves.
    """
    import json
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    out_dir = os.path.join(repo_root, "data", "daily_analyses")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "version": 1,
        "date": str(target),
        "generated_at": datetime.now(KST).isoformat(),
        "title": title,
        "body": body,
        "level": level.value if hasattr(level, "value") else str(level),
        "daily": daily,
        "spy_pct": spy_pct,
        "positions": positions,
        "funnel": funnel,
    }
    out = os.path.join(out_dir, f"{target}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out)
    return out


async def main() -> None:
    # Target = previous trading day (US close was ~5 hours ago at 06:00 KST)
    now_kst = datetime.now(KST)
    raw_target = (now_kst - timedelta(days=1)).date()

    # M16 (2026-06-07): skip non-trading days. Sunday-KST 06:00 was
    # firing against Saturday-ET (no session) and still posting a
    # neutral verdict + spawning generate_recommendations.py — wasted
    # LLM quota and seeded the dashboard with empty "latest" rows.
    # Walk back to the last real US trading day; if that doesn't move
    # (yesterday WAS a trading day) proceed normally.
    from services.holiday_calendar import (
        is_us_trading_day,
        previous_us_trading_day,
    )

    if not is_us_trading_day(raw_target):
        target = previous_us_trading_day(raw_target)
        print(
            f"Non-trading day skip: raw={raw_target} → target={target} "
            f"(US session inactive on raw target)"
        )
        # If we're more than 3 days behind the last session (weekend +
        # long holiday), still produce the report for the last actual
        # session so the operator gets a Monday-morning snapshot. But
        # don't post twice on consecutive non-trading days — only fire
        # when raw_target is exactly the day AFTER the last trading
        # day (i.e. we're on Saturday-KST = Friday-ET+1).
        gap_days = (raw_target - target).days
        if gap_days > 1:
            print(
                f"Already-reported gap (gap_days={gap_days}); "
                f"skipping duplicate run"
            )
            return
    else:
        target = raw_target

    daily = await _daily_aggregates(target)
    spy_pct = _spy_same_day(target)
    positions = await _open_positions()
    funnel = await _funnel_today()

    title, body, fields, level = _format_report(daily, spy_pct, positions, funnel)

    # Persist artifact for the dashboard
    artifact_path = _save_artifact(
        target, daily, spy_pct, positions, funnel, title, body, level,
    )
    print(f"Artifact: {artifact_path}")

    # Console output (for systemd journal)
    print(f"{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(body)
    print()

    # Discord push (env-driven via pydantic-settings — same path as backend)
    webhook = NotificationConfig().discord_webhook_url
    if not webhook:
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if webhook:
        adapter = DiscordAdapter(webhook_url=webhook)
        sent = await adapter.send_rich(
            title=title, body=body, level=level, fields=fields,
        )
        print(f"Discord: {'✓ sent' if sent else '✗ send failed'}")
    else:
        print("Discord webhook not configured — skipped push")

    # Hermes Phase 0+1: exposure snapshot + sentinel.
    # Runs before the LLM step so Phase 2 (funnel-aware prompt) can
    # later read today's snapshot. Failures here must not break the
    # daily report — wrap defensively.
    await _run_exposure_snapshot()

    # Chain the LLM recommendation generators (#60). Run in-line (await)
    # instead of detached subprocess: systemd's default KillMode=
    # control-group was reaping the spawned children when the daily
    # service exited (no LLM artifacts written 5-23 ~ 5-27). The daily
    # report is the source-of-truth either way — LLM failures only
    # affect the recommendation queue, not the deterministic numbers.
    # Weekly only fires on Monday runs (KST).
    await _run_llm_recommendations(now_kst)


async def _run_exposure_snapshot() -> None:
    """Compute + persist today's exposure snapshot, push sentinel flags."""
    from pathlib import Path

    try:
        history_dir = Path(__file__).resolve().parent.parent / "data" / "exposure_history"
        snap = await collect_snapshot(history_dir=history_dir)
        out = write_snapshot(snap, history_dir)
        print(f"Exposure snapshot: {out}")
        for market, exp in snap.markets.items():
            print(
                f"  {market}: deployed={exp.deployed_pct*100:.1f}% "
                f"slot_fill={exp.slot_fill_ratio*100:.1f}% "
                f"placeholders={exp.placeholder_count}/{exp.position_count} "
                f"idle_days={exp.cash_idle_days}"
            )
        if snap.flags:
            print(f"⚠ Sentinel flags ({len(snap.flags)}):")
            for f in snap.flags:
                print(f"  [{f.severity}] {f.market} {f.flag}: {f.detail}")
            await _push_sentinel_discord(snap.flags)
        else:
            print("✓ No sentinel flags")
    except Exception as e:
        print(f"Exposure snapshot failed: {e}")


async def _push_sentinel_discord(flags) -> None:
    webhook = NotificationConfig().discord_webhook_url or os.environ.get(
        "DISCORD_WEBHOOK_URL", "",
    )
    if not webhook:
        return
    title = f"⚠ Exposure sentinel — {len(flags)} flag(s)"
    body = "\n".join(
        f"**{f.market}** `{f.flag}`: {f.detail}" for f in flags
    )
    try:
        adapter = DiscordAdapter(webhook_url=webhook)
        await adapter.send_rich(
            title=title, body=body, level=AlertLevel.WARNING, fields={},
        )
    except Exception as e:
        print(f"Sentinel Discord push failed: {e}")


async def _run_llm_recommendations(now_kst: datetime) -> None:
    script_path = os.path.join(os.path.dirname(__file__), "generate_recommendations.py")
    venv_python = sys.executable
    for mode in ("daily",) + (("weekly",) if now_kst.weekday() == 0 else ()):
        try:
            proc = await asyncio.create_subprocess_exec(
                venv_python, script_path, "--mode", mode,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
            except asyncio.TimeoutError:
                proc.kill()
                print(f"LLM {mode} recommendations: timed out at 10min")
                continue
            print(
                f"LLM {mode} recommendations: rc={proc.returncode}"
                + (f" (last 200 chars: {stdout[-200:].decode(errors='replace')})"
                   if proc.returncode != 0 else "")
            )
        except Exception as e:
            print(f"LLM {mode} run failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
