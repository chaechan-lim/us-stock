"""Hermes Phase 3 C2 — counterfactual replay over FunnelEvent data.

When a recommendation's `param_path` isn't in `_BACKTEST_PARAM_MAP`
(the path doesn't affect a backtest-config knob), the validator
previously stored `{skip: "..."}`. With C2, we instead simulate the
proposed change against the last N days of funnel events and report
how many rejections would have passed.

This is a complement to backtest, not a replacement: backtest measures
*portfolio* impact (Ret/Sharpe/MDD); replay measures *funnel* impact
(would-pass count, deploy lift estimate). Together they give the
operator more information than the binary skip/proceed gate.

Currently replayable paths:

  markets.{KR|US}.evaluation_loop.daily_buy_limit
    Funnel rows with reject_reason='daily_limit' would pass under a
    higher limit, modulo the per-day count cap. Count daily rejects
    that fall within the new headroom.

  markets.{KR|US}.evaluation_loop.opening_avoidance_minutes
    Rows with reject_reason='opening_avoidance' would pass if the
    event ts is past the new shorter window (or stay rejected under
    a longer one). Pure ts/minute math.

  markets.{KR|US}.evaluation_loop.sell_cooldown_days
    Rows with reject_reason='sell_cooldown'. For each, we look up the
    most recent SELL of that symbol in the Orders table before the
    event ts → compute days_since → would pass if days_since >= proposed
    days. Requires DB join.

Other paths (whipsaw, same_signal_24h, sizing tokens, ...) need
context not currently captured in FunnelEvent rows. Return
`{not_replayable: "<reason>"}` so the operator still gets the path
listed (not silently dropped).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.timeutil import now_utc_naive
from typing import Any

from sqlalchemy import and_, func, select

from core.models import FunnelEvent, Order

logger = logging.getLogger(__name__)


# Path → (market, reject_reason filter, replay-fn name)
_REPLAY_HANDLERS: dict[str, tuple[str, str, str]] = {
    "markets.KR.evaluation_loop.daily_buy_limit":
        ("KR", "daily_limit", "_replay_daily_limit"),
    "markets.US.evaluation_loop.daily_buy_limit":
        ("US", "daily_limit", "_replay_daily_limit"),
    "markets.KR.evaluation_loop.opening_avoidance_minutes":
        ("KR", "opening_avoidance", "_replay_opening_avoidance"),
    "markets.US.evaluation_loop.opening_avoidance_minutes":
        ("US", "opening_avoidance", "_replay_opening_avoidance"),
    "markets.KR.evaluation_loop.sell_cooldown_days":
        ("KR", "sell_cooldown", "_replay_sell_cooldown"),
    "markets.US.evaluation_loop.sell_cooldown_days":
        ("US", "sell_cooldown", "_replay_sell_cooldown"),
}


# Per-market trading session start time, in market-local wall-clock.
#
# Bug-002 (2026-06-05): the prior comment claimed UTC offset was
# "baked into ts" — wrong. FunnelEvent.ts uses `now_utc_naive` which
# is naive UTC (after the utcnow→core.timeutil migration). Comparing
# UTC `ts.hour*60+ts.minute` to a 09:00 KST anchor wrapped via +1440
# and made `would_pass_under_proposed` ≈100% for every realistic
# threshold, silently biasing operator-facing replay summaries.
# Now: convert ts (naive UTC) to the market's local timezone before
# extracting hour/minute.
_MARKET_OPEN = {
    "KR": (9, 0),    # 09:00 KST
    "US": (9, 30),   # 09:30 ET — DST handled by zoneinfo
}
_MARKET_TZ = {
    "KR": ZoneInfo("Asia/Seoul"),
    "US": ZoneInfo("America/New_York"),
}


def is_replayable(param_path: str) -> bool:
    return param_path in _REPLAY_HANDLERS


async def replay_recommendation(
    session,
    param_path: str,
    current_value: Any,
    proposed_value: Any,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Run counterfactual replay for a recommendation's proposed change.

    Returns a dict suitable for AgentRecommendation.backtest_result.
    On not-replayable paths returns `{not_replayable: ...}` rather than
    raising — the operator still sees something useful in the UI.
    """
    if param_path not in _REPLAY_HANDLERS:
        return {
            "not_replayable": f"path {param_path!r} has no replay handler",
        }

    market, reject_reason, handler_name = _REPLAY_HANDLERS[param_path]
    handler = globals().get(handler_name)
    if handler is None:  # programmer error — handler name typo
        return {"error": f"handler {handler_name!r} not defined"}

    since = now_utc_naive() - timedelta(days=lookback_days)
    return await handler(
        session=session,
        market=market,
        reject_reason=reject_reason,
        since=since,
        current_value=current_value,
        proposed_value=proposed_value,
    )


async def _replay_daily_limit(
    session, market: str, reject_reason: str, since: datetime,
    current_value: Any, proposed_value: Any,
) -> dict[str, Any]:
    """Daily_buy_limit replay.

    Count rejected events per trading day under reject_reason=daily_limit.
    Under the proposed (higher) limit, the first `proposed - current`
    additional rejections per day would have passed.
    """
    try:
        cur = int(current_value)
        prop = int(proposed_value)
    except (TypeError, ValueError):
        return {"not_replayable": "non-integer values"}
    if prop <= cur:
        return {"replayed": 0, "would_pass": 0, "note": "proposed not higher than current"}

    # Group rejections by date
    q = select(
        func.date(FunnelEvent.ts).label("d"),
        func.count().label("n"),
    ).where(
        and_(
            FunnelEvent.market == market,
            FunnelEvent.decision == "rejected",
            FunnelEvent.reject_reason == reject_reason,
            FunnelEvent.ts >= since,
        )
    ).group_by(func.date(FunnelEvent.ts))

    rows = (await session.execute(q)).all()
    total_rejects = sum(int(r.n) for r in rows)
    headroom = prop - cur
    # Each day's would-pass = min(rejects_that_day, headroom)
    would_pass = sum(min(int(r.n), headroom) for r in rows)
    days = len(rows)

    return {
        "method": "funnel_replay",
        "lookback_days": (now_utc_naive() - since).days,
        "filter": f"market={market} reject_reason={reject_reason}",
        "replayed_rejections": total_rejects,
        "would_pass_under_proposed": would_pass,
        "would_pass_rate": (
            round(would_pass / total_rejects, 3) if total_rejects else 0.0
        ),
        "trading_days_with_rejections": days,
        "avg_additional_buys_per_day": (
            round(would_pass / days, 2) if days else 0.0
        ),
    }


async def _replay_opening_avoidance(
    session, market: str, reject_reason: str, since: datetime,
    current_value: Any, proposed_value: Any,
) -> dict[str, Any]:
    """opening_avoidance_minutes replay.

    Event ts already passed the new threshold? Then it would pass under
    proposed. Note: we don't know if other gates would still block it
    after this one — the count is an upper bound on funnel-stage lift.
    """
    try:
        cur = int(current_value)
        prop = int(proposed_value)
    except (TypeError, ValueError):
        return {"not_replayable": "non-integer values"}

    open_h, open_m = _MARKET_OPEN.get(market, (9, 30))
    open_total_min = open_h * 60 + open_m
    local_tz = _MARKET_TZ.get(market, ZoneInfo("UTC"))

    q = select(FunnelEvent.ts).where(
        and_(
            FunnelEvent.market == market,
            FunnelEvent.decision == "rejected",
            FunnelEvent.reject_reason == reject_reason,
            FunnelEvent.ts >= since,
        )
    )
    rows = (await session.execute(q)).all()

    total = len(rows)
    would_pass = 0
    for row in rows:
        # Bug-002: ts is naive UTC (now_utc_naive default). Attach the
        # UTC zone, then convert to market-local before extracting
        # hour/minute so the comparison against `open_total_min` (in
        # market-local wall-clock) is unit-consistent.
        ts = row.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local_ts = ts.astimezone(local_tz)
        minute_of_day = local_ts.hour * 60 + local_ts.minute
        elapsed = minute_of_day - open_total_min
        if elapsed < 0:
            elapsed += 24 * 60     # pre-open noise (rare); wrap
        if elapsed >= prop:
            would_pass += 1

    return {
        "method": "funnel_replay",
        "lookback_days": (now_utc_naive() - since).days,
        "filter": f"market={market} reject_reason={reject_reason}",
        "replayed_rejections": total,
        "would_pass_under_proposed": would_pass,
        "would_pass_rate": (
            round(would_pass / total, 3) if total else 0.0
        ),
        "current_minutes": cur,
        "proposed_minutes": prop,
    }


async def _replay_sell_cooldown(
    session, market: str, reject_reason: str, since: datetime,
    current_value: Any, proposed_value: Any,
) -> dict[str, Any]:
    """sell_cooldown_days replay.

    For each rejected event, find the symbol's most recent SELL order
    before the event ts; compute days_since; check if >= proposed.
    """
    try:
        cur = int(current_value)
        prop = int(proposed_value)
    except (TypeError, ValueError):
        return {"not_replayable": "non-integer values"}

    q = select(FunnelEvent.symbol, FunnelEvent.ts).where(
        and_(
            FunnelEvent.market == market,
            FunnelEvent.decision == "rejected",
            FunnelEvent.reject_reason == reject_reason,
            FunnelEvent.ts >= since,
        )
    )
    rows = (await session.execute(q)).all()
    total = len(rows)
    would_pass = 0
    not_resolved = 0

    for symbol, event_ts in rows:
        # Most recent SELL before event_ts
        last_sell_q = select(func.max(Order.filled_at)).where(
            and_(
                Order.market == market,
                Order.symbol == symbol,
                Order.side == "SELL",
                Order.status == "filled",
                Order.filled_at < event_ts,
            )
        )
        last_sell = (await session.execute(last_sell_q)).scalar()
        if last_sell is None:
            not_resolved += 1
            continue
        days_since = (event_ts - last_sell).total_seconds() / 86400.0
        if days_since >= prop:
            would_pass += 1

    return {
        "method": "funnel_replay",
        "lookback_days": (now_utc_naive() - since).days,
        "filter": f"market={market} reject_reason={reject_reason}",
        "replayed_rejections": total,
        "would_pass_under_proposed": would_pass,
        "would_pass_rate": (
            round(would_pass / total, 3) if total else 0.0
        ),
        "no_sell_history": not_resolved,
        "current_days": cur,
        "proposed_days": prop,
    }
