"""Backfill cash_flow column with P1-F.2 strong-rel rule.

Existing portfolio_snapshots rows that were under the prior dual-
threshold (rel ≥ 20% + abs ≥ ₩5M/$5K) had cash_flow=0 even when the
swing was a real deposit/withdrawal. P1-F.2 adds a strong-rel rule
(≥30% single-snapshot swing + token abs floor) that catches small-
account flows like the KR live ₩4.3M withdrawal that fell below the
fixed ₩5M abs floor.

Run once after deploying the P1-F.2 code change:

    cd /home/chans/us-stock
    venv/bin/python scripts/backfill_cash_flow_p1f2.py            # dry-run
    venv/bin/python scripts/backfill_cash_flow_p1f2.py --apply    # write

Idempotent: only updates rows where cash_flow was previously 0 and
the new logic detects a non-zero flow. Rows already non-zero are
left alone.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import select, update

from core.models import PortfolioSnapshot
from db.session import get_session_factory
from engine.portfolio_manager import (
    CASH_FLOW_ABS_THRESHOLD_KR,
    CASH_FLOW_ABS_THRESHOLD_US,
    CASH_FLOW_REL_THRESHOLD,
    CASH_FLOW_STRONG_ABS_THRESHOLD_KR,
    CASH_FLOW_STRONG_ABS_THRESHOLD_US,
    CASH_FLOW_STRONG_REL_THRESHOLD,
    detect_cash_flow,
)


async def backfill(apply: bool) -> None:
    f = get_session_factory()
    updates: list[tuple[int, str, float, float, float]] = []

    async with f() as s:
        for market in ("US", "KR"):
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.market == market)
                .order_by(PortfolioSnapshot.recorded_at)
            )
            rows = (await s.execute(stmt)).scalars().all()
            print(f"[{market}] {len(rows)} snapshots")
            prev = None
            for row in rows:
                if prev is None:
                    prev = row
                    continue
                if prev.total_value_usd and prev.total_value_usd > 0:
                    new_cf = detect_cash_flow(
                        prev_total=prev.total_value_usd,
                        new_total=row.total_value_usd or 0,
                        rel_threshold=CASH_FLOW_REL_THRESHOLD,
                        abs_threshold=(
                            CASH_FLOW_ABS_THRESHOLD_KR if market == "KR"
                            else CASH_FLOW_ABS_THRESHOLD_US
                        ),
                        strong_rel_threshold=CASH_FLOW_STRONG_REL_THRESHOLD,
                        strong_abs_threshold=(
                            CASH_FLOW_STRONG_ABS_THRESHOLD_KR if market == "KR"
                            else CASH_FLOW_STRONG_ABS_THRESHOLD_US
                        ),
                    )
                    existing = float(getattr(row, "cash_flow", 0.0) or 0.0)
                    if existing == 0.0 and new_cf != 0.0:
                        updates.append((
                            row.id, market, prev.total_value_usd or 0.0,
                            row.total_value_usd or 0.0, new_cf,
                        ))
                prev = row

    if not updates:
        print("No updates needed.")
        return

    print(f"\n{'Would update' if not apply else 'Updating'} {len(updates)} rows:")
    for rid, m, p, n, cf in updates:
        print(f"  id={rid} [{m}] prev={p:,.0f} new={n:,.0f} → cf={cf:+,.0f}")

    if apply:
        async with f() as s:
            for rid, _, _, _, cf in updates:
                await s.execute(
                    update(PortfolioSnapshot)
                    .where(PortfolioSnapshot.id == rid)
                    .values(cash_flow=cf)
                )
            await s.commit()
        print(f"\n✓ Applied {len(updates)} updates.")
    else:
        print("\n(dry-run; re-run with --apply to write)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="Write to DB (default: dry-run)")
    args = p.parse_args()
    asyncio.run(backfill(apply=args.apply))


if __name__ == "__main__":
    main()
