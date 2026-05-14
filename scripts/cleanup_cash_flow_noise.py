"""P1-F (2026-05-15) — zero out cash_flow records that fail the new
dual-threshold detection.

Before this PR detect_cash_flow used a single 5%/10% relative threshold
that triggered on normal intraday position-value swings. Live DB has
~14 KR false-positive records in the past month (e.g. -1.03M on a 14%
position-value drop, no actual withdrawal). These corrupt the TWR
metrics computation by subtracting nonexistent deposits.

This script re-evaluates each non-zero cash_flow record using the new
dual threshold (rel 20% AND abs ₩5M for KR / $5k for US) and zeros
out the ones that don't meet both bars.

Safe to run multiple times — idempotent. Outputs a summary of what
was cleaned.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import select, update

from db.session import get_session_factory
from core.models import PortfolioSnapshot
from engine.portfolio_manager import (
    CASH_FLOW_ABS_THRESHOLD_KR,
    CASH_FLOW_ABS_THRESHOLD_US,
    CASH_FLOW_REL_THRESHOLD,
    detect_cash_flow,
)


async def main(dry_run: bool = False) -> None:
    factory = get_session_factory()

    async with factory() as session:
        # Pull all snapshots ordered by (market, recorded_at) so we can
        # compute prev_total per market.
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.cash_flow != 0)
            .order_by(PortfolioSnapshot.market, PortfolioSnapshot.recorded_at)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

        # For each non-zero cash_flow row, find prev snapshot by market
        prev_by_market: dict[str, PortfolioSnapshot] = {}
        all_stmt = (
            select(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.market, PortfolioSnapshot.recorded_at)
        )
        all_rows = (await session.execute(all_stmt)).scalars().all()
        # Build index: (market, recorded_at) → prev_total_usd
        prev_index: dict[tuple[str, str], float] = {}
        last_per_market: dict[str, float] = {}
        for r in all_rows:
            key = (r.market or "", r.recorded_at.isoformat() if r.recorded_at else "")
            prev_index[key] = last_per_market.get(r.market or "", 0.0)
            if r.total_value_usd:
                last_per_market[r.market or ""] = float(r.total_value_usd)

        cleared = []
        kept = []
        for r in rows:
            market = r.market or ""
            key = (market, r.recorded_at.isoformat() if r.recorded_at else "")
            prev_total = prev_index.get(key, 0.0)
            cur_total = float(r.total_value_usd or 0)
            abs_thr = (
                CASH_FLOW_ABS_THRESHOLD_KR
                if market == "KR"
                else CASH_FLOW_ABS_THRESHOLD_US
            )
            new_cf = detect_cash_flow(
                prev_total=prev_total,
                new_total=cur_total,
                rel_threshold=CASH_FLOW_REL_THRESHOLD,
                abs_threshold=abs_thr,
            )
            old_cf = float(r.cash_flow or 0)
            if abs(new_cf) < 0.5 and abs(old_cf) > 0:
                cleared.append((r, old_cf, prev_total, cur_total))
            else:
                kept.append((r, old_cf, new_cf))

        # Report
        print(f"=== cash_flow re-evaluation ===")
        print(f"Total non-zero records:     {len(rows)}")
        print(f"  to be zeroed (noise):     {len(cleared)}")
        print(f"  to be kept (real):        {len(kept)}")
        print()
        print("--- KEPT (real deposits/withdrawals) ---")
        for r, old, new in kept:
            print(f"  {r.recorded_at}  {r.market}  {old:>+15,.0f}  (kept)")
        print()
        print("--- ZEROED (noise) ---")
        for r, old, prev, cur in cleared:
            rel = abs(old) / prev * 100 if prev > 0 else 0
            print(f"  {r.recorded_at}  {r.market}  "
                  f"prev={prev:>15,.0f}  cur={cur:>15,.0f}  "
                  f"old_cf={old:>+15,.0f}  rel={rel:.1f}%")

        if dry_run:
            print("\n(dry run — no changes written)")
            return

        if not cleared:
            print("\nNothing to clean.")
            return

        # Apply zeros
        ids = [r.id for r, _, _, _ in cleared]
        await session.execute(
            update(PortfolioSnapshot)
            .where(PortfolioSnapshot.id.in_(ids))
            .values(cash_flow=0.0)
        )
        await session.commit()
        print(f"\n✓ Zeroed {len(cleared)} records.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry))
