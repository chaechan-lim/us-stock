"""Portfolio state tracker with DB snapshots.

Tracks balance, positions, equity, and PnL using cached market data.
Saves periodic snapshots to the portfolio_snapshots table for
equity curve tracking and daily PnL calculation.
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models import PortfolioSnapshot
from core.timeutil import now_utc_naive
from data.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

# Anomaly detection: skip snapshot if total_value drops more than this
# fraction vs the previous snapshot (e.g. 0.5 = 50% drop).
ANOMALY_DROP_THRESHOLD = 0.5

# STOCK-46 / P1-F (2026-05-15): Cash flow detection thresholds.
# Old single-threshold (5% for US, 10% for KR) generated noise records —
# intraday position-value swings 5-15% routinely produced 1-2M KRW false
# deposit/withdrawal entries, distorting the TWR metrics that subtract
# them. Examples (live, 2026-05-07~13): −1.03M, +1.66M, −0.89M, +1.13M /
# −1.15M same day — none real. Real deposit on 2026-05-14 was +17.4M.
#
# New: BOTH a relative AND absolute threshold must be exceeded. Real
# deposits are typically large in both. Noise is small in absolute terms
# even when relative threshold trips.
CASH_FLOW_REL_THRESHOLD = 0.20    # 20% of prev_equity (was 5%/10%)
CASH_FLOW_ABS_THRESHOLD_US = 5_000      # $5,000
CASH_FLOW_ABS_THRESHOLD_KR = 5_000_000  # ₩5,000,000
# P1-F.2 (2026-05-22): a single-snapshot swing this large is almost
# always a cash flow regardless of absolute size. KR live (#?) had two
# 44% intraday drops that were real withdrawals but fell below the
# fixed ₩5M abs floor → cash_flow=0 → TWR treated them as -65% MDD.
CASH_FLOW_STRONG_REL_THRESHOLD = 0.30
CASH_FLOW_STRONG_ABS_THRESHOLD_US = 100     # $100 — token floor to skip rounding noise
CASH_FLOW_STRONG_ABS_THRESHOLD_KR = 100_000 # ₩100,000


def detect_cash_flow(
    prev_total: float,
    new_total: float,
    rel_threshold: float | None = None,
    abs_threshold: float | None = None,
    strong_rel_threshold: float | None = None,
    strong_abs_threshold: float | None = None,
) -> float:
    """Detect external deposit/withdrawal between two snapshots.

    P1-F.2 (2026-05-22): two-rule detection.

    1. **Strong rel rule** — single-snapshot swing ≥ strong_rel_threshold
       (30%) with a token absolute floor (KR ₩100k, US $100) almost
       always indicates a cash flow. Position-value swings of 30%+ in
       one snapshot are rare; legitimate market moves of that
       magnitude happen over a day, not seconds. KR live (5-22) had
       two 44% intraday equity drops that were real withdrawals; the
       previous dual-threshold rule missed them because they were
       under the fixed ₩5M abs floor (the account was only ~₩10M).

    2. **Dual-threshold rule** — moderate swings (20–30% rel) need to
       also clear an absolute floor. Preserves the P1-F false-positive
       protection from STOCK-46.

    Returns the detected cash flow amount (positive=deposit, negative=
    withdrawal). 0.0 when neither rule fires.
    """
    if prev_total <= 0:
        return 0.0

    raw_cf = new_total - prev_total
    eff_rel = rel_threshold if rel_threshold is not None else CASH_FLOW_REL_THRESHOLD
    eff_abs = abs_threshold if abs_threshold is not None else CASH_FLOW_ABS_THRESHOLD_US
    eff_strong_rel = (
        strong_rel_threshold if strong_rel_threshold is not None
        else CASH_FLOW_STRONG_REL_THRESHOLD
    )
    eff_strong_abs = (
        strong_abs_threshold if strong_abs_threshold is not None
        else CASH_FLOW_STRONG_ABS_THRESHOLD_US
    )

    abs_cf = abs(raw_cf)
    rel = abs_cf / prev_total

    # Rule 1: strong relative move — always count.
    if rel >= eff_strong_rel and abs_cf > eff_strong_abs:
        return raw_cf
    # Rule 2: dual-threshold for moderate moves.
    if abs_cf > eff_rel * prev_total and abs_cf > eff_abs:
        return raw_cf
    return 0.0


class PortfolioManager:
    """Track portfolio state and persist snapshots to DB."""

    def __init__(
        self,
        market_data: MarketDataService,
        session_factory: async_sessionmaker[AsyncSession],
        market: str = "US",
    ):
        self._market_data = market_data
        self._session_factory = session_factory
        self._market = market

    async def get_summary(self) -> dict:
        """Get current portfolio state: balance, positions, total equity, unrealized PnL."""
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()

        invested = sum(p.quantity * p.avg_price for p in positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_equity = balance.total  # already includes position market value

        return {
            "cash": balance.available,
            "invested": invested,
            "total_equity": total_equity,
            "unrealized_pnl": unrealized_pnl,
            "position_count": len(positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ],
        }

    async def save_snapshot(self) -> None:
        """Save current portfolio state to portfolio_snapshots table.

        Includes anomaly detection: if total_value drops more than
        ANOMALY_DROP_THRESHOLD (50%) vs the previous snapshot, the
        snapshot is skipped and a warning is logged.  This guards
        against timing issues where balance.total does not yet include
        position market value (STOCK-45).
        """
        balance = await self._market_data.get_balance()
        positions = await self._market_data.get_positions()

        invested = sum(p.quantity * p.avg_price for p in positions)
        position_market_value = sum(
            p.quantity * p.current_price for p in positions if p.current_price > 0
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_equity = balance.total  # already includes position market value

        # STOCK-45: Warn about positions with stale/missing price data.
        zero_price = [p.symbol for p in positions if p.quantity > 0 and p.current_price <= 0]
        if zero_price:
            logger.warning(
                "[%s] %d positions with current_price<=0: %s",
                self._market,
                len(zero_price),
                zero_price,
            )

        # STOCK-45: Detect when balance.total excludes position value.
        # If balance.total < cash + 50% of position market value,
        # positions are likely missing from the total.
        cash_plus_half_pos = balance.available + position_market_value * 0.5
        if position_market_value > 0 and total_equity < cash_plus_half_pos:
            logger.warning(
                "[%s] Snapshot anomaly: total_equity=%.2f < cash=%.2f + "
                "50%% position_value=%.2f — positions may be excluded from "
                "balance.total. Skipping snapshot.",
                self._market,
                total_equity,
                balance.available,
                position_market_value,
            )
            return

        # STOCK-45: Compare with previous snapshot — skip on anomalous drop.
        prev = await self._get_last_snapshot()
        if prev is not None and prev.total_value_usd > 0:
            drop_ratio = 1.0 - total_equity / prev.total_value_usd
            if drop_ratio > ANOMALY_DROP_THRESHOLD:
                logger.warning(
                    "[%s] Snapshot anomaly: total_equity=%.2f vs "
                    "previous=%.2f (%.1f%% drop). Skipping snapshot.",
                    self._market,
                    total_equity,
                    prev.total_value_usd,
                    drop_ratio * 100,
                )
                return

        daily_pnl = await self._calculate_daily_pnl(total_equity)

        # P1-F (2026-05-15): dual-threshold cash-flow detection. Old
        # single-threshold (10% KR / 5% US) was triggering on intraday
        # position-value swings. New: BOTH relative (20%) AND absolute
        # (₩5M for KR, $5K for US) must be exceeded.
        # P1-F.2 (2026-05-22): added strong-rel rule (≥30% single-snapshot
        # swing → always count) so small accounts whose real ₩2-4M cash
        # flows fell below the fixed ₩5M abs floor aren't silently missed.
        cash_flow = 0.0
        abs_thr = (
            CASH_FLOW_ABS_THRESHOLD_KR
            if self._market == "KR"
            else CASH_FLOW_ABS_THRESHOLD_US
        )
        strong_abs_thr = (
            CASH_FLOW_STRONG_ABS_THRESHOLD_KR
            if self._market == "KR"
            else CASH_FLOW_STRONG_ABS_THRESHOLD_US
        )
        # P1-F.3 (2026-06-04): suppress cash-flow detection during this
        # market's REGULAR session. Live observed false positive:
        # 6/4 01:32 UTC (= 10:32 KST, KR market active) detected
        # ₩+5.03M "deposit" that was actually a transient KIS CTRP6548R
        # accounting hiccup (integrated total briefly recomputed after
        # an ETF buy + sell round-trip). True deposits land in 입출금
        # outside trading hours; intraday CTRP swings are normally just
        # evaluation timing. Only the strong rule (≥30% rel) still fires
        # during sessions — a real intraday flow of that magnitude would
        # be impossible to miss anyway.
        try:
            from engine.scheduler import (
                MarketPhase, get_kr_market_phase, get_market_phase,
            )
            phase = (
                get_kr_market_phase() if self._market == "KR"
                else get_market_phase()
            )
            in_session = phase == MarketPhase.REGULAR
        except Exception:
            in_session = False
        if prev is not None and prev.total_value_usd > 0:
            # Raise thresholds dramatically during session to make only
            # genuine 30%+ swings count.
            session_abs = abs_thr * 100 if in_session else abs_thr
            session_rel = 1.0 if in_session else CASH_FLOW_REL_THRESHOLD
            cash_flow = detect_cash_flow(
                prev_total=prev.total_value_usd,
                new_total=total_equity,
                rel_threshold=session_rel,
                abs_threshold=session_abs,
                strong_abs_threshold=strong_abs_thr,
            )
            if cash_flow != 0.0:
                action = "deposit" if cash_flow > 0 else "withdrawal"
                logger.info(
                    "[%s] Cash flow detected: %.2f (%s) (in_session=%s)",
                    self._market, cash_flow, action, in_session,
                )

        # STOCK-58: Capture exchange rate at snapshot time for accurate historical conversions.
        # 2026-04-14: Removed `if self._market == "US"` guard — KR snapshots
        # also need the rate for equity timeline building. Without it,
        # usd_krw_rate=0 was stored for all KR snapshots, breaking dashboard
        # return calculations when US+KR timelines were combined.
        usd_krw_rate = None
        try:
            usd_krw_rate = await self._market_data.get_exchange_rate()
            if usd_krw_rate is not None and usd_krw_rate <= 0:
                usd_krw_rate = None
        except Exception as e:
            logger.debug("[%s] Failed to fetch exchange rate for snapshot: %s", self._market, e)

        # 2026-05-06: Capture KIS CTRP6548R integrated total (KRW) when
        # the underlying adapter has it cached. Equity-history "combined"
        # mode reads this so it doesn't have to add KR.total + US.total
        # (which double-counts the shared deposit pool under 통합증거금).
        integrated_total_krw = None
        adapter = getattr(self._market_data, "_adapter", None)
        cached = getattr(adapter, "_integrated_total_asset", None)
        try:
            if cached is not None and float(cached) > 0:
                integrated_total_krw = float(cached)
        except (TypeError, ValueError):
            pass  # AsyncMock or other non-numeric — leave None

        snapshot = PortfolioSnapshot(
            market=self._market,
            total_value_usd=total_equity,
            cash_usd=balance.available,
            invested_usd=invested,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            cash_flow=cash_flow,
            usd_krw_rate=usd_krw_rate,
            integrated_total_krw=integrated_total_krw,
            recorded_at=now_utc_naive(),
        )

        async with self._session_factory() as session:
            session.add(snapshot)
            await session.commit()

        logger.info(
            "Portfolio snapshot saved: equity=%.2f, cash=%.2f, pnl=%.2f",
            total_equity,
            balance.available,
            daily_pnl or 0.0,
        )

    async def _get_last_snapshot(self) -> PortfolioSnapshot | None:
        """Fetch the most recent snapshot for this market."""
        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(desc(PortfolioSnapshot.recorded_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def delete_snapshots_by_ids(self, ids: list[int]) -> int:
        """Delete snapshots by ID list. Returns count of deleted rows.

        Admin utility for correcting bad snapshots (e.g. STOCK-45
        anomalous data from timing issues).
        """
        if not ids:
            return 0

        async with self._session_factory() as session:
            stmt = (
                delete(PortfolioSnapshot)
                .where(PortfolioSnapshot.id.in_(ids))
                .where(PortfolioSnapshot.market == self._market)
            )
            result = await session.execute(stmt)
            await session.commit()
            deleted = result.rowcount

        logger.info(
            "[%s] Deleted %d anomalous snapshots (ids=%s)",
            self._market,
            deleted,
            ids,
        )
        return deleted

    async def _calculate_daily_pnl(self, current_equity: float) -> float | None:
        """Calculate PnL vs the first snapshot of today."""
        today_start = now_utc_naive().replace(hour=0, minute=0, second=0, microsecond=0)

        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.recorded_at >= today_start)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(PortfolioSnapshot.recorded_at.asc())
                .limit(1)
            )
            result = await session.execute(stmt)
            first_today = result.scalar_one_or_none()

        if first_today is None:
            return None

        return current_equity - first_today.total_value_usd

    async def get_daily_pnl(self) -> float:
        """Calculate today's PnL from snapshots."""
        balance = await self._market_data.get_balance()
        current_equity = balance.total  # already includes position market value

        pnl = await self._calculate_daily_pnl(current_equity)
        return pnl if pnl is not None else 0.0

    async def get_equity_history(self, days: int = 30) -> list[dict]:
        """Get equity curve from snapshots."""
        since = now_utc_naive() - timedelta(days=days)

        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.recorded_at >= since)
                .where(PortfolioSnapshot.market == self._market)
                .order_by(PortfolioSnapshot.recorded_at.asc())
            )
            result = await session.execute(stmt)
            snapshots = result.scalars().all()

        return [
            {
                "date": s.recorded_at.strftime("%Y-%m-%d %H:%M") if s.recorded_at else None,
                "total_value_usd": s.total_value_usd,
                "cash_usd": s.cash_usd,
                "invested_usd": s.invested_usd,
                "unrealized_pnl": s.unrealized_pnl,
                "daily_pnl": s.daily_pnl,
                "cash_flow": getattr(s, "cash_flow", 0.0) or 0.0,
                "integrated_total_krw": getattr(s, "integrated_total_krw", None),
            }
            for s in snapshots
        ]

    async def get_combined_equity_history(self, days: int = 30) -> list[dict]:
        """Equity curve using KIS CTRP6548R integrated total (KRW).

        Pulls KR snapshots' `integrated_total_krw` field (the KR adapter is
        the only one that calls CTRP6548R). Single source of truth — no
        US+KR addition, no double-counting of the shared deposit pool.

        Returns rows with `total_value_krw` so the field name matches the
        unit. Rows where the field is NULL (legacy snapshots before
        2026-05-06) are skipped.
        """
        since = now_utc_naive() - timedelta(days=days)

        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.recorded_at >= since)
                .where(PortfolioSnapshot.market == "KR")
                .where(PortfolioSnapshot.integrated_total_krw.isnot(None))
                .order_by(PortfolioSnapshot.recorded_at.asc())
            )
            result = await session.execute(stmt)
            snapshots = result.scalars().all()

        return [
            {
                "date": s.recorded_at.strftime("%Y-%m-%d %H:%M") if s.recorded_at else None,
                "total_value_krw": s.integrated_total_krw,
                "usd_krw_rate": s.usd_krw_rate,
                # P1-D (2026-05-14): forward cash_flow so the TWR metrics
                # path can exclude deposit/withdrawal effects.
                "cash_flow": getattr(s, "cash_flow", 0.0) or 0.0,
            }
            for s in snapshots
        ]
