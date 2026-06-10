"""Agent context service — persistent memory for AI agents.

Stores insights from past analyses and retrieves relevant context
for future LLM calls, with automatic token budget management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models import AgentMemory
from core.timeutil import now_utc_naive

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4

# Default retention per category
DEFAULT_TTL_DAYS = {
    "symbol": 7,       # per-stock insight: 1 week
    "sector": 14,      # sector trend: 2 weeks
    "market": 7,       # market regime: 1 week
    "lesson": 30,      # trade lesson: 1 month
}

# Max stored entries per agent_type
MAX_ENTRIES_PER_AGENT = 200


class AgentContextService:
    """Manages persistent memory for AI agents.

    Usage:
        ctx = AgentContextService(session_factory)

        # After analysis — save insight
        await ctx.save("market_analyst", "symbol", "AAPL",
                       "Bearish RSI divergence forming on daily chart", importance=7)

        # Before analysis — build context string within token budget
        context = await ctx.build_context("market_analyst", symbol="AAPL", max_tokens=1500)
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def save(
        self,
        agent_type: str,
        category: str,
        symbol: str | None,
        content: str,
        importance: int = 5,
        ttl_days: int | None = None,
    ) -> None:
        """Save an insight to agent memory.

        Args:
            agent_type: Which agent (market_analyst, risk, trade_review)
            category: Type of insight (symbol, sector, market, lesson)
            symbol: Stock ticker (None for market/lesson)
            content: The insight text
            importance: Priority 1-10 (higher = more important)
            ttl_days: Override default retention days
        """
        if not content or not content.strip():
            return

        days = ttl_days or DEFAULT_TTL_DAYS.get(category, 7)
        token_count = max(1, len(content) // CHARS_PER_TOKEN)

        async with self._session_factory() as session:
            entry = AgentMemory(
                agent_type=agent_type,
                category=category,
                symbol=symbol,
                content=content.strip(),
                token_count=token_count,
                importance=min(10, max(1, importance)),
                expires_at=now_utc_naive() + timedelta(days=days),
            )
            session.add(entry)
            await session.commit()

    async def build_context(
        self,
        agent_type: str,
        *,
        symbol: str | None = None,
        sector: str | None = None,
        max_tokens: int = 1500,
    ) -> str:
        """Build a context string from relevant memories within token budget.

        Priority order:
        1. Same symbol insights (most relevant)
        2. Same sector insights
        3. Market-wide observations
        4. Trade lessons

        Returns empty string if no relevant memories found.
        """
        async with self._session_factory() as session:
            now = now_utc_naive()

            # Fetch non-expired memories for this agent, ordered by importance desc
            stmt = (
                select(AgentMemory)
                .where(
                    AgentMemory.agent_type == agent_type,
                    AgentMemory.expires_at > now,
                )
                .order_by(AgentMemory.importance.desc(), AgentMemory.created_at.desc())
            )
            result = await session.execute(stmt)
            all_memories = list(result.scalars().all())

        if not all_memories:
            return ""

        # Bucket by relevance tier
        symbol_hits = []
        sector_hits = []
        market_hits = []
        lesson_hits = []

        for m in all_memories:
            if symbol and m.symbol == symbol:
                symbol_hits.append(m)
            elif m.category == "sector" and sector and m.symbol and m.symbol.lower() == sector.lower():
                sector_hits.append(m)
            elif m.category == "market":
                market_hits.append(m)
            elif m.category == "lesson":
                lesson_hits.append(m)
            elif m.category == "symbol" and m.symbol != symbol:
                # Other symbol insights — low priority but may be useful
                pass

        # Assemble within budget
        parts = []
        remaining = max_tokens

        for label, bucket in [
            ("Previous insights for this symbol", symbol_hits),
            ("Sector observations", sector_hits),
            ("Market context", market_hits),
            ("Lessons learned", lesson_hits),
        ]:
            if not bucket or remaining <= 0:
                continue

            section_items = []
            for m in bucket:
                if m.token_count > remaining:
                    continue
                days_ago = (now - m.created_at).days
                age = f"{days_ago}d ago" if days_ago > 0 else "today"
                section_items.append(f"- [{age}] {m.content}")
                remaining -= m.token_count

            if section_items:
                parts.append(f"### {label}:")
                parts.extend(section_items)

        if not parts:
            return ""

        return "## Agent Memory (past insights)\n" + "\n".join(parts)

    async def cleanup_expired(self) -> int:
        """Delete expired memory entries. Returns count deleted."""
        async with self._session_factory() as session:
            stmt = delete(AgentMemory).where(
                AgentMemory.expires_at <= now_utc_naive()
            )
            result = await session.execute(stmt)
            await session.commit()
            count = result.rowcount
            if count:
                logger.info("Cleaned up %d expired agent memories", count)
            return count

    async def enforce_limits(self, agent_type: str) -> int:
        """Trim oldest low-importance entries if over MAX_ENTRIES_PER_AGENT."""
        async with self._session_factory() as session:
            # Count entries
            count_stmt = (
                select(AgentMemory.id)
                .where(AgentMemory.agent_type == agent_type)
            )
            result = await session.execute(count_stmt)
            ids = list(result.scalars().all())

            excess = len(ids) - MAX_ENTRIES_PER_AGENT
            if excess <= 0:
                return 0

            # Find oldest, lowest-importance entries to delete
            trim_stmt = (
                select(AgentMemory.id)
                .where(AgentMemory.agent_type == agent_type)
                .order_by(AgentMemory.importance.asc(), AgentMemory.created_at.asc())
                .limit(excess)
            )
            result = await session.execute(trim_stmt)
            to_delete = list(result.scalars().all())

            if to_delete:
                await session.execute(
                    delete(AgentMemory).where(AgentMemory.id.in_(to_delete))
                )
                await session.commit()
                logger.info("Trimmed %d agent memories for %s", len(to_delete), agent_type)

            return len(to_delete)
