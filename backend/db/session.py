"""Database session management."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import DatabaseConfig

_engine = None
_session_factory = None


def get_engine(config: DatabaseConfig | None = None):
    global _engine
    if _engine is None:
        if config is None:
            config = DatabaseConfig()
        _engine = create_async_engine(
            config.url,
            echo=config.echo,
            pool_size=10,
            max_overflow=20,
            # RES-H2 (2026-06-05): bounded pool wait. SQLAlchemy default
            # is 30s, but with 35 scheduler tasks + 2 eval loops + per-
            # signal FunnelEvent persists, pool exhaustion under burst
            # used to silently block for 30s before TimeoutError. Cap
            # at 10s so callers get a fast failure they can log.
            pool_timeout=10,
            # Detect Postgres connections killed by the network /
            # restart / idle timeout before handing them out.
            pool_pre_ping=True,
            # Recycle long-idle connections so we don't reuse one that
            # the broker silently closed.
            pool_recycle=300,
        )
    return _engine


def get_session_factory(config: DatabaseConfig | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine(config)
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncSession:
    factory = get_session_factory()
    async with factory() as session:
        yield session
