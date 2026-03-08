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
