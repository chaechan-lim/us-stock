"""Database session management."""

import logging
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import DatabaseConfig
from core.models import Base

logger = logging.getLogger(__name__)

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


async def ensure_schema_columns(engine: AsyncEngine) -> list[str]:
    """Ensure all ORM model columns exist in the database tables.

    After create_all() handles table creation, this function checks for
    any columns defined in ORM models but missing from the actual DB schema,
    and adds them via ALTER TABLE ADD COLUMN.

    This bridges the gap between ORM model updates and alembic migrations.
    For example, ``is_paper`` was added to the Order model (STOCK-2) but the
    initial alembic migration did not include it, causing queries that filter
    on ``Order.is_paper`` to fail with "column does not exist".

    Returns:
        List of added columns in "table.column" format.
    """
    added: list[str] = []

    async with engine.begin() as conn:
        missing_cols: list[tuple[str, str, str, str]] = await conn.run_sync(_find_missing_columns)

        for table_name, col_name, col_type_str, default_clause in missing_cols:
            sql = (
                f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" {col_type_str}{default_clause}'
            )
            await conn.execute(text(sql))
            added.append(f"{table_name}.{col_name}")
            logger.info("Added missing column %s.%s to database", table_name, col_name)

    return added


def _find_missing_columns(
    sync_conn: Any,
) -> list[tuple[str, str, str, str]]:
    """Inspect DB schema and find columns defined in ORM but missing from DB.

    Runs in synchronous context via ``run_sync()``.

    Returns:
        List of (table_name, col_name, compiled_type, default_clause) tuples.
    """
    insp = inspect(sync_conn)
    dialect = sync_conn.engine.dialect
    missing: list[tuple[str, str, str, str]] = []

    for table in Base.metadata.tables.values():
        if not insp.has_table(table.name):
            continue  # Table doesn't exist yet; create_all() handles this

        existing = {c["name"] for c in insp.get_columns(table.name)}

        for col in table.columns:
            if col.name in existing:
                continue

            col_type_str = col.type.compile(dialect=dialect)
            default_clause = _build_default_clause(col)
            missing.append((table.name, col.name, col_type_str, default_clause))

    return missing


def _build_default_clause(col: Any) -> str:
    """Build a DEFAULT clause for ALTER TABLE ADD COLUMN.

    Only handles static defaults (bool, int, float, str).
    Callable defaults (e.g. datetime.utcnow) are skipped since they are
    handled at the ORM level, not in DDL.
    """
    if col.default is None:
        return ""

    val = col.default.arg
    if callable(val):
        return ""  # ORM-level default, not DDL

    if isinstance(val, bool):
        return f" DEFAULT {str(val).upper()}"
    if isinstance(val, (int, float)):
        return f" DEFAULT {val}"
    if isinstance(val, str):
        return f" DEFAULT '{val}'"

    return ""
