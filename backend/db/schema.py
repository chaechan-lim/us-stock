"""Startup schema migrations for columns that create_all() cannot add.

SQLAlchemy's Base.metadata.create_all() creates new tables but does NOT
alter existing tables to add missing columns. This module provides an
idempotent ensure_schema() that inspects the live DB and applies
lightweight ALTER TABLE statements for any missing columns.

Called once during application startup (lifespan), after create_all().
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Each migration is (table, column, SQL type with constraints).
# Order matters when there are dependencies between migrations.
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("orders", "is_paper", "BOOLEAN NOT NULL DEFAULT FALSE"),
]

# Indexes to create alongside column migrations.
# (index_name, table, column)
_INDEX_MIGRATIONS: list[tuple[str, str, str]] = [
    ("idx_orders_is_paper", "orders", "is_paper"),
]


async def ensure_schema(engine: AsyncEngine) -> list[str]:
    """Ensure all expected columns and indexes exist in the database.

    Inspects the live schema via SQLAlchemy's inspector and applies
    ALTER TABLE / CREATE INDEX statements for anything missing.

    Returns a list of applied migration descriptions (empty if up-to-date).
    """
    applied: list[str] = []

    async with engine.begin() as conn:
        # Detect dialect for SQL syntax differences
        dialect = engine.dialect.name

        # --- Column migrations ---
        for table, column, col_sql in _COLUMN_MIGRATIONS:
            has_column = await _column_exists(conn, table, column, dialect)
            if not has_column:
                if dialect == "sqlite":
                    # SQLite uses simpler ALTER TABLE syntax
                    stmt = f"ALTER TABLE {table} ADD COLUMN {column} {col_sql}"
                else:
                    # PostgreSQL / others
                    stmt = f"ALTER TABLE {table} ADD COLUMN {column} {col_sql}"

                await conn.execute(text(stmt))
                msg = f"Added column {table}.{column}"
                logger.info(msg)
                applied.append(msg)

        # --- Index migrations ---
        for idx_name, table, column in _INDEX_MIGRATIONS:
            has_index = await _index_exists(conn, idx_name, table, dialect)
            if not has_index:
                # Check the column exists first (may have been just added above)
                has_col = await _column_exists(conn, table, column, dialect)
                if has_col:
                    stmt = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})"
                    # SQLite doesn't support IF NOT EXISTS for indexes in all versions,
                    # but modern SQLite (3.3+) does. For safety, catch errors.
                    try:
                        await conn.execute(text(stmt))
                        msg = f"Created index {idx_name} on {table}({column})"
                        logger.info(msg)
                        applied.append(msg)
                    except Exception as e:
                        logger.debug("Index %s creation skipped: %s", idx_name, e)

    if not applied:
        logger.debug("Schema is up-to-date, no migrations needed")
    else:
        logger.info("Applied %d schema migration(s)", len(applied))

    return applied


async def _column_exists(
    conn,
    table: str,
    column: str,
    dialect: str,
) -> bool:
    """Check if a column exists in a table."""
    if dialect == "sqlite":
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        rows = result.fetchall()
        return any(row[1] == column for row in rows)
    else:
        # PostgreSQL: use information_schema
        result = await conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :table AND column_name = :column"
            ),
            {"table": table, "column": column},
        )
        return result.scalar() is not None


async def _index_exists(
    conn,
    index_name: str,
    table: str,
    dialect: str,
) -> bool:
    """Check if an index exists."""
    if dialect == "sqlite":
        result = await conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='index' AND name=:name"),
            {"name": index_name},
        )
        return result.scalar() is not None
    else:
        # PostgreSQL: use pg_indexes
        result = await conn.execute(
            text("SELECT 1 FROM pg_indexes WHERE indexname = :name AND tablename = :table"),
            {"name": index_name, "table": table},
        )
        return result.scalar() is not None
