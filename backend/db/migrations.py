"""Auto-migration for missing columns and indexes on existing tables.

SQLAlchemy's ``Base.metadata.create_all`` only creates tables that do not
exist yet — it does **not** add new columns or indexes to tables that are
already present.  When the ORM model evolves (e.g. ``is_paper`` added to
``Order``), the physical DB schema lags behind until an explicit ALTER
TABLE / CREATE INDEX is applied.

This module inspects the live schema on startup and adds any columns or
indexes that are defined in the model but absent from the physical table.
"""

import logging
from typing import Sequence

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Columns that may be missing from existing deployments.
# Format: (table_name, column_name, SQL_type, default_value, not_null)
_EXPECTED_COLUMNS: Sequence[tuple[str, str, str, str | None, bool]] = [
    ("orders", "is_paper", "BOOLEAN", "FALSE", True),
    ("orders", "buy_strategy", "VARCHAR(50)", None, False),
    ("orders", "pnl_pct", "FLOAT", None, False),
    ("portfolio_snapshots", "cash_flow", "FLOAT", "0.0", False),
]

# Indexes that may be missing from existing deployments.
# Format: (table_name, index_name, column_name)
_EXPECTED_INDEXES: Sequence[tuple[str, str, str]] = [
    ("orders", "idx_orders_is_paper", "is_paper"),
]


async def ensure_columns(engine: AsyncEngine) -> list[str]:
    """Inspect the live DB schema and add any missing columns.

    Returns a list of ``"table.column"`` strings that were added.
    Designed to be safe to run on every startup (idempotent).
    """
    added: list[str] = []

    async with engine.begin() as conn:

        def _sync_check_and_add(sync_conn) -> None:  # type: ignore[no-untyped-def]
            insp = inspect(sync_conn)

            for table, column, sql_type, default, not_null in _EXPECTED_COLUMNS:
                if not insp.has_table(table):
                    continue

                existing_cols = {c["name"] for c in insp.get_columns(table)}
                if column in existing_cols:
                    continue

                not_null_clause = " NOT NULL" if not_null else ""
                default_clause = f" DEFAULT {default}" if default else ""
                ddl = (
                    f"ALTER TABLE {table} ADD COLUMN {column} "
                    f"{sql_type}{not_null_clause}{default_clause}"
                )
                sync_conn.execute(text(ddl))
                added.append(f"{table}.{column}")
                logger.info(
                    "Added missing column: %s.%s (%s%s%s)",
                    table, column, sql_type, not_null_clause, default_clause,
                )

        await conn.run_sync(_sync_check_and_add)

    if added:
        logger.info("Auto-migration complete: added %d column(s): %s", len(added), ", ".join(added))
    else:
        logger.debug("Auto-migration: all expected columns already present")

    return added


async def ensure_indexes(engine: AsyncEngine) -> list[str]:
    """Create any indexes that are defined in the model but missing from the DB.

    Returns a list of index names that were created.
    Designed to be safe to run on every startup (idempotent).
    Uses ``CREATE INDEX IF NOT EXISTS`` to avoid errors on repeated runs.
    """
    created: list[str] = []

    async with engine.begin() as conn:

        def _sync_check_and_create(sync_conn) -> None:  # type: ignore[no-untyped-def]
            insp = inspect(sync_conn)

            for table, index_name, column in _EXPECTED_INDEXES:
                if not insp.has_table(table):
                    continue

                # Check if the column exists (index on missing column would fail)
                existing_cols = {c["name"] for c in insp.get_columns(table)}
                if column not in existing_cols:
                    logger.debug(
                        "Skipping index %s: column %s.%s does not exist yet",
                        index_name, table, column,
                    )
                    continue

                # Check if index already exists
                existing_indexes = {
                    idx["name"] for idx in insp.get_indexes(table)
                }
                if index_name in existing_indexes:
                    continue

                ddl = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
                sync_conn.execute(text(ddl))
                created.append(index_name)
                logger.info(
                    "Created missing index: %s ON %s(%s)",
                    index_name, table, column,
                )

        await conn.run_sync(_sync_check_and_create)

    if created:
        logger.info(
            "Auto-migration complete: created %d index(es): %s",
            len(created), ", ".join(created),
        )
    else:
        logger.debug("Auto-migration: all expected indexes already present")

    return created
