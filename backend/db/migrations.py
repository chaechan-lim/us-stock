"""Auto-migration for missing columns on existing tables.

SQLAlchemy's ``Base.metadata.create_all`` only creates tables that do not
exist yet — it does **not** add new columns to tables that are already
present.  When the ORM model evolves (e.g. ``is_paper`` added to
``Order``), the physical DB schema lags behind until an explicit ALTER
TABLE is applied.

This module inspects the live schema on startup and adds any columns
that are defined in the model but absent from the physical table.
"""

import logging
from typing import Sequence

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Columns that may be missing from existing deployments.
# Format: (table_name, column_name, SQL_type, default_value_or_None)
_EXPECTED_COLUMNS: Sequence[tuple[str, str, str, str | None]] = [
    ("orders", "is_paper", "BOOLEAN", "FALSE"),
    ("orders", "buy_strategy", "VARCHAR(50)", None),
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

            for table, column, sql_type, default in _EXPECTED_COLUMNS:
                if not insp.has_table(table):
                    continue

                existing_cols = {c["name"] for c in insp.get_columns(table)}
                if column in existing_cols:
                    continue

                default_clause = f" DEFAULT {default}" if default else ""
                ddl = f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}{default_clause}"
                sync_conn.execute(text(ddl))
                added.append(f"{table}.{column}")
                logger.info(
                    "Added missing column: %s.%s (%s%s)",
                    table, column, sql_type, default_clause,
                )

        await conn.run_sync(_sync_check_and_add)

    if added:
        logger.info("Auto-migration complete: added %d column(s): %s", len(added), ", ".join(added))
    else:
        logger.debug("Auto-migration: all expected columns already present")

    return added
