"""Auto-migration for missing columns and indexes on existing tables.

SQLAlchemy's ``Base.metadata.create_all`` only creates tables that do not
exist yet — it does **not** add new columns or indexes to tables that are
already present.  When the ORM model evolves (e.g. ``is_paper`` added to
``Order``), the physical DB schema lags behind until an explicit ALTER
TABLE / CREATE INDEX is applied.

This module inspects the live schema on startup and adds any columns or
indexes that are defined in the model but absent from the physical table.

``run_alembic_upgrade`` is also provided to run ``alembic upgrade head`` as a
subprocess so that all versioned schema migrations are applied on every server
start.  Using a subprocess avoids event-loop conflicts with alembic's async
``env.py`` which calls ``asyncio.run()`` internally.
"""

import asyncio
import functools
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# The first alembic revision (initial schema).  Used when stamping a legacy DB
# that has tables but no alembic_version tracking table.
_INITIAL_REVISION: str = "cfbdf0cd6e1f"

# Column from migration 607feca4f8b7 (STOCK-58).  Its presence indicates that
# all migrations up to STOCK-58 have already been applied on this DB.
# MUST UPDATE: set this to a column added in the latest migration whenever a
# new versioned migration is added, so that fully-migrated legacy DBs are
# stamped at head rather than at _INITIAL_REVISION.
_STOCK58_SENTINEL_TABLE: str = "portfolio_snapshots"
_STOCK58_SENTINEL_COLUMN: str = "usd_krw_rate"

# PostgreSQL advisory lock ID used to serialise ``run_alembic_upgrade`` across
# multiple workers that start up simultaneously (e.g. ``uvicorn --workers N``).
# The value is an arbitrary project-specific constant.
_PG_MIGRATION_LOCK_ID: int = 7625765983


def _run_alembic_cmd(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run an alembic sub-command as a subprocess.

    Returns the completed process.  Caller is responsible for checking
    ``returncode``.

    Raises ``subprocess.TimeoutExpired`` if the command does not complete
    within 180 seconds.
    """
    cmd = [sys.executable, "-m", "alembic"] + args
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=180)


async def run_alembic_upgrade(
    engine: AsyncEngine,
    alembic_dir: Path | None = None,
) -> None:
    """Run ``alembic upgrade head`` as a subprocess on server startup.

    Handles three DB states safely:

    1. **Fresh DB** (no tables yet): alembic creates all tables from scratch via
       the migration chain.
    2. **Legacy DB** (tables exist, no ``alembic_version`` table): the DB was
       bootstrapped via ``Base.metadata.create_all`` without alembic tracking.
       We stamp at the appropriate revision so alembic only applies *pending*
       migrations, then run upgrade.
    3. **Tracked DB** (``alembic_version`` table exists): alembic applies only
       unapplied migrations as normal.

    ``alembic_dir`` is the directory that contains ``alembic.ini``.  Defaults to
    the ``backend/`` directory (parent of this file's ``db/`` package).

    On PostgreSQL the entire inspect→stamp→upgrade sequence is protected by a
    session-level advisory lock (``pg_advisory_lock``) so that concurrent
    worker processes serialise correctly instead of racing on ``ALTER TABLE``.

    Raises ``RuntimeError`` if any alembic command exits with a non-zero code
    or times out, which blocks server startup as required.
    """
    backend_dir: Path = alembic_dir if alembic_dir is not None else Path(__file__).parent.parent
    is_postgres: bool = engine.dialect.name == "postgresql"

    # Keep a single connection open for the full migration sequence so that
    # the PostgreSQL advisory lock (session-scoped) is held throughout.
    async with engine.connect() as conn:
        if is_postgres:
            logger.info(
                "alembic: acquiring pg_advisory_lock(%d) to serialise multi-worker startup",
                _PG_MIGRATION_LOCK_ID,
            )
            await conn.execute(text(f"SELECT pg_advisory_lock({_PG_MIGRATION_LOCK_ID})"))

        try:
            # ── 1. Inspect current DB state ──────────────────────────────────────────
            has_alembic_version: bool = False
            has_base_tables: bool = False
            has_stock58_columns: bool = False

            def _inspect_state(sync_conn) -> tuple[bool, bool, bool]:  # type: ignore[no-untyped-def]
                insp = inspect(sync_conn)
                _has_alembic = insp.has_table("alembic_version")
                _has_base = insp.has_table("orders")
                _has_58 = False
                if insp.has_table(_STOCK58_SENTINEL_TABLE):
                    cols = {c["name"] for c in insp.get_columns(_STOCK58_SENTINEL_TABLE)}
                    _has_58 = _STOCK58_SENTINEL_COLUMN in cols
                return _has_alembic, _has_base, _has_58

            has_alembic_version, has_base_tables, has_stock58_columns = await conn.run_sync(
                _inspect_state
            )

            # ── 2. Stamp legacy deployments so alembic upgrade is incremental ────────
            if not has_alembic_version and has_base_tables:
                # Tables were created via create_all without alembic.
                # Determine which revision to stamp based on what columns exist.
                if has_stock58_columns:
                    # All current migrations are already reflected in the schema
                    # (e.g. fresh DB where create_all picked up the latest ORM models).
                    stamp_rev = "head"
                else:
                    # Missing STOCK-58 columns → DB is at the initial revision.
                    stamp_rev = _INITIAL_REVISION

                logger.info("alembic: no version table found; stamping legacy DB at %s", stamp_rev)
                try:
                    stamp_proc = await asyncio.to_thread(
                        functools.partial(_run_alembic_cmd, ["stamp", stamp_rev], backend_dir)
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"alembic stamp {stamp_rev} timed out after 180 s") from exc
                if stamp_proc.stdout:
                    logger.info("alembic stamp stdout:\n%s", stamp_proc.stdout)
                if stamp_proc.stderr:
                    logger.info("alembic stamp stderr:\n%s", stamp_proc.stderr)
                if stamp_proc.returncode != 0:
                    raise RuntimeError(
                        f"alembic stamp {stamp_rev} failed "
                        f"(exit {stamp_proc.returncode}):\n{stamp_proc.stderr}"
                    )

            # ── 3. Run upgrade head ───────────────────────────────────────────────────
            logger.info("alembic: running upgrade head …")
            try:
                upgrade_proc = await asyncio.to_thread(
                    functools.partial(_run_alembic_cmd, ["upgrade", "head"], backend_dir)
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("alembic upgrade head timed out after 180 s") from exc
            if upgrade_proc.stdout:
                logger.info("alembic upgrade stdout:\n%s", upgrade_proc.stdout)
            if upgrade_proc.stderr:
                logger.info("alembic upgrade stderr:\n%s", upgrade_proc.stderr)
            if upgrade_proc.returncode != 0:
                raise RuntimeError(
                    f"alembic upgrade head failed "
                    f"(exit {upgrade_proc.returncode}):\n{upgrade_proc.stderr}"
                )

            logger.info("alembic: upgrade head complete")

        finally:
            if is_postgres:
                try:
                    await conn.execute(text(f"SELECT pg_advisory_unlock({_PG_MIGRATION_LOCK_ID})"))
                except Exception:
                    logger.warning(
                        "alembic: failed to release pg_advisory_lock(%d); "
                        "lock will expire when connection closes",
                        _PG_MIGRATION_LOCK_ID,
                    )


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
                    table,
                    column,
                    sql_type,
                    not_null_clause,
                    default_clause,
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
                        index_name,
                        table,
                        column,
                    )
                    continue

                # Check if index already exists
                existing_indexes = {idx["name"] for idx in insp.get_indexes(table)}
                if index_name in existing_indexes:
                    continue

                ddl = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
                sync_conn.execute(text(ddl))
                created.append(index_name)
                logger.info(
                    "Created missing index: %s ON %s(%s)",
                    index_name,
                    table,
                    column,
                )

        await conn.run_sync(_sync_check_and_create)

    if created:
        logger.info(
            "Auto-migration complete: created %d index(es): %s",
            len(created),
            ", ".join(created),
        )
    else:
        logger.debug("Auto-migration: all expected indexes already present")

    return created
