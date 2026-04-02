"""Add account_id column to trading tables

Revision ID: e3a8f2c1d4b5
Revises: 607feca4f8b7
Create Date: 2026-04-03 07:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3a8f2c1d4b5"
down_revision: Union[str, Sequence[str], None] = "607feca4f8b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    STOCK-83: Add account_id column to orders, positions, portfolio_snapshots,
    and strategy_logs tables with default 'ACC001' for existing data.
    Add composite indexes for efficient per-account queries.
    """
    # Add account_id to orders table
    op.add_column(
        "orders",
        sa.Column(
            "account_id",
            sa.String(20),
            nullable=False,
            server_default="ACC001",
        ),
    )
    op.create_index(
        "idx_orders_account_market_symbol",
        "orders",
        ["account_id", "market", "symbol"],
    )

    # Add account_id to positions table
    op.add_column(
        "positions",
        sa.Column(
            "account_id",
            sa.String(20),
            nullable=False,
            server_default="ACC001",
        ),
    )
    op.create_index(
        "idx_positions_account_market_symbol",
        "positions",
        ["account_id", "market", "symbol"],
    )

    # Add account_id to portfolio_snapshots table
    op.add_column(
        "portfolio_snapshots",
        sa.Column(
            "account_id",
            sa.String(20),
            nullable=False,
            server_default="ACC001",
        ),
    )
    op.create_index(
        "idx_snapshots_account_market",
        "portfolio_snapshots",
        ["account_id", "market"],
    )

    # Add account_id to strategy_logs table (signals)
    op.add_column(
        "strategy_logs",
        sa.Column(
            "account_id",
            sa.String(20),
            nullable=False,
            server_default="ACC001",
        ),
    )
    op.create_index(
        "idx_strategy_logs_account_symbol",
        "strategy_logs",
        ["account_id", "symbol"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes before dropping columns

    # strategy_logs
    op.drop_index("idx_strategy_logs_account_symbol", table_name="strategy_logs")
    op.drop_column("strategy_logs", "account_id")

    # portfolio_snapshots
    op.drop_index("idx_snapshots_account_market", table_name="portfolio_snapshots")
    op.drop_column("portfolio_snapshots", "account_id")

    # positions
    op.drop_index("idx_positions_account_market_symbol", table_name="positions")
    op.drop_column("positions", "account_id")

    # orders
    op.drop_index("idx_orders_account_market_symbol", table_name="orders")
    op.drop_column("orders", "account_id")
