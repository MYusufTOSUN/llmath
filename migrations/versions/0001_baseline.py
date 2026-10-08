"""Baseline: the pilot schema, created with create_all before migrations existed.

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Pilot databases already have these tables; new databases are built by init_db() with
    # create_all and stamped at head, so this revision only marks the starting point.
    pass


def downgrade() -> None:
    pass
