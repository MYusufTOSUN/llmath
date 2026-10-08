"""Question difficulty (MATH level 1-5) and the reason a model stopped generating.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-06
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("questions") as batch:
        batch.add_column(sa.Column("difficulty", sa.Integer(), nullable=True))
    with op.batch_alter_table("responses") as batch:
        batch.add_column(sa.Column("finish_reason", sa.String(32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("responses") as batch:
        batch.drop_column("finish_reason")
    with op.batch_alter_table("questions") as batch:
        batch.drop_column("difficulty")
