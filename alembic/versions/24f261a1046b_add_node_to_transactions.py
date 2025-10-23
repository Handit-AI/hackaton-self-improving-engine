"""add_node_to_transactions

Revision ID: 24f261a1046b
Revises: e2ef107075d6
Create Date: 2025-10-23 13:35:02.339247

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '24f261a1046b'
down_revision: Union[str, None] = 'e2ef107075d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add node column to transactions table
    op.add_column('transactions', sa.Column('node', sa.String(length=50), nullable=True))


def downgrade() -> None:
    # Remove node column from transactions table
    op.drop_column('transactions', 'node')

