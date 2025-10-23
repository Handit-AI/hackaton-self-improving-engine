"""increase_decision_column_sizes_to_2000

Revision ID: e4c5279f5d8b
Revises: f4160a9b8676
Create Date: 2025-10-23 13:58:32.044748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4c5279f5d8b'
down_revision: Union[str, None] = 'f4160a9b8676'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Increase predicted_decision column size from VARCHAR(500) to VARCHAR(2000)
    op.alter_column('transactions', 'predicted_decision',
                    existing_type=sa.String(length=500),
                    type_=sa.String(length=2000),
                    existing_nullable=False)
    
    # Increase correct_decision column size from VARCHAR(500) to VARCHAR(2000)
    op.alter_column('transactions', 'correct_decision',
                    existing_type=sa.String(length=500),
                    type_=sa.String(length=2000),
                    existing_nullable=False)


def downgrade() -> None:
    # Revert predicted_decision column size back to VARCHAR(500)
    op.alter_column('transactions', 'predicted_decision',
                    existing_type=sa.String(length=2000),
                    type_=sa.String(length=500),
                    existing_nullable=False)
    
    # Revert correct_decision column size back to VARCHAR(500)
    op.alter_column('transactions', 'correct_decision',
                    existing_type=sa.String(length=2000),
                    type_=sa.String(length=500),
                    existing_nullable=False)

