"""increase_decision_column_sizes

Revision ID: f4160a9b8676
Revises: 24f261a1046b
Create Date: 2025-10-23 13:56:12.229733

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4160a9b8676'
down_revision: Union[str, None] = '24f261a1046b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Increase predicted_decision column size from VARCHAR(20) to VARCHAR(500)
    op.alter_column('transactions', 'predicted_decision',
                    existing_type=sa.String(length=20),
                    type_=sa.String(length=500),
                    existing_nullable=False)
    
    # Increase correct_decision column size from VARCHAR(20) to VARCHAR(500)
    op.alter_column('transactions', 'correct_decision',
                    existing_type=sa.String(length=20),
                    type_=sa.String(length=500),
                    existing_nullable=False)


def downgrade() -> None:
    # Revert predicted_decision column size back to VARCHAR(20)
    op.alter_column('transactions', 'predicted_decision',
                    existing_type=sa.String(length=500),
                    type_=sa.String(length=20),
                    existing_nullable=False)
    
    # Revert correct_decision column size back to VARCHAR(20)
    op.alter_column('transactions', 'correct_decision',
                    existing_type=sa.String(length=500),
                    type_=sa.String(length=20),
                    existing_nullable=False)

