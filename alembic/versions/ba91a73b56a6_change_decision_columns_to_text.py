"""change_decision_columns_to_text

Revision ID: ba91a73b56a6
Revises: e4c5279f5d8b
Create Date: 2025-10-23 14:24:35.361740

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ba91a73b56a6'
down_revision: Union[str, None] = 'e4c5279f5d8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Change predicted_decision and correct_decision from VARCHAR to TEXT (no length limit)
    op.alter_column('transactions', 'predicted_decision',
                    type_=sa.Text(),
                    existing_type=sa.String(length=2000),
                    existing_nullable=False)
    
    op.alter_column('transactions', 'correct_decision',
                    type_=sa.Text(),
                    existing_type=sa.String(length=2000),
                    existing_nullable=False)


def downgrade() -> None:
    # Revert back to VARCHAR(2000)
    op.alter_column('transactions', 'predicted_decision',
                    type_=sa.String(length=2000),
                    existing_type=sa.Text(),
                    existing_nullable=False)
    
    op.alter_column('transactions', 'correct_decision',
                    type_=sa.String(length=2000),
                    existing_type=sa.Text(),
                    existing_nullable=False)

