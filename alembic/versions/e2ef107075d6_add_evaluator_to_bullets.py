"""add_evaluator_to_bullets

Revision ID: e2ef107075d6
Revises: 0c930717c667
Create Date: 2025-10-23 12:58:02.550623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2ef107075d6'
down_revision: Union[str, None] = '0c930717c667'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add evaluator column to bullets table
    op.add_column('bullets', sa.Column('evaluator', sa.String(length=100), nullable=True))
    
    # Create index for faster lookups
    op.create_index('ix_bullets_evaluator', 'bullets', ['evaluator'])


def downgrade() -> None:
    # Drop index
    op.drop_index('ix_bullets_evaluator', 'bullets')
    
    # Drop column
    op.drop_column('bullets', 'evaluator')

