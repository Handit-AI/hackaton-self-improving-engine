"""add_mode_to_session_run_metrics

Revision ID: 9cd7757aea6d
Revises: b44cd9fee57b
Create Date: 2025-10-23 15:25:50.827732

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9cd7757aea6d'
down_revision: Union[str, None] = 'b44cd9fee57b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old unique constraint
    op.drop_constraint('unique_session_run_evaluator', 'session_run_metrics', type_='unique')
    
    # Add mode column
    op.add_column('session_run_metrics', sa.Column('mode', sa.String(length=50), nullable=True))
    
    # Set default mode for existing records
    op.execute("UPDATE session_run_metrics SET mode = 'online' WHERE mode IS NULL")
    
    # Make mode not nullable
    op.alter_column('session_run_metrics', 'mode', nullable=False)
    
    # Create new unique constraint including mode
    op.create_unique_constraint('unique_session_run_evaluator_mode', 'session_run_metrics', 
                               ['session_id', 'run_id', 'node', 'evaluator', 'mode'])


def downgrade() -> None:
    # Drop the new unique constraint
    op.drop_constraint('unique_session_run_evaluator_mode', 'session_run_metrics', type_='unique')
    
    # Drop the mode column
    op.drop_column('session_run_metrics', 'mode')
    
    # Recreate the old unique constraint
    op.create_unique_constraint('unique_session_run_evaluator', 'session_run_metrics', 
                               ['session_id', 'run_id', 'node', 'evaluator'])

