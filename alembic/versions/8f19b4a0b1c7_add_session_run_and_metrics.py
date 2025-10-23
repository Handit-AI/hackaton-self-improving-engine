"""add_session_run_and_metrics

Revision ID: 8f19b4a0b1c7
Revises: ba91a73b56a6
Create Date: 2025-10-23 15:00:08.937958

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8f19b4a0b1c7'
down_revision: Union[str, None] = 'ba91a73b56a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add session_id and run_id to transactions table
    op.add_column('transactions', sa.Column('session_id', sa.String(length=100), nullable=True))
    op.add_column('transactions', sa.Column('run_id', sa.String(length=100), nullable=True))
    
    # Create indexes for session_id and run_id
    op.create_index('ix_transactions_session_id', 'transactions', ['session_id'])
    op.create_index('ix_transactions_run_id', 'transactions', ['run_id'])
    
    # Create session_run_metrics table
    op.create_table(
        'session_run_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=100), nullable=False),
        sa.Column('run_id', sa.String(length=100), nullable=False),
        sa.Column('node', sa.String(length=100), nullable=False),
        sa.Column('evaluator', sa.String(length=100), nullable=False),
        sa.Column('correct_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('accuracy', sa.Float(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'run_id', 'node', 'evaluator', name='unique_session_run_evaluator'),
        sa.CheckConstraint('correct_count >= 0 AND total_count >= 0', name='valid_counts'),
        sa.CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='valid_accuracy')
    )
    op.create_index('ix_session_run_metrics_session_id', 'session_run_metrics', ['session_id'])
    op.create_index('ix_session_run_metrics_run_id', 'session_run_metrics', ['run_id'])


def downgrade() -> None:
    # Drop session_run_metrics table
    op.drop_index('ix_session_run_metrics_run_id', 'session_run_metrics')
    op.drop_index('ix_session_run_metrics_session_id', 'session_run_metrics')
    op.drop_table('session_run_metrics')
    
    # Drop indexes and columns from transactions
    op.drop_index('ix_transactions_run_id', 'transactions')
    op.drop_index('ix_transactions_session_id', 'transactions')
    op.drop_column('transactions', 'run_id')
    op.drop_column('transactions', 'session_id')

