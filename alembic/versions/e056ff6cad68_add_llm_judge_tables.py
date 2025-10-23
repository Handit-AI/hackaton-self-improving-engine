"""add_llm_judge_tables

Revision ID: e056ff6cad68
Revises: 001_initial_schema
Create Date: 2025-10-23 11:25:30.308535

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e056ff6cad68'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create llm_judges table
    op.create_table(
        'llm_judges',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('node', sa.String(length=100), nullable=False),
        sa.Column('model', sa.String(length=50), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('evaluation_criteria', sa.JSON(), nullable=True),
        sa.Column('domain', sa.String(length=100), nullable=True),
        sa.Column('total_evaluations', sa.Integer(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.CheckConstraint('temperature >= 0 AND temperature <= 2', name='valid_temperature'),
        sa.CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='valid_accuracy'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('node')
    )
    op.create_index(op.f('ix_llm_judges_id'), 'llm_judges', ['id'], unique=False)
    
    # Create judge_evaluations table
    op.create_table(
        'judge_evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('judge_id', sa.Integer(), nullable=False),
        sa.Column('transaction_id', sa.Integer(), nullable=False),
        sa.Column('input_text', sa.Text(), nullable=False),
        sa.Column('output_text', sa.Text(), nullable=False),
        sa.Column('ground_truth', sa.Text(), nullable=True),
        sa.Column('is_correct', sa.Boolean(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('judge_was_correct', sa.Boolean(), nullable=True),
        sa.Column('evaluated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['judge_id'], ['llm_judges.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_judge_evaluations_id'), 'judge_evaluations', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_judge_evaluations_id'), table_name='judge_evaluations')
    op.drop_table('judge_evaluations')
    op.drop_index(op.f('ix_llm_judges_id'), table_name='llm_judges')
    op.drop_table('llm_judges')

