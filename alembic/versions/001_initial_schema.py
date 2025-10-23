"""initial schema for ACE system

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-12-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create input_patterns table
    op.create_table(
        'input_patterns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_embedding', ARRAY(sa.Float()), nullable=False),
        sa.Column('normalized_features', JSONB(), nullable=True),
        sa.Column('frequency', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('first_seen_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_seen_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('problem_type', sa.String(length=100), nullable=True),
        sa.CheckConstraint('array_length(query_embedding, 1) = 1536', name='valid_embedding'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_input_patterns_id'), 'input_patterns', ['id'], unique=False)
    
    # Create bullets table
    op.create_table(
        'bullets',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('node', sa.String(length=100), nullable=False),
        sa.Column('helpful_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('harmful_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('times_selected', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_used', sa.TIMESTAMP(), nullable=True),
        sa.CheckConstraint('helpful_count >= 0 AND harmful_count >= 0', name='valid_performance'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create bullet_input_effectiveness table
    op.create_table(
        'bullet_input_effectiveness',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('input_pattern_id', sa.Integer(), nullable=False),
        sa.Column('bullet_id', sa.String(length=50), nullable=False),
        sa.Column('node', sa.String(length=100), nullable=False),
        sa.Column('helpful_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('harmful_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('times_selected', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('first_tested_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_tested_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['input_pattern_id'], ['input_patterns.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['bullet_id'], ['bullets.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('input_pattern_id', 'bullet_id', name='unique_bullet_input'),
        sa.CheckConstraint('helpful_count >= 0 AND harmful_count >= 0', name='valid_performance'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_bullet_input_effectiveness_id'), 'bullet_input_effectiveness', ['id'], unique=False)
    
    # Create transactions table
    op.create_table(
        'transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('transaction_data', JSONB(), nullable=False),
        sa.Column('mode', sa.String(length=50), nullable=False),
        sa.Column('predicted_decision', sa.String(length=20), nullable=False),
        sa.Column('correct_decision', sa.String(length=20), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('input_pattern_id', sa.Integer(), nullable=True),
        sa.Column('analyzed_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['input_pattern_id'], ['input_patterns.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transactions_id'), 'transactions', ['id'], unique=False)
    
    # Create bullet_selections table
    op.create_table(
        'bullet_selections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('transaction_id', sa.Integer(), nullable=False),
        sa.Column('bullet_id', sa.String(length=50), nullable=False),
        sa.Column('bullet_input_effectiveness_id', sa.Integer(), nullable=True),
        sa.Column('node', sa.String(length=100), nullable=False),
        sa.Column('selection_score', sa.Float(), nullable=True),
        sa.Column('selected_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['bullet_id'], ['bullets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['bullet_input_effectiveness_id'], ['bullet_input_effectiveness.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_bullet_selections_id'), 'bullet_selections', ['id'], unique=False)
    
    # Create additional indexes
    op.create_index('idx_bullets_node', 'bullets', ['node'])
    op.create_index('idx_bullets_performance', 'bullets', [sa.text('(helpful_count + harmful_count)')])
    op.create_index('idx_bullet_input_pattern', 'bullet_input_effectiveness', ['input_pattern_id'])
    op.create_index('idx_bullet_input_bullet', 'bullet_input_effectiveness', ['bullet_id'])
    op.create_index('idx_bullet_input_node', 'bullet_input_effectiveness', ['node'])
    op.create_index('idx_transactions_mode', 'transactions', ['mode'])
    op.create_index('idx_transactions_pattern', 'transactions', ['input_pattern_id'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('idx_transactions_pattern', table_name='transactions')
    op.drop_index('idx_transactions_mode', table_name='transactions')
    op.drop_index('idx_bullet_input_node', table_name='bullet_input_effectiveness')
    op.drop_index('idx_bullet_input_bullet', table_name='bullet_input_effectiveness')
    op.drop_index('idx_bullet_input_pattern', table_name='bullet_input_effectiveness')
    op.drop_index('idx_bullets_performance', table_name='bullets')
    op.drop_index('idx_bullets_node', table_name='bullets')
    
    op.drop_index(op.f('ix_bullet_selections_id'), table_name='bullet_selections')
    op.drop_table('bullet_selections')
    
    op.drop_index(op.f('ix_transactions_id'), table_name='transactions')
    op.drop_table('transactions')
    
    op.drop_index(op.f('ix_bullet_input_effectiveness_id'), table_name='bullet_input_effectiveness')
    op.drop_table('bullet_input_effectiveness')
    
    op.drop_table('bullets')
    
    op.drop_index(op.f('ix_input_patterns_id'), table_name='input_patterns')
    op.drop_table('input_patterns')

