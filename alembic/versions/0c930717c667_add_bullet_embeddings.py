"""add_bullet_embeddings

Revision ID: 0c930717c667
Revises: e056ff6cad68
Create Date: 2025-10-23 12:11:46.669911

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0c930717c667'
down_revision: Union[str, None] = 'e056ff6cad68'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add content_embedding column to bullets table
    op.add_column('bullets', sa.Column('content_embedding', sa.ARRAY(sa.Float), nullable=True))
    
    # Add constraint for embedding dimension
    op.create_check_constraint(
        'valid_embedding',
        'bullets',
        'content_embedding IS NULL OR array_length(content_embedding, 1) = 1536'
    )


def downgrade() -> None:
    # Drop constraint
    op.drop_constraint('valid_embedding', 'bullets', type_='check')
    
    # Drop column
    op.drop_column('bullets', 'content_embedding')

