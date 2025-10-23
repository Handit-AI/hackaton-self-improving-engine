"""
Database models for ACE system.
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func
from database import Base


class InputPattern(Base):
    """Store input patterns with embeddings for similarity matching."""
    
    __tablename__ = "input_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Input data
    query_text = Column(Text, nullable=False)
    query_embedding = Column(ARRAY(Float), nullable=False)  # 1536 dimensions
    
    # Normalized features
    normalized_features = Column(JSONB, nullable=True)
    
    # Metadata
    frequency = Column(Integer, default=1)
    first_seen_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    last_seen_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    
    # Classification
    problem_type = Column(String(100), nullable=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('array_length(query_embedding, 1) = 1536', name='valid_embedding'),
    )


class Bullet(Base):
    """Store learned bullets."""
    
    __tablename__ = "bullets"
    
    id = Column(String(50), primary_key=True)
    
    # Content
    content = Column(Text, nullable=False)
    node = Column(String(100), nullable=False)  # Which agent node
    
    # Performance tracking (global)
    helpful_count = Column(Integer, default=0)
    harmful_count = Column(Integer, default=0)
    times_selected = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    last_used = Column(TIMESTAMP, nullable=True)
    source = Column(String(20), nullable=True, default='online')  # 'offline' or 'online'
    
    # Constraints
    __table_args__ = (
        CheckConstraint('helpful_count >= 0 AND harmful_count >= 0', name='valid_performance'),
    )


class BulletInputEffectiveness(Base):
    """Association table - links bullets to input patterns with effectiveness metrics."""
    
    __tablename__ = "bullet_input_effectiveness"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Links
    input_pattern_id = Column(Integer, ForeignKey("input_patterns.id", ondelete="CASCADE"), nullable=False)
    bullet_id = Column(String(50), ForeignKey("bullets.id", ondelete="CASCADE"), nullable=False)
    node = Column(String(100), nullable=False)  # Which node this bullet is for
    
    # Performance for THIS specific input pattern
    helpful_count = Column(Integer, default=0)
    harmful_count = Column(Integer, default=0)
    times_selected = Column(Integer, default=0)
    
    # Metadata
    first_tested_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    last_tested_at = Column(TIMESTAMP, nullable=True)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('input_pattern_id', 'bullet_id', name='unique_bullet_input'),
        CheckConstraint('helpful_count >= 0 AND harmful_count >= 0', name='valid_performance'),
    )


class Transaction(Base):
    """Audit trail for transactions analyzed."""
    
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Transaction data
    transaction_data = Column(JSONB, nullable=False)
    
    # Analysis metadata
    mode = Column(String(50), nullable=False)  # 'vanilla', 'offline_ace', 'online_ace'
    
    # Results
    predicted_decision = Column(String(20), nullable=False)
    correct_decision = Column(String(20), nullable=False)
    is_correct = Column(Boolean, nullable=False, default=False)
    
    # Links
    input_pattern_id = Column(Integer, ForeignKey("input_patterns.id", ondelete="SET NULL"), nullable=True)
    
    # Timestamps
    analyzed_at = Column(TIMESTAMP, server_default=func.current_timestamp())


class BulletSelection(Base):
    """Track which bullets were selected for each analysis."""
    
    __tablename__ = "bullet_selections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Links
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False)
    bullet_id = Column(String(50), ForeignKey("bullets.id", ondelete="CASCADE"), nullable=False)
    bullet_input_effectiveness_id = Column(Integer, ForeignKey("bullet_input_effectiveness.id", ondelete="SET NULL"), nullable=True)
    
    # Selection details
    node = Column(String(100), nullable=False)
    selection_score = Column(Float, nullable=True)
    
    # Timestamps
    selected_at = Column(TIMESTAMP, server_default=func.current_timestamp())

