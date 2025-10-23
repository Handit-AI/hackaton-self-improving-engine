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
    evaluator = Column(String(100), nullable=True)  # Which evaluator/perspective (e.g., 'formatter', 'correctness')
    
    # Embedding for semantic search (cached!)
    content_embedding = Column(ARRAY(Float), nullable=True)  # 1536 dimensions
    
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
        CheckConstraint('content_embedding IS NULL OR array_length(content_embedding, 1) = 1536', name='valid_embedding'),
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
    node = Column(String(50), nullable=True)  # Agent node name (e.g., 'fraud_detection')
    session_id = Column(String(100), nullable=True, index=True)  # Session identifier
    run_id = Column(String(100), nullable=True, index=True)  # Run identifier within session
    
    # Results
    predicted_decision = Column(Text, nullable=False)  # No length limit
    correct_decision = Column(Text, nullable=False)  # No length limit
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


class LLMJudge(Base):
    """Configure LLM judges per node with customizable prompts."""
    
    __tablename__ = "llm_judges"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Configuration
    node = Column(String(100), nullable=False)  # Which agent node
    evaluator = Column(String(100), nullable=False)  # Evaluator name
    model = Column(String(50), nullable=False, default='gpt-4o-mini')  # Model to use
    temperature = Column(Float, default=0.0)  # Temperature for judging
    
    # Prompt configuration
    system_prompt = Column(Text, nullable=False)  # System prompt for judge
    evaluation_criteria = Column(JSONB, nullable=True)  # Specific criteria to evaluate
    
    # Domain context
    domain = Column(String(100), default='fraud detection')
    
    # Performance tracking
    total_evaluations = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)  # Accuracy of judge when ground truth available
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('node', 'evaluator', name='unique_node_evaluator'),
        CheckConstraint('temperature >= 0 AND temperature <= 2', name='valid_temperature'),
        CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='valid_accuracy'),
    )


class JudgeEvaluation(Base):
    """Audit trail for judge evaluations."""
    
    __tablename__ = "judge_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Links
    judge_id = Column(Integer, ForeignKey("llm_judges.id", ondelete="CASCADE"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False)
    
    # Evaluation data
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    
    # Judge results
    is_correct = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    
    # Judge accuracy (if ground truth available)
    judge_was_correct = Column(Boolean, nullable=True)
    
    # Timestamps
    evaluated_at = Column(TIMESTAMP, server_default=func.current_timestamp())


class SessionRunMetrics(Base):
    """Track metrics per session, run, and evaluator."""
    
    __tablename__ = "session_run_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Session and run identifiers
    session_id = Column(String(100), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    node = Column(String(100), nullable=False)  # Agent node name
    evaluator = Column(String(100), nullable=False)  # Evaluator name
    mode = Column(String(50), nullable=False)  # Mode: 'vanilla', 'offline_online', 'online'
    
    # Metrics
    correct_count = Column(Integer, default=0)  # Number of correct predictions
    total_count = Column(Integer, default=0)  # Total number of predictions
    accuracy = Column(Float, default=0.0)  # Calculated accuracy (correct_count / total_count)
    
    # Metadata
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('session_id', 'run_id', 'node', 'evaluator', 'mode', name='unique_session_run_evaluator_mode'),
        CheckConstraint('correct_count >= 0 AND total_count >= 0', name='valid_counts'),
        CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='valid_accuracy'),
    )


