"""add_evaluator_and_insert_judges

Revision ID: b44cd9fee57b
Revises: 8f19b4a0b1c7
Create Date: 2025-10-23 15:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b44cd9fee57b'
down_revision: Union[str, None] = '8f19b4a0b1c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Drop the unique constraint on node
    op.drop_constraint('llm_judges_node_key', 'llm_judges', type_='unique')
    
    # Step 2: Add evaluator column
    op.add_column('llm_judges', sa.Column('evaluator', sa.String(length=100), nullable=True))
    
    # Step 3: Set default evaluator for existing records
    op.execute("UPDATE llm_judges SET evaluator = 'default' WHERE evaluator IS NULL")
    
    # Step 4: Make evaluator not nullable
    op.alter_column('llm_judges', 'evaluator', nullable=False)
    
    # Step 5: Create unique constraint on (node, evaluator)
    op.create_unique_constraint('unique_node_evaluator', 'llm_judges', ['node', 'evaluator'])
    
    # Step 6: Import and insert judges
    import sys
    import os
    
    # Add the current directory to the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Import the judge definitions
    from behavioral_analizer_judges import (
        BEHAVIORAL_PATTERN_CONSISTENCY_JUDGE,
        TEMPORAL_ANOMALY_DETECTION_JUDGE,
        PROFILE_EVOLUTION_VS_FRAUD_JUDGE
    )
    from decision_aggregator_judges import (
        OUTPUT_FORMAT_COMPLIANCE_JUDGE,
        DECISION_LOGIC_CONSISTENCY_JUDGE,
        HOLISTIC_RISK_SYNTHESIS_JUDGE
    )
    
    # Create connection
    conn = op.get_bind()
    
    # Behavioral Analyzer Judges
    behavioral_judges = [
        {
            'node': 'behavioral_analizer',
            'evaluator': 'pattern_consistency',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': BEHAVIORAL_PATTERN_CONSISTENCY_JUDGE,
            'evaluation_criteria': [
                'Spending Pattern Recognition',
                'Merchant Category Consistency',
                'Time Pattern Analysis',
                'Velocity Pattern Recognition',
                'Profile Stability Assessment'
            ],
            'domain': 'behavioral pattern analysis'
        },
        {
            'node': 'behavioral_analizer',
            'evaluator': 'temporal_anomaly',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': TEMPORAL_ANOMALY_DETECTION_JUDGE,
            'evaluation_criteria': [
                'Time-of-Day Anomalies',
                'Velocity Spike Detection',
                'Sequential Pattern Analysis',
                'Behavioral Timing Shifts',
                'Account Lifecycle Timing'
            ],
            'domain': 'temporal anomaly detection'
        },
        {
            'node': 'behavioral_analizer',
            'evaluator': 'profile_evolution',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': PROFILE_EVOLUTION_VS_FRAUD_JUDGE,
            'evaluation_criteria': [
                'Life Event Recognition',
                'Gradual vs Sudden Change Analysis',
                'Account Takeover Indicators',
                'Synthetic Identity Maturation',
                'Context Consistency Validation'
            ],
            'domain': 'profile evolution analysis'
        }
    ]
    
    # Decision Aggregator Judges
    decision_judges = [
        {
            'node': 'decision_aggregator',
            'evaluator': 'format_compliance',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': OUTPUT_FORMAT_COMPLIANCE_JUDGE,
            'evaluation_criteria': [
                'final_decision field validation',
                'conclusion field compliance',
                'recommendations array validation',
                'reason field quality',
                'business logic alignment'
            ],
            'domain': 'output format compliance'
        },
        {
            'node': 'decision_aggregator',
            'evaluator': 'logic_consistency',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': DECISION_LOGIC_CONSISTENCY_JUDGE,
            'evaluation_criteria': [
                'Analyzer Consensus Requirements',
                'Weighted Risk Assessment',
                'Critical Override Patterns',
                'Context Integration',
                'Decision-Reasoning Alignment'
            ],
            'domain': 'decision logic consistency'
        },
        {
            'node': 'decision_aggregator',
            'evaluator': 'risk_synthesis',
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'system_prompt': HOLISTIC_RISK_SYNTHESIS_JUDGE,
            'evaluation_criteria': [
                'Information Completeness',
                'Risk vs Friction Balance',
                'Pattern Recognition Synthesis',
                'Business Impact Assessment',
                'Special Situation Handling'
            ],
            'domain': 'holistic risk synthesis'
        }
    ]
    
    # Insert behavioral judges
    for judge in behavioral_judges:
        import json
        
        conn.execute(
            sa.text("""
                INSERT INTO llm_judges (node, evaluator, model, temperature, system_prompt, evaluation_criteria, domain, is_active, total_evaluations, accuracy)
                VALUES (:node, :evaluator, :model, :temperature, :system_prompt, CAST(:evaluation_criteria AS jsonb), :domain, true, 0, 0.0)
                ON CONFLICT (node, evaluator) DO NOTHING
            """),
            {
                'node': judge['node'],
                'evaluator': judge['evaluator'],
                'model': judge['model'],
                'temperature': judge['temperature'],
                'system_prompt': judge['system_prompt'],
                'evaluation_criteria': json.dumps(judge['evaluation_criteria']),
                'domain': judge['domain']
            }
        )
    
    # Insert decision judges
    for judge in decision_judges:
        import json
        
        conn.execute(
            sa.text("""
                INSERT INTO llm_judges (node, evaluator, model, temperature, system_prompt, evaluation_criteria, domain, is_active, total_evaluations, accuracy)
                VALUES (:node, :evaluator, :model, :temperature, :system_prompt, CAST(:evaluation_criteria AS jsonb), :domain, true, 0, 0.0)
                ON CONFLICT (node, evaluator) DO NOTHING
            """),
            {
                'node': judge['node'],
                'evaluator': judge['evaluator'],
                'model': judge['model'],
                'temperature': judge['temperature'],
                'system_prompt': judge['system_prompt'],
                'evaluation_criteria': json.dumps(judge['evaluation_criteria']),
                'domain': judge['domain']
            }
        )


def downgrade() -> None:
    # Remove the judges
    conn = op.get_bind()
    conn.execute(sa.text("DELETE FROM llm_judges WHERE node IN ('behavioral_analizer', 'decision_aggregator')"))
    
    # Drop the unique constraint
    op.drop_constraint('unique_node_evaluator', 'llm_judges', type_='unique')
    
    # Drop the evaluator column
    op.drop_column('llm_judges', 'evaluator')
    
    # Recreate the unique constraint on node
    op.create_unique_constraint('llm_judges_node_key', 'llm_judges', ['node'])
