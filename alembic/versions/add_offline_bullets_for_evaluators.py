"""add_offline_bullets_for_evaluators

Revision ID: a1b2c3d4e5f6
Revises: e4c5279f5d8b
Create Date: 2025-10-23 16:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import uuid


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'e4c5279f5d8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    
    # Offline bullets for evaluators based on challenging edge cases
    bullets = [
        # Decision Aggregator - FORMAT_COMPLIANCE (3 bullets)
        {
            'id': f"decision_aggregator_format_{str(uuid.uuid4())[:8]}",
            'content': 'When final_decision field is present, ensure it uses EXACTLY one of: "APPROVE", "REVIEW", or "DECLINE" in uppercase. Never use variations like "approve", "Approve", "APPROVED", "OK", or "YES".',
            'node': 'decision_aggregator',
            'evaluator': 'format_compliance',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_format_{str(uuid.uuid4())[:8]}",
            'content': 'The recommendations field must be an array of strings, never a string representation of an array. Use ["item1", "item2"] not "[\'item1\', \'item2\']". Maximum 5 recommendations, each under 100 characters.',
            'node': 'decision_aggregator',
            'evaluator': 'format_compliance',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_format_{str(uuid.uuid4())[:8]}",
            'content': 'The conclusion field must be 1-2 sentences (under 200 characters), professional language, no emoji or formatting. If conclusion contradicts final_decision (e.g., APPROVE + negative conclusion), flag as format violation.',
            'node': 'decision_aggregator',
            'evaluator': 'format_compliance',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        
        # Decision Aggregator - LOGIC_CONSISTENCY (3 bullets)
        {
            'id': f"decision_aggregator_logic_{str(uuid.uuid4())[:8]}",
            'content': 'For DECLINE decisions, at least 3/5 analyzers must indicate high risk OR 2/5 with critical risk (ATO, synthetic identity) OR 1/5 with confirmed fraud pattern. Reasoning MUST cite specific analyzer findings.',
            'node': 'decision_aggregator',
            'evaluator': 'logic_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_logic_{str(uuid.uuid4())[:8]}",
            'content': 'For APPROVE decisions, maximum 1/5 analyzers showing risk, no critical risk indicators, all high-weight analyzers (pattern, velocity) must be clean, and must explain why any concerns are mitigated.',
            'node': 'decision_aggregator',
            'evaluator': 'logic_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_logic_{str(uuid.uuid4())[:8]}",
            'content': 'For REVIEW decisions, expect 2-3 analyzers showing moderate risk OR conflicting signals (some high, some low) OR legitimate but unusual patterns. Must specify what needs manual review.',
            'node': 'decision_aggregator',
            'evaluator': 'logic_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        
        # Decision Aggregator - RISK_SYNTHESIS (3 bullets)
        {
            'id': f"decision_aggregator_risk_{str(uuid.uuid4())[:8]}",
            'content': 'For emergency situations (hospitals, pharmacies, natural disasters, stranded travelers), bias toward APPROVE/REVIEW, never auto-DECLINE. These are time-critical legitimate needs.',
            'node': 'decision_aggregator',
            'evaluator': 'risk_synthesis',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_risk_{str(uuid.uuid4())[:8]}",
            'content': 'For life transitions (new baby, gender transition, foster placement, refugee status), expect spending pattern changes but still APPROVE if contextual factors align. Look for gradual evolution over sudden changes.',
            'node': 'decision_aggregator',
            'evaluator': 'risk_synthesis',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"decision_aggregator_risk_{str(uuid.uuid4())[:8]}",
            'content': 'For authorized users (teenage children, disabled helpers, caregivers), check for authorized_user field, authorized_helpers list, or FAMILY card type. Different behavioral patterns from primary cardholder are expected.',
            'node': 'decision_aggregator',
            'evaluator': 'risk_synthesis',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        
        # Behavioral Analyzer - PATTERN_CONSISTENCY (3 bullets)
        {
            'id': f"behavioral_pattern_{str(uuid.uuid4())[:8]}",
            'content': 'When transaction amount is >3x typical_transaction_amount, flag as suspicious UNLESS context justifies (life event, emergency, professional purchase). Check customer segment: STUDENT vs PREMIUM have different thresholds.',
            'node': 'behavioral_analizer',
            'evaluator': 'pattern_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_pattern_{str(uuid.uuid4())[:8]}",
            'content': 'Check merchant_category_consistency score: >0.7 = consistent, <0.4 = unusual. For new merchant categories, consider if they align with life events (baby products, medical, college supplies) vs fraud indicators.',
            'node': 'behavioral_analizer',
            'evaluator': 'pattern_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_pattern_{str(uuid.uuid4())[:8]}",
            'content': 'Account age <90 days has limited pattern reliability. For new accounts, focus on velocity analysis rather than spending patterns. Students/young accounts: Higher variance tolerance (stability <0.5 is normal).',
            'node': 'behavioral_analizer',
            'evaluator': 'pattern_consistency',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        
        # Behavioral Analyzer - TEMPORAL_ANOMALY (3 bullets)
        {
            'id': f"behavioral_temporal_{str(uuid.uuid4())[:8]}",
            'content': 'Flag transactions outside typical_time_of_day UNLESS: customer has night_activity_normal=true OR transaction type justifies timing (emergency pharmacy, 24hr services) OR timezone differences for travel.',
            'node': 'behavioral_analizer',
            'evaluator': 'temporal_anomaly',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_temporal_{str(uuid.uuid4())[:8]}",
            'content': 'Detect velocity spikes: >3 transactions in 1 hour when typical is 1-2/day, or spending 50% of monthly typical in 1 hour. Distinguish fraud (random merchants, decreasing amounts) from legitimate (related merchants, consistent amounts).',
            'node': 'behavioral_analizer',
            'evaluator': 'temporal_anomaly',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_temporal_{str(uuid.uuid4())[:8]}",
            'content': 'For sequential pattern analysis, detect rapid sequential transactions: <1 minute between distant locations = impossible travel, consistent 30-60 second gaps = automated/bot behavior.',
            'node': 'behavioral_analizer',
            'evaluator': 'temporal_anomaly',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        
        # Behavioral Analyzer - PROFILE_EVOLUTION (3 bullets)
        {
            'id': f"behavioral_evolution_{str(uuid.uuid4())[:8]}",
            'content': 'Legitimate life events show gradual changes over weeks/months: income increase → gradual spending increase, new hobby → consistent new merchant category. Fraud shows sudden changes within hours/days.',
            'node': 'behavioral_analizer',
            'evaluator': 'profile_evolution',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_evolution_{str(uuid.uuid4())[:8]}",
            'content': 'Account takeover indicators: password/email/phone changed + immediate transactions, complete abandonment of historical merchants, digital goods focus when physical was normal, gift card purchases when never before.',
            'node': 'behavioral_analizer',
            'evaluator': 'profile_evolution',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
        {
            'id': f"behavioral_evolution_{str(uuid.uuid4())[:8]}",
            'content': 'Synthetic identity patterns: too perfect patterns (exact same time daily), no organic variation in amounts, credit building pattern (small → medium → bust-out), missing human inconsistencies, all transactions on specific dates.',
            'node': 'behavioral_analizer',
            'evaluator': 'profile_evolution',
            'source': 'offline',
            'helpful_count': 0,
            'harmful_count': 0,
            'times_selected': 0
        },
    ]
    
    # Insert bullets
    for bullet in bullets:
        conn.execute(
            sa.text("""
                INSERT INTO bullets (id, content, node, evaluator, source, helpful_count, harmful_count, times_selected)
                VALUES (:id, :content, :node, :evaluator, :source, :helpful_count, :harmful_count, :times_selected)
                ON CONFLICT (id) DO NOTHING
            """),
            bullet
        )


def downgrade() -> None:
    conn = op.get_bind()
    
    # Delete the offline bullets we inserted
    conn.execute(
        sa.text("""
            DELETE FROM bullets 
            WHERE source = 'offline' 
            AND evaluator IN ('format_compliance', 'logic_consistency', 'risk_synthesis', 'pattern_consistency', 'temporal_anomaly', 'profile_evolution')
        """)
    )

