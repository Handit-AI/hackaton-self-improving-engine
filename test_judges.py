"""
LLM Judge Configurations for Testing

Three different judge perspectives:
1. Conservative Judge - Prevents false positives (lower risk tolerance)
2. Balanced Judge - Balanced approach (moderate risk tolerance)
3. Aggressive Judge - Prevents false negatives (higher risk tolerance)
"""
from models import LLMJudge


def create_conservative_judge(db_session):
    """Create a risk-focused judge - evaluates risk assessment correctness."""
    # Check if already exists
    existing = db_session.query(LLMJudge).filter(
        LLMJudge.node == "fraud_detection_risk_focused"
    ).first()
    
    if existing:
        return existing
    
    judge = LLMJudge(
        node="fraud_detection_risk_focused",
        model="gpt-4o-mini",
        temperature=0.0,
        system_prompt="""You are a risk assessment expert evaluating fraud detection decisions.

Focus ONLY on: Does the decision correctly assess the risk factors?

Evaluate:
- Are high-risk factors (VPN, suspicious location, new user, etc.) properly weighted?
- Does the decision match the risk level implied by the transaction details?
- Would a risk analyst make the same decision?

Ignore: User experience, business rules, policy decisions.
Focus on: Technical risk assessment accuracy.""",
        evaluation_criteria=[
            "Do the risk factors support this decision?",
            "Are suspicious patterns appropriately flagged?",
            "Is the risk level correctly assessed?"
        ],
        domain="fraud detection - risk assessment",
        is_active=True
    )
    db_session.add(judge)
    db_session.commit()
    return judge


def create_balanced_judge(db_session):
    """Create a pattern-focused judge - evaluates pattern matching correctness."""
    # Check if already exists
    existing = db_session.query(LLMJudge).filter(
        LLMJudge.node == "fraud_detection_pattern_focused"
    ).first()
    
    if existing:
        return existing
    
    judge = LLMJudge(
        node="fraud_detection_pattern_focused",
        model="gpt-4o-mini",
        temperature=0.1,
        system_prompt="""You are a fraud pattern expert evaluating fraud detection decisions.

Focus ONLY on: Does the decision match known fraud patterns?

Evaluate:
- Does this transaction show patterns of known fraud schemes?
- Are legitimate transaction patterns being recognized?
- Does the decision match what fraud detection patterns suggest?

Ignore: Risk assessment, business rules, user experience.
Focus on: Pattern matching accuracy.""",
        evaluation_criteria=[
            "Does this match known fraud patterns?",
            "Does this match legitimate transaction patterns?",
            "Are pattern indicators correctly interpreted?"
        ],
        domain="fraud detection - pattern matching",
        is_active=True
    )
    db_session.add(judge)
    db_session.commit()
    return judge


def create_aggressive_judge(db_session):
    """Create a context-focused judge - evaluates contextual correctness."""
    # Check if already exists
    existing = db_session.query(LLMJudge).filter(
        LLMJudge.node == "fraud_detection_context_focused"
    ).first()
    
    if existing:
        return existing
    
    judge = LLMJudge(
        node="fraud_detection_context_focused",
        model="gpt-4o-mini",
        temperature=0.2,
        system_prompt="""You are a fraud context expert evaluating fraud detection decisions.

Focus ONLY on: Does the decision consider the full transaction context?

Evaluate:
- Does the decision account for user history and behavior?
- Is the transaction context (time, location, merchant) properly considered?
- Would this decision make sense given the complete picture?

Ignore: Risk factors in isolation, pattern matching alone.
Focus on: Contextual appropriateness.""",
        evaluation_criteria=[
            "Is user history properly considered?",
            "Is transaction context appropriately used?",
            "Does the decision fit the complete picture?"
        ],
        domain="fraud detection - contextual analysis",
        is_active=True
    )
    db_session.add(judge)
    db_session.commit()
    return judge


def create_all_judges(db_session):
    """Create all three judges for testing."""
    return {
        'risk_focused': create_conservative_judge(db_session),
        'pattern_focused': create_balanced_judge(db_session),
        'context_focused': create_aggressive_judge(db_session)
    }


def get_judge_for_mode(mode: str, judges: dict):
    """Get the appropriate judge for a test mode."""
    if mode == 'vanilla':
        return judges['pattern_focused']
    elif mode == 'offline_online':
        return judges['risk_focused']
    elif mode == 'online_only':
        return judges['context_focused']
    else:
        return judges['pattern_focused']

