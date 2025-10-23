"""
Mock fraud detection agent for testing.
"""
import logging
from typing import Dict, Any, List
import random

logger = logging.getLogger(__name__)


class MockFraudAgent:
    """Mock fraud detection agent for testing."""
    
    def __init__(self):
        self.nodes = ["behavior_analyzer", "risk_scorer", "pattern_detector", "transaction_validator", "history_checker"]
    
    async def analyze(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transaction and return decision.
        
        Args:
            transaction: Transaction data
        
        Returns:
            Dict with decision, risk_score, nodes, reasoning
        """
        # Mock analysis
        risk_score = random.randint(0, 100)
        decision = "DECLINE" if risk_score > 60 else "APPROVE"
        
        return {
            "decision": decision,
            "risk_score": risk_score,
            "nodes": self.nodes,
            "reasoning": f"Mock analysis for transaction"
        }

