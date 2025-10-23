"""
Reflector - Generates new bullets from successes/failures.

This module handles the reflection process that creates new bullets
based on analysis results.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class Reflector:
    """Generates new bullets from successes/failures."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Reflector.
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def reflect(
        self,
        query: str,
        predicted: str,
        correct: str,
        node: str,
        agent_reasoning: str = "",
        judge_reasoning: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Generate new bullet from analysis result.
        
        Args:
            query: Input query text
            predicted: Predicted decision
            correct: Correct decision
            node: Agent node name
            agent_reasoning: Agent's reasoning
            judge_reasoning: Judge's reasoning/insight (optional)
        
        Returns:
            Dict with new_bullet, problem_types, confidence or None
        """
        is_correct = (predicted == correct)
        
        # Include judge reasoning if provided
        judge_context = ""
        if judge_reasoning:
            judge_context = f"\n\nJudge's Insight:\n{judge_reasoning}"
        
        prompt = f"""You are a fraud detection expert learning from {'success' if is_correct else 'failure'}.

Transaction: {query}

Agent ({node}) Analysis:
Predicted: {predicted}
Correct: {correct}
Agent Reasoning: {agent_reasoning}{judge_context}

Extract ONE specific, actionable fraud detection heuristic for the {node}.

Output JSON:
{{
  "new_bullet": "Specific, measurable heuristic with thresholds",
  "problem_types": ["type1", "type2"],
  "confidence": 0.0-1.0
}}

Examples:
✓ "VPN from high-risk countries (Nigeria, Romania) + crypto merchant = 95% fraud"
✓ "Velocity > 10 transactions/hour for new users (< 60 days) = card testing"
✗ "Be careful with suspicious users" (too vague)
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use 4o-mini for improvements
                messages=[
                    {"role": "system", "content": f"You are learning fraud detection patterns for {node}."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if not result.get("new_bullet"):
                logger.warning("Reflector returned empty bullet")
                return None
            
            logger.info(f"Generated bullet for {node}: {result['new_bullet'][:50]}...")
            return result
        
        except Exception as e:
            logger.error(f"Reflector error: {e}")
            return None

