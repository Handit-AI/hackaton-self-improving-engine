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
        
        # Analyze judge reasoning to determine problem type
        if judge_reasoning:
            # If judge feedback exists, make it the PRIMARY focus
            prompt = f"""You are learning from {'success' if is_correct else 'failure'}.

PRIMARY FOCUS - The LLM Judge has identified this specific issue:
{judge_reasoning}

This is the MAIN problem to address. Generate a bullet that specifically fixes THIS issue.

Agent ({node}) Analysis:
Predicted: {predicted}
Correct: {correct}
Agent Reasoning: {agent_reasoning}

Input Context (for reference only):
{query[:500]}...

TASK: Generate ONE generalized, actionable heuristic/bullet for the {node} that addresses the pattern behind this issue.

CRITICAL CONTEXT: This bullet will be used as INSTRUCTION/HEURISTIC in prompts to guide the LLM. It must be written as an instruction that tells the LLM how to behave, not as code or programmatic logic.

CRITICAL REQUIREMENTS:
- The bullet MUST be written as an INSTRUCTION for the LLM agent
- The bullet MUST address the GENERAL PATTERN behind the judge's specific concern
- The bullet MUST be generalized to catch similar variations (not hardcoded to one example)
- The bullet MUST be a clear directive: "When X, do Y" or "Ensure that X" or "Check if X"
- The bullet MUST help prevent this type of error pattern from happening again
- Focus on the JUDGE's feedback, not the transaction details

Output JSON:
{{
  "new_bullet": "Generalized instruction/heuristic that addresses the pattern behind the judge's concern",
  "problem_types": ["type1", "type2"],
  "confidence": 0.0-1.0
}}

Examples based on judge feedback:
If judge says "format issue - final_decision is 'approve' instead of 'APPROVE'":
✓ "When outputting final_decision field, ensure it uses EXACTLY uppercase versions ('APPROVE', 'REVIEW', 'DECLINE'), not lowercase or mixed case variants"
✓ "Normalize final_decision values to uppercase before outputting: convert any case variation to the exact uppercase version"

If judge says "format issue - recommendations is string instead of array":
✓ "When outputting recommendations field, ensure it's a proper JSON array structure with square brackets, not a string representation"
✓ "For recommendations field, use JSON array syntax (e.g., [\"item1\", \"item2\"]), never stringified arrays or comma-separated strings"

If judge says "logic issue - DECLINE without sufficient analyzer support":
✓ "When deciding DECLINE, verify that at least 3/5 analyzers indicate high risk and cite specific analyzer findings in your reasoning"
✓ "For DECLINE decisions, ensure multiple analyzers support the decision and include specific evidence from those analyzers"

BAD examples (too specific, irrelevant, or not instructions):
✗ "Ensure that the final_decision field is set to 'REVIEW' exactly" (too specific to one value)
✗ "Flag transactions involving unknown merchants" (irrelevant to judge's concern)
✗ "Check user behavior patterns" (too vague)
✗ "if value == 'review': return 'REVIEW'" (not an instruction, this is code)
"""
        else:
            # No judge feedback - use generic reflection
            prompt = f"""You are learning from {'success' if is_correct else 'failure'}.

Input: {query}

Agent ({node}) Analysis:
Predicted: {predicted}
Correct: {correct}
Agent Reasoning: {agent_reasoning}

TASK: Generate ONE specific, actionable heuristic/bullet for the {node}.

CRITICAL REQUIREMENTS:
- The bullet MUST be specific and measurable (include thresholds, conditions, patterns)
- The bullet should help the agent improve on this type of issue

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
                    {"role": "system", "content": f"You are a learning system that generates actionable heuristics for {node} based on LLM judge feedback."},
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

