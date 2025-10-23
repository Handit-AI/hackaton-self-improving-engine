"""
LLM Judge - Evaluates correctness of agent outputs.

This module provides an LLM-based judge to evaluate whether agent outputs
are correct, especially useful when ground truth is not available.
"""
import os
import logging
from typing import Optional
from openai import OpenAI
from sqlalchemy.orm import Session
from models import LLMJudge, JudgeEvaluation

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-based judge for evaluating agent outputs.
    
    Uses GPT-4o-mini to determine if outputs are correct.
    Can be configured per node via database.
    """
    
    def __init__(self, api_key: Optional[str] = None, db_session: Optional[Session] = None):
        """
        Initialize LLM Judge.
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            db_session: Database session for loading judge config
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.db_session = db_session
        self._judge_cache = {}  # Cache for judge configs
    
    def _get_judge_config(self, node: str) -> dict:
        """Get judge configuration for a node from database or cache."""
        # Check cache first
        if node in self._judge_cache:
            return self._judge_cache[node]
        
        # Load from database if available
        if self.db_session:
            try:
                judge_config = self.db_session.query(LLMJudge).filter(
                    LLMJudge.node == node,
                    LLMJudge.is_active == True
                ).first()
                
                if judge_config:
                    config = {
                        'model': judge_config.model,
                        'temperature': judge_config.temperature,
                        'system_prompt': judge_config.system_prompt,
                        'evaluation_criteria': judge_config.evaluation_criteria,
                        'domain': judge_config.domain
                    }
                    self._judge_cache[node] = config
                    return config
            except Exception as e:
                logger.warning(f"Failed to load judge config from DB: {e}")
        
        # Default configuration
        default_config = {
            'model': 'gpt-4o-mini',
            'temperature': 0.0,
            'system_prompt': f"You are an expert judge evaluating fraud detection outputs for {node}.",
            'evaluation_criteria': None,
            'domain': 'fraud detection'
        }
        self._judge_cache[node] = default_config
        return default_config
    
    async def judge(
        self,
        input_text: str,
        output: str,
        ground_truth: Optional[str] = None,
        node: str = "fraud_detection",
        save_to_db: bool = True
    ) -> tuple[bool, float, str]:
        """
        Judge if output is correct.
        
        Args:
            input_text: Input to the model
            output: Model output
            ground_truth: Ground truth (if available)
            node: Agent node name
            save_to_db: Whether to save evaluation to database
        
        Returns:
            Tuple of (is_correct, confidence, reason)
        """
        # Get judge configuration for this node
        config = self._get_judge_config(node)
        
        # If ground truth provided, use exact match
        if ground_truth:
            is_correct = output.strip().lower() == ground_truth.strip().lower()
            reason = "Matches ground truth" if is_correct else "Does not match ground truth"
            return is_correct, 1.0 if is_correct else 0.0, reason
        
        # Otherwise use LLM judge
        try:
            # Build evaluation criteria into prompt
            criteria_text = ""
            if config.get('evaluation_criteria'):
                criteria_list = config['evaluation_criteria']
                criteria_text = f"\n\nEvaluation Criteria:\n" + "\n".join(f"- {c}" for c in criteria_list)
            
            response = self.client.chat.completions.create(
                model=config['model'],
                messages=[
                    {
                        "role": "system",
                        "content": config['system_prompt']
                    },
                    {
                        "role": "user",
                        "content": f"""Input: {input_text}

Output: {output}

Is this output correct and appropriate for this {config['domain']} scenario?{criteria_text}

Provide your assessment in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation of why it is correct or incorrect"
}}"""
                    }
                ],
                response_format={"type": "json_object"},
                temperature=config['temperature']
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            is_correct = result.get("is_correct", False)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reason", "")
            
            # Save to database if enabled
            if save_to_db and self.db_session:
                try:
                    judge_config = self.db_session.query(LLMJudge).filter(
                        LLMJudge.node == node
                    ).first()
                    
                    if judge_config:
                        evaluation = JudgeEvaluation(
                            judge_id=judge_config.id,
                            input_text=input_text,
                            output_text=output,
                            ground_truth=ground_truth,
                            is_correct=is_correct,
                            confidence=confidence,
                            reasoning=reasoning
                        )
                        self.db_session.add(evaluation)
                        self.db_session.commit()
                        
                        # Update judge stats
                        judge_config.total_evaluations += 1
                        self.db_session.commit()
                except Exception as e:
                    logger.error(f"Failed to save judge evaluation: {e}")
            
            logger.info(f"LLM Judge ({node}): correct={is_correct}, confidence={confidence:.2f}, reason={reasoning[:50]}...")
            return is_correct, confidence, reasoning
        
        except Exception as e:
            logger.error(f"Error in LLM judge: {e}")
            # Default to conservative judgment
            return False, 0.0, "Error in judgment"
    
    async def judge_batch(
        self,
        evaluations: list[dict[str, str]],
        node: str = "fraud_detection"
    ) -> list[tuple[bool, float, str]]:
        """
        Judge multiple outputs in batch.
        
        Args:
            evaluations: List of dicts with 'input', 'output', 'ground_truth'
            node: Agent node name
        
        Returns:
            List of (is_correct, confidence, reason) tuples
        """
        results = []
        for eval_data in evaluations:
            result = await self.judge(
                input_text=eval_data['input'],
                output=eval_data['output'],
                ground_truth=eval_data.get('ground_truth'),
                node=node
            )
            results.append(result)
        return results


# Global instance
_default_judge = None


def get_judge() -> LLMJudge:
    """Get or create default judge instance."""
    global _default_judge
    if _default_judge is None:
        _default_judge = LLMJudge()
    return _default_judge

