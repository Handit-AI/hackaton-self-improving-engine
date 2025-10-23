"""
Comprehensive Testing Suite for ACE System

Tests 3 modes:
1. Vanilla - No improvements
2. Offline + Online - Pre-trained with offline bullets
3. Online Only - Real-time learning only

Uses GPT-4o mini for simple prompts and LLM judge for evaluation.
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

from openai import OpenAI
from ace.llm_judge import LLMJudge
from test_judges import create_all_judges, get_judge_for_mode
from database import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results for a single test."""
    mode: str
    total: int
    correct: int
    accuracy: float
    details: List[Dict[str, Any]]


class SimpleJudge:
    """Simple LLM judge using GPT-4o mini."""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Keep 4o-mini for quality judgment
    
    def judge(self, input_text: str, output: str, ground_truth: str) -> Tuple[bool, float, str]:
        """Judge if output matches ground truth."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a judge evaluating fraud detection outputs. Answer in JSON format."
                    },
                    {
                        "role": "user",
                        "content": f"""Input: {input_text}

Predicted Output: {output}
Ground Truth: {ground_truth}

Is the predicted output correct? Respond in JSON:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation of why it is correct or incorrect"
}}"""
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("is_correct", False), result.get("confidence", 0.5), result.get("reason", "")
        
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return False, 0.0, "Error in judgment"


class SimpleFraudAgent:
    """Simple fraud detection agent using GPT-3.5-turbo for speed."""
    
    def __init__(self, api_key: str = None, bullets: List[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"  # Use 3.5-turbo for fast agent responses
        self.bullets = bullets or []
    
    def analyze(self, transaction: str) -> Dict[str, Any]:
        """Analyze transaction and return decision."""
        try:
            bullets_context = ""
            if self.bullets:
                bullets_context = "\n\nGuidelines:\n" + "\n".join(f"- {b}" for b in self.bullets)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a fraud detection expert. Analyze transactions and return structured output.

Your response must be JSON with:
{
    "decision": "APPROVE" or "DECLINE",
    "reasoning": "brief explanation of your decision"
}"""
                    },
                    {
                        "role": "user",
                        "content": f"""Transaction: {transaction}{bullets_context}

Analyze this transaction and determine if it should be APPROVED or DECLINED.

Return JSON:
{{
    "decision": "APPROVE" or "DECLINE",
    "reasoning": "brief explanation"
}}"""
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "decision": result.get("decision", "DECLINE"),
                "reasoning": result.get("reasoning", "")
            }
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"decision": "DECLINE", "reasoning": f"Error: {e}"}


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def split_dataset(dataset: List[Dict[str, Any]], train_ratio: float = 0.7) -> Tuple[List, List]:
    """Split dataset into train and test sets."""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def run_vanilla_mode(test_set: List[Dict[str, Any]], judge) -> TestResult:
    """Test vanilla mode - no improvements."""
    logger.info("=" * 60)
    logger.info("Testing VANILLA mode")
    logger.info("=" * 60)
    
    agent = SimpleFraudAgent()
    results = []
    correct = 0
    
    for idx, item in enumerate(test_set):
        logger.info(f"\n[{idx+1}/{len(test_set)}] Testing vanilla mode...")
        
        result = agent.analyze(item['query'])
        predicted = result['decision']
        ground_truth = item['answer']
        
        # Compare with ground truth directly (no LLM judge for offline evaluation)
        is_correct = predicted == ground_truth
        confidence = 1.0 if is_correct else 0.0
        reason = "Matches ground truth" if is_correct else "Does not match ground truth"
        
        if is_correct:
            correct += 1
        
        results.append({
            "query": item['query'][:100] + "...",
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "confidence": confidence,
            "reason": reason
        })
        
        logger.info(f"Query: {item['query'][:80]}...")
        logger.info(f"Predicted: {predicted} | Ground Truth: {ground_truth} | Correct: {is_correct}")
    
    accuracy = correct / len(test_set) if test_set else 0.0
    logger.info(f"\nVanilla Mode Accuracy: {accuracy:.2%}")
    
    return TestResult(
        mode="vanilla",
        total=len(test_set),
        correct=correct,
        accuracy=accuracy,
        details=results
    )


async def run_offline_online_mode(
    train_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    judge: SimpleJudge
) -> TestResult:
    """Test offline + online mode."""
    logger.info("=" * 60)
    logger.info("Testing OFFLINE + ONLINE mode")
    logger.info("=" * 60)
    
    # Step 1: Offline training
    logger.info("\n--- Step 1: Offline Training ---")
    
    from ace.training_pipeline import TrainingPipeline
    from ace.bullet_playbook import BulletPlaybook
    from ace.hybrid_selector import HybridSelector
    from ace.reflector import Reflector
    from ace.curator import Curator
    
    # Mock agent for offline training
    class MockAgent:
        async def analyze(self, transaction):
            return {"decision": "DECLINE", "reasoning": "Default"}
    
    mock_agent = MockAgent()
    selector = HybridSelector()
    reflector = Reflector()
    curator = Curator()
    training_pipeline = TrainingPipeline(mock_agent, selector, reflector, curator)
    playbook = BulletPlaybook()
    
    # Generate bullets from offline training
    logger.info(f"Training on {len(train_set)} examples...")
    offline_bullets = []
    
    for idx, item in enumerate(train_set):
        logger.info(f"  [{idx+1}/{len(train_set)}] Generating bullet...")
        
        bullet_id = await training_pipeline.add_bullet_from_reflection(
            query=item['query'],
            predicted=item['answer'],
            correct=item['answer'],
            node="fraud_detection",
            agent_reasoning=item['answer'],
            playbook=playbook,
            source="offline"
        )
        
        if bullet_id:
            offline_bullets.append(bullet_id)
    
    # Get bullets from playbook
    bullets = playbook.get_bullets_for_node("fraud_detection")
    bullet_texts = [b.content for b in bullets]
    
    logger.info(f"Generated {len(bullet_texts)} bullets from offline training")
    
    # Step 2: Test with offline bullets
    logger.info("\n--- Step 2: Testing with Offline Bullets ---")
    
    agent = SimpleFraudAgent(bullets=bullet_texts)
    results = []
    correct = 0
    
    for idx, item in enumerate(test_set):
        logger.info(f"\n[{idx+1}/{len(test_set)}] Testing offline+online mode...")
        
        result = agent.analyze(item['query'])
        predicted = result['decision']
        ground_truth = item['answer']
        
        # Compare with ground truth directly (no LLM judge - we have labels)
        is_correct = predicted == ground_truth
        confidence = 1.0 if is_correct else 0.0
        reason = "Matches ground truth" if is_correct else "Does not match ground truth"
        
        if is_correct:
            correct += 1
        
        results.append({
            "query": item['query'][:100] + "...",
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "confidence": confidence,
            "reason": reason,
            "bullets_used": len(bullet_texts)
        })
        
        logger.info(f"Query: {item['query'][:80]}...")
        logger.info(f"Predicted: {predicted} | Ground Truth: {ground_truth} | Correct: {is_correct}")
    
    accuracy = correct / len(test_set) if test_set else 0.0
    logger.info(f"\nOffline + Online Mode Accuracy: {accuracy:.2%}")
    
    return TestResult(
        mode="offline_online",
        total=len(test_set),
        correct=correct,
        accuracy=accuracy,
        details=results
    )


async def run_online_only_mode(test_set: List[Dict[str, Any]], judge: SimpleJudge) -> TestResult:
    """Test online only mode."""
    logger.info("=" * 60)
    logger.info("Testing ONLINE ONLY mode")
    logger.info("=" * 60)
    
    from ace.training_pipeline import TrainingPipeline
    from ace.bullet_playbook import BulletPlaybook
    from ace.hybrid_selector import HybridSelector
    from ace.reflector import Reflector
    from ace.curator import Curator
    
    # Mock agent
    class MockAgent:
        async def analyze(self, transaction):
            return {"decision": "DECLINE", "reasoning": "Default"}
    
    mock_agent = MockAgent()
    selector = HybridSelector()
    reflector = Reflector()
    curator = Curator()
    training_pipeline = TrainingPipeline(mock_agent, selector, reflector, curator)
    playbook = BulletPlaybook()
    
    agent = SimpleFraudAgent()
    results = []
    correct = 0
    
    for idx, item in enumerate(test_set):
        logger.info(f"\n[{idx+1}/{len(test_set)}] Testing online only mode...")
        
        # Analyze
        result = agent.analyze(item['query'])
        predicted = result['decision']
        ground_truth = item['answer']
        
        # Judge (SimpleJudge is synchronous)
        is_correct, confidence, reason = judge.judge(item['query'], predicted, ground_truth)
        
        if is_correct:
            correct += 1
        
        # Online learning with judge reasoning
        bullet_id = await training_pipeline.add_bullet_from_reflection(
            query=item['query'],
            predicted=predicted,
            correct=ground_truth,
            node="fraud_detection",
            agent_reasoning=predicted,
            playbook=playbook,
            source="online",
            judge_reasoning=reason  # Pass judge reasoning to bullet generation
        )
        
        results.append({
            "query": item['query'][:100] + "...",
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "confidence": confidence,
            "reason": reason,
            "bullet_added": bullet_id is not None
        })
        
        logger.info(f"Query: {item['query'][:80]}...")
        logger.info(f"Predicted: {predicted} | Ground Truth: {ground_truth} | Correct: {is_correct}")
    
    accuracy = correct / len(test_set) if test_set else 0.0
    logger.info(f"\nOnline Only Mode Accuracy: {accuracy:.2%}")
    
    return TestResult(
        mode="online_only",
        total=len(test_set),
        correct=correct,
        accuracy=accuracy,
        details=results
    )


def main():
    """Main test runner."""
    logger.info("Starting Comprehensive ACE Test Suite")
    logger.info("=" * 60)
    
    # Load datasets
    logger.info("\nLoading datasets...")
    complex_dataset = load_dataset("agents/complex_fraud_detection.json")
    ultra_hard_dataset = load_dataset("agents/ultra_hard_fraud_detection.json")
    ambiguous_dataset = load_dataset("agents/ambiguous_fraud_detection.json")
    
    logger.info(f"Complex dataset: {len(complex_dataset)} examples")
    logger.info(f"Ultra hard dataset: {len(ultra_hard_dataset)} examples")
    logger.info(f"Ambiguous dataset: {len(ambiguous_dataset)} examples")
    
    # Combine datasets
    full_dataset = complex_dataset + ultra_hard_dataset + ambiguous_dataset
    logger.info(f"Combined dataset: {len(full_dataset)} examples")
    
    # Split into train/test
    train_set, test_set = split_dataset(full_dataset, train_ratio=0.7)
    logger.info(f"Train set: {len(train_set)} examples")
    logger.info(f"Test set: {len(test_set)} examples")
    
    # Initialize LLM judges with different perspectives
    db_session = SessionLocal()
    try:
        judge_configs = create_all_judges(db_session)
        logger.info(f"Created {len(judge_configs)} LLM judges: {list(judge_configs.keys())}")
        
        # Create SimpleJudge instances for each judge
        judge_instances = {
            'risk_focused': SimpleJudge(),
            'pattern_focused': SimpleJudge(),
            'context_focused': SimpleJudge()
        }
    except Exception as e:
        logger.warning(f"Could not create judges in DB: {e}")
        # Fallback to SimpleJudge
        judge_instances = {
            'risk_focused': SimpleJudge(),
            'pattern_focused': SimpleJudge(),
            'context_focused': SimpleJudge()
        }
    
    # Run tests with appropriate judges
    results = []
    
    # 1. Vanilla mode - Pattern-focused judge (not used, just ground truth)
    logger.info("\n" + "="*60)
    logger.info("Vanilla Mode - Ground Truth Comparison")
    logger.info("="*60)
    vanilla_judge = judge_instances['pattern_focused']  # Not used, just for compatibility
    vanilla_result = run_vanilla_mode(test_set, vanilla_judge)
    results.append(vanilla_result)
    
    # 2. Offline + Online mode - Risk-focused judge (not used, just ground truth)
    logger.info("\n" + "="*60)
    logger.info("Offline + Online Mode - Ground Truth Comparison")
    logger.info("="*60)
    offline_online_judge = judge_instances['risk_focused']  # Not used, just for compatibility
    import asyncio
    offline_online_result = asyncio.run(run_offline_online_mode(train_set, test_set, offline_online_judge))
    results.append(offline_online_result)
    
    # 3. Online only mode - Context-focused judge (USES LLM JUDGE!)
    logger.info("\n" + "="*60)
    logger.info("Online Only Mode - Using CONTEXT-FOCUSED LLM Judge")
    logger.info("="*60)
    online_only_judge = judge_instances['context_focused']
    online_only_result = asyncio.run(run_online_only_mode(test_set, online_only_judge))
    results.append(online_only_result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"\n{result.mode.upper()}:")
        logger.info(f"  Total: {result.total}")
        logger.info(f"  Correct: {result.correct}")
        logger.info(f"  Accuracy: {result.accuracy:.2%}")
    
    # Save results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                r.mode: {
                    "total": r.total,
                    "correct": r.correct,
                    "accuracy": r.accuracy
                }
                for r in results
            },
            "details": {
                r.mode: r.details
                for r in results
            }
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

