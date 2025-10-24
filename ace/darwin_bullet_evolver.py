"""
Darwin-Gödel Machine for Bullet Evolution

Implements evolutionary bullet generation:
1. Generate initial population (6 bullets)
2. Test on ultra-hard problems (fitness evaluation)
3. Select top 3 performers
4. Crossover to create 3 new bullets
5. Test again and keep top 2
"""
from typing import List, Dict, Any, Tuple, Optional
import random
import logging
from openai import OpenAI
import os
import json
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class DarwinBulletEvolver:
    """
    Evolves bullets using Darwin-Gödel Machine principles.
    
    Process:
    1. Generate 6 candidate bullets
    2. Test on saved transactions using LLM judges (fitness)
    3. Select top 3
    4. Crossover top 3 to create 3 children
    5. Test children and keep top 2
    """
    
    def __init__(self, api_key: Optional[str] = None, db_session=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.db_session = db_session
        self.test_dataset = None  # Will be loaded from saved transactions
    
    def load_test_dataset_from_db(self, n_samples: int = 10, node: str = None):
        """Load transactions from database for fitness evaluation."""
        try:
            from models import Transaction
            
            # Get recent transactions, optionally filtered by node
            query = self.db_session.query(Transaction)
            
            if node:
                query = query.filter(Transaction.node == node)
            
            transactions = query.order_by(
                Transaction.analyzed_at.desc()
            ).limit(n_samples).all()
            
            self.test_dataset = []
            for txn in transactions:
                self.test_dataset.append({
                    'query': str(txn.transaction_data),
                    'answer': txn.correct_decision,
                    'predicted': txn.predicted_decision
                })
            
            logger.info(f"Loaded {len(self.test_dataset)} transactions from database for fitness evaluation (node: {node})")
            
        except Exception as e:
            logger.error(f"Error loading transactions from DB: {e}")
            self.test_dataset = []
    
    async def evolve_bullets(
        self,
        initial_bullets: List[str],
        node: str,
        n_samples: int = 5,
        min_transactions: int = 3,
        max_transactions: int = 5,
        evaluator: Optional[str] = None
    ) -> List[str]:
        """
        Evolve bullets through Darwin-Gödel process (FAST MODE).
        
        Generate 4 bullets, test on 5 random transactions, keep top 2.
        
        Args:
            initial_bullets: List of bullet content strings to evolve
            node: Agent node name
            n_samples: Number of transactions to test on (default: 5)
            min_transactions: Minimum transactions needed to start evolution
            max_transactions: Maximum transactions to use for evaluation
            evaluator: Evaluator/perspective name (optional)
        
        Returns:
            List of top 2 evolved bullet contents
        """
        # Load test dataset from database
        if not self.test_dataset:
            if self.db_session:
                self.load_test_dataset_from_db(n_samples=max_transactions, node=node)
            else:
                logger.warning("No database session, skipping evolution")
                return initial_bullets[:2]
        
        # Check if we have enough transactions
        if len(self.test_dataset) < min_transactions:
            logger.info(f"Not enough transactions ({len(self.test_dataset)} < {min_transactions}) for evolution, returning initial bullets")
            return initial_bullets[:2]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Darwin-Gödel Evolution for {node} (evaluator: {evaluator})")
        logger.info(f"Using {len(self.test_dataset)} transactions from database")
        logger.info(f"{'='*60}")
        
        # Use limited number of samples
        actual_samples = min(n_samples, len(self.test_dataset))
        
        # Step 1: Generate 4 candidate bullets via crossover
        logger.info(f"\nStep 1: Generating 4 candidate bullets")
        candidate_bullets = await self._generate_candidates(initial_bullets[:6], node, n_candidates=4)
        
        if not candidate_bullets:
            logger.warning("No candidates generated")
            return initial_bullets[:2]
        
        logger.info(f"Generated {len(candidate_bullets)} candidates")
        
        # Step 2: Test candidates on random transactions
        logger.info(f"\nStep 2: Testing {len(candidate_bullets)} candidates on {actual_samples} transactions")
        fitness_scores = await self._evaluate_bullets(candidate_bullets, node, actual_samples, evaluator)
        
        # Sort by fitness
        sorted_bullets = sorted(zip(candidate_bullets, fitness_scores), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Fitness scores:")
        for i, (bullet, score) in enumerate(sorted_bullets):
            logger.info(f"  {i+1}. Score: {score:.2%} - {bullet[:60]}...")
        
        # Step 3: Keep top 2 bullets
        top_2 = [bullet for bullet, score in sorted_bullets[:2]]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evolution Complete!")
        logger.info(f"  Input: {len(initial_bullets)} bullets")
        logger.info(f"  Candidates generated: {len(candidate_bullets)}")
        logger.info(f"  Top 2 evolved bullets: {[b[:40] + '...' for b in top_2]}")
        logger.info(f"{'='*60}\n")
        
        return top_2
    
    async def _generate_candidates(self, bullets: List[str], node: str, n_candidates: int = 4) -> List[str]:
        """Generate candidate bullets via crossover."""
        candidates = []
        
        if len(bullets) < 2:
            return candidates
        
        # Generate n_candidates via crossover
        for i in range(n_candidates):
            # Select two random parents
            parent1, parent2 = random.sample(bullets, 2)
            
            # Crossover
            child = await self._crossover_two_bullets(parent1, parent2, node)
            if child:
                candidates.append(child)
        
        return candidates
    
    async def _evaluate_bullets(
        self,
        bullets: List[str],
        node: str,
        n_samples: int,
        evaluator: Optional[str] = None
    ) -> List[float]:
        """
        Evaluate bullets using LLM judges on saved transactions.
        
        Returns fitness scores (accuracy) for each bullet.
        
        Args:
            evaluator: Evaluator/perspective name (uses LLM judge domain if available)
        """
        # Sample transactions
        test_samples = random.sample(self.test_dataset, min(n_samples, len(self.test_dataset)))
        
        # Get evaluator context from LLM judges
        evaluator_context = evaluator
        if self.db_session:
            from models import LLMJudge
            judge = self.db_session.query(LLMJudge).filter(
                LLMJudge.node == node
            ).first()
            
            if judge and judge.domain:
                evaluator_context = judge.domain
        
        fitness_scores = []
        
        for bullet in bullets:
            correct = 0
            total = 0
            
            for sample in test_samples:
                # Use LLM judge to evaluate if bullet would help with this transaction
                evaluation = await self._judge_bullet_performance(
                    bullet=bullet,
                    transaction_query=sample['query'],
                    ground_truth=sample.get('answer', 'DECLINE'),
                    node=node,
                    evaluator_context=evaluator_context
                )
                
                if evaluation['is_helpful']:
                    correct += 1
                total += 1
            
            # Fitness is accuracy on all samples
            fitness = correct / total if total > 0 else 0.0
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    async def _judge_bullet_performance(
        self,
        bullet: str,
        transaction_query: str,
        ground_truth: str,
        node: str,
        evaluator_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM judge to evaluate if bullet would be helpful for this transaction.
        
        Returns dict with is_helpful boolean and reasoning.
        """
        try:
            # Use evaluator context if provided
            context_text = f" (Context: {evaluator_context})" if evaluator_context else ""
            
            prompt = f"""You are an LLM judge evaluating if a fraud detection heuristic would be helpful for a transaction.

HEURISTIC (Bullet):
{bullet}

TRANSACTION:
{transaction_query}

CORRECT DECISION: {ground_truth}

Node: {node}{context_text}

TASK: Determine if this heuristic would help correctly identify this transaction as fraudulent or legitimate.

Output JSON:
{{
  "is_helpful": true/false,
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}}

is_helpful is true if:
- The heuristic applies to this transaction AND
- The heuristic would lead to the correct decision ({ground_truth})

is_helpful is false if:
- The heuristic doesn't apply to this transaction OR
- The heuristic would lead to the wrong decision
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are an expert LLM judge for {node}."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "is_helpful": result.get("is_helpful", False),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error judging bullet performance: {e}")
            return {"is_helpful": False, "reasoning": str(e), "confidence": 0.0}
    
    async def _check_bullet_relevance(self, bullet: str, query: str, node: str) -> bool:
        """Check if bullet is relevant to the query using LLM."""
        try:
            prompt = f"""You are evaluating if a fraud detection heuristic is relevant to a transaction.

HEURISTIC (Bullet):
{bullet}

TRANSACTION:
{query}

Node: {node}

Is this heuristic relevant to evaluating this transaction?

Output JSON:
{{
  "relevant": true/false,
  "reason": "brief explanation"
}}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a fraud detection expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("relevant", False)
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return True  # Default to relevant if check fails
    
    async def _crossover_top_3(self, top_3_bullets: List[str], node: str) -> List[str]:
        """
        Crossover top 3 bullets to create 3 children.
        
        Strategy:
        - Child 1: Combine bullet 1 + bullet 2
        - Child 2: Combine bullet 2 + bullet 3
        - Child 3: Combine bullet 1 + bullet 3
        """
        children = []
        
        # Child 1: Bullet 1 × Bullet 2
        child1 = await self._crossover_two_bullets(top_3_bullets[0], top_3_bullets[1], node)
        if child1:
            children.append(child1)
        
        # Child 2: Bullet 2 × Bullet 3
        child2 = await self._crossover_two_bullets(top_3_bullets[1], top_3_bullets[2], node)
        if child2:
            children.append(child2)
        
        # Child 3: Bullet 1 × Bullet 3
        child3 = await self._crossover_two_bullets(top_3_bullets[0], top_3_bullets[2], node)
        if child3:
            children.append(child3)
        
        return children
    
    async def _crossover_two_bullets(
        self,
        parent1: str,
        parent2: str,
        node: str
    ) -> Optional[str]:
        """
        Crossover: Combine two parent bullets into a child.
        
        Uses LLM to intelligently merge patterns.
        """
        prompt = f"""You are a genetic programming system evolving fraud detection heuristics.

CROSSOVER OPERATION: Combine two successful parent bullets into one superior child.

Parent 1:
{parent1}

Parent 2:
{parent2}

Node: {node}

TASK: Create a child bullet that intelligently combines features from both parents.

CROSSOVER STRATEGIES:
1. **Conjunction** - Combine conditions with AND logic
   Example: "New user" + "VPN usage" → "New user AND VPN usage"

2. **Threshold Blending** - Average or select better threshold
   Example: "> 90 days" + "> 60 days" → "> 75 days"

3. **Pattern Fusion** - Merge complementary fraud indicators
   Example: "high amount + crypto" + "new user + night time" → "new user + high amount + crypto + night time"

4. **Risk Escalation** - Combine to create stronger signal
   Example: "80% fraud" + "85% fraud" → "90% fraud"

REQUIREMENTS:
- Child must be MORE SPECIFIC than either parent alone
- Include MEASURABLE thresholds (exact numbers, percentages)
- Maintain logical consistency
- Create something NOVEL but GROUNDED in parent logic

OUTPUT JSON:
{{
  "child_bullet": "Specific combined fraud heuristic with thresholds",
  "inherited_from_parent1": "What was taken from parent 1",
  "inherited_from_parent2": "What was taken from parent 2"
}}

Example:
Parent1: "New user (< 90 days) + large amount (> $1000) = 80% fraud"
Parent2: "VPN usage + crypto merchant = 95% fraud"
→ Child: "New user (< 90 days) + VPN + crypto merchant + amount > $1000 = 97% fraud"
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a genetic programming system for fraud detection."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("child_bullet")
            
        except Exception as e:
            logger.error(f"Crossover error: {e}")
            return None

