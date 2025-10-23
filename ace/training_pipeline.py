"""
TrainingPipeline - Orchestrates offline and online training with Darwin-Gödel evolution.

This module handles the training process for both offline (pre-training)
and online (real-time learning) modes, with integrated Darwin-Gödel evolution.
"""
import logging
from typing import List, Dict, Any, Optional
from ace.bullet_playbook import BulletPlaybook
from ace.hybrid_selector import HybridSelector
from ace.reflector import Reflector
from ace.curator import Curator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates training for ACE system with Darwin-Gödel evolution.
    
    Supports:
    - Offline training: Pre-train on historical data
    - Online training: Real-time learning during operation
    - Darwin-Gödel evolution: Integrated into bullet generation
    """
    
    def __init__(
        self,
        fraud_agent,
        selector: HybridSelector,
        reflector: Reflector,
        curator: Curator,
        darwin_evolver=None
    ):
        """
        Initialize TrainingPipeline.
        
        Args:
            fraud_agent: Fraud detection agent instance
            selector: HybridSelector instance
            reflector: Reflector instance
            curator: Curator instance
            darwin_evolver: DarwinBulletEvolver instance (optional)
        """
        self.fraud_agent = fraud_agent
        self.selector = selector
        self.reflector = reflector
        self.curator = curator
        self.darwin_evolver = darwin_evolver
    
    async def train_offline(
        self,
        dataset: List[Dict[str, Any]],
        playbook: BulletPlaybook,
        n_bullets: int = 5
    ) -> Dict[str, Any]:
        """
        Offline training: Build playbook from historical data.
        
        This creates bullets with source='offline' and saves them to database.
        
        Args:
            dataset: List of training examples
            playbook: BulletPlaybook to populate
            n_bullets: Number of bullets to select per analysis
        
        Returns:
            Dict with training results and metrics
        """
        results = []
        iteration_metrics = []
        
        logger.info(f"Starting offline training on {len(dataset)} transactions...")
        
        for i, problem in enumerate(dataset):
            try:
                # Analyze with current playbook
                agent_result = await self.fraud_agent.analyze(problem['transaction'])
                
                # Determine correctness
                is_correct = (agent_result['decision'] == problem['answer'])
                
                # For each agent node, select bullets and enhance
                node_results = {}
                for node_name in agent_result.get('nodes', []):
                    # Select bullets for this node
                    if playbook.get_stats()['total_bullets'] > 0:
                        selected_bullets, _ = self.selector.select_bullets(
                            query=problem.get('query', str(problem['transaction'])),
                            node=node_name,
                            playbook=playbook,
                            n_bullets=n_bullets,
                            iteration=i
                        )
                    else:
                        selected_bullets = []
                    
                    # Update bullet stats based on correctness
                    for bullet in selected_bullets:
                        playbook.update_bullet_stats(bullet.id, is_correct)
                    
                    # Reflect if wrong or periodically
                    if not is_correct or i % 5 == 0:
                        reflection = await self.reflector.reflect(
                            query=problem.get('query', str(problem['transaction'])),
                            predicted=agent_result['decision'],
                            correct=problem['answer'],
                            node=node_name,
                            agent_reasoning=agent_result.get('reasoning', ''),
                            judge_reasoning=""  # No judge reasoning in offline training
                        )
                        
                        if reflection and reflection.get('new_bullet'):
                            self.curator.merge_bullet(
                                content=reflection['new_bullet'],
                                node=node_name,
                                playbook=playbook,
                                source="offline"  # Mark as offline
                            )
                    
                    node_results[node_name] = is_correct
                
                # Track overall correctness
                results.append(is_correct)
                
                # Log metrics
                accuracy = sum(results) / len(results)
                iteration_metrics.append({
                    'iteration': i,
                    'accuracy': accuracy,
                    'playbook_size': len(playbook.bullets),
                    'is_correct': is_correct,
                    'node_results': node_results
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Iteration {i+1}/{len(dataset)}: Accuracy={accuracy:.2%}, Playbook={len(playbook.bullets)} bullets")
            
            except Exception as e:
                logger.error(f"Error in offline training iteration {i}: {e}")
                continue
        
        final_accuracy = sum(results) / len(results) if results else 0.0
        
        logger.info(f"\nOffline training complete:")
        logger.info(f"  Final Accuracy: {final_accuracy:.2%}")
        logger.info(f"  Total Bullets: {len(playbook.bullets)}")
        logger.info(f"  Bullets per node: {playbook.get_stats()['bullets_per_node']}")
        
        return {
            'playbook': playbook,
            'final_accuracy': final_accuracy,
            'iteration_metrics': iteration_metrics,
            'problems_processed': len(dataset)
        }
    
    async def train_online(
        self,
        transaction: Dict[str, Any],
        playbook: BulletPlaybook,
        n_bullets: int = 5
    ) -> Dict[str, Any]:
        """
        Online training: Real-time learning from single transaction.
        
        This creates bullets with source='online' and updates stats.
        
        Args:
            transaction: Transaction to learn from
            playbook: BulletPlaybook to update
            n_bullets: Number of bullets to select
        
        Returns:
            Dict with analysis result and learning outcome
        """
        # Analyze transaction
        agent_result = await self.fraud_agent.analyze(transaction)
        
        # For each node, select bullets and update
        for node_name in agent_result.get('nodes', []):
            # Select bullets (can use offline + online)
            selected_bullets, _ = self.selector.select_bullets(
                query=str(transaction),
                node=node_name,
                playbook=playbook,
                n_bullets=n_bullets,
                iteration=0
            )
            
            # Update stats based on actual performance
            # (In real deployment, you'd check against ground truth)
            for bullet in selected_bullets:
                # Mark as used
                bullet.times_selected += 1
                
                # Performance update would happen after validation
                # For now, just track selection
        
        return {
            'decision': agent_result['decision'],
            'risk_score': agent_result.get('risk_score', 0),
            'nodes': agent_result.get('nodes', []),
            'selected_bullets': [b.id for b in selected_bullets] if selected_bullets else []
        }
    
    async def add_bullet_from_reflection(
        self,
        query: str,
        predicted: str,
        correct: str,
        node: str,
        agent_reasoning: str,
        playbook: BulletPlaybook,
        source: str = "online",
        judge_reasoning: str = "",
        evaluator: Optional[str] = None
    ) -> Optional[str]:
        """
        Add a new bullet from reflection with Darwin-Gödel evolution.
        
        Evolution happens EVERY time a bullet is generated:
        1. Generate bullet from reflection
        2. Generate 4 candidates via crossover
        3. Test all 5 bullets on transactions
        4. Keep only top 2 bullets
        
        Args:
            query: Input query
            predicted: Predicted decision
            correct: Correct decision
            node: Agent node
            agent_reasoning: Agent's reasoning
            playbook: BulletPlaybook to update
            source: 'offline' or 'online'
            judge_reasoning: Judge's reasoning/insight (optional)
            evaluator: Evaluator/perspective name (optional)
        
        Returns:
            Bullet ID if added, None if duplicate
        """
        # Step 1: Generate bullet from reflection
        reflection = await self.reflector.reflect(
            query=query,
            predicted=predicted,
            correct=correct,
            node=node,
            agent_reasoning=agent_reasoning,
            judge_reasoning=judge_reasoning
        )
        
        if not reflection or not reflection.get('new_bullet'):
            return None
        
        new_bullet_content = reflection['new_bullet']
        
        # Step 2: Darwin-Gödel Evolution (if enabled)
        if self.darwin_evolver:
            # Get recent bullets for crossover (NOT for testing - only used as parents)
            recent_bullets = playbook.get_bullets_for_node(node, evaluator=evaluator)
            
            if len(recent_bullets) >= 2:
                logger.info(f"🧬 Darwin-Gödel Evolution: Testing new bullet for {node} (evaluator: {evaluator})")
                
                # Get last 6 bullets for crossover (these are NOT tested, only used as parents)
                bullet_texts = [b.content for b in recent_bullets[-6:]]
                
                # Step 2a: Generate 4 candidates via crossover
                candidate_bullets = await self.darwin_evolver._generate_candidates(
                    bullet_texts, 
                    node, 
                    n_candidates=4
                )
                
                # Step 2b: Create test set: ONLY newly generated bullets (1 new + 4 candidates = 5 total)
                # IMPORTANT: We are NOT testing old bullets from playbook, only these new ones
                all_bullets_to_test = [new_bullet_content] + candidate_bullets
                
                logger.info(f"  Testing {len(all_bullets_to_test)} NEW bullets (1 new + 4 candidates)")
                
                # Step 2c: Load test dataset if not already loaded
                if not self.darwin_evolver.test_dataset:
                    if self.darwin_evolver.db_session:
                        self.darwin_evolver.load_test_dataset_from_db(n_samples=4, node=node)  # Reduced to 4 scenarios
                    else:
                        logger.warning("No database session for evolution, skipping fitness evaluation")
                        # Just return the new bullet without evolution
                        initial_bullet_id = self.curator.merge_bullet(
                            content=new_bullet_content,
                            node=node,
                            playbook=playbook,
                            source=source,
                            evaluator=evaluator
                        )
                        return initial_bullet_id
                
                # Step 2d: Test all 5 NEW bullets on transactions (reduced to 4 scenarios)
                fitness_scores = await self.darwin_evolver._evaluate_bullets(
                    all_bullets_to_test,
                    node=node,
                    n_samples=4,  # Reduced from 5 to 4
                    evaluator=evaluator
                )
                
                # Step 2e: Sort by fitness and keep ONLY top 1 bullet
                sorted_bullets = sorted(
                    zip(all_bullets_to_test, fitness_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                logger.info(f"  Fitness scores:")
                for i, (bullet, score) in enumerate(sorted_bullets):
                    logger.info(f"    {i+1}. Score: {score:.2%} - {bullet[:60]}...")
                
                top_bullet = sorted_bullets[0] if sorted_bullets else None
                
                if top_bullet:
                    logger.info(f"  ✓ Keeping best bullet (score: {top_bullet[1]:.2%})")
                    
                    # Step 2f: Add the best bullet (skip duplicates)
                    bullet_id = self.curator.merge_bullet(
                        content=top_bullet[0],
                        node=node,
                        playbook=playbook,
                        source="evolution",
                        evaluator=evaluator
                    )
                    
                    if bullet_id:
                        logger.info(f"    ✓ Added evolved bullet: {bullet_id[:50]}...")
                        return bullet_id
                
                return None
            else:
                # Not enough bullets for evolution, just add the new bullet
                logger.info(f"Not enough bullets for evolution ({len(recent_bullets)} < 2), adding new bullet directly")
                initial_bullet_id = self.curator.merge_bullet(
                    content=new_bullet_content,
                    node=node,
                    playbook=playbook,
                    source=source,
                    evaluator=evaluator
                )
                return initial_bullet_id
        else:
            # No evolution, just add the bullet
            initial_bullet_id = self.curator.merge_bullet(
                content=new_bullet_content,
                node=node,
                playbook=playbook,
                source=source,
                evaluator=evaluator
            )
            return initial_bullet_id

