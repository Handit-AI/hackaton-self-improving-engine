"""
TrainingPipeline - Orchestrates offline and online training.

This module handles the training process for both offline (pre-training)
and online (real-time learning) modes.
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
    Orchestrates training for ACE system.
    
    Supports:
    - Offline training: Pre-train on historical data
    - Online training: Real-time learning during operation
    """
    
    def __init__(
        self,
        fraud_agent,
        selector: HybridSelector,
        reflector: Reflector,
        curator: Curator
    ):
        """
        Initialize TrainingPipeline.
        
        Args:
            fraud_agent: Fraud detection agent instance
            selector: HybridSelector instance
            reflector: Reflector instance
            curator: Curator instance
        """
        self.fraud_agent = fraud_agent
        self.selector = selector
        self.reflector = reflector
        self.curator = curator
    
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
                            agent_reasoning=agent_result.get('reasoning', '')
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
        source: str = "online"
    ) -> Optional[str]:
        """
        Add a new bullet from reflection.
        
        Args:
            query: Input query
            predicted: Predicted decision
            correct: Correct decision
            node: Agent node
            agent_reasoning: Agent's reasoning
            playbook: BulletPlaybook to update
            source: 'offline' or 'online'
        
        Returns:
            Bullet ID if added, None if duplicate
        """
        reflection = await self.reflector.reflect(
            query=query,
            predicted=predicted,
            correct=correct,
            node=node,
            agent_reasoning=agent_reasoning
        )
        
        if not reflection or not reflection.get('new_bullet'):
            return None
        
        return self.curator.merge_bullet(
            content=reflection['new_bullet'],
            node=node,
            playbook=playbook,
            source=source
        )

