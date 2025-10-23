"""
HybridSelector - 5-stage hybrid bullet selection algorithm.

This module implements the hybrid selector that combines:
1. Quality filtering
2. Semantic similarity
3. Thompson sampling
4. Diversity promotion
"""
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import numpy as np
from scipy.stats import beta as beta_dist
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)


class HybridSelector:
    """
    5-Stage Hybrid Bullet Selection Algorithm
    
    Selects bullets based on:
    1. Contextual filtering (by node and problem types)
    2. Quality filtering (success rate threshold)
    3. Semantic filtering (embedding similarity)
    4. Hybrid scoring (Thompson sampling + quality + semantic)
    5. Diversity promotion (avoid redundant bullets)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_threshold: float = 0.3,
        semantic_threshold: float = 0.5,
        diversity_weight: float = 0.15,
        weights: Optional[Dict[str, float]] = None,
        embedding_model: str = "text-embedding-3-small",
        db_session = None,
        pattern_manager = None
    ):
        """
        Initialize HybridSelector.
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            quality_threshold: Minimum success rate for bullets
            semantic_threshold: Minimum similarity for selection
            diversity_weight: Weight for diversity bonus
            weights: Weights for scoring components
            embedding_model: OpenAI embedding model to use
            db_session: Database session for pattern-based effectiveness
            pattern_manager: PatternManager instance for pattern-based selection
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        
        self.quality_threshold = quality_threshold
        self.semantic_threshold = semantic_threshold
        self.diversity_weight = diversity_weight
        
        self.weights = weights or {
            'quality': 0.3,
            'semantic': 0.4,
            'thompson': 0.3
        }
        
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Pattern-based effectiveness
        self.db_session = db_session
        self.pattern_manager = pattern_manager
    
    def select_bullets(
        self,
        query: str,
        node: str,
        playbook: 'BulletPlaybook',
        n_bullets: int = 5,
        iteration: int = 0,
        source: Optional[str] = None,  # 'offline', 'online', or None for all
        pattern_id: Optional[int] = None  # Pattern ID for pattern-based selection
    ) -> Tuple[List['Bullet'], List[Dict]]:
        """
        Select bullets for a specific agent node.
        
        Args:
            query: Input query text
            node: Agent node name
            playbook: BulletPlaybook instance
            n_bullets: Number of bullets to select
            iteration: Current iteration (for exploration decay)
            source: Filter by source ('offline', 'online', or None for all)
            pattern_id: Pattern ID for pattern-based effectiveness boost
        
        Returns:
            Tuple of (selected_bullets, score_details)
        """
        # Get bullets for this node (optionally filtered by source)
        bullets = playbook.get_bullets_for_node(node, source=source)
        
        if not bullets:
            logger.warning(f"No bullets found for node: {node}")
            return [], []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Stage 1: Contextual Filter (already done by get_bullets_for_node)
        logger.debug(f"Stage 1: Found {len(bullets)} bullets for node {node}")
        
        # Stage 2: Quality Filter
        stage2_bullets = self._quality_filter(bullets, n_bullets)
        logger.debug(f"Stage 2: Quality filter: {len(stage2_bullets)} bullets")
        
        # Stage 3: Semantic Filter
        stage3_bullets, semantic_scores = self._semantic_filter(
            stage2_bullets, query_embedding, n_bullets
        )
        logger.debug(f"Stage 3: Semantic filter: {len(stage3_bullets)} bullets")
        
        # Stage 4: Hybrid Scoring (with pattern boost)
        scored_bullets = self._hybrid_scoring(
            stage3_bullets, semantic_scores, iteration, pattern_id
        )
        
        # Stage 5: Diversity Promotion
        selected_bullets, final_scores = self._diversity_promotion(
            scored_bullets, n_bullets
        )
        logger.debug(f"Stage 5: Selected {len(selected_bullets)} bullets")
        
        # Fallback: If no bullets selected by score, return last 5 bullets
        if not selected_bullets and bullets:
            logger.info(f"No bullets selected by score for node {node}. Using fallback: last {min(5, len(bullets))} bullets")
            # Sort by created_at descending (most recent first)
            def sort_key(bullet):
                if hasattr(bullet, 'created_at') and bullet.created_at:
                    try:
                        # If created_at is a string, parse it
                        if isinstance(bullet.created_at, str):
                            return datetime.fromisoformat(bullet.created_at.replace('Z', '+00:00'))
                        # If created_at is a datetime object
                        return bullet.created_at
                    except Exception:
                        return datetime.min
                return datetime.min
            
            sorted_bullets = sorted(bullets, key=sort_key, reverse=True)
            selected_bullets = sorted_bullets[:min(5, len(sorted_bullets))]  # Last 5 bullets
            final_scores = [
                {
                    'bullet_id': bullet.id,
                    'quality': bullet.get_success_rate(),
                    'semantic': 0.0,
                    'thompson': 0.0,
                    'pattern_boost': 0.0,
                    'combined': 0.0,
                    'diversity_bonus': 0.0,
                    'final_score': 0.0,
                    'fallback': True
                }
                for bullet in selected_bullets
            ]
        
        return selected_bullets, final_scores
    
    def _quality_filter(self, bullets: List['Bullet'], n_bullets: int) -> List['Bullet']:
        """Stage 2: Filter bullets by quality threshold."""
        filtered = [
            b for b in bullets 
            if b.get_success_rate() >= self.quality_threshold
        ]
        
        # Relax threshold if not enough bullets
        if len(filtered) < n_bullets:
            relaxed_threshold = self.quality_threshold * 0.8
            filtered = [
                b for b in bullets 
                if b.get_success_rate() >= relaxed_threshold
            ]
        
        return filtered if len(filtered) >= n_bullets else bullets
    
    def _semantic_filter(
        self, 
        bullets: List['Bullet'], 
        query_embedding: List[float],
        n_bullets: int
    ) -> Tuple[List['Bullet'], Dict[str, float]]:
        """Stage 3: Filter bullets by semantic similarity."""
        # Ensure bullets have embeddings
        for bullet in bullets:
            if bullet.embedding is None:
                # Generate embedding and cache it
                bullet.embedding = self._get_embedding(bullet.content)
                # Note: Embedding will be saved to DB when bullet is updated
        
        # Calculate semantic scores
        semantic_scores = {}
        for bullet in bullets:
            similarity = self._cosine_similarity(query_embedding, bullet.embedding)
            semantic_scores[bullet.id] = similarity
        
        # Filter by threshold
        filtered = [
            b for b in bullets 
            if semantic_scores[b.id] >= self.semantic_threshold
        ]
        
        # Relax threshold if not enough bullets
        if len(filtered) < n_bullets:
            relaxed_threshold = self.semantic_threshold * 0.8
            filtered = [
                b for b in bullets 
                if semantic_scores[b.id] >= relaxed_threshold
            ]
        
        return (filtered if len(filtered) >= n_bullets else bullets), semantic_scores
    
    def _hybrid_scoring(
        self, 
        bullets: List['Bullet'], 
        semantic_scores: Dict[str, float],
        iteration: int,
        pattern_id: Optional[int] = None
    ) -> List[Tuple['Bullet', float, Dict[str, float]]]:
        """Stage 4: Hybrid scoring with Thompson sampling and pattern-based effectiveness."""
        scored = []
        
        # Exponential decay for exploration
        exploration_weight = max(0.1, self.weights['thompson'] * np.exp(-iteration / 100))
        
        for bullet in bullets:
            # Component scores
            quality_score = bullet.get_success_rate()
            semantic_score = semantic_scores.get(bullet.id, 0.5)
            thompson_score = self._thompson_sample(bullet)
            
            # Pattern-based effectiveness boost
            pattern_boost = 0.0
            if pattern_id and self.db_session:
                pattern_boost = self._get_pattern_effectiveness(bullet.id, pattern_id)
            
            # Combined score
            combined = (
                self.weights['quality'] * quality_score +
                self.weights['semantic'] * semantic_score +
                exploration_weight * thompson_score +
                pattern_boost  # Boost for pattern-specific effectiveness
            )
            
            breakdown = {
                'quality': quality_score,
                'semantic': semantic_score,
                'thompson': thompson_score,
                'pattern_boost': pattern_boost,
                'combined': combined
            }
            
            scored.append((bullet, combined, breakdown))
        
        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _get_pattern_effectiveness(self, bullet_id: str, pattern_id: int) -> float:
        """
        Get pattern-specific effectiveness for a bullet.
        
        Returns effectiveness boost (0.0 to 0.3) based on how well this bullet
        works for this specific pattern.
        """
        if not self.db_session:
            return 0.0
        
        try:
            from models import BulletInputEffectiveness
            
            # Get effectiveness record
            record = self.db_session.query(BulletInputEffectiveness).filter(
                BulletInputEffectiveness.bullet_id == bullet_id,
                BulletInputEffectiveness.input_pattern_id == pattern_id
            ).first()
            
            if record:
                total = record.helpful_count + record.harmful_count
                if total > 0:
                    success_rate = record.helpful_count / total
                    # Boost proportional to success rate (max 0.3 boost)
                    return success_rate * 0.3
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting pattern effectiveness: {e}")
            return 0.0
    
    def _thompson_sample(self, bullet: 'Bullet') -> float:
        """Thompson sampling for exploration-exploitation balance."""
        alpha = bullet.helpful_count + 1
        beta = bullet.harmful_count + 1
        return float(beta_dist.rvs(alpha, beta))
    
    def _diversity_promotion(
        self, 
        scored_bullets: List[Tuple['Bullet', float, Dict]], 
        n_bullets: int
    ) -> Tuple[List['Bullet'], List[Dict]]:
        """Stage 5: Promote diversity to avoid redundant bullets."""
        selected = []
        final_scores = []
        
        for bullet, score, breakdown in scored_bullets:
            if len(selected) >= n_bullets:
                break
            
            # Calculate diversity bonus
            diversity_bonus = 0.0
            if selected and bullet.embedding:
                similarities = [
                    self._cosine_similarity(bullet.embedding, s.embedding)
                    for s in selected
                    if s.embedding
                ]
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    diversity_bonus = (1 - avg_similarity) * self.diversity_weight
            
            final_score = score + diversity_bonus
            
            selected.append(bullet)
            final_scores.append({
                'bullet_id': bullet.id,
                **breakdown,
                'diversity_bonus': diversity_bonus,
                'final_score': final_score
            })
        
        return selected, final_scores
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def has_relevant_bullets(
        self,
        query: str,
        bullets: List['Bullet'],
        similarity_threshold: float = 0.7
    ) -> bool:
        """
        Check if there are relevant bullets for a query.
        
        Args:
            query: Input query text
            bullets: List of bullets to check
            similarity_threshold: Minimum similarity score (default: 0.7)
        
        Returns:
            True if at least one bullet has similarity > threshold
        """
        if not bullets or len(bullets) < 3:
            return False
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Check similarity with existing bullets
        for bullet in bullets:
            if bullet.embedding:
                similarity = self._cosine_similarity(query_embedding, bullet.embedding)
                if similarity > similarity_threshold:
                    return True
        
        return False
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()

