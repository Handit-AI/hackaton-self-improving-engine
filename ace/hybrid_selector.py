"""
HybridSelector - 5-stage hybrid bullet selection algorithm.

This module implements the hybrid selector that combines:
1. Quality filtering
2. Semantic similarity
3. Thompson sampling
4. Diversity promotion
"""
from typing import List, Tuple, Dict, Optional
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
        embedding_model: str = "text-embedding-3-small"
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
    
    def select_bullets(
        self,
        query: str,
        node: str,
        playbook: 'BulletPlaybook',
        n_bullets: int = 5,
        iteration: int = 0,
        source: Optional[str] = None  # 'offline', 'online', or None for all
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
        
        # Stage 4: Hybrid Scoring
        scored_bullets = self._hybrid_scoring(
            stage3_bullets, semantic_scores, iteration
        )
        
        # Stage 5: Diversity Promotion
        selected_bullets, final_scores = self._diversity_promotion(
            scored_bullets, n_bullets
        )
        logger.debug(f"Stage 5: Selected {len(selected_bullets)} bullets")
        
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
                bullet.embedding = self._get_embedding(bullet.content)
        
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
        iteration: int
    ) -> List[Tuple['Bullet', float, Dict[str, float]]]:
        """Stage 4: Hybrid scoring with Thompson sampling."""
        scored = []
        
        # Exponential decay for exploration
        exploration_weight = max(0.1, self.weights['thompson'] * np.exp(-iteration / 100))
        
        for bullet in bullets:
            # Component scores
            quality_score = bullet.get_success_rate()
            semantic_score = semantic_scores.get(bullet.id, 0.5)
            thompson_score = self._thompson_sample(bullet)
            
            # Combined score
            combined = (
                self.weights['quality'] * quality_score +
                self.weights['semantic'] * semantic_score +
                exploration_weight * thompson_score
            )
            
            breakdown = {
                'quality': quality_score,
                'semantic': semantic_score,
                'thompson': thompson_score,
                'combined': combined
            }
            
            scored.append((bullet, combined, breakdown))
        
        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
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
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()

