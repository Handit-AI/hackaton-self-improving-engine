"""
PatternManager - Handles input pattern extraction, classification, and bullet effectiveness tracking.

This module provides a general way to:
1. Extract and classify input patterns
2. Find similar patterns
3. Track which bullets work for which patterns
4. Select bullets based on pattern similarity
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from openai import OpenAI
import os
import numpy as np
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PatternManager:
    """
    Manages input patterns for intelligent bullet selection.
    
    This is a general solution that works across different use cases by:
    1. Extracting key features from inputs
    2. Creating pattern embeddings
    3. Finding similar patterns
    4. Tracking bullet effectiveness per pattern
    """
    
    def __init__(self, db_session: Optional[Session] = None, similarity_threshold: float = 0.85):
        """
        Initialize PatternManager.
        
        Args:
            db_session: Database session for persistence
            similarity_threshold: Minimum similarity to consider patterns as similar
        """
        self.db_session = db_session
        self.client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment
        self.similarity_threshold = similarity_threshold
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def extract_pattern_features(self, query: str) -> Dict[str, Any]:
        """
        Extract key features from a query using LLM for intelligent classification.
        
        This uses an LLM to understand the semantic characteristics of the input.
        
        Args:
            query: Input query text
        
        Returns:
            Dict with extracted features and classification
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a pattern classification expert. Analyze queries and extract key characteristics.

Return a JSON object with these fields:
{
    "primary_category": "Brief category (e.g., 'high_value_transaction', 'new_user_pattern', 'unusual_time')",
    "risk_level": "low" | "medium" | "high",
    "key_indicators": ["indicator1", "indicator2"],
    "pattern_type": "Brief description of the pattern type"
}

Be general and focus on semantic characteristics, not specific values."""
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nClassify this query and extract its key characteristics."
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Add raw features for fallback
            features = {
                "llm_classification": result,
                "query_length": len(query),
                "word_count": len(query.split()),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error in LLM pattern extraction: {e}")
            # Fallback to simple features
            return {
                "query_length": len(query),
                "word_count": len(query.split()),
                "has_amount": "$" in query,
            }
    
    def are_patterns_similar(self, query1: str, query2: str) -> Tuple[bool, float]:
        """
        Use LLM to determine if two queries are similar patterns.
        
        Args:
            query1: First query
            query2: Second query
        
        Returns:
            Tuple of (is_similar, confidence)
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a pattern similarity expert. Determine if two queries represent similar patterns.

Consider:
- Transaction characteristics
- User behavior patterns
- Risk factors
- Context similarity

Return JSON with:
{
    "similar": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}"""
                    },
                    {
                        "role": "user",
                        "content": f"""Query 1: {query1}

Query 2: {query2}

Are these similar patterns?"""
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            is_similar = result.get("similar", False)
            confidence = result.get("confidence", 0.5)
            
            return is_similar, confidence
            
        except Exception as e:
            logger.error(f"Error in LLM pattern similarity: {e}")
            # Fallback to embedding similarity
            embedding1 = self.get_embedding(query1)
            embedding2 = self.get_embedding(query2)
            similarity = self.cosine_similarity(embedding1, embedding2)
            return similarity >= self.similarity_threshold, similarity
    
    
    def classify_input_to_category(self, input_summary: str, node: str) -> Tuple[str, float]:
        """
        Classify input summary into a category and find similar pattern.
        
        Args:
            input_summary: Text summary of the input
            node: Agent node name
        
        Returns:
            Tuple of (pattern_id, confidence)
        """
        if not self.db_session:
            return None, 0.0
        
        try:
            from models import InputPattern
            
            # Get embedding for the summary
            summary_embedding = self.get_embedding(input_summary)
            
            # Get all patterns for this node (or all patterns if no node filter)
            existing_patterns = self.db_session.query(InputPattern).all()
            
            best_match = None
            best_confidence = 0.0
            
            for pattern in existing_patterns:
                # Calculate similarity
                similarity = self.cosine_similarity(summary_embedding, pattern.query_embedding)
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = pattern
            
            # If we found a good match, return it
            if best_match and best_confidence >= self.similarity_threshold:
                logger.debug(f"Found similar pattern {best_match.id} for {node} (confidence: {best_confidence:.2f})")
                return best_match.id, best_confidence
            
            # Otherwise, create new pattern
            features = self.extract_pattern_features(input_summary)
            new_pattern = InputPattern(
                query_text=input_summary,
                query_embedding=summary_embedding,
                normalized_features=features,
                frequency=1
            )
            self.db_session.add(new_pattern)
            self.db_session.commit()
            
            logger.info(f"Created new pattern {new_pattern.id} for {node}")
            return new_pattern.id, 1.0
            
        except Exception as e:
            logger.error(f"Error classifying input: {e}")
            return None, 0.0
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 1536
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_or_create_pattern(self, query: str) -> int:
        """
        Find existing similar pattern or create new one using LLM.
        
        Args:
            query: Input query
        
        Returns:
            Pattern ID
        """
        if not self.db_session:
            return None
        
        try:
            from models import InputPattern
            from sqlalchemy.sql import func
            
            # Extract features using LLM
            features = self.extract_pattern_features(query)
            
            # Get embedding
            embedding = self.get_embedding(query)
            
            # Find similar patterns using LLM
            existing_patterns = self.db_session.query(InputPattern).all()
            
            for pattern in existing_patterns:
                # Use LLM to check similarity
                is_similar, confidence = self.are_patterns_similar(query, pattern.query_text)
                
                if is_similar and confidence >= self.similarity_threshold:
                    # Update frequency
                    pattern.frequency += 1
                    pattern.last_seen_at = func.current_timestamp()
                    self.db_session.commit()
                    logger.debug(f"Found similar pattern {pattern.id} (confidence: {confidence:.2f})")
                    return pattern.id
            
            # Create new pattern
            new_pattern = InputPattern(
                query_text=query,
                query_embedding=embedding,
                normalized_features=features,
                frequency=1
            )
            self.db_session.add(new_pattern)
            self.db_session.commit()
            
            logger.info(f"Created new pattern {new_pattern.id}")
            return new_pattern.id
            
        except Exception as e:
            logger.error(f"Error in find_or_create_pattern: {e}")
            return None
    
    
    def record_bullet_effectiveness(
        self,
        pattern_id: int,
        bullet_id: str,
        node: str,
        is_helpful: bool
    ):
        """
        Record whether a bullet was helpful for a pattern.
        
        Args:
            pattern_id: Pattern ID
            bullet_id: Bullet ID
            node: Agent node name
            is_helpful: Whether the bullet was helpful
        """
        if not self.db_session:
            return
        
        try:
            from models import BulletInputEffectiveness
            from sqlalchemy.sql import func
            
            # Find or create effectiveness record
            record = self.db_session.query(BulletInputEffectiveness).filter(
                BulletInputEffectiveness.input_pattern_id == pattern_id,
                BulletInputEffectiveness.bullet_id == bullet_id
            ).first()
            
            if record:
                if is_helpful:
                    record.helpful_count += 1
                else:
                    record.harmful_count += 1
                record.times_selected += 1
                record.last_tested_at = func.current_timestamp()
            else:
                record = BulletInputEffectiveness(
                    input_pattern_id=pattern_id,
                    bullet_id=bullet_id,
                    node=node,
                    helpful_count=1 if is_helpful else 0,
                    harmful_count=0 if is_helpful else 1,
                    times_selected=1
                )
                self.db_session.add(record)
            
            # Flush to ensure any pending bullets are available
            self.db_session.flush()
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error recording bullet effectiveness: {e}")
            self.db_session.rollback()

