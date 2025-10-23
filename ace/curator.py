"""
Curator - Manages bullet quality and prevents duplicates.

This module handles deduplication and quality control for bullets.
"""
import logging
from typing import List, Optional
from difflib import SequenceMatcher
from ace.bullet_playbook import BulletPlaybook

logger = logging.getLogger(__name__)


class Curator:
    """Manages bullet quality and prevents duplicates."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize Curator.
        
        Args:
            similarity_threshold: Threshold for considering bullets as duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def merge_bullet(
        self,
        content: str,
        node: str,
        playbook: BulletPlaybook,
        bullet_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Add bullet if not duplicate.
        
        Args:
            content: Bullet content
            node: Agent node name
            playbook: BulletPlaybook instance
            bullet_id: Optional bullet ID
        
        Returns:
            Bullet ID if added, None if duplicate
        """
        # Check duplicates within this node's bullets
        node_bullets = playbook.get_bullets_for_node(node)
        
        for existing in node_bullets:
            similarity = self._text_similarity(content, existing.content)
            if similarity > self.similarity_threshold:
                logger.debug(f"Duplicate bullet detected for {node} (similarity: {similarity:.2f})")
                return None
        
        # Add new bullet
        bullet_id = playbook.add_bullet(
            content=content,
            node=node,
            bullet_id=bullet_id
        )
        
        logger.info(f"Added bullet [{bullet_id}] to {node}: {content[:50]}...")
        return bullet_id
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

