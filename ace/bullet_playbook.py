"""
BulletPlaybook - Manages bullets with per-node organization.

This class handles storage, retrieval, and performance tracking of bullets.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Bullet:
    """Single bullet in playbook."""
    id: str
    content: str
    node: str  # Which agent node uses this bullet
    
    # Performance tracking
    helpful_count: int = 0
    harmful_count: int = 0
    times_selected: int = 0
    
    # Semantic search
    embedding: Optional[List[float]] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    source: str = "online"  # 'offline' or 'online'
    
    def get_success_rate(self) -> float:
        """Calculate empirical success rate."""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5  # Neutral prior
        return self.helpful_count / total
    
    def update_stats(self, is_helpful: bool):
        """Update bullet statistics."""
        if is_helpful:
            self.helpful_count += 1
        else:
            self.harmful_count += 1
        
        self.times_selected += 1
        self.last_used = datetime.now().isoformat()


class BulletPlaybook:
    """
    Manages bullets with per-node organization.
    
    This is a simplified in-memory version. In production, this would
    interact with the database models.
    """
    
    def __init__(self):
        self.bullets: List[Bullet] = []
        self._bullet_index: Dict[str, Bullet] = {}
        self._node_index: Dict[str, List[Bullet]] = {}
    
    def add_bullet(
        self,
        content: str,
        node: str,
        bullet_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        source: str = "online"
    ) -> str:
        """Add bullet to specific node's playbook."""
        if bullet_id is None:
            import uuid
            bullet_id = f"{node}_{str(uuid.uuid4())[:8]}"
        
        bullet = Bullet(
            id=bullet_id,
            content=content,
            node=node,
            embedding=embedding,
            source=source
        )
        
        self.bullets.append(bullet)
        self._bullet_index[bullet_id] = bullet
        
        # Add to node index
        if node not in self._node_index:
            self._node_index[node] = []
        self._node_index[node].append(bullet)
        
        logger.info(f"Added bullet [{bullet_id}] to {node} (source: {source}): {content[:50]}...")
        return bullet_id
    
    def get_bullets_for_node(self, node: str, source: Optional[str] = None) -> List[Bullet]:
        """
        Get bullets for specific node, optionally filtered by source.
        
        Args:
            node: Agent node name
            source: 'offline', 'online', or None for all
        
        Returns:
            List of bullets
        """
        bullets = self._node_index.get(node, [])
        
        if source:
            bullets = [b for b in bullets if b.source == source]
        
        return bullets
    
    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        """Get bullet by ID."""
        return self._bullet_index.get(bullet_id)
    
    def get_all_bullets(self) -> List[Bullet]:
        """Get all bullets across all nodes."""
        return self.bullets
    
    def update_bullet_stats(self, bullet_id: str, is_helpful: bool):
        """Update bullet performance."""
        bullet = self.get_bullet(bullet_id)
        if bullet:
            bullet.update_stats(is_helpful)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get playbook statistics."""
        if not self.bullets:
            return {
                "total_bullets": 0,
                "bullets_per_node": {},
                "avg_success_rate": 0.0
            }
        
        bullets_per_node = {
            node: len(bullets)
            for node, bullets in self._node_index.items()
        }
        
        success_rates = [b.get_success_rate() for b in self.bullets]
        
        return {
            "total_bullets": len(self.bullets),
            "bullets_per_node": bullets_per_node,
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "nodes": list(self._node_index.keys())
        }
    
    def clear(self):
        """Clear all bullets."""
        self.bullets = []
        self._bullet_index = {}
        self._node_index = {}

