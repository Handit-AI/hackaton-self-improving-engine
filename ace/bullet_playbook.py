"""
BulletPlaybook - Manages bullets with per-node organization.

This class handles storage, retrieval, and performance tracking of bullets.
Supports both in-memory and database-backed storage.
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
    evaluator: Optional[str] = None  # Which evaluator/perspective
    
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
    
    Supports both in-memory and database-backed storage.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize playbook.
        
        Args:
            db_session: SQLAlchemy session for database persistence (optional)
        """
        self.db_session = db_session
        self.bullets: List[Bullet] = []
        self._bullet_index: Dict[str, Bullet] = {}
        self._node_index: Dict[str, List[Bullet]] = {}
        
        # Load from database if session provided
        if self.db_session:
            self._load_from_db()
    
    def _load_from_db(self):
        """Load bullets from database."""
        try:
            from models import Bullet as BulletModel
            
            db_bullets = self.db_session.query(BulletModel).all()
            
            for db_bullet in db_bullets:
                bullet = Bullet(
                    id=db_bullet.id,
                    content=db_bullet.content,
                    node=db_bullet.node,
                    evaluator=db_bullet.evaluator,
                    helpful_count=db_bullet.helpful_count,
                    harmful_count=db_bullet.harmful_count,
                    times_selected=db_bullet.times_selected,
                    created_at=db_bullet.created_at.isoformat() if db_bullet.created_at else datetime.now().isoformat(),
                    last_used=db_bullet.last_used.isoformat() if db_bullet.last_used else None,
                    source=db_bullet.source or "online",
                    embedding=db_bullet.content_embedding  # Load cached embedding!
                )
                
                self.bullets.append(bullet)
                self._bullet_index[bullet.id] = bullet
                
                if bullet.node not in self._node_index:
                    self._node_index[bullet.node] = []
                self._node_index[bullet.node].append(bullet)
            
            logger.info(f"Loaded {len(self.bullets)} bullets from database")
            
        except Exception as e:
            logger.error(f"Error loading bullets from database: {e}")
    
    def _save_to_db(self, bullet: Bullet):
        """Save bullet to database."""
        if not self.db_session:
            return
        
        try:
            from models import Bullet as BulletModel
            from sqlalchemy.sql import func
            
            # Check if bullet exists
            db_bullet = self.db_session.query(BulletModel).filter(
                BulletModel.id == bullet.id
            ).first()
            
            if db_bullet:
                # Update existing
                db_bullet.content = bullet.content
                db_bullet.node = bullet.node
                db_bullet.evaluator = bullet.evaluator
                db_bullet.helpful_count = bullet.helpful_count
                db_bullet.harmful_count = bullet.harmful_count
                db_bullet.times_selected = bullet.times_selected
                db_bullet.source = bullet.source
                if bullet.last_used:
                    db_bullet.last_used = datetime.fromisoformat(bullet.last_used)
                # Update embedding if it exists
                if bullet.embedding:
                    db_bullet.content_embedding = bullet.embedding
            else:
                # Create new
                db_bullet = BulletModel(
                    id=bullet.id,
                    content=bullet.content,
                    node=bullet.node,
                    evaluator=bullet.evaluator,
                    helpful_count=bullet.helpful_count,
                    harmful_count=bullet.harmful_count,
                    times_selected=bullet.times_selected,
                    source=bullet.source,
                    created_at=datetime.fromisoformat(bullet.created_at) if bullet.created_at else func.now(),
                    last_used=datetime.fromisoformat(bullet.last_used) if bullet.last_used else None,
                    content_embedding=bullet.embedding  # Save embedding!
                )
                self.db_session.add(db_bullet)
            
            # Flush to ensure bullet is available for foreign key relationships
            self.db_session.flush()
            # Commit to persist the bullet
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error saving bullet to database: {e}")
            self.db_session.rollback()
    
    def generate_embedding(self, content: str) -> Optional[List[float]]:
        """
        Generate embedding for content.
        
        This should be called explicitly when creating new bullets.
        """
        from openai import OpenAI
        import os
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def add_bullet(
        self,
        content: str,
        node: str,
        bullet_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        source: str = "online",
        evaluator: Optional[str] = None
    ) -> str:
        """
        Add bullet to specific node's playbook.
        
        Args:
            content: Bullet content
            node: Agent node name
            bullet_id: Optional bullet ID
            embedding: Optional embedding (will be generated if not provided)
            source: Source of bullet ('offline' or 'online')
            evaluator: Evaluator/perspective name (e.g., 'formatter', 'correctness')
        """
        if bullet_id is None:
            import uuid
            bullet_id = f"{node}_{str(uuid.uuid4())[:8]}"
        
        # Only generate embedding if not provided (embeddings should be pre-calculated)
        if embedding is None:
            logger.warning(f"No embedding provided for bullet {bullet_id}. Embeddings should be pre-calculated.")
            # Don't generate on-the-fly - embeddings should be pre-calculated
            embedding = None
        
        bullet = Bullet(
            id=bullet_id,
            content=content,
            node=node,
            evaluator=evaluator,
            embedding=embedding,
            source=source
        )
        
        self.bullets.append(bullet)
        self._bullet_index[bullet_id] = bullet
        
        # Add to node index
        if node not in self._node_index:
            self._node_index[node] = []
        self._node_index[node].append(bullet)
        
        # Save to database
        self._save_to_db(bullet)
        
        logger.info(f"Added bullet [{bullet_id}] to {node} (source: {source}): {content[:50]}...")
        return bullet_id
    
    def get_bullets_for_node(self, node: str, source: Optional[str] = None, evaluator: Optional[str] = None) -> List[Bullet]:
        """
        Get bullets for specific node, optionally filtered by source and evaluator.
        
        Args:
            node: Agent node name
            source: 'offline', 'online', or None for all
            evaluator: Evaluator/perspective name, or None for all
        
        Returns:
            List of bullets
        """
        bullets = self._node_index.get(node, [])
        
        if source:
            bullets = [b for b in bullets if b.source == source]
        
        if evaluator:
            bullets = [b for b in bullets if b.evaluator == evaluator]
        
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
            # Save to database (including any new embeddings)
            self._save_to_db(bullet)
    
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

