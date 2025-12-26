"""
Episodic Memory - Memories of specific events and interactions.

This module manages episodic memories, which are memories of specific
events, interactions, and experiences. These memories have temporal
context and decay over time based on recency and importance.

Example:
    >>> from src.core.memory import EpisodicMemory, EpisodeMemoryItem
    >>> memory = EpisodicMemory(character_id="gandalf")
    >>> await memory.store("Met Frodo at Bag End", importance=0.9)
    >>> memories = await memory.retrieve("Frodo", top_k=5)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.logging_config import get_logger, LoggerMixin
from src.config.settings import get_settings

logger = get_logger(__name__)


class EpisodeMemoryItem(BaseModel):
    """
    A single episodic memory.
    
    Represents a memory of a specific event or interaction with
    temporal and contextual information.
    
    Attributes:
        memory_id: Unique identifier
        character_id: Character this memory belongs to
        content: The memory content
        timestamp: When the memory was created
        importance: How important this memory is (0.0 to 1.0)
        emotion: Emotional tag associated with the memory
        participants: Who was involved
        location: Where it happened
        tags: Searchable tags
        access_count: How many times this memory was accessed
        last_accessed: When it was last retrieved
        consolidation_level: How well consolidated (0.0 to 1.0)
    """
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    character_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: Optional[str] = None
    participants: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    consolidation_level: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def calculate_relevance_score(
        self,
        current_time: Optional[datetime] = None,
        recency_weight: float = 0.3,
        importance_weight: float = 0.5,
        access_weight: float = 0.2,
    ) -> float:
        """
        Calculate relevance score for retrieval.
        
        Combines recency, importance, and access frequency into a single
        score for ranking memories during retrieval.
        
        Args:
            current_time: Current time (defaults to now)
            recency_weight: Weight for recency component
            importance_weight: Weight for importance
            access_weight: Weight for access frequency
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        current_time = current_time or datetime.now()
        
        # Recency score (exponential decay)
        time_diff = (current_time - self.timestamp).total_seconds()
        days_old = time_diff / 86400.0  # Convert to days
        recency_score = max(0.0, 1.0 - (days_old / 365.0))  # Decay over a year
        
        # Importance score (directly from importance)
        importance_score = self.importance
        
        # Access score (logarithmic with access count)
        import math
        access_score = min(1.0, math.log(self.access_count + 1) / math.log(100))
        
        # Combine scores
        total_score = (
            recency_weight * recency_score +
            importance_weight * importance_score +
            access_weight * access_score
        )
        
        # Boost by consolidation level
        total_score *= (1.0 + 0.2 * self.consolidation_level)
        
        return min(1.0, total_score)
    
    def mark_accessed(self) -> None:
        """Mark this memory as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def should_consolidate(self, threshold: int = 5) -> bool:
        """Check if this memory should be consolidated."""
        return self.access_count >= threshold and self.consolidation_level < 1.0


class EpisodicMemory(LoggerMixin):
    """
    Episodic memory system for a character.
    
    Manages storage and retrieval of specific event memories with
    time-based decay and importance weighting.
    
    Attributes:
        character_id: Character this memory system belongs to
        memories: List of all episodic memories
        max_memories: Maximum number of memories to keep
        
    Example:
        >>> memory = EpisodicMemory(character_id="gandalf")
        >>> await memory.store("Discussed the ring with Frodo", importance=0.9)
        >>> results = await memory.retrieve("ring", top_k=5)
    """
    
    def __init__(
        self,
        character_id: str,
        max_memories: Optional[int] = None,
    ):
        """
        Initialize episodic memory system.
        
        Args:
            character_id: Character this memory belongs to
            max_memories: Maximum memories to keep (from settings if None)
        """
        self.character_id = character_id
        self._settings = get_settings()
        self.max_memories = max_memories or self._settings.memory_max_episodic_items
        
        # In-memory storage (Phase 2 will use vector DB)
        self._memories: dict[str, EpisodeMemoryItem] = {}
        
        self.logger.info(
            "Initialized EpisodicMemory",
            character_id=character_id,
            max_memories=self.max_memories,
        )
    
    async def store(
        self,
        content: str,
        importance: float = 0.5,
        emotion: Optional[str] = None,
        participants: Optional[list[str]] = None,
        location: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **metadata: Any,
    ) -> str:
        """
        Store a new episodic memory.
        
        Args:
            content: Memory content
            importance: How important (0.0 to 1.0)
            emotion: Associated emotion
            participants: Who was involved
            location: Where it happened
            tags: Searchable tags
            **metadata: Additional metadata
            
        Returns:
            memory_id: ID of the stored memory
        """
        memory = EpisodeMemoryItem(
            character_id=self.character_id,
            content=content,
            importance=importance,
            emotion=emotion,
            participants=participants or [],
            location=location,
            tags=tags or [],
            metadata=metadata,
        )
        
        self._memories[memory.memory_id] = memory
        
        # Prune if necessary
        if len(self._memories) > self.max_memories:
            await self._prune_memories()
        
        self.logger.debug(
            "Stored episodic memory",
            character_id=self.character_id,
            memory_id=memory.memory_id,
            importance=importance,
        )
        
        return memory.memory_id
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        time_window: Optional[timedelta] = None,
        emotion_filter: Optional[str] = None,
    ) -> list[EpisodeMemoryItem]:
        """
        Retrieve relevant episodic memories.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_importance: Minimum importance threshold
            time_window: Only return memories within this timeframe
            emotion_filter: Filter by emotion
            
        Returns:
            List of relevant memories sorted by relevance
        """
        current_time = datetime.now()
        query_lower = query.lower()
        
        # Pre-filter by importance and time window first (before text matching)
        # This is more efficient than checking text match on every item
        cutoff_time = current_time - time_window if time_window else None
        
        # Filter and score memories in a single pass
        candidates: list[tuple[float, EpisodeMemoryItem]] = []
        
        for memory in self._memories.values():
            # Apply quick filters first (avoid expensive operations)
            if memory.importance < min_importance:
                continue
            
            if cutoff_time and memory.timestamp < cutoff_time:
                continue
            
            if emotion_filter and memory.emotion != emotion_filter:
                continue
            
            # Simple text matching (Phase 2 will use embeddings)
            if self._matches_query(memory, query_lower):
                relevance = memory.calculate_relevance_score(current_time)
                candidates.append((relevance, memory))
                
                # Early termination if we have enough high-relevance results
                # (only if we have significantly more than needed)
                if len(candidates) > top_k * 3:
                    # Sort what we have so far and trim
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates = candidates[:top_k * 2]
        
        # Sort by relevance and return top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in candidates[:top_k]]
        
        # Mark as accessed (batch operation)
        current_access_time = datetime.now()
        for memory in results:
            memory.access_count += 1
            memory.last_accessed = current_access_time
        
        self.logger.debug(
            "Retrieved episodic memories",
            character_id=self.character_id,
            query_length=len(query),
            results=len(results),
        )
        
        return results
    
    def _matches_query(self, memory: EpisodeMemoryItem, query_lower: str) -> bool:
        """Check if memory matches query (simple text matching for MVP)."""
        content_lower = memory.content.lower()
        
        # Check content
        if query_lower in content_lower:
            return True
        
        # Check tags
        for tag in memory.tags:
            if query_lower in tag.lower():
                return True
        
        # Check participants
        for participant in memory.participants:
            if query_lower in participant.lower():
                return True
        
        # Check location
        if memory.location and query_lower in memory.location.lower():
            return True
        
        return False
    
    async def _prune_memories(self) -> None:
        """
        Prune least relevant memories when at capacity.
        
        Removes memories with lowest relevance scores while
        preserving important and recently accessed memories.
        """
        if len(self._memories) <= self.max_memories:
            return
        
        current_time = datetime.now()
        to_remove = len(self._memories) - self.max_memories
        
        # Use heap for more efficient top-k selection instead of full sort
        # We only need to find the lowest scoring memories to remove
        import heapq
        
        # Calculate relevance for all memories (unavoidable for correctness)
        scored_memories = [
            (mem.calculate_relevance_score(current_time), mem_id)
            for mem_id, mem in self._memories.items()
        ]
        
        # Use heapq.nsmallest to efficiently find lowest scoring items
        # This is O(n log k) instead of O(n log n) for full sort
        to_remove_items = heapq.nsmallest(to_remove, scored_memories, key=lambda x: x[0])
        
        # Remove lowest scoring memories
        for _, mem_id in to_remove_items:
            del self._memories[mem_id]
        
        self.logger.info(
            "Pruned episodic memories",
            character_id=self.character_id,
            removed=to_remove,
            remaining=len(self._memories),
        )
    
    async def consolidate(self, memory_id: str) -> bool:
        """
        Consolidate a memory (increase its consolidation level).
        
        Consolidated memories are more resistant to pruning.
        
        Args:
            memory_id: ID of memory to consolidate
            
        Returns:
            True if consolidated, False if not found
        """
        if memory_id not in self._memories:
            return False
        
        memory = self._memories[memory_id]
        memory.consolidation_level = min(1.0, memory.consolidation_level + 0.2)
        
        self.logger.debug(
            "Consolidated memory",
            memory_id=memory_id,
            new_level=memory.consolidation_level,
        )
        
        return True
    
    def get_by_id(self, memory_id: str) -> Optional[EpisodeMemoryItem]:
        """Get a specific memory by ID."""
        return self._memories.get(memory_id)
    
    def get_all(self) -> list[EpisodeMemoryItem]:
        """Get all memories."""
        return list(self._memories.values())
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self._memories:
            return {
                "total": 0,
                "character_id": self.character_id,
            }
        
        memories = list(self._memories.values())
        return {
            "total": len(memories),
            "character_id": self.character_id,
            "avg_importance": sum(m.importance for m in memories) / len(memories),
            "oldest": min(m.timestamp for m in memories),
            "newest": max(m.timestamp for m in memories),
            "most_accessed": max(m.access_count for m in memories),
        }
    
    async def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self.logger.info("Cleared all episodic memories", character_id=self.character_id)
    
    def __len__(self) -> int:
        """Number of memories."""
        return len(self._memories)
