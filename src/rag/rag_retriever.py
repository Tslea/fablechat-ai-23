"""
RAG Retriever - Retrieves relevant knowledge from vector store.

This module handles the retrieval of relevant character knowledge
from the multi-tier vector store system.

Example:
    >>> from src.rag import RAGRetriever
    >>> retriever = RAGRetriever()
    >>> results = await retriever.retrieve(
    ...     character_id="gandalf",
    ...     query="How do you feel about Bilbo?"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.logging_config import get_logger, LoggerMixin
from src.config.settings import get_settings
from src.rag.knowledge_indexer import KnowledgeTier

logger = get_logger(__name__)


class RetrievalMode(str, Enum):
    """Retrieval modes for different use cases."""
    BALANCED = "balanced"  # All tiers weighted equally
    ESSENCE_FIRST = "essence_first"  # Prioritize core identity
    KNOWLEDGE_FIRST = "knowledge_first"  # Prioritize lore
    RELATIONSHIP_FOCUS = "relationship_focus"  # Emphasize relationships
    STYLE_FOCUS = "style_focus"  # Emphasize communication style


class RetrievalResult(BaseModel):
    """
    A single retrieval result.
    
    Attributes:
        chunk_id: ID of the retrieved chunk
        character_id: Character this belongs to
        tier: Knowledge tier
        content: The retrieved content
        score: Relevance score (0.0 to 1.0)
        metadata: Additional metadata
    """
    chunk_id: str
    character_id: str
    tier: KnowledgeTier
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """
    Context built from retrieval results.
    
    Attributes:
        query: Original query
        results: List of retrieval results
        by_tier: Results organized by tier
        total_score: Sum of all scores
        retrieval_time_ms: Time taken for retrieval
    """
    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    by_tier: dict[str, list[RetrievalResult]] = Field(default_factory=dict)
    total_score: float = 0.0
    retrieval_time_ms: int = 0
    
    def get_formatted_context(self) -> str:
        """Format results as context for LLM prompt."""
        lines = []
        
        for tier in KnowledgeTier:
            tier_results = self.by_tier.get(tier.value, [])
            if tier_results:
                lines.append(f"\n### {tier.value.title()}")
                for result in tier_results:
                    lines.append(f"- {result.content}")
        
        return "\n".join(lines)


class RAGRetriever(LoggerMixin):
    """
    Retrieves relevant knowledge from vector store.
    
    Handles multi-tier retrieval with configurable weighting
    and filtering strategies.
    
    Attributes:
        collection_name: Qdrant collection name
        
    Example:
        >>> retriever = RAGRetriever()
        >>> context = await retriever.retrieve_context(
        ...     character_id="gandalf",
        ...     query="Tell me about the Ring",
        ...     mode=RetrievalMode.KNOWLEDGE_FIRST
        ... )
    """
    
    # Tier weights for different retrieval modes
    TIER_WEIGHTS: dict[RetrievalMode, dict[KnowledgeTier, float]] = {
        RetrievalMode.BALANCED: {
            KnowledgeTier.ESSENCE: 1.0,
            KnowledgeTier.KNOWLEDGE: 1.0,
            KnowledgeTier.RELATIONSHIPS: 1.0,
            KnowledgeTier.STYLE: 1.0,
            KnowledgeTier.CONTEXT: 1.0,
        },
        RetrievalMode.ESSENCE_FIRST: {
            KnowledgeTier.ESSENCE: 1.5,
            KnowledgeTier.KNOWLEDGE: 0.8,
            KnowledgeTier.RELATIONSHIPS: 0.6,
            KnowledgeTier.STYLE: 0.8,
            KnowledgeTier.CONTEXT: 0.5,
        },
        RetrievalMode.KNOWLEDGE_FIRST: {
            KnowledgeTier.ESSENCE: 0.7,
            KnowledgeTier.KNOWLEDGE: 1.5,
            KnowledgeTier.RELATIONSHIPS: 0.8,
            KnowledgeTier.STYLE: 0.6,
            KnowledgeTier.CONTEXT: 0.7,
        },
        RetrievalMode.RELATIONSHIP_FOCUS: {
            KnowledgeTier.ESSENCE: 0.7,
            KnowledgeTier.KNOWLEDGE: 0.6,
            KnowledgeTier.RELATIONSHIPS: 1.5,
            KnowledgeTier.STYLE: 0.8,
            KnowledgeTier.CONTEXT: 0.7,
        },
        RetrievalMode.STYLE_FOCUS: {
            KnowledgeTier.ESSENCE: 0.8,
            KnowledgeTier.KNOWLEDGE: 0.5,
            KnowledgeTier.RELATIONSHIPS: 0.6,
            KnowledgeTier.STYLE: 1.5,
            KnowledgeTier.CONTEXT: 0.6,
        },
    }
    
    def __init__(
        self,
        collection_name: str = "fantasy_characters",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Name of the embedding model to use
        """
        self._settings = get_settings()
        self._collection_name = collection_name
        self._embedding_model = embedding_model or self._settings.embedding_model
        
        # Qdrant client (initialized lazily)
        self._qdrant_client = None
        
        # Embedding function (initialized lazily)
        self._embed_function = None
        
        self.logger.info(
            "Initialized RAGRetriever",
            collection=collection_name,
        )
    
    async def _ensure_client(self) -> None:
        """Ensure Qdrant client is initialized."""
        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
                
                self._qdrant_client = QdrantClient(
                    host=self._settings.qdrant_host,
                    port=self._settings.qdrant_port,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to initialize Qdrant client",
                    error=str(e),
                )
                raise
    
    async def _ensure_embedder(self) -> None:
        """Ensure embedding function is initialized."""
        if self._embed_function is None:
            try:
                if self._settings.openai_api_key:
                    from openai import AsyncOpenAI
                    from functools import lru_cache
                    
                    # Create client once and reuse
                    client = AsyncOpenAI(api_key=self._settings.openai_api_key)
                    
                    # Cache to avoid redundant embedding calls for same text
                    embedding_cache: dict[str, list[float]] = {}
                    
                    async def embed(text: str) -> list[float]:
                        # Check cache first
                        if text in embedding_cache:
                            return embedding_cache[text]
                        
                        response = await client.embeddings.create(
                            model=self._embedding_model,
                            input=[text],
                        )
                        embedding = response.data[0].embedding
                        
                        # Cache the result (limit cache size to prevent memory issues)
                        if len(embedding_cache) < 1000:
                            embedding_cache[text] = embedding
                        
                        return embedding
                    
                    self._embed_function = embed
                else:
                    # Mock embedder for development
                    import hashlib
                    
                    # Cache mock embeddings too for consistency
                    mock_cache: dict[str, list[float]] = {}
                    
                    async def mock_embed(text: str) -> list[float]:
                        if text in mock_cache:
                            return mock_cache[text]
                        
                        hash_bytes = hashlib.sha256(text.encode()).digest()
                        embedding = [
                            (b - 128) / 128.0
                            for b in hash_bytes[:self._settings.embedding_dimension // 8]
                        ] * 8
                        embedding = embedding[:self._settings.embedding_dimension]
                        while len(embedding) < self._settings.embedding_dimension:
                            embedding.append(0.0)
                        
                        if len(mock_cache) < 1000:
                            mock_cache[text] = embedding
                        
                        return embedding
                    
                    self._embed_function = mock_embed
                    self.logger.warning("Using mock embedder for retrieval")
            except Exception as e:
                self.logger.error("Failed to initialize embedder", error=str(e))
                raise
    
    async def retrieve(
        self,
        character_id: str,
        query: str,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.BALANCED,
        tier_filter: Optional[list[KnowledgeTier]] = None,
        min_score: float = 0.5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant knowledge for a query.
        
        Args:
            character_id: Character to retrieve for
            query: The query text
            top_k: Maximum results to return
            mode: Retrieval mode for tier weighting
            tier_filter: Optional filter to specific tiers
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        await self._ensure_client()
        await self._ensure_embedder()
        
        self.logger.debug(
            "Retrieving knowledge",
            character_id=character_id,
            query_length=len(query),
            mode=mode.value,
        )
        
        # Embed the query
        query_embedding = await self._embed_function(query)
        
        # Build filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        filter_conditions = [
            FieldCondition(
                key="character_id",
                match=MatchValue(value=character_id),
            ),
        ]
        
        if tier_filter:
            tier_values = [t.value for t in tier_filter]
            # For multiple tiers, we need OR condition
            # Simplified: search all and filter in Python
        
        # Search
        search_result = self._qdrant_client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=filter_conditions),
            limit=top_k * 2,  # Get more, then filter/rerank
        )
        
        # Convert to RetrievalResult and apply weighting
        weights = self.TIER_WEIGHTS[mode]
        results = []
        
        for hit in search_result:
            tier = KnowledgeTier(hit.payload.get("tier", "knowledge"))
            
            # Apply tier filter
            if tier_filter and tier not in tier_filter:
                continue
            
            # Apply tier weight
            weighted_score = hit.score * weights.get(tier, 1.0)
            
            # Apply min score filter
            if weighted_score < min_score:
                continue
            
            results.append(RetrievalResult(
                chunk_id=str(hit.id),
                character_id=hit.payload.get("character_id", character_id),
                tier=tier,
                content=hit.payload.get("content", ""),
                score=min(1.0, weighted_score),  # Cap at 1.0
                metadata=hit.payload.get("metadata", {}),
            ))
        
        # Sort by weighted score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]
        
        self.logger.debug(
            "Retrieved knowledge",
            character_id=character_id,
            num_results=len(results),
        )
        
        return results
    
    async def retrieve_context(
        self,
        character_id: str,
        query: str,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.BALANCED,
        tier_filter: Optional[list[KnowledgeTier]] = None,
        min_score: float = 0.5,
    ) -> RetrievalContext:
        """
        Retrieve and organize knowledge as context.
        
        Args:
            character_id: Character to retrieve for
            query: The query text
            top_k: Maximum results to return
            mode: Retrieval mode for tier weighting
            tier_filter: Optional filter to specific tiers
            min_score: Minimum score threshold
            
        Returns:
            RetrievalContext with organized results
        """
        start_time = datetime.now()
        
        results = await self.retrieve(
            character_id=character_id,
            query=query,
            top_k=top_k,
            mode=mode,
            tier_filter=tier_filter,
            min_score=min_score,
        )
        
        # Organize by tier
        by_tier: dict[str, list[RetrievalResult]] = {}
        for result in results:
            tier_key = result.tier.value
            if tier_key not in by_tier:
                by_tier[tier_key] = []
            by_tier[tier_key].append(result)
        
        retrieval_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return RetrievalContext(
            query=query,
            results=results,
            by_tier=by_tier,
            total_score=sum(r.score for r in results),
            retrieval_time_ms=retrieval_time,
        )
    
    async def retrieve_for_conversation(
        self,
        character_id: str,
        conversation_history: list[dict[str, str]],
        current_message: str,
        top_k: int = 15,
    ) -> RetrievalContext:
        """
        Retrieve context for a conversation.
        
        Uses conversation history to improve retrieval relevance.
        
        Args:
            character_id: Character to retrieve for
            conversation_history: List of previous messages
            current_message: Current user message
            top_k: Maximum results to return
            
        Returns:
            RetrievalContext optimized for conversation
        """
        # Build enhanced query from conversation
        recent_messages = conversation_history[-5:]  # Last 5 messages
        context_parts = [msg.get("content", "") for msg in recent_messages]
        context_parts.append(current_message)
        
        enhanced_query = " ".join(context_parts)
        
        # Determine best retrieval mode based on message content
        mode = self._determine_mode(current_message)
        
        return await self.retrieve_context(
            character_id=character_id,
            query=enhanced_query,
            top_k=top_k,
            mode=mode,
        )
    
    def _determine_mode(self, message: str) -> RetrievalMode:
        """Determine best retrieval mode based on message content."""
        message_lower = message.lower()
        
        # Relationship questions
        if any(word in message_lower for word in ["friend", "enemy", "relationship", "know", "think of"]):
            return RetrievalMode.RELATIONSHIP_FOCUS
        
        # Knowledge questions
        if any(word in message_lower for word in ["history", "story", "happened", "did you", "have you"]):
            return RetrievalMode.KNOWLEDGE_FIRST
        
        # Identity questions
        if any(word in message_lower for word in ["who are you", "what are you", "your purpose", "believe"]):
            return RetrievalMode.ESSENCE_FIRST
        
        # Default to balanced
        return RetrievalMode.BALANCED
    
    async def get_character_summary(
        self,
        character_id: str,
    ) -> dict[str, list[str]]:
        """
        Get a summary of indexed knowledge for a character.
        
        Args:
            character_id: Character to summarize
            
        Returns:
            Dictionary with content by tier
        """
        await self._ensure_client()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        summary: dict[str, list[str]] = {}
        
        for tier in KnowledgeTier:
            results = self._qdrant_client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="character_id",
                            match=MatchValue(value=character_id),
                        ),
                        FieldCondition(
                            key="tier",
                            match=MatchValue(value=tier.value),
                        ),
                    ],
                ),
                limit=100,
            )
            
            summary[tier.value] = [
                point.payload.get("content", "")
                for point in results[0]
            ]
        
        return summary
