"""
Character Engine - Main orchestrator for character behavior.

This module provides the main CharacterEngine class that coordinates
all character components including personality, memory, emotions,
knowledge retrieval, and response generation.

The CharacterEngine is the primary interface for interacting with
a character - it processes user input and generates authentic,
in-character responses.

Example:
    >>> from src.core.character import CharacterEngine
    >>> gandalf = CharacterEngine.from_character_id("gandalf")
    >>> response = await gandalf.process_message("What is the one ring?")
    >>> print(response.text)
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.logging_config import get_logger, LoggerMixin
from src.config.settings import get_settings
from src.core.character.personality_core import PersonalityCore
from src.core.character.emotional_state import EmotionalState, Emotion
from src.core.character.advanced_emotions import (
    AdvancedEmotionalState,
    create_advanced_emotional_state,
    AppraisalResult,
    AffectiveSystem,
    EmotionLabel,
)
from src.core.character.consistency_checker import ConsistencyChecker, ValidationResult

logger = get_logger(__name__)

# Flag to enable advanced emotional system
USE_ADVANCED_EMOTIONS = True


class MemorySystemProtocol(Protocol):
    """Protocol for memory system integration."""
    
    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        """Store a memory."""
        ...
    
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        ...


class RAGSystemProtocol(Protocol):
    """Protocol for RAG system integration."""
    
    async def retrieve(self, query: str, character_id: str) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge."""
        ...


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider integration."""
    
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response."""
        ...


class CharacterMessage(BaseModel):
    """
    A message in a conversation.
    
    Attributes:
        message_id: Unique identifier
        role: Who sent the message (user, character, system)
        content: Message content
        timestamp: When the message was sent
        metadata: Additional metadata
    """
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., pattern="^(user|character|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterResponse(BaseModel):
    """
    Response from a character.
    
    Attributes:
        text: The response text
        emotion: Current emotional state
        thinking: Internal reasoning (optional, for debug)
        retrieved_knowledge: Knowledge used to generate response
        consistency_check: Result of consistency validation
        metadata: Additional metadata
    """
    text: str
    emotion: str
    emotion_intensity: float = 0.5
    thinking: Optional[str] = None
    retrieved_knowledge: list[str] = Field(default_factory=list)
    consistency_check: Optional[ValidationResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """
    Context for an ongoing conversation.
    
    Attributes:
        conversation_id: Unique conversation identifier
        character_id: Character in this conversation
        user_id: User in this conversation
        messages: Message history
        started_at: When conversation started
        last_activity: Last activity timestamp
    """
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    character_id: str
    user_id: str = "anonymous"
    messages: list[CharacterMessage] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, **metadata: Any) -> CharacterMessage:
        """Add a message to the conversation."""
        message = CharacterMessage(
            role=role,
            content=content,
            metadata=metadata,
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_recent_messages(self, count: int = 10) -> list[CharacterMessage]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def to_prompt_history(self, count: int = 10) -> str:
        """Convert recent messages to prompt format."""
        recent = self.get_recent_messages(count)
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Character"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)


class CharacterEngine(LoggerMixin):
    """
    Main orchestrator for character behavior and interactions.
    
    The CharacterEngine coordinates all aspects of character behavior:
    - Personality-driven responses
    - Emotional state management
    - Knowledge retrieval from RAG
    - Memory management
    - Response consistency validation
    
    Attributes:
        character_id: Unique identifier for this character
        personality: The character's personality core
        emotional_state: Current emotional state
        consistency_checker: Validates response consistency
        
    Example:
        >>> engine = CharacterEngine.from_character_id("gandalf")
        >>> response = await engine.process_message(
        ...     "Tell me about the One Ring",
        ...     conversation_context=context
        ... )
        >>> print(response.text)
    """
    
    def __init__(
        self,
        personality: PersonalityCore,
        emotional_state: Optional[EmotionalState] = None,
        memory_system: Optional[MemorySystemProtocol] = None,
        rag_system: Optional[RAGSystemProtocol] = None,
        llm_provider: Optional[LLMProviderProtocol] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        use_advanced_emotions: bool = USE_ADVANCED_EMOTIONS,
    ):
        """
        Initialize the character engine.
        
        Args:
            personality: The character's personality core
            emotional_state: Initial emotional state (created if not provided)
            memory_system: Memory system for episodic/semantic memory
            rag_system: RAG system for knowledge retrieval
            llm_provider: LLM provider for response generation
            consistency_checker: Checker for response validation
            use_advanced_emotions: Whether to use the advanced neuroscience-based emotional system
        """
        self.personality = personality
        self.character_id = personality.character_id
        self._use_advanced_emotions = use_advanced_emotions
        
        # Initialize emotional state based on selected system
        if use_advanced_emotions:
            # Use advanced neuroscience-based emotional system
            # Derive baseline from personality traits with slight positive bias
            baseline_valence = 0.15  # Default slightly positive
            baseline_arousal = 0.1   # Default slightly engaged
            baseline_dominance = 0.0
            
            # Adjust baseline based on archetype
            archetype = personality.primary_archetype.value
            if archetype in ["hero", "warrior"]:
                baseline_valence = 0.2   # Heroes are generally positive
                baseline_dominance = 0.4
                baseline_arousal = 0.25
            elif archetype in ["sage", "mentor"]:
                baseline_valence = 0.3   # Mentors are warm and positive
                baseline_dominance = 0.3
                baseline_arousal = 0.1   # Calm
            elif archetype in ["trickster"]:
                baseline_valence = 0.4   # Tricksters are jovial
                baseline_arousal = 0.35
            elif archetype in ["elf", "fae"]:
                baseline_valence = 0.25  # Elves are serene but positive
                baseline_arousal = -0.1
                baseline_dominance = 0.2
            elif archetype in ["dragon", "villain"]:
                baseline_dominance = 0.5
                baseline_valence = -0.1  # Slightly negative but not much
                baseline_arousal = 0.2
            
            # Emotional inertia from personality stability
            emotional_inertia = 0.3  # Default moderate stability
            
            self.advanced_emotional_state = create_advanced_emotional_state(
                character_id=self.character_id,
                baseline_valence=baseline_valence,
                baseline_arousal=baseline_arousal,
                baseline_dominance=baseline_dominance,
                emotional_inertia=emotional_inertia,
            )
            # Keep old system for backward compatibility
            self.emotional_state = emotional_state or EmotionalState(
                character_id=self.character_id,
                baseline_mood=Emotion.NEUTRAL,
            )
        else:
            # Use classic emotional system
            self.emotional_state = emotional_state or EmotionalState(
                character_id=self.character_id,
                baseline_mood=Emotion.NEUTRAL,
            )
            self.advanced_emotional_state = None
        
        # External systems (can be None for basic operation)
        self._memory_system = memory_system
        self._rag_system = rag_system
        self._llm_provider = llm_provider
        
        # Consistency checker
        self._consistency_checker = consistency_checker or ConsistencyChecker()
        
        # Active conversations
        self._conversations: dict[str, ConversationContext] = {}
        
        # Settings
        self._settings = get_settings()
        
        self.logger.info(
            "Initialized CharacterEngine",
            character_id=self.character_id,
            character_name=self.personality.name,
        )
    
    @classmethod
    def from_character_id(
        cls,
        character_id: str,
        data_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> "CharacterEngine":
        """
        Create a CharacterEngine from a character ID.
        
        Loads character data from the data directory.
        
        Args:
            character_id: The character's ID
            data_path: Path to character data (default: data/characters/)
            **kwargs: Additional arguments for CharacterEngine
            
        Returns:
            Configured CharacterEngine instance
        """
        data_path = data_path or Path("data/characters")
        personality_path = data_path / character_id / "personality.yaml"
        
        if not personality_path.exists():
            raise FileNotFoundError(
                f"Character personality not found: {personality_path}"
            )
        
        personality = PersonalityCore.from_yaml(personality_path)
        
        return cls(personality=personality, **kwargs)
    
    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> "CharacterEngine":
        """
        Create a CharacterEngine from a personality YAML file.
        
        Args:
            path: Path to the personality YAML file
            **kwargs: Additional arguments for CharacterEngine
            
        Returns:
            Configured CharacterEngine instance
        """
        personality = PersonalityCore.from_yaml(path)
        return cls(personality=personality, **kwargs)
    
    def set_memory_system(self, memory_system: MemorySystemProtocol) -> None:
        """Set the memory system."""
        self._memory_system = memory_system
    
    def set_rag_system(self, rag_system: RAGSystemProtocol) -> None:
        """Set the RAG system."""
        self._rag_system = rag_system
    
    def set_llm_provider(self, llm_provider: LLMProviderProtocol) -> None:
        """Set the LLM provider."""
        self._llm_provider = llm_provider
    
    def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous",
    ) -> ConversationContext:
        """
        Get an existing conversation or create a new one.
        
        Args:
            conversation_id: ID of existing conversation (creates new if None)
            user_id: User ID for the conversation
            
        Returns:
            ConversationContext instance
        """
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        
        context = ConversationContext(
            conversation_id=conversation_id or str(uuid4()),
            character_id=self.character_id,
            user_id=user_id,
        )
        self._conversations[context.conversation_id] = context
        
        return context
    
    async def process_message(
        self,
        message: str,
        conversation_context: Optional[ConversationContext] = None,
        user_id: str = "anonymous",
        include_thinking: bool = False,
        validate_response: bool = True,
    ) -> CharacterResponse:
        """
        Process a user message and generate a character response.
        
        This is the main method for interacting with the character.
        It orchestrates knowledge retrieval, emotion updates, and
        response generation.
        
        Args:
            message: The user's message
            conversation_context: Existing conversation (creates new if None)
            user_id: User ID if no context provided
            include_thinking: Include internal reasoning in response
            validate_response: Run consistency check on response
            
        Returns:
            CharacterResponse with the generated response
            
        Example:
            >>> response = await engine.process_message("Hello Gandalf!")
            >>> print(response.text)
            "Well met, friend! What brings you to seek the counsel of an old wizard?"
        """
        self.logger.info(
            "Processing message",
            character_id=self.character_id,
            message_length=len(message),
        )
        
        # Get or create conversation context
        context = conversation_context or self.get_or_create_conversation(
            user_id=user_id
        )
        
        # Add user message to context
        context.add_message("user", message)
        
        # Update emotional state based on message (must be done first)
        await self._analyze_and_update_emotion(message, context)
        
        # Retrieve relevant knowledge and memories in parallel (independent operations)
        retrieved_knowledge, memories = await asyncio.gather(
            self._retrieve_knowledge(message, context),
            self._retrieve_memories(message, context),
            return_exceptions=True,
        )
        
        # Handle exceptions from parallel retrieval
        if isinstance(retrieved_knowledge, Exception):
            self.logger.error("Failed to retrieve knowledge", error=str(retrieved_knowledge))
            retrieved_knowledge = []
        if isinstance(memories, Exception):
            self.logger.error("Failed to retrieve memories", error=str(memories))
            memories = []
        
        # Generate response
        response_text, thinking = await self._generate_response(
            message=message,
            context=context,
            knowledge=retrieved_knowledge,
            memories=memories,
            include_thinking=include_thinking,
        )
        
        # Validate response if requested
        consistency_result = None
        if validate_response:
            consistency_result = self._consistency_checker.validate_response(
                self.personality,
                response_text,
                {"context": context.to_prompt_history()},
            )
            
            # If critical issues, try to regenerate
            if consistency_result.has_critical_issues:
                self.logger.warning(
                    "Response failed consistency check",
                    character_id=self.character_id,
                    issues=len(consistency_result.issues),
                )
                # TODO: Implement response regeneration with constraints
        
        # Add character response to context
        context.add_message("character", response_text)
        
        # Store in memory if available
        await self._store_interaction_memory(message, response_text, context)
        
        # Build metadata with emotional information
        metadata = {
            "conversation_id": context.conversation_id,
            "message_count": len(context.messages),
        }
        
        # Add advanced emotional data if available
        if self._use_advanced_emotions and self.advanced_emotional_state:
            metadata["advanced_emotions"] = self.advanced_emotional_state.get_summary()
        
        # Determine emotion label and intensity
        if self._use_advanced_emotions and self.advanced_emotional_state:
            emotion_label = self.advanced_emotional_state.dominant_emotion
            emotion_intensity = self.advanced_emotional_state.dominant_intensity
        else:
            emotion_label = self.emotional_state.dominant_emotion.value
            emotion_intensity = self.emotional_state.dominant_intensity
        
        return CharacterResponse(
            text=response_text,
            emotion=emotion_label,
            emotion_intensity=emotion_intensity,
            thinking=thinking if include_thinking else None,
            retrieved_knowledge=[k.get("content", "")[:100] for k in retrieved_knowledge],
            consistency_check=consistency_result,
            metadata=metadata,
        )
    
    async def _analyze_and_update_emotion(
        self,
        message: str,
        context: ConversationContext,
    ) -> None:
        """
        Analyze message and update emotional state.
        
        Uses:
        - Advanced system: Cognitive appraisal based on neuroscience
        - Classic system: Simple keyword analysis
        """
        if self._use_advanced_emotions and self.advanced_emotional_state:
            # Use advanced neuroscience-based emotional system
            # First apply time-based decay
            self.advanced_emotional_state.decay()
            
            # Perform cognitive appraisal of the message
            appraisal = self.advanced_emotional_state.appraise_message(message)
            
            # Update affect based on appraisal
            self.advanced_emotional_state.update_affect(appraisal)
            
            # Log the emotional analysis
            summary = self.advanced_emotional_state.get_summary()
            self.logger.debug(
                "Advanced emotion analysis",
                emotion=summary["emotion"],
                valence=f"{summary['valence']:.2f}",
                arousal=f"{summary['arousal']:.2f}",
                dominant_system=summary.get("dominant_system"),
            )
            
            # Sync with classic system for backward compatibility
            emotion_label = summary["emotion"]
            try:
                # Map advanced emotions to classic Emotion enum
                emotion_mapping = {
                    "joy": Emotion.JOY,
                    "excitement": Emotion.JOY,
                    "enthusiasm": Emotion.JOY,
                    "contentment": Emotion.SERENITY,
                    "serenity": Emotion.SERENITY,
                    "calm": Emotion.SERENITY,
                    "peace": Emotion.SERENITY,
                    "anger": Emotion.ANGER,
                    "frustration": Emotion.ANGER,
                    "fear": Emotion.FEAR,
                    "anxiety": Emotion.FEAR,
                    "panic": Emotion.FEAR,
                    "sadness": Emotion.SADNESS,
                    "grief": Emotion.SADNESS,
                    "melancholy": Emotion.SADNESS,
                    "despair": Emotion.SADNESS,
                    "love": Emotion.LOVE,
                    "compassion": Emotion.LOVE,
                    "curiosity": Emotion.CURIOSITY,
                    "interest": Emotion.CURIOSITY,
                    "surprise": Emotion.SURPRISE,
                    "disgust": Emotion.DISGUST,
                    "contempt": Emotion.CONTEMPT,
                    "pride": Emotion.JOY,
                    "shame": Emotion.SADNESS,
                    "guilt": Emotion.SADNESS,
                    "hope": Emotion.ANTICIPATION,
                    "determination": Emotion.DETERMINATION,
                    "awe": Emotion.AWE,
                    "gratitude": Emotion.LOVE,
                    "neutral": Emotion.NEUTRAL,
                }
                classic_emotion = emotion_mapping.get(emotion_label, Emotion.NEUTRAL)
                intensity = (summary["arousal"] + 1) / 2  # Convert -1..1 to 0..1
                self.emotional_state.apply_emotion(
                    classic_emotion, intensity=intensity, trigger="advanced_appraisal"
                )
            except Exception as e:
                self.logger.warning(f"Failed to map emotion: {e}")
        else:
            # Use classic keyword-based emotion detection
            message_lower = message.lower()
            
            # Positive triggers
            if any(word in message_lower for word in ["thank", "grateful", "happy", "wonderful"]):
                self.emotional_state.apply_emotion(
                    Emotion.JOY, intensity=0.4, trigger="positive_interaction"
                )
            
            # Threat triggers
            if any(word in message_lower for word in ["danger", "enemy", "attack", "threat"]):
                self.emotional_state.apply_emotion(
                    Emotion.FEAR, intensity=0.3, trigger="threat_mentioned"
                )
                self.emotional_state.apply_emotion(
                    Emotion.DETERMINATION, intensity=0.5, trigger="threat_mentioned"
                )
            
            # Curiosity triggers
            if any(word in message_lower for word in ["what", "why", "how", "tell me"]):
                self.emotional_state.apply_emotion(
                    Emotion.CURIOSITY, intensity=0.3, trigger="question_asked"
                )
            
            # Sadness triggers
            if any(word in message_lower for word in ["death", "died", "lost", "gone"]):
                self.emotional_state.apply_emotion(
                    Emotion.SADNESS, intensity=0.4, trigger="loss_mentioned"
                )
    
    async def _retrieve_knowledge(
        self,
        message: str,
        context: ConversationContext,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge from RAG system."""
        if not self._rag_system:
            return []
        
        try:
            return await self._rag_system.retrieve(message, self.character_id)
        except Exception as e:
            self.logger.error("Failed to retrieve knowledge", error=str(e))
            return []
    
    async def _retrieve_memories(
        self,
        message: str,
        context: ConversationContext,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        if not self._memory_system:
            return []
        
        try:
            return await self._memory_system.retrieve(message, top_k=5)
        except Exception as e:
            self.logger.error("Failed to retrieve memories", error=str(e))
            return []
    
    async def _generate_response(
        self,
        message: str,
        context: ConversationContext,
        knowledge: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        include_thinking: bool = False,
    ) -> tuple[str, Optional[str]]:
        """
        Generate a response using the LLM.
        
        Returns:
            Tuple of (response_text, thinking_text)
        """
        # Build the prompt
        prompt = self._build_prompt(message, context, knowledge, memories)
        
        if not self._llm_provider:
            # Return a placeholder response if no LLM is configured
            return self._generate_placeholder_response(message), None
        
        try:
            llm_response = await self._llm_provider.generate(
                prompt,
                temperature=self._settings.openai_temperature,
                max_tokens=self._settings.openai_max_tokens,
            )
            
            # Extract content from LLMResponse object
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Parse thinking if included
            thinking = None
            if include_thinking and "<thinking>" in response_text:
                # Extract thinking section
                parts = response_text.split("<thinking>")
                if len(parts) > 1:
                    thinking_parts = parts[1].split("</thinking>")
                    thinking = thinking_parts[0].strip()
                    response_text = thinking_parts[1].strip() if len(thinking_parts) > 1 else response_text
            
            return response_text.strip(), thinking
            
        except Exception as e:
            self.logger.error("Failed to generate response", error=str(e))
            return self._generate_placeholder_response(message), None
    
    def _build_prompt(
        self,
        message: str,
        context: ConversationContext,
        knowledge: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> str:
        """Build the full prompt for LLM generation."""
        # Use list for efficient string building
        parts: list[str] = [
            "You are a living fantasy character. Stay completely in character.",
            "",
            self.personality.generate_personality_prompt(),
            "",
        ]
        
        # Add emotional state based on system in use
        if self._use_advanced_emotions and self.advanced_emotional_state:
            parts.append(self.advanced_emotional_state.get_response_modifier())
            summary = self.advanced_emotional_state.get_summary()
            # Build emotional state info efficiently
            parts.extend([
                f"\nValence: {summary['valence']:.2f} (negative←→positive)",
                f"Arousal: {summary['arousal']:.2f} (calm←→activated)",
                f"Dominance: {summary['dominance']:.2f} (submissive←→dominant)",
            ])
            if summary.get("dominant_system"):
                parts.append(f"Primary drive: {summary['dominant_system']}")
        else:
            parts.append(self.emotional_state.generate_emotion_prompt_section())
        parts.append("")
        
        # Add retrieved knowledge (limit and build efficiently)
        if knowledge:
            parts.append("\n## Relevant Knowledge")
            # Use list comprehension with filtering for better performance
            parts.extend(
                f"- {k.get('content', '')[:200]}"
                for k in knowledge[:5]
                if k.get('content')
            )
        
        # Add relevant memories (limit and build efficiently)
        if memories:
            parts.append("\n## Relevant Memories")
            # Use list comprehension with filtering
            parts.extend(
                f"- {m.get('content', '')[:150]}"
                for m in memories[:3]
                if m.get('content')
            )
        
        # Add conversation history
        history = context.to_prompt_history(count=6)
        if history:
            parts.extend(["\n## Recent Conversation", history])
        
        # Add the current message
        parts.extend([
            "\n## Current Message from User",
            message,
            "\n## Instructions",
            "Respond as this character would, staying true to their personality, knowledge, and current emotional state.",
            "Keep the response natural and in character. Do not break character or acknowledge being an AI.",
        ])
        
        # Use join for efficient final string construction
        return "\n".join(parts)
    
    def _generate_placeholder_response(self, message: str) -> str:
        """Generate a placeholder response when LLM is not available."""
        name = self.personality.display_name
        archetype = self.personality.primary_archetype.value
        
        placeholders = {
            "mentor": f"*{name} strokes their beard thoughtfully* That is a question worthy of contemplation, my friend.",
            "hero": f"*{name} stands ready* I hear your words. What would you have me do?",
            "sage": f"*{name} considers your words carefully* There is wisdom to be found in your question.",
            "trickster": f"*{name} grins mischievously* Ah, now that's an interesting thing to ask!",
            "dragon": f"*{name}'s eyes gleam* You dare address me with such questions, mortal?",
        }
        
        return placeholders.get(
            archetype,
            f"*{name} acknowledges you* Your words have been heard."
        )
    
    async def _store_interaction_memory(
        self,
        user_message: str,
        character_response: str,
        context: ConversationContext,
    ) -> None:
        """Store the interaction in memory."""
        if not self._memory_system:
            return
        
        try:
            memory_content = f"User said: {user_message}\nI responded: {character_response}"
            await self._memory_system.store(
                content=memory_content,
                metadata={
                    "type": "interaction",
                    "conversation_id": context.conversation_id,
                    "user_id": context.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "emotion": self.emotional_state.dominant_emotion.value,
                },
            )
        except Exception as e:
            self.logger.error("Failed to store memory", error=str(e))
    
    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the character.
        
        Returns:
            Dictionary with character status information
        """
        status = {
            "character_id": self.character_id,
            "name": self.personality.name,
            "display_name": self.personality.display_name,
            "archetype": self.personality.primary_archetype.value,
            "alignment": self.personality.alignment.value,
            "emotional_state": self.emotional_state.get_emotional_summary(),
            "active_conversations": len(self._conversations),
            "has_memory_system": self._memory_system is not None,
            "has_rag_system": self._rag_system is not None,
            "has_llm_provider": self._llm_provider is not None,
            "use_advanced_emotions": self._use_advanced_emotions,
        }
        
        # Add advanced emotional data if enabled
        if self._use_advanced_emotions and self.advanced_emotional_state:
            status["advanced_emotional_state"] = self.advanced_emotional_state.get_summary()
        
        return status
    
    def __str__(self) -> str:
        """String representation."""
        return f"CharacterEngine({self.personality.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"CharacterEngine(id={self.character_id!r}, "
            f"name={self.personality.name!r}, "
            f"emotion={self.emotional_state.dominant_emotion.value})"
        )
