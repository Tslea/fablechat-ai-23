"""
Chat routes - Real-time chat with characters.

This module provides REST and WebSocket endpoints for chatting with characters.
"""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.db.session import get_session_dependency, get_session
from src.db.models.character import Character
from src.db.models.conversation import Conversation, Message, MessageRole
from src.core.character.character_engine import CharacterEngine, CharacterResponse as EngineResponse
from src.core.character.personality_core import PersonalityCore, PersonalityTrait, CoreValue, Archetype, Alignment
from src.core.character.emotional_state import EmotionalState, Emotion
from src.rag.rag_system import RAGSystem
from src.llm import get_chat_llm

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Initialize systems
_rag_system: Optional[RAGSystem] = None
_character_engines: dict[str, CharacterEngine] = {}
_character_cache: dict[str, tuple[Character, float]] = {}  # Cache with timestamp
_cache_ttl = 300  # 5 minutes cache TTL


def get_rag_system() -> RAGSystem:
    """Get or create RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


async def get_cached_character(character_id: str, session: AsyncSession) -> Optional[Character]:
    """Get character from cache or database."""
    import time
    current_time = time.time()
    
    # Check cache
    if character_id in _character_cache:
        character, timestamp = _character_cache[character_id]
        # Return cached if not expired
        if current_time - timestamp < _cache_ttl:
            return character
    
    # Fetch from database
    result = await session.execute(
        select(Character).where(Character.id == character_id)
    )
    character = result.scalar_one_or_none()
    
    # Cache the result
    if character:
        _character_cache[character_id] = (character, current_time)
    
    return character


async def get_character_engine(character: Character) -> CharacterEngine:
    """Get or create CharacterEngine for a character."""
    if character.id not in _character_engines:
        # Convert string traits to PersonalityTrait objects
        trait_strings = character.personality_json.get("dominant_traits", [])
        traits = [
            PersonalityTrait(name=t, intensity=0.7, description=f"{character.name}'s {t}")
            for t in trait_strings if isinstance(t, str)
        ]
        
        # Convert string values to CoreValue objects
        value_strings = character.personality_json.get("values", [])
        values = [
            CoreValue(name=v, priority=i+1, description=f"Core value: {v}")
            for i, v in enumerate(value_strings) if isinstance(v, str)
        ]
        
        # Map archetype string to enum
        try:
            archetype = Archetype(character.archetype.value.lower())
        except ValueError:
            archetype = Archetype.HERO
        
        # Map alignment string to enum
        try:
            alignment = Alignment(character.alignment.value.lower())
        except ValueError:
            alignment = Alignment.TRUE_NEUTRAL
        
        # Build PersonalityCore from database character
        personality = PersonalityCore(
            character_id=character.id,
            name=character.name,
            primary_archetype=archetype,
            alignment=alignment,
            traits=traits,
            values=values,
            fears=character.personality_json.get("fears", []),
            motivations=character.personality_json.get("motivations", []),
        )
        
        # Create emotional state
        emotional_state = EmotionalState(
            character_id=character.id,
            baseline_mood=Emotion.NEUTRAL,
        )
        
        # Create engine
        engine = CharacterEngine(
            personality=personality,
            emotional_state=emotional_state,
        )
        
        # Set LLM provider
        try:
            llm = get_chat_llm()
            engine.set_llm_provider(llm)
        except Exception as e:
            logger.warning("Could not set LLM provider", error=str(e))
        
        # Set RAG system
        try:
            rag = get_rag_system()
            engine.set_rag_system(rag)
        except Exception as e:
            logger.warning("Could not set RAG system", error=str(e))
        
        _character_engines[character.id] = engine
    
    return _character_engines[character.id]


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat."""
    character_id: str
    message: str
    conversation_id: Optional[str] = None
    user_id: str = "anonymous"
    context: Any = Field(default_factory=dict)  # Can be dict or list of messages


class ChatResponse(BaseModel):
    """Response model for chat - includes Character Engine data."""
    response: str
    character_id: str
    character_name: str
    conversation_id: str
    message_id: str
    # Character Engine integration
    emotional_state: str = "neutral"
    emotion_intensity: float = 0.5
    retrieved_knowledge: list[str] = Field(default_factory=list)
    memory_used: bool = False
    consistency_score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamToken(BaseModel):
    """Model for streaming token."""
    token: str
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info("WebSocket connected", client_id=client_id)
    
    def disconnect(self, client_id: str) -> None:
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info("WebSocket disconnected", client_id=client_id)
    
    async def send_message(self, client_id: str, message: dict[str, Any]) -> None:
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


# Routes
@router.post("/send", response_model=ChatResponse)
async def send_message(
    data: ChatRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> ChatResponse:
    """
    Send a message to a character.
    
    Args:
        data: Chat request data
        
    Returns:
        Character's response
        
    Raises:
        HTTPException: If character not found
    """
    # Get character from cache
    character = await get_cached_character(data.character_id, session)
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Get or create conversation
    if data.conversation_id:
        result = await session.execute(
            select(Conversation).where(Conversation.id == data.conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            id=str(uuid4()),
            character_id=data.character_id,
            user_id=data.user_id,
            started_at=datetime.now(),
        )
        session.add(conversation)
    
    # Add user message
    user_message = Message(
        id=str(uuid4()),
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=data.message,
        tokens=len(data.message) // 4,
    )
    session.add(user_message)
    conversation.message_count += 1
    
    # Generate response using Character Engine
    response_data = await _generate_response(
        character=character,
        message=data.message,
        context=data.context,
    )
    
    response_text = response_data["text"]
    
    # Add character response
    char_message = Message(
        id=str(uuid4()),
        conversation_id=conversation.id,
        role=MessageRole.CHARACTER,
        content=response_text,
        tokens=len(response_text) // 4,
        metadata_json={
            "emotion": response_data["emotion"],
            "emotion_intensity": response_data["emotion_intensity"],
        }
    )
    session.add(char_message)
    conversation.message_count += 1
    
    await session.commit()
    
    logger.info(
        "Chat message processed",
        character_id=data.character_id,
        conversation_id=conversation.id,
        emotion=response_data["emotion"],
    )
    
    return ChatResponse(
        response=response_text,
        character_id=character.id,
        character_name=character.name,
        conversation_id=conversation.id,
        message_id=char_message.id,
        emotional_state=response_data["emotion"],
        emotion_intensity=response_data["emotion_intensity"],
        retrieved_knowledge=response_data["retrieved_knowledge"],
        memory_used=response_data["memory_used"],
        consistency_score=response_data["consistency_score"],
        metadata={
            "tokens_used": user_message.tokens + char_message.tokens,
        },
    )


@router.websocket("/ws/{client_id}")
async def websocket_chat(
    websocket: WebSocket,
    client_id: str,
) -> None:
    """
    WebSocket endpoint for real-time chat.
    
    Message format:
    {
        "type": "message",
        "character_id": "...",
        "content": "...",
        "conversation_id": "..." (optional)
    }
    
    Response format:
    {
        "type": "response" | "token" | "error",
        "content": "...",
        "metadata": {...}
    }
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                await _handle_ws_message(
                    client_id=client_id,
                    character_id=data.get("character_id"),
                    content=data.get("content", ""),
                    conversation_id=data.get("conversation_id"),
                    user_id=data.get("user_id", client_id),
                )
            elif data.get("type") == "ping":
                await manager.send_message(client_id, {"type": "pong"})
            else:
                await manager.send_message(client_id, {
                    "type": "error",
                    "content": "Unknown message type",
                })
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), client_id=client_id)
        manager.disconnect(client_id)


async def _handle_ws_message(
    client_id: str,
    character_id: str,
    content: str,
    conversation_id: Optional[str],
    user_id: str,
) -> None:
    """Handle a WebSocket chat message."""
    async with get_session() as session:
        # Get character from cache
        character = await get_cached_character(character_id, session)
        
        if not character:
            await manager.send_message(client_id, {
                "type": "error",
                "content": "Character not found",
            })
            return
        
        # Send typing indicator
        await manager.send_message(client_id, {
            "type": "typing",
            "character_id": character_id,
            "character_name": character.name,
        })
        
        # Generate response using Character Engine
        response_data = await _generate_response(
            character=character,
            message=content,
            context={},
        )
        
        # Send response with Character Engine data (including advanced emotions)
        response_message = {
            "type": "response",
            "character_id": character_id,
            "character_name": character.name,
            "content": response_data["text"],
            "conversation_id": conversation_id,
            "emotional_state": response_data["emotion"],
            "emotion_intensity": response_data["emotion_intensity"],
            "retrieved_knowledge": response_data["retrieved_knowledge"],
            "memory_used": response_data["memory_used"],
            "consistency_score": response_data["consistency_score"],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
            },
        }
        
        # Include advanced emotional data if available
        if response_data.get("advanced_emotions"):
            response_message["advanced_emotions"] = response_data["advanced_emotions"]
        
        await manager.send_message(client_id, response_message)


async def _generate_response(
    character: Character,
    message: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a character response using Character Engine + RAG.
    
    Returns:
        Dict with response text and Character Engine data
    """
    try:
        # Get or create CharacterEngine for this character
        engine = await get_character_engine(character)
        
        # Process message through Character Engine
        response = await engine.process_message(
            message=message,
            include_thinking=False,
            validate_response=True,
        )
        
        result = {
            "text": response.text,
            "emotion": response.emotion,
            "emotion_intensity": response.emotion_intensity,
            "retrieved_knowledge": response.retrieved_knowledge,
            "consistency_score": response.consistency_check.confidence if response.consistency_check else None,
            "memory_used": len(response.retrieved_knowledge) > 0,
        }
        
        # Add advanced emotional data if available in metadata
        if response.metadata.get("advanced_emotions"):
            result["advanced_emotions"] = response.metadata["advanced_emotions"]
        
        return result
        
    except Exception as e:
        logger.error("Character Engine error, falling back to simple response", error=str(e))
        
        # Fallback to simple response
        return await _generate_fallback_response(character, message)


async def _generate_fallback_response(
    character: Character,
    message: str,
) -> dict[str, Any]:
    """
    Generate a fallback response when Character Engine fails.
    """
    # Placeholder response based on character
    personality = character.personality_json
    style = character.speaking_style_json
    
    # Build a simple response
    name = character.name
    archetype = character.archetype.value
    
    # Get some personality traits
    traits = personality.get("dominant_traits", personality.get("traits", []))
    trait_str = ", ".join(traits[:3]) if traits else "thoughtful"
    
    # Get catchphrases if available
    catchphrases = style.get("catchphrases", [])
    
    # Simple template-based response
    responses = [
        f"*{name} considers your words carefully* Indeed, that is a matter worth pondering.",
        f"*As a {archetype}, {name} responds thoughtfully* Your question speaks to matters I hold dear.",
        f"*{name}'s {trait_str} nature shows as they reply* Let me share my thoughts with you.",
    ]
    
    # Select based on message length for variety
    idx = len(message) % len(responses)
    base_response = responses[idx]
    
    # Add message acknowledgment
    if "?" in message:
        base_response += f" You ask about '{message[:50]}...' - a question that deserves careful consideration."
    else:
        base_response += f" I hear your words regarding '{message[:50]}...' and they resonate with me."
    
    # Add a catchphrase if available
    if catchphrases:
        catchphrase = catchphrases[len(message) % len(catchphrases)]
        base_response = f'"{catchphrase}" ' + base_response
    
    return {
        "text": base_response,
        "emotion": "neutral",
        "emotion_intensity": 0.5,
        "retrieved_knowledge": [],
        "consistency_score": None,
        "memory_used": False,
    }


@router.get("/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    limit: int = 50,
    session: AsyncSession = Depends(get_session_dependency),
) -> dict[str, Any]:
    """
    Get chat history for a conversation.
    
    Args:
        conversation_id: Conversation UUID
        limit: Maximum messages to return
        
    Returns:
        Conversation history
        
    Raises:
        HTTPException: If conversation not found
    """
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    
    return {
        "conversation_id": conversation_id,
        "character_id": conversation.character_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role.value,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in reversed(messages)
        ],
    }
