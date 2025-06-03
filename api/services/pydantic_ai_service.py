import asyncio
from datetime import datetime, timezone
import json
from .rag_agent import agent, RAGDeps
from .lightrag_service import get_lightrag
from .database_service import Database
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from schemas.chat import ChatMessage
from typing import AsyncIterator


async def get_agent_deps():
    rag = await get_lightrag()
    deps = RAGDeps(lightrag=rag)
    return deps


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


async def stream_agent_response(user_input: str, message_history: list[ModelMessage], db: Database) -> AsyncIterator[bytes]:
    """Streams a newline-delimited JSON payload to the client."""
    try:
        deps = await get_agent_deps()
        
        # Send user message
        user_message = json.dumps({
            'role': 'user',
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'content': user_input
        }).encode('utf-8') + b'\n'
        yield user_message

        # Get message history from database
        messages = await db.get_messages()

        # Run agent with streaming
        async with agent.run_stream(user_input, deps=deps, message_history=messages) as result:
            if not result.stream:
                raise ValueError("Agent run did not return a stream.")
                
            async for text in result.stream(debounce_by=0.01):
                m = ModelResponse(
                    parts=[TextPart(text)],
                    timestamp=result.timestamp()
                )
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

            # Store new messages in database using the persistent connection
            await db.add_messages(result.new_messages_json())

    except Exception as e:
        raise Exception(f"Error in stream_agent_response: {str(e)}")


async def agent_response(user_input, message_history):
    try:
        deps = await get_agent_deps()
        result = await agent.run(user_input, deps=deps, message_history=message_history)
        return result
    except Exception as e:
        raise Exception(f"Error in agent_response: {str(e)}")
