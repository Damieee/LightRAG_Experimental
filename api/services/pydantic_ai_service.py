# services/pydantic_ai_service.py

import asyncio
from datetime import datetime, timezone
import json
from typing import AsyncIterator

from .rag_agent import stream_rag_answer, run_rag_agent


async def stream_agent_response(
    user_input: str,
    message_history: list,
    db=None
) -> AsyncIterator[bytes]:
    """
    Streams newline-delimited JSON back to the HTTP client using LightRAG.

    1. Immediately yield the user's own message.
    2. Stream model response from LightRAG.
    """

    try:
        # ─── 1. Immediately send the user’s own message as a JSON line ───
        user_message = json.dumps({
            "role": "user",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "content": user_input
        }).encode("utf-8") + b"\n"
        yield user_message

        # ─── 2. Stream model response ───
        async for chunk in stream_rag_answer(user_input, stream=True):
            # If chunk is an async generator, consume it
            if hasattr(chunk, "__aiter__"):
                content = ""
                async for part in chunk:
                    content += str(part)
            else:
                content = str(chunk)
            yield json.dumps({
                "role": "model",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "content": content
            }).encode("utf-8") + b"\n"

    except Exception as e:
        raise Exception(f"Error in stream_agent_response: {e}")


async def agent_response(user_input, message_history):
    """
    Non-streaming fallback: return the full response once completed using LightRAG.
    """
    try:
        return await run_rag_agent(user_input)
    except Exception as e:
        raise Exception(f"Error in agent_response: {e}")
