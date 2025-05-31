import asyncio
from .rag_agent import agent, RAGDeps
from .lightrag_service import get_lightrag

async def get_agent_deps():
    rag = await get_lightrag()
    deps = RAGDeps(lightrag=rag)
    return deps

async def stream_agent_response(user_input, message_history):
    try:
        deps = await get_agent_deps()
        
        # Since run_stream() returns an AsyncGeneratorContextManager,
        # we must open it with "async with" to get the actual generator inside.
        async with agent.run_stream(user_input, deps=deps, message_history=message_history) as response:
            # Now `response` should be the object that supports async iteration.
            # In many implementations, `response` will itself be an async generator
            # or expose a `.stream_text(...)` method that yields each chunk.
            if hasattr(response, "stream_text"):
                # If response has a `stream_text(...)` method (as in your original code)
                async for message in response.stream_text(delta=True):
                    yield message
            else:
                # Otherwise, assume `response` is itself an async generator
                async for message in response:
                    yield message

    except Exception as e:
        raise Exception(f"Error in stream_agent_response: {str(e)}")



async def agent_response(user_input, message_history):
    try:
        deps = await get_agent_deps()
        result = await agent.run(user_input, deps=deps, message_history=message_history)
        return result
    except Exception as e:
        raise Exception(f"Error in agent_response: {str(e)}")
