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
        async with agent.run_stream(user_input, deps=deps, message_history=message_history) as result:
            async for message in result.stream_text(delta=True):
                yield message
    except Exception as e:
        raise Exception(f"Error in stream_agent_response: {str(e)}")

async def agent_response(user_input, message_history):
    try:
        deps = await get_agent_deps()
        result = await agent.run(user_input, deps=deps, message_history=message_history)
        return await result.output()
    except Exception as e:
        raise Exception(f"Error in agent_response: {str(e)}")
