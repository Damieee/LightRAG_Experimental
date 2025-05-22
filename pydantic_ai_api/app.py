from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest
from services.pydantic_ai_service import stream_agent_response, agent_response
import asyncio

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    user_input = data.get("user_input")
    message_history = data.get("message_history", [])
    async def event_stream():
        async for chunk in stream_agent_response(user_input, message_history):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("user_input")
    message_history = data.get("message_history", [])
    response = await agent_response(user_input, message_history)
    return {"response": response}



