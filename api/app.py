from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest
from services.pydantic_ai_service import stream_agent_response, agent_response
from schemas.docs import InsertDocRequest, UpdateDocRequest, RemoveDocRequest
from services.lightrag_service import insert_document, update_document, remove_document
import asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

@app.post("/docs/insert")
async def docs_insert(req: InsertDocRequest):
    doc_id = await insert_document(req.content)
    return {"doc_id": doc_id}

@app.post("/docs/update")
async def docs_update(req: UpdateDocRequest):
    result = await update_document(req.doc_id, req.content)
    return {"result": result}

@app.post("/docs/remove")
async def docs_remove(req: RemoveDocRequest):
    result = await remove_document(req.doc_id)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


