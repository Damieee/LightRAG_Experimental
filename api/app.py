from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.chat import ChatRequest, ErrorResponse
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
async def chat_stream(chat_request: ChatRequest):
    try:
        async def event_stream():
            async for chunk in stream_agent_response(chat_request.user_input, chat_request.message_history):
                yield chunk
        return StreamingResponse(event_stream(), media_type="text/plain")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Streaming Error", details=str(e)).model_dump()
        )

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        response = await agent_response(chat_request.user_input, chat_request.message_history)
        return {"response": response}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Chat Error", details=str(e)).model_dump()
        )

@app.post("/docs/insert")
async def docs_insert(req: InsertDocRequest):
    try:
        doc_id = await insert_document(req.content)
        return {"doc_id": doc_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Document Insert Error", details=str(e)).model_dump()
        )

@app.post("/docs/update")
async def docs_update(req: UpdateDocRequest):
    try:
        result = await update_document(req.doc_id, req.content)
        return {"result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Document Update Error", details=str(e)).model_dump()
        )

@app.post("/docs/remove")
async def docs_remove(req: RemoveDocRequest):
    try:
        result = await remove_document(req.doc_id)
        return {"result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Document Remove Error", details=str(e)).model_dump()
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


