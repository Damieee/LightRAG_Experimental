# ===================== Imports =====================
import os
import sys
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass

# --- External dependencies ---
import dotenv
from openai import AsyncOpenAI

# --- lightrag dependencies ---
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# ===================== Schemas =====================

class MessagePart(BaseModel):
    part_kind: str
    content: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "part_kind": "text",
                "content": "Hello, how can I help you today?"
            }
        }
    }

class Message(BaseModel):
    parts: List[MessagePart]

    model_config = {
        "json_schema_extra": {
            "example": {
                "parts": [
                    {
                        "part_kind": "text",
                        "content": "Hello, how can I help you today?"
                    },
                    {
                        "part_kind": "code",
                        "content": "print('Hello, World!')"
                    }
                ]
            }
        }
    }

class ChatRequest(BaseModel):
    user_input: str
    message_history: Optional[List[Any]] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_input": "How do I print hello world in Python?",
                "message_history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ]
            }
        }
    }

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Processing Error",
                "details": "Failed to process the chat request"
            }
        }
    }

class InsertDocRequest(BaseModel):
    content: str = Field(
        example="This is a detailed document about machine learning. Machine learning (ML) is a subset of artificial intelligence that focuses on developing systems that can learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data. Common ML techniques include supervised learning, unsupervised learning, and reinforcement learning. Applications range from image recognition to natural language processing."
    )

class UpdateDocRequest(BaseModel):
    doc_id: str = Field(example="doc_123abc456")
    content: str = Field(
        example="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn like humans. It encompasses various subfields including machine learning, neural networks, and deep learning. AI systems can perform tasks such as visual perception, speech recognition, decision-making, and language translation. The field continues to evolve with applications in healthcare, finance, autonomous vehicles, and more."
    )

class RemoveDocRequest(BaseModel):
    doc_id: str = Field(example="doc_123abc456")

# ===================== RAG Agent =====================

# Load environment variables from .env file
dotenv.load_dotenv()

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

# --- Agent setup ---
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

@dataclass
class RAGDeps:
    lightrag: LightRAG

agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions about Machine Learning based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the Machine Learning documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response."
)

@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="local")
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    await rag.initialize_storages()
    return rag

# ===================== Services =====================

async def get_lightrag():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    await rag.initialize_storages()
    return rag

async def get_lightrag_for_insertion():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def insert_document(content: str):
    try:
        rag = await get_lightrag_for_insertion()
        return await rag.ainsert(content)
    except Exception as e:
        raise Exception(f"Failed to insert document: {str(e)}")

async def update_document(doc_id: str, content: str):
    try:
        rag = await get_lightrag()
        return await rag.update(doc_id, content)
    except Exception as e:
        raise Exception(f"Failed to update document {doc_id}: {str(e)}")

async def remove_document(doc_id: str):
    try:
        rag = await get_lightrag()
        return await rag.remove(doc_id)
    except Exception as e:
        raise Exception(f"Failed to remove document {doc_id}: {str(e)}")

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
        return result
    except Exception as e:
        raise Exception(f"Error in agent_response: {str(e)}")

# ===================== FastAPI App =====================

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
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
