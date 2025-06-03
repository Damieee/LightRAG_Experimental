"""Pydantic AI agent that leverages RAG with a local LightRAG for Pydantic documentation."""

import os
import sys
import argparse
import asyncio
from dataclasses import dataclass

import dotenv
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed, gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables from .env file
dotenv.load_dotenv()

async def custom_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "your-model-name"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        **kwargs,
    )

def custom_embedding_func(texts):
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=lambda texts: openai_embed(
            texts,
            embed_model=os.getenv("EMBEDDING_MODEL", "your-embedding-model"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST"),
        ),
    )(texts)

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Check for OpenAI API key (optional, only warn)
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set. Custom LLM may require its own key.")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
        # llm_model_func=custom_llm_model_func,
    )
    await rag.initialize_storages()
    return rag

@dataclass
class RAGDeps:
    lightrag: LightRAG

async def stream_rag_answer(question: str, stream: bool = True):
    """
    Stream the answer to a question using LightRAG.
    If streaming is not supported, yield the full answer at once.
    """
    rag = await initialize_rag()
    param = QueryParam(mode="local", history_turns=5, only_need_context=False, stream=stream)
    # Try streaming, fallback to non-streaming
    if hasattr(rag, "aquery_stream"):
        async for chunk in rag.aquery_stream(question, param=param):
            yield chunk
    else:
        # Fallback: yield the full answer at once
        result = await rag.aquery(question, param=param)
        yield result

async def run_rag_agent(question: str) -> str:
    """
    Get the full answer to a question using LightRAG (non-streaming).
    """
    rag = await initialize_rag()
    param = QueryParam(mode="local", history_turns=5, only_need_context=False)
    result = await rag.aquery(question, param=param)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run a LightRAG agent")
    parser.add_argument("--question", help="The question to answer")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    args = parser.parse_args()

    if args.stream:
        async def run_stream():
            async for chunk in stream_rag_answer(args.question, stream=True):
                print(chunk, end="", flush=True)
        asyncio.run(run_stream())
    else:
        response = asyncio.run(run_rag_agent(args.question))
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()
