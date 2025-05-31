import os
import asyncio
from lightrag.lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

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
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8001"),
        ),
    )(texts)

WORKING_DIR = "./pydantic-docs"

async def get_lightrag():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=custom_llm_model_func
    )
    await rag.initialize_storages()
    return rag

async def get_lightrag_for_insertion():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=custom_llm_model_func
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
