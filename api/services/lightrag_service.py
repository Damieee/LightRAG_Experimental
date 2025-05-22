import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./pydantic-docs"

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

async def insert_document(content: str):
    rag = await get_lightrag()
    await initialize_pipeline_status()
    return rag.insert(content)

async def update_document(doc_id: str, content: str):
    rag = await get_lightrag()
    await initialize_pipeline_status()
    return rag.update(doc_id, content)

async def remove_document(doc_id: str):
    rag = await get_lightrag()
    await initialize_pipeline_status()
    return rag.remove(doc_id)
