import os
import asyncio
from lightrag.lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent

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
