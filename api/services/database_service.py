import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
from typing_extensions import LiteralString
import asyncpg

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

@dataclass
class Database:
    pool: asyncpg.Pool

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        dsn: str = "postgresql://ditech_owner:npg_v5Xzkf0yrTjE@ep-wispy-term-a5h18slq-pooler.us-east-2.aws.neon.tech/ditech?sslmode=require"
    ) -> AsyncIterator['Database']:
        pool = await asyncpg.create_pool(dsn)
        async with pool.acquire() as con:
            await con.execute(
                '''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    message_list TEXT
                );
                '''
            )
        db = cls(pool)
        try:
            yield db
        finally:
            await pool.close()

    async def close(self):
        await self.pool.close()

    async def add_messages(self, messages: bytes):
        async with self.pool.acquire() as con:
            await con.execute(
                'INSERT INTO messages (message_list) VALUES ($1);',
                messages.decode() if isinstance(messages, bytes) else messages
            )

    async def get_messages(self) -> list[ModelMessage]:
        async with self.pool.acquire() as con:
            rows = await con.fetch('SELECT message_list FROM messages ORDER BY id')
            messages: list[ModelMessage] = []
            for row in rows:
                messages.extend(ModelMessagesTypeAdapter.validate_json(row['message_list']))
            return messages
