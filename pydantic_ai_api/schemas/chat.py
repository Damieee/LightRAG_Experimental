from pydantic import BaseModel
from typing import List, Optional, Any

class MessagePart(BaseModel):
    part_kind: str
    content: str

class Message(BaseModel):
    parts: List[MessagePart]

class ChatRequest(BaseModel):
    user_input: str
    message_history: Optional[List[Any]] = []

# You can expand Message/MessagePart as needed to match your agent's message structure.
