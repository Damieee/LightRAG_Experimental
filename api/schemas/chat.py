from pydantic import BaseModel
from typing import List, Optional, Any, Literal
from typing_extensions import LiteralString, ParamSpec, TypedDict


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

class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str