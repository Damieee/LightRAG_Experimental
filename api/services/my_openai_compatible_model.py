import os
import httpx
from typing import List, AsyncIterator, Optional
from datetime import datetime
from pydantic_ai.models import (
    Model, ModelMessage, ModelSettings, ModelRequestParameters,
    ModelResponse, StreamedResponse, check_allow_model_requests, Usage
)
from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

class MyOpenAICompatibleModel(Model):
    @property
    def model_name(self) -> str:
        return os.getenv("LLM_MODEL", "your-model-name")

    @property
    def system(self) -> str:
        return "openai-compatible"

    @property
    def base_url(self) -> Optional[str]:
        return os.getenv("LLM_BINDING_HOST_PYDANTIC", "http://localhost:8000")

    def convert_message(self, m: ModelMessage):
        # Handle ModelRequest type
        if not hasattr(m, 'parts'):
            return {"role": "user", "content": m.user_text_prompt}
            
        # Extract content from parts
        if len(m.parts) > 0:
            if isinstance(m.parts[0], SystemPromptPart):
                return {"role": "system", "content": m.parts[0].content}
            else:
                return {"role": "user", "content": m.parts[-1].content}
        return {"role": "user", "content": ""}

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        formatted_messages = [self.convert_message(m) for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": False,
        }
        api_key = os.getenv("LLM_BINDING_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            
        print(f"Response from OpenAI API: {data}")  # Debugging line to check response
        # Create response part
        response_content = data["choices"][0]["message"]["content"]
        response_part = TextPart(content=response_content)
        
        # Create usage info
        usage_data = data["usage"]
        usage = Usage(
            request_tokens=usage_data["prompt_tokens"],
            response_tokens=usage_data["completion_tokens"],
            total_tokens=usage_data["total_tokens"]
        )
        
        return ModelResponse(
            parts=[response_part],
            usage=usage,
            model_name=self.model_name,
            vendor_details={
                "finish_reason": data["choices"][0]["finish_reason"]
            },
            vendor_id=data["id"]
        )

    async def request_stream(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        formatted_messages = [self.convert_message(m) for m in messages]
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,
        }
        api_key = os.getenv("LLM_BINDING_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        yield StreamedResponse.from_openai_chunk(data)

    @property
    def profile(self):
        # Optionally implement this if you want to provide a profile
        return super().profile

    @property
    def timestamp(self) -> datetime:
        return datetime.now()