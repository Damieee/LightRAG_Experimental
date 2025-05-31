import os
import httpx
from typing import List, AsyncIterator, Optional
from datetime import datetime
from pydantic_ai.models import Model, ModelMessage, ModelSettings, ModelRequestParameters, ModelResponse, StreamedResponse, check_allow_model_requests
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
        return os.getenv("LLM_BINDING_HOST", "http://localhost:8000")

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        # Compose the payload as OpenAI expects
        payload = {
            "model": self.model_name,
            "messages": [m.dict() for m in messages],
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
        # You may need to adapt this to your API's response format
        return ModelResponse.from_openai(data)

    async def request_stream(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        payload = {
            "model": self.model_name,
            "messages": [m.dict() for m in messages],
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