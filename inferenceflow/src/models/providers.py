from abc import ABC, abstractmethod
from src.models.response import InferenceResponse
from datetime import datetime

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        pass

class OpenAIProvider(LLMProvider):
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        # Dummy implementation
        return InferenceResponse(
            response=f"OpenAI response for prompt: {prompt}",
            provider="openai",
            model=model,
            latency=0.5,
            tokens_used=50,
            cost=0.0005,
            cache_hit=False,
            request_id="dummy_request_id",
            created_at=datetime.now(),
        )

class AnthropicProvider(LLMProvider):
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        # Dummy implementation
        return InferenceResponse(
            response=f"Anthropic response for prompt: {prompt}",
            provider="anthropic",
            model=model,
            latency=0.6,
            tokens_used=60,
            cost=0.0006,
            cache_hit=False,
            request_id="dummy_request_id",
            created_at=datetime.now(),
        )

class GoogleProvider(LLMProvider):
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        # Dummy implementation
        return InferenceResponse(
            response=f"Google response for prompt: {prompt}",
            provider="google",
            model=model,
            latency=0.4,
            tokens_used=40,
            cost=0.0004,
            cache_hit=False,
            request_id="dummy_request_id",
            created_at=datetime.now(),
        )

class CohereProvider(LLMProvider):
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        # Dummy implementation
        return InferenceResponse(
            response=f"Cohere response for prompt: {prompt}",
            provider="cohere",
            model=model,
            latency=0.7,
            tokens_used=70,
            cost=0.0007,
            cache_hit=False,
            request_id="dummy_request_id",
            created_at=datetime.now(),
        )


providers = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "cohere": CohereProvider(),
}
