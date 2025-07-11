from abc import ABC, abstractmethod
from src.models.response import InferenceResponse
from datetime import datetime
from typing import Dict, Any
import aiohttp
import asyncio
from src.utils.config import settings
from src.utils.logger import logger

class ProviderStatus:
    def __init__(self):
        self.is_healthy: bool = True
        self.last_failure: datetime | None = None
        self.failure_count: int = 0
        self.last_success: datetime | None = None
        self.latency: float = 0.0
        self.error_rate: float = 0.0

    def record_success(self, latency: float):
        self.is_healthy = True
        self.last_success = datetime.now()
        self.latency = latency
        self.failure_count = 0 # Reset failure count on success
        self.error_rate = 0.0

    def record_failure(self):
        self.is_healthy = False
        self.last_failure = datetime.now()
        self.failure_count += 1
        # Simple error rate calculation, could be more sophisticated
        self.error_rate = min(1.0, self.error_rate + 0.1) 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "last_failure": str(self.last_failure) if self.last_failure else None,
            "failure_count": self.failure_count,
            "last_success": str(self.last_success) if self.last_success else None,
            "latency": self.latency,
            "error_rate": self.error_rate,
        }

class LLMProvider(ABC):
    def __init__(self):
        self.status = ProviderStatus()

    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        pass

    def get_provider_status(self) -> ProviderStatus:
        return self.status

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Dummy cost estimation, implement real logic per provider
        return (len(prompt) / 1000 * 0.001) + (max_tokens / 1000 * 0.002)

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        start_time = datetime.now()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status() # Raise an exception for bad status codes
                    data = await response.json()
                    
                    latency = (datetime.now() - start_time).total_seconds()
                    self.status.record_success(latency)

                    # Extract relevant information from OpenAI response
                    text_response = data["choices"][0]["message"]["content"]
                    tokens_used = data["usage"]["total_tokens"]
                    cost = self.estimate_cost(prompt, tokens_used) # More accurate cost calculation needed

                    return InferenceResponse(
                        response=text_response,
                        provider="openai",
                        model=model,
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=data.get("id", ""),
                        created_at=datetime.now(),
                    )
        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API error: {e}")
            self.status.record_failure()
            raise
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI API: {e}")
            self.status.record_failure()
            raise

class AnthropicProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.api_key = settings.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
        }

        start_time = datetime.now()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    latency = (datetime.now() - start_time).total_seconds()
                    self.status.record_success(latency)

                    text_response = data["content"][0]["text"]
                    tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                    cost = self.estimate_cost(prompt, tokens_used)

                    return InferenceResponse(
                        response=text_response,
                        provider="anthropic",
                        model=model,
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=data.get("id", ""),
                        created_at=datetime.now(),
                    )
        except aiohttp.ClientError as e:
            logger.error(f"Anthropic API error: {e}")
            self.status.record_failure()
            raise
        except Exception as e:
            logger.error(f"Unexpected error with Anthropic API: {e}")
            self.status.record_failure()
            raise

class GoogleProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.api_key = settings.GOOGLE_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }
        }

        start_time = datetime.now()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}?key={self.api_key}", headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    latency = (datetime.now() - start_time).total_seconds()
                    self.status.record_success(latency)

                    text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    # Google API does not directly return token usage in this endpoint, estimate or fetch separately
                    tokens_used = len(prompt.split()) + len(text_response.split()) # Basic estimation
                    cost = self.estimate_cost(prompt, tokens_used)

                    return InferenceResponse(
                        response=text_response,
                        provider="google",
                        model=model,
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id="", # Google API response doesn't seem to have a direct request ID
                        created_at=datetime.now(),
                    )
        except aiohttp.ClientError as e:
            logger.error(f"Google API error: {e}")
            self.status.record_failure()
            raise
        except Exception as e:
            logger.error(f"Unexpected error with Google API: {e}")
            self.status.record_failure()
            raise

class CohereProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.api_key = settings.COHERE_API_KEY
        self.base_url = "https://api.cohere.ai/v1/generate"

    async def generate(self, prompt: str, model: str, **kwargs) -> InferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        start_time = datetime.now()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    latency = (datetime.now() - start_time).total_seconds()
                    self.status.record_success(latency)

                    text_response = data["generations"][0]["text"]
                    # Cohere API response doesn't directly provide token usage in this endpoint
                    tokens_used = len(prompt.split()) + len(text_response.split()) # Basic estimation
                    cost = self.estimate_cost(prompt, tokens_used)

                    return InferenceResponse(
                        response=text_response,
                        provider="cohere",
                        model=model,
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=data.get("id", ""),
                        created_at=datetime.now(),
                    )
        except aiohttp.ClientError as e:
            logger.error(f"Cohere API error: {e}")
            self.status.record_failure()
            raise
        except Exception as e:
            logger.error(f"Unexpected error with Cohere API: {e}")
            self.status.record_failure()
            raise


providers = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "cohere": CohereProvider(),
}
