from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.utils.config import settings
from src.utils.logger import LoggerMixin
import uuid

class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ProviderInfo:
    name: str
    models: List[str]
    base_url: str
    api_key: str
    status: ProviderStatus = ProviderStatus.HEALTHY
    error_count: int = 0
    last_error: Optional[str] = None
    avg_latency: float = 0.0
    total_requests: int = 0

class BaseProvider(ABC, LoggerMixin):
    """Base class for all LLM providers"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.status = ProviderStatus.HEALTHY
        self.error_count = 0
        self.total_requests = 0
        self.total_latency = 0.0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.timeout_seconds)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response for the given request"""
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for the request"""
        pass
    
    def get_provider_status(self) -> ProviderInfo:
        """Get current provider status"""
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0.0
        return ProviderInfo(
            name=self.__class__.__name__,
            models=self.get_available_models(),
            base_url=self.base_url,
            api_key=self.api_key[:10] + "..." if self.api_key else None,
            status=self.status,
            error_count=self.error_count,
            avg_latency=avg_latency,
            total_requests=self.total_requests
        )
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    def _update_metrics(self, latency: float, success: bool = True):
        """Update provider metrics"""
        self.total_requests += 1
        self.total_latency += latency
        
        if not success:
            self.error_count += 1
            if self.error_count > 5:
                self.status = ProviderStatus.UNHEALTHY
            elif self.error_count > 2:
                self.status = ProviderStatus.DEGRADED
        else:
            if self.error_count > 0:
                self.error_count = max(0, self.error_count - 1)
            if self.error_count == 0:
                self.status = ProviderStatus.HEALTHY

class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": request.model_preference or "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens_used = data["usage"]["total_tokens"]
                    cost = self.estimate_cost(request.prompt, tokens_used)
                    
                    self._update_metrics(latency, success=True)
                    
                    return InferenceResponse(
                        response=response_text,
                        provider="openai",
                        model=payload["model"],
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=request_id,
                        metadata={"usage": data["usage"]}
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    self._update_metrics(time.time() - start_time, success=False)
                    raise Exception(f"OpenAI API error: {response.status}")
                    
        except Exception as e:
            latency = time.time() - start_time
            self._update_metrics(latency, success=False)
            self.logger.error(f"Error in OpenAI provider: {str(e)}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Rough cost estimation (prices may vary)
        # GPT-3.5-turbo: $0.002 per 1K tokens
        # GPT-4: $0.03 per 1K tokens input, $0.06 per 1K tokens output
        input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        output_tokens = max_tokens
        
        if "gpt-4" in (self.models[1] or "gpt-3.5-turbo"):
            return (input_tokens * 0.03 + output_tokens * 0.06) / 1000
        else:
            return ((input_tokens + output_tokens) * 0.002) / 1000
    
    def get_available_models(self) -> List[str]:
        return self.models

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.models = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": request.model_preference or "claude-3-sonnet-20240229",
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    
                    response_text = data["content"][0]["text"]
                    tokens_used = data["usage"]["output_tokens"]
                    cost = self.estimate_cost(request.prompt, tokens_used)
                    
                    self._update_metrics(latency, success=True)
                    
                    return InferenceResponse(
                        response=response_text,
                        provider="anthropic",
                        model=payload["model"],
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=request_id,
                        metadata={"usage": data["usage"]}
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"Anthropic API error: {response.status} - {error_text}")
                    self._update_metrics(time.time() - start_time, success=False)
                    raise Exception(f"Anthropic API error: {response.status}")
                    
        except Exception as e:
            latency = time.time() - start_time
            self._update_metrics(latency, success=False)
            self.logger.error(f"Error in Anthropic provider: {str(e)}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Claude-3-sonnet: $3 per 1M input tokens, $15 per 1M output tokens
        # Claude-3-haiku: $0.25 per 1M input tokens, $1.25 per 1M output tokens
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        if "haiku" in (self.models[1] or "claude-3-sonnet-20240229"):
            return (input_tokens * 0.25 + output_tokens * 1.25) / 1000000
        else:
            return (input_tokens * 3 + output_tokens * 15) / 1000000
    
    def get_available_models(self) -> List[str]:
        return self.models

class GoogleProvider(BaseProvider):
    """Google Gemini API provider implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://generativelanguage.googleapis.com/v1beta")
        self.models = ["gemini-pro"]
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            payload = {
                "contents": [{"parts": [{"text": request.prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": request.max_tokens,
                    "temperature": request.temperature
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    tokens_used = data["usageMetadata"]["totalTokenCount"]
                    cost = self.estimate_cost(request.prompt, tokens_used)
                    
                    self._update_metrics(latency, success=True)
                    
                    return InferenceResponse(
                        response=response_text,
                        provider="google",
                        model="gemini-pro",
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=request_id,
                        metadata={"usage": data["usageMetadata"]}
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"Google API error: {response.status} - {error_text}")
                    self._update_metrics(time.time() - start_time, success=False)
                    raise Exception(f"Google API error: {response.status}")
                    
        except Exception as e:
            latency = time.time() - start_time
            self._update_metrics(latency, success=False)
            self.logger.error(f"Error in Google provider: {str(e)}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Gemini Pro: $0.00025 per 1K characters input, $0.0005 per 1K characters output
        input_chars = len(prompt)
        output_chars = max_tokens * 4  # Rough estimation
        
        return (input_chars * 0.00025 + output_chars * 0.0005) / 1000
    
    def get_available_models(self) -> List[str]:
        return self.models

class CohereProvider(BaseProvider):
    """Cohere API provider implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.cohere.com/v1")
        self.models = ["command-r-plus", "command-r"]
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": request.model_preference or "command-r-plus",
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            async with self.session.post(
                f"{self.base_url}/generate",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    
                    response_text = data["generations"][0]["text"]
                    tokens_used = data["meta"]["billed_units"]["input_tokens"] + data["meta"]["billed_units"]["output_tokens"]
                    cost = self.estimate_cost(request.prompt, tokens_used)
                    
                    self._update_metrics(latency, success=True)
                    
                    return InferenceResponse(
                        response=response_text,
                        provider="cohere",
                        model=payload["model"],
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=request_id,
                        metadata={"billed_units": data["meta"]["billed_units"]}
                    )
                else:
                    error_text = await response.text()
                    self.logger.error(f"Cohere API error: {response.status} - {error_text}")
                    self._update_metrics(time.time() - start_time, success=False)
                    raise Exception(f"Cohere API error: {response.status}")
                    
        except Exception as e:
            latency = time.time() - start_time
            self._update_metrics(latency, success=False)
            self.logger.error(f"Error in Cohere provider: {str(e)}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Command-R+: $0.0003 per 1K input tokens, $0.0015 per 1K output tokens
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        return (input_tokens * 0.0003 + output_tokens * 0.0015) / 1000
    
    def get_available_models(self) -> List[str]:
        return self.models

class ProviderManager:
    """Manages multiple LLM providers with load balancing and health monitoring"""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.logger = LoggerMixin().logger
        
    def add_provider(self, name: str, provider: BaseProvider):
        """Add a provider to the manager"""
        self.providers[name] = provider
        self.logger.info(f"Added provider: {name}")
    
    async def get_healthy_providers(self) -> List[BaseProvider]:
        """Get list of healthy providers"""
        healthy_providers = []
        for provider in self.providers.values():
            if provider.status != ProviderStatus.UNHEALTHY:
                healthy_providers.append(provider)
        return healthy_providers
    
    async def get_best_provider(self, request: InferenceRequest) -> Optional[BaseProvider]:
        """Get the best provider based on health, latency, and cost"""
        healthy_providers = await self.get_healthy_providers()
        
        if not healthy_providers:
            return None
        
        # Simple selection: prefer healthy providers with lower average latency
        best_provider = min(healthy_providers, key=lambda p: p.total_latency / max(p.total_requests, 1))
        return best_provider
    
    def get_all_provider_status(self) -> Dict[str, ProviderInfo]:
        """Get status of all providers"""
        return {name: provider.get_provider_status() for name, provider in self.providers.items()}
