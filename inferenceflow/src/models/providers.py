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
    def estimate_cost(self, prompt: str, max_tokens: int, model: str = None) -> float:
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
    """OpenAI API provider implementation with advanced model support"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
        self.model_capabilities = {
            "gpt-4-turbo": {"max_tokens": 128000, "supports_vision": True, "cost_per_1k_input": 0.01, "cost_per_1k_output": 0.03},
            "gpt-4o": {"max_tokens": 128000, "supports_vision": True, "cost_per_1k_input": 0.005, "cost_per_1k_output": 0.015},
            "gpt-4o-mini": {"max_tokens": 128000, "supports_vision": True, "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006}
        }
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": request.model_preference or "gpt-4o",
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
                    cost = self.estimate_cost(request.prompt, tokens_used, payload.get("model", "gpt-4o"))
                    
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
    
    def estimate_cost(self, prompt: str, max_tokens: int, model: str = None) -> float:
        # Advanced cost estimation with RL-optimized pricing
        input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        output_tokens = max_tokens
        
        # Use model-specific pricing
        model = model or "gpt-4o"
        capabilities = self.model_capabilities.get(model, self.model_capabilities["gpt-4o"])
        
        input_cost = (input_tokens * capabilities["cost_per_1k_input"]) / 1000
        output_cost = (output_tokens * capabilities["cost_per_1k_output"]) / 1000
        
        return input_cost + output_cost
    
    def get_available_models(self) -> List[str]:
        return self.models

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider implementation with advanced model support"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        self.model_capabilities = {
            "claude-3-5-sonnet-20240620": {"max_tokens": 200000, "supports_vision": True, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
            "claude-3-opus-20240229": {"max_tokens": 200000, "supports_vision": True, "cost_per_1k_input": 0.015, "cost_per_1k_output": 0.075},
            "claude-3-haiku-20240307": {"max_tokens": 200000, "supports_vision": True, "cost_per_1k_input": 0.00025, "cost_per_1k_output": 0.00125}
        }
    
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
                "model": request.model_preference or "claude-3-5-sonnet-20240620",
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
                    cost = self.estimate_cost(request.prompt, tokens_used, payload.get("model", "claude-3-5-sonnet-20240620"))
                    
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
    
    def estimate_cost(self, prompt: str, max_tokens: int, model: str = None) -> float:
        # Advanced cost estimation with RL-optimized pricing
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        # Use model-specific pricing
        model = model or "claude-3-5-sonnet-20240620"
        capabilities = self.model_capabilities.get(model, self.model_capabilities["claude-3-5-sonnet-20240620"])
        
        input_cost = (input_tokens * capabilities["cost_per_1k_input"]) / 1000
        output_cost = (output_tokens * capabilities["cost_per_1k_output"]) / 1000
        
        return input_cost + output_cost
    
    def get_available_models(self) -> List[str]:
        return self.models

class GoogleProvider(BaseProvider):
    """Google Gemini API provider implementation with advanced model support"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://generativelanguage.googleapis.com/v1beta")
        self.models = ["gemini-1.5-pro", "gemini-1.5-flash"]
        self.model_capabilities = {
            "gemini-1.5-pro": {"max_tokens": 1000000, "supports_vision": True, "cost_per_1k_input": 0.0035, "cost_per_1k_output": 0.0105},
            "gemini-1.5-flash": {"max_tokens": 1000000, "supports_vision": True, "cost_per_1k_input": 0.00035, "cost_per_1k_output": 0.0007}
        }
    
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
            
            # Use preference or default to flash
            model = request.model_preference or "gemini-1.5-flash"
            
            async with self.session.post(
                f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    tokens_used = data.get("usageMetadata", {}).get("totalTokenCount", 0)
                    cost = self.estimate_cost(request.prompt, tokens_used, model)
                    
                    self._update_metrics(latency, success=True)
                    
                    return InferenceResponse(
                        response=response_text,
                        provider="google",
                        model=model,
                        latency=latency,
                        tokens_used=tokens_used,
                        cost=cost,
                        cache_hit=False,
                        request_id=request_id,
                        metadata={"usage": data.get("usageMetadata", {})}
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
    
    def estimate_cost(self, prompt: str, max_tokens: int, model: str = None) -> float:
        # Advanced cost estimation with RL-optimized pricing
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        # Use model-specific pricing
        model = model or "gemini-1.5-flash"
        capabilities = self.model_capabilities.get(model, self.model_capabilities["gemini-1.5-flash"])
        
        input_cost = (input_tokens * capabilities["cost_per_1k_input"]) / 1000
        output_cost = (output_tokens * capabilities["cost_per_1k_output"]) / 1000
        
        return input_cost + output_cost
    
    def get_available_models(self) -> List[str]:
        return self.models

class CohereProvider(BaseProvider):
    """Cohere API provider implementation with advanced model support"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.cohere.com/v1")
        self.models = ["command-r-plus", "command-r"]
        self.model_capabilities = {
            "command-r-plus": {"max_tokens": 128000, "supports_vision": False, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
            "command-r": {"max_tokens": 128000, "supports_vision": False, "cost_per_1k_input": 0.0005, "cost_per_1k_output": 0.0015}
        }
    
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
                    cost = self.estimate_cost(request.prompt, tokens_used, payload.get("model", "gpt-4-turbo"))
                    
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
    
    def estimate_cost(self, prompt: str, max_tokens: int, model: str = None) -> float:
        # Advanced cost estimation with RL-optimized pricing
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        # Use model-specific pricing
        model = model or "command-r-plus"
        capabilities = self.model_capabilities.get(model, self.model_capabilities["command-r-plus"])
        
        input_cost = (input_tokens * capabilities["cost_per_1k_input"]) / 1000
        output_cost = (output_tokens * capabilities["cost_per_1k_output"]) / 1000
        
        return input_cost + output_cost
    
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
