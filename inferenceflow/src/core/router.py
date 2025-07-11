from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from datetime import datetime, timedelta
from src.core.cache import cache
from src.models.providers import providers, LLMProvider
from src.services.metrics import metrics
from src.core.scaler import scaler
from src.utils.logger import logger
from src.utils.config import settings
from src.services.embedding import embedding_service
import random
import uuid
import time
import asyncio
from collections import deque
from typing import List, Optional, Dict, Any

import numpy as np
from dataclasses import dataclass
from enum import Enum
from src.models.providers import BaseProvider, ProviderManager, ProviderStatus
from src.utils.config import settings
from src.utils.logger import LoggerMixin
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque

class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"

@dataclass
class RoutingDecision:
    provider: BaseProvider
    confidence: float
    reasoning: str
    request_type: RequestType
    estimated_latency: float
    estimated_cost: float

class SmartRouter(LoggerMixin):
    """Intelligent request routing based on semantic similarity and load balancing"""
    
    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager
        self.embedding_model: Optional[SentenceTransformer] = None
        self.request_history: deque = deque(maxlen=1000)
        self.provider_latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        
        # Request type patterns for classification
        self.request_patterns = {
            RequestType.CHAT: [
                "hello", "how are you", "what's up", "conversation", "chat"
            ],
            RequestType.COMPLETION: [
                "continue", "complete", "finish", "next", "following"
            ],
            RequestType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview", "tl;dr"
            ],
            RequestType.TRANSLATION: [
                "translate", "translation", "language", "convert"
            ],
            RequestType.CODE_GENERATION: [
                "code", "program", "function", "class", "algorithm", "python", "javascript"
            ],
            RequestType.ANALYSIS: [
                "analyze", "analysis", "examine", "evaluate", "assess"
            ]
        }
    
    async def initialize(self):
        """Initialize the router with embedding model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Router initialized with embedding model")
        except Exception as e:
            self.logger.error(f"Failed to initialize router: {str(e)}")
            raise
    
    async def route_request(self, request: InferenceRequest) -> RoutingDecision:
        """Route request to the optimal provider"""
        
        # Classify request type
        request_type = self._classify_request_type(request.prompt)
        
        # Get available providers
        available_providers = await self.provider_manager.get_healthy_providers()
        
        if not available_providers:
            raise Exception("No healthy providers available")
        
        # Apply circuit breaker checks
        available_providers = [
            p for p in available_providers 
            if not self._is_circuit_breaker_open(p)
        ]
        
        if not available_providers:
            # Reset circuit breakers and try again
            self._reset_circuit_breakers()
            available_providers = await self.provider_manager.get_healthy_providers()
        
        # Select optimal provider
        optimal_provider = await self._select_optimal_provider(
            request, available_providers, request_type
        )
        
        # Calculate confidence and reasoning
        confidence, reasoning = self._calculate_routing_confidence(
            request, optimal_provider, request_type
        )
        
        # Estimate performance metrics
        estimated_latency = self._estimate_latency(optimal_provider)
        estimated_cost = optimal_provider.estimate_cost(
            request.prompt, request.max_tokens
        )
        
        # Record routing decision
        self._record_routing_decision(request, optimal_provider, request_type)
        
        return RoutingDecision(
            provider=optimal_provider,
            confidence=confidence,
            reasoning=reasoning,
            request_type=request_type,
            estimated_latency=estimated_latency,
            estimated_cost=estimated_cost
        )
    
    def _classify_request_type(self, prompt: str) -> RequestType:
        """Classify request type based on prompt content"""
        prompt_lower = prompt.lower()
        
        # Calculate similarity scores for each request type
        type_scores = {}
        
        for req_type, patterns in self.request_patterns.items():
            max_similarity = 0.0
            for pattern in patterns:
                if pattern in prompt_lower:
                    max_similarity = max(max_similarity, 0.8)  # High similarity for keyword match
                else:
                    # Calculate semantic similarity if embedding model is available
                    if self.embedding_model:
                        try:
                            prompt_embedding = self.embedding_model.encode(prompt)
                            pattern_embedding = self.embedding_model.encode(pattern)
                            similarity = self._cosine_similarity(prompt_embedding, pattern_embedding)
                            max_similarity = max(max_similarity, similarity)
                        except Exception:
                            continue
            
            type_scores[req_type] = max_similarity
        
        # Return the request type with highest similarity
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.3:  # Threshold for classification
                return best_type[0]
        
        # Default to chat if no clear classification
        return RequestType.CHAT
    
    async def _select_optimal_provider(
        self, 
        request: InferenceRequest, 
        providers: List[BaseProvider], 
        request_type: RequestType
    ) -> BaseProvider:
        """Select the optimal provider based on multiple factors"""
        
        if len(providers) == 1:
            return providers[0]
        
        # Calculate scores for each provider
        provider_scores = []
        
        for provider in providers:
            score = await self._calculate_provider_score(
                provider, request, request_type
            )
            provider_scores.append((provider, score))
        
        # Sort by score (higher is better) and return the best
        provider_scores.sort(key=lambda x: x[1], reverse=True)
        return provider_scores[0][0]
    
    async def _calculate_provider_score(
        self, 
        provider: BaseProvider, 
        request: InferenceRequest, 
        request_type: RequestType
    ) -> float:
        """Calculate a score for a provider based on multiple factors"""
        
        score = 0.0
        
        # Health status (40% weight)
        if provider.status == ProviderStatus.HEALTHY:
            score += 0.4
        elif provider.status == ProviderStatus.DEGRADED:
            score += 0.2
        
        # Latency performance (30% weight)
        avg_latency = provider.total_latency / max(provider.total_requests, 1)
        if avg_latency > 0:
            # Lower latency = higher score
            latency_score = max(0, 1 - (avg_latency / 10))  # Normalize to 0-1
            score += latency_score * 0.3
        
        # Cost efficiency (20% weight)
        estimated_cost = provider.estimate_cost(request.prompt, request.max_tokens)
        # Lower cost = higher score (normalize to reasonable range)
        cost_score = max(0, 1 - (estimated_cost / 0.1))  # Assume $0.1 is high cost
        score += cost_score * 0.2
        
        # Request type compatibility (10% weight)
        type_compatibility = self._get_type_compatibility(provider, request_type)
        score += type_compatibility * 0.1
        
        return score
    
    def _get_type_compatibility(self, provider: BaseProvider, request_type: RequestType) -> float:
        """Get compatibility score between provider and request type"""
        
        # Simple compatibility matrix (could be more sophisticated)
        compatibility_matrix = {
            "OpenAIProvider": {
                RequestType.CHAT: 0.9,
                RequestType.COMPLETION: 0.8,
                RequestType.SUMMARIZATION: 0.7,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.9,
                RequestType.ANALYSIS: 0.8
            },
            "AnthropicProvider": {
                RequestType.CHAT: 0.9,
                RequestType.COMPLETION: 0.7,
                RequestType.SUMMARIZATION: 0.8,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.8,
                RequestType.ANALYSIS: 0.9
            },
            "GoogleProvider": {
                RequestType.CHAT: 0.7,
                RequestType.COMPLETION: 0.6,
                RequestType.SUMMARIZATION: 0.8,
                RequestType.TRANSLATION: 0.9,
                RequestType.CODE_GENERATION: 0.7,
                RequestType.ANALYSIS: 0.8
            },
            "CohereProvider": {
                RequestType.CHAT: 0.6,
                RequestType.COMPLETION: 0.8,
                RequestType.SUMMARIZATION: 0.7,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.6,
                RequestType.ANALYSIS: 0.7
            }
        }
        
        provider_name = provider.__class__.__name__
        return compatibility_matrix.get(provider_name, {}).get(request_type, 0.5)
    
    def _calculate_routing_confidence(
        self, 
        request: InferenceRequest, 
        provider: BaseProvider, 
        request_type: RequestType
    ) -> tuple[float, str]:
        """Calculate confidence score and reasoning for routing decision"""
        
        confidence = 0.0
        reasoning_parts = []
        
        # Provider health confidence
        if provider.status == ProviderStatus.HEALTHY:
            confidence += 0.4
            reasoning_parts.append("provider is healthy")
        elif provider.status == ProviderStatus.DEGRADED:
            confidence += 0.2
            reasoning_parts.append("provider is degraded but available")
        
        # Request type confidence
        type_compatibility = self._get_type_compatibility(provider, request_type)
        confidence += type_compatibility * 0.3
        reasoning_parts.append(f"good compatibility with {request_type.value} requests")
        
        # Historical performance confidence
        if provider.total_requests > 10:
            avg_latency = provider.total_latency / provider.total_requests
            if avg_latency < 2.0:  # Good historical performance
                confidence += 0.3
                reasoning_parts.append("good historical performance")
        
        reasoning = f"Selected {provider.__class__.__name__} because: {', '.join(reasoning_parts)}"
        
        return min(confidence, 1.0), reasoning
    
    def _estimate_latency(self, provider: BaseProvider) -> float:
        """Estimate expected latency for a provider"""
        if provider.total_requests > 0:
            return provider.total_latency / provider.total_requests
        return 2.0  # Default estimate
    
    def _record_routing_decision(
        self, 
        request: InferenceRequest, 
        provider: BaseProvider, 
        request_type: RequestType
    ):
        """Record routing decision for analytics"""
        decision = {
            "timestamp": time.time(),
            "provider": provider.__class__.__name__,
            "request_type": request_type.value,
            "prompt_length": len(request.prompt),
            "priority": request.priority.value
        }
        self.request_history.append(decision)
    
    def _is_circuit_breaker_open(self, provider: BaseProvider) -> bool:
        """Check if circuit breaker is open for a provider"""
        provider_name = provider.__class__.__name__
        
        if provider_name not in self.circuit_breaker_states:
            return False
        
        state = self.circuit_breaker_states[provider_name]
        
        # Check if circuit breaker is open
        if state.get("is_open", False):
            # Check if timeout has passed
            if time.time() - state.get("opened_at", 0) > settings.circuit_breaker_timeout:
                # Try to close circuit breaker
                state["is_open"] = False
                state["failure_count"] = 0
                return False
            return True
        
        return False
    
    def _reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        for state in self.circuit_breaker_states.values():
            state["is_open"] = False
            state["failure_count"] = 0
    
    def record_provider_failure(self, provider: BaseProvider):
        """Record a provider failure for circuit breaker"""
        provider_name = provider.__class__.__name__
        
        if provider_name not in self.circuit_breaker_states:
            self.circuit_breaker_states[provider_name] = {
                "failure_count": 0,
                "is_open": False,
                "opened_at": 0
            }
        
        state = self.circuit_breaker_states[provider_name]
        state["failure_count"] += 1
        
        # Open circuit breaker if failure threshold is reached
        if state["failure_count"] >= 5:  # Configurable threshold
            state["is_open"] = True
            state["opened_at"] = time.time()
            self.logger.warning(f"Circuit breaker opened for {provider_name}")
    
    def record_provider_success(self, provider: BaseProvider):
        """Record a provider success for circuit breaker"""
        provider_name = provider.__class__.__name__
        
        if provider_name in self.circuit_breaker_states:
            state = self.circuit_breaker_states[provider_name]
            state["failure_count"] = max(0, state["failure_count"] - 1)
            
            # Close circuit breaker if it was open
            if state["is_open"]:
                state["is_open"] = False
                self.logger.info(f"Circuit breaker closed for {provider_name}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and analytics"""
        stats = {
            "total_routing_decisions": len(self.request_history),
            "circuit_breaker_states": {},
            "request_type_distribution": defaultdict(int),
            "provider_distribution": defaultdict(int)
        }
        
        # Circuit breaker states
        for provider_name, state in self.circuit_breaker_states.items():
            stats["circuit_breaker_states"][provider_name] = {
                "is_open": state.get("is_open", False),
                "failure_count": state.get("failure_count", 0)
            }
        
        # Request type and provider distribution
        for decision in self.request_history:
            stats["request_type_distribution"][decision["request_type"]] += 1
            stats["provider_distribution"][decision["provider"]] += 1
        
        return stats

# Global router instance (will be initialized in orchestrator)
smart_router: Optional[SmartRouter] = None
