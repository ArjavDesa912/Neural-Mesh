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

class Router:
    def __init__(self):
        self.in_flight_requests: Dict[str, asyncio.Lock] = {}
        self.request_history: deque = deque(maxlen=1000) # For deduplication

    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        embeddings = embedding_service.encode([prompt1, prompt2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return float(similarity)

    def select_optimal_provider(self, request: InferenceRequest) -> LLMProvider:
        available_providers = []
        for name, provider in providers.items():
            # Check circuit breaker status
            if not self.check_circuit_breaker(provider):
                logger.warning(f"Provider {name} is unhealthy due to circuit breaker.")
                continue
            available_providers.append((name, provider))

        if not available_providers:
            raise Exception("No healthy LLM providers available.")

        # Prioritize based on model preference if specified and available
        if request.model_preference:
            for name, provider in available_providers:
                if name == request.model_preference:
                    logger.info(f"Selected preferred provider: {name}")
                    return provider

        # Simple load balancing: choose the one with lowest latency and error rate
        # Could be extended with more sophisticated algorithms (e.g., weighted round-robin, least connections)
        # Sort by error rate (ascending), then latency (ascending), then estimated cost (ascending)
        available_providers.sort(key=lambda x: (
            x[1].status.error_rate,
            x[1].status.latency,
            x[1].estimate_cost(request.prompt, request.max_tokens) # Incorporate cost into sorting
        ))
        selected_provider_name, selected_provider = available_providers[0]
        logger.info(f"Selected optimal provider: {selected_provider_name} (Latency: {selected_provider.status.latency:.2f}s, Error Rate: {selected_provider.status.error_rate:.2f}, Estimated Cost: {selected_provider.estimate_cost(request.prompt, request.max_tokens):.4f})")
        return selected_provider

    def check_circuit_breaker(self, provider: LLMProvider) -> bool:
        # If provider is healthy, allow requests
        if provider.status.is_healthy:
            return True

        # If not healthy, check if enough time has passed to attempt a retry
        if provider.status.last_failure and \
           (datetime.now() - provider.status.last_failure) > timedelta(seconds=settings.CIRCUIT_BREAKER_TIMEOUT):
            logger.info(f"Attempting to re-enable provider {provider.__class__.__name__} after timeout.")
            provider.status.is_healthy = True  # Tentatively re-enable
            return True
        return False

    async def route_request(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        logger.info(f"Routing request {request_id} for prompt: {request.prompt[:50]}...")

        # Request Deduplication
        # Check if a similar request is already in flight or recently completed
        # This is a simplified in-memory deduplication. For production, consider Redis or a dedicated service.
        for past_request_prompt, past_response in self.request_history:
            if self.calculate_similarity(request.prompt, past_request_prompt) > settings.SIMILARITY_THRESHOLD:
                logger.info(f"Request {request_id}: Similar request found in history. Returning cached response.")
                metrics.increment_cache_hit() # Treat as a cache hit for deduplication
                latency = time.time() - start_time
                metrics.record_request_latency(latency, "deduplicated")
                return InferenceResponse(
                    response=past_response.response,
                    provider=past_response.provider,
                    model=past_response.model,
                    latency=latency,
                    tokens_used=past_response.tokens_used,
                    cost=past_response.cost,
                    cache_hit=True,
                    request_id=request_id,
                    created_at=datetime.now(),
                )

        # Use a lock for the exact prompt to prevent multiple identical requests from hitting LLM providers simultaneously
        if request.prompt not in self.in_flight_requests:
            self.in_flight_requests[request.prompt] = asyncio.Lock()

        async with self.in_flight_requests[request.prompt]:
            # Double check cache after acquiring lock, in case another request populated it
            cached_response = cache.get_similar_cached_response(request.prompt)
            if cached_response:
                metrics.increment_cache_hit()
                latency = time.time() - start_time
                metrics.record_request_latency(latency, "cache")
                logger.info(f"Request {request_id}: Cache hit after acquiring lock.")
                return InferenceResponse(
                    response=cached_response,
                    provider="cache",
                    model="cached",
                    latency=latency,
                    tokens_used=0,
                    cost=0.0,
                    cache_hit=True,
                    request_id=request_id,
                    created_at=datetime.now(),
                )

            metrics.increment_cache_miss()

            # 1. Check cache (semantic cache)
            cached_response = cache.get_similar_cached_response(request.prompt)
            if cached_response:
                metrics.increment_cache_hit()
                latency = time.time() - start_time
                metrics.record_request_latency(latency, "cache")
                logger.info(f"Request {request_id}: Cache hit for prompt.")
                return InferenceResponse(
                    response=cached_response,
                    provider="cache",
                    model="cached",
                    latency=latency,
                    tokens_used=0,
                    cost=0.0,
                    cache_hit=True,
                    request_id=request_id,
                    created_at=datetime.now(),
                )

            metrics.increment_cache_miss()

            # 2. Select an optimal provider
            provider = self.select_optimal_provider(request)
            provider_name = provider.__class__.__name__.replace("Provider", "").lower()

            provider_response = None
            try:
                # 3. Generate response from the provider
                logger.info(f"Request {request_id}: Sending request to {provider_name} for model {request.model_preference or 'default_model'}")
                provider_response = await provider.generate(request.prompt, request.model_preference or "default_model")
                provider_response.request_id = request_id
                
                # Record metrics and update provider status
                latency = time.time() - start_time
                metrics.record_request_latency(latency, provider_response.provider)
                metrics.record_cost_per_request(provider_response.cost, provider_response.provider)
                metrics.increment_throughput()
                provider.status.record_success(latency)
                logger.info(f"Request {request_id}: Successfully received response from {provider_name}.")

                # 4. Cache the new response
                prompt_embedding = embedding_service.encode(request.prompt)
                cache.cache_response(request.prompt, provider_response.response, prompt_embedding)
                self.request_history.append((request.prompt, provider_response)) # Add to deduplication history

            except Exception as e:
                logger.error(f"Request {request_id}: Error from provider {provider_name}: {e}", exc_info=True)
                metrics.record_provider_error(provider_name, str(type(e).__name__))
                provider.status.record_failure()
                raise e

            # Dummy traffic update for scaler
            scaler.traffic_history.append(1.0) # Placeholder for actual traffic data
            predicted_traffic = scaler.predict_traffic(1)
            current_load = 1.0 # Placeholder
            target_replicas = scaler.calculate_required_capacity(current_load, predicted_traffic[0])
            scaler.scale_infrastructure(target_replicas)

            return provider_response

router = Router() # Instantiate the router
route_request = router.route_request # Expose the method as a function for gateway.py compatibility
