from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from datetime import datetime
from src.core.cache import cache
from src.models.providers import providers
from src.services.metrics import metrics
from src.core.scaler import scaler
import random
import uuid
import time

async def route_request(request: InferenceRequest) -> InferenceResponse:
    start_time = time.time()
    
    # 1. Check cache
    cached_response = cache.get_similar_cached_response(request.prompt)
    if cached_response:
        metrics.increment_cache_hit()
        latency = time.time() - start_time
        metrics.record_request_latency(latency, "cache")
        return InferenceResponse(
            response=cached_response,
            provider="cache",
            model="cached",
            latency=latency,
            tokens_used=0,
            cost=0.0,
            cache_hit=True,
            request_id=str(uuid.uuid4()),
            created_at=datetime.now(),
        )

    metrics.increment_cache_miss()

    # 2. If not in cache, select a provider
    if request.model_preference and request.model_preference in providers:
        provider_name = request.model_preference
    else:
        provider_name = random.choice(list(providers.keys()))
    
    provider = providers[provider_name]
    
    provider_response = None
    try:
        # 3. Generate response from the provider
        provider_response = await provider.generate(request.prompt, request.model_preference or "default_model")
        provider_response.request_id = str(uuid.uuid4())
        
        # Record metrics
        latency = time.time() - start_time
        metrics.record_request_latency(latency, provider_response.provider)
        metrics.record_cost_per_request(provider_response.cost, provider_response.provider)
        metrics.increment_throughput()

        # 4. Cache the new response
        prompt_embedding = cache.model.encode(request.prompt).tolist()
        cache.cache_response(request.prompt, provider_response.response, prompt_embedding)

    except Exception as e:
        metrics.record_provider_error(provider_name, str(type(e).__name__))
        raise e

    # Dummy traffic update for scaler
    scaler.traffic_history.append(1.0) # Placeholder for actual traffic data
    predicted_traffic = scaler.predict_traffic(1)
    current_load = 1.0 # Placeholder
    target_replicas = scaler.calculate_required_capacity(current_load, predicted_traffic[0])
    scaler.scale_infrastructure(target_replicas)

    return provider_response