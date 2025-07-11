import asyncio
import time
import uuid
from typing import Optional, Dict, Any, List
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.models.providers import (
    ProviderManager, OpenAIProvider, AnthropicProvider, 
    GoogleProvider, CohereProvider, BaseProvider
)
from src.core.cache import semantic_cache
from src.core.router import SmartRouter
from src.services.metrics import metrics_service
from src.utils.config import settings
from src.utils.logger import LoggerMixin

class Orchestrator(LoggerMixin):
    """Main orchestrator that coordinates all inference components"""
    
    def __init__(self):
        self.provider_manager = ProviderManager()
        self.router: Optional[SmartRouter] = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing orchestrator...")
            
            # Initialize providers
            await self._initialize_providers()
            
            # Initialize cache
            await semantic_cache.initialize()
            
            # Initialize router
            self.router = SmartRouter(self.provider_manager)
            await self.router.initialize()
            
            # Initialize metrics
            await metrics_service.initialize()
            
            self.is_initialized = True
            self.logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise
    
    async def _initialize_providers(self):
        """Initialize all LLM providers"""
        providers_config = [
            ("openai", OpenAIProvider, settings.openai_api_key),
            ("anthropic", AnthropicProvider, settings.anthropic_api_key),
            ("google", GoogleProvider, settings.google_api_key),
            ("cohere", CohereProvider, settings.cohere_api_key)
        ]
        
        for name, provider_class, api_key in providers_config:
            if api_key:
                try:
                    provider = provider_class(api_key)
                    self.provider_manager.add_provider(name, provider)
                    self.logger.info(f"Initialized {name} provider")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {name} provider: {str(e)}")
            else:
                self.logger.warning(f"Skipping {name} provider - no API key provided")
    
    async def process_inference_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request through the complete pipeline"""
        
        if not self.is_initialized:
            raise Exception("Orchestrator not initialized")
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Processing inference request {request_id}")
            
            # 1. Check semantic cache first
            cached_result = await semantic_cache.get_similar_cached_response(request.prompt)
            if cached_result:
                cached_response, similarity = cached_result
                latency = time.time() - start_time
                
                # Record cache hit metrics
                await metrics_service.record_cache_hit()
                await metrics_service.record_request_latency(latency, "cache")
                
                return InferenceResponse(
                    response=cached_response["response"],
                    provider="cache",
                    model="cached",
                    latency=latency,
                    tokens_used=0,
                    cost=0.0,
                    cache_hit=True,
                    request_id=request_id,
                    metadata={"similarity": similarity, "cache_source": cached_response}
                )
            
            # 2. Route request to optimal provider
            routing_decision = await self.router.route_request(request)
            provider = routing_decision.provider
            
            # 3. Generate response from provider
            async with provider:
                try:
                    provider_response = await provider.generate(request)
                    provider_response.request_id = request_id
                    
                    # Record success metrics
                    latency = time.time() - start_time
                    await metrics_service.record_request_latency(latency, provider_response.provider)
                    await metrics_service.record_provider_success(provider_response.provider)
                    
                    # Update router with success
                    self.router.record_provider_success(provider)
                    
                    # 4. Cache the response
                    await semantic_cache.cache_response(
                        request.prompt,
                        provider_response.response,
                        {
                            "provider": provider_response.provider,
                            "model": provider_response.model,
                            "tokens_used": provider_response.tokens_used,
                            "cost": provider_response.cost
                        }
                    )
                    
                    self.logger.info(f"Successfully processed request {request_id} via {provider_response.provider}")
                    return provider_response
                    
                except Exception as e:
                    # Record failure metrics
                    latency = time.time() - start_time
                    await metrics_service.record_request_latency(latency, provider.__class__.__name__)
                    await metrics_service.record_provider_error(provider.__class__.__name__, str(type(e).__name__))
                    
                    # Update router with failure
                    self.router.record_provider_failure(provider)
                    
                    self.logger.error(f"Provider error for request {request_id}: {str(e)}")
                    raise
                    
        except Exception as e:
            latency = time.time() - start_time
            await metrics_service.record_request_latency(latency, "error")
            self.logger.error(f"Error processing request {request_id}: {str(e)}")
            raise
    
    async def process_batch_requests(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Process multiple inference requests in parallel"""
        
        if not self.is_initialized:
            raise Exception("Orchestrator not initialized")
        
        self.logger.info(f"Processing batch of {len(requests)} requests")
        
        # Process requests concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(settings.max_workers)
        
        async def process_single_request(req: InferenceRequest) -> InferenceResponse:
            async with semaphore:
                return await self.process_inference_request(req)
        
        # Create tasks for all requests
        tasks = [process_single_request(req) for req in requests]
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch request {i} failed: {str(response)}")
                # Create error response
                error_response = InferenceResponse(
                    response=f"Error: {str(response)}",
                    provider="error",
                    model="error",
                    latency=0.0,
                    tokens_used=0,
                    cost=0.0,
                    cache_hit=False,
                    request_id=str(uuid.uuid4()),
                    metadata={"error": str(response)}
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        try:
            # Get provider statuses
            provider_statuses = self.provider_manager.get_all_provider_status()
            
            # Get cache stats
            cache_stats = await semantic_cache.get_cache_stats()
            
            # Get router stats
            router_stats = self.router.get_routing_stats() if self.router else {}
            
            # Get metrics summary
            metrics_summary = await metrics_service.get_performance_summary()
            
            # Get orchestrator stats
            orchestrator_stats = {
                "is_initialized": self.is_initialized,
                "total_providers": len(provider_statuses),
                "healthy_providers": len([p for p in provider_statuses.values() if p.status.value == "healthy"]),
                "degraded_providers": len([p for p in provider_statuses.values() if p.status.value == "degraded"]),
                "unhealthy_providers": len([p for p in provider_statuses.values() if p.status.value == "unhealthy"])
            }
            
            return {
                "status": "healthy",
                "orchestrator": orchestrator_stats,
                "providers": provider_statuses,
                "cache": cache_stats,
                "router": router_stats,
                "metrics": metrics_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components"""
        
        health_status = {
            "overall": "healthy",
            "components": {}
        }
        
        # Check providers
        try:
            healthy_providers = await self.provider_manager.get_healthy_providers()
            health_status["components"]["providers"] = {
                "status": "healthy" if healthy_providers else "unhealthy",
                "healthy_count": len(healthy_providers)
            }
        except Exception as e:
            health_status["components"]["providers"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        # Check cache
        try:
            cache_stats = await semantic_cache.get_cache_stats()
            health_status["components"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_stats.get("hit_rate_percent", 0)
            }
        except Exception as e:
            health_status["components"]["cache"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        # Check router
        try:
            if self.router:
                router_stats = self.router.get_routing_stats()
                health_status["components"]["router"] = {
                    "status": "healthy",
                    "total_decisions": router_stats.get("total_routing_decisions", 0)
                }
            else:
                health_status["components"]["router"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            health_status["components"]["router"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        return health_status
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        self.logger.info("Shutting down orchestrator...")
        
        try:
            # Close cache connection
            await semantic_cache.close()
            
            # Close provider sessions
            for provider in self.provider_manager.providers.values():
                if hasattr(provider, 'session') and provider.session:
                    await provider.session.close()
            
            self.logger.info("Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

# Global orchestrator instance
orchestrator = Orchestrator()
