import pytest
import asyncio
from src.models.request import InferenceRequest, Priority
from src.models.response import InferenceResponse
from src.utils.config import settings
from src.core.orchestrator import orchestrator
from src.services.metrics import metrics_service

@pytest.fixture
async def test_orchestrator():
    """Setup test orchestrator"""
    # Initialize with minimal configuration for testing
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()

@pytest.mark.asyncio
async def test_inference_request_model():
    """Test InferenceRequest model validation"""
    request = InferenceRequest(
        prompt="Hello, how are you?",
        user_id="test_user",
        max_tokens=100,
        temperature=0.7
    )
    
    assert request.prompt == "Hello, how are you?"
    assert request.user_id == "test_user"
    assert request.max_tokens == 100
    assert request.temperature == 0.7
    assert request.priority == Priority.NORMAL

@pytest.mark.asyncio
async def test_inference_response_model():
    """Test InferenceResponse model validation"""
    response = InferenceResponse(
        response="I'm doing well, thank you!",
        provider="test_provider",
        model="test_model",
        latency=1.5,
        tokens_used=20,
        cost=0.001,
        cache_hit=False,
        request_id="test_request_id"
    )
    
    assert response.response == "I'm doing well, thank you!"
    assert response.provider == "test_provider"
    assert response.latency == 1.5
    assert response.cost == 0.001

@pytest.mark.asyncio
async def test_metrics_service():
    """Test metrics service functionality"""
    # Test recording metrics
    await metrics_service.record_request_latency(1.5, "test_provider")
    await metrics_service.record_cache_hit()
    await metrics_service.record_provider_success("test_provider")
    
    # Get performance summary
    summary = await metrics_service.get_performance_summary()
    
    assert "total_requests" in summary
    assert "cache_hits" in summary
    assert summary["cache_hits"] >= 1

@pytest.mark.asyncio
async def test_config_settings():
    """Test configuration settings"""
    assert hasattr(settings, 'redis_host')
    assert hasattr(settings, 'cache_ttl')
    assert hasattr(settings, 'similarity_threshold')
    assert settings.cache_ttl > 0
    assert 0 <= settings.similarity_threshold <= 1

@pytest.mark.asyncio
async def test_priority_enum():
    """Test Priority enum values"""
    assert Priority.LOW.value == "low"
    assert Priority.NORMAL.value == "normal"
    assert Priority.HIGH.value == "high"

if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_inference_request_model())
    asyncio.run(test_inference_response_model())
    asyncio.run(test_metrics_service())
    asyncio.run(test_config_settings())
    asyncio.run(test_priority_enum())
    print("All basic tests passed!") 