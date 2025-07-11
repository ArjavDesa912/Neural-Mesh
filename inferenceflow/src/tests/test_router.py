import pytest
from src.core.router import Router
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.models.providers import OpenAIProvider, AnthropicProvider, GoogleProvider, CohereProvider
from src.core.cache import SemanticCache
from src.services.embedding import embedding_service
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

@pytest.fixture
def router_instance():
    return Router()

@pytest.fixture
def mock_providers():
    mock_openai = MagicMock(spec=OpenAIProvider)
    mock_anthropic = MagicMock(spec=AnthropicProvider)
    mock_google = MagicMock(spec=GoogleProvider)
    mock_cohere = MagicMock(spec=CohereProvider)

    mock_openai.status = MagicMock(error_rate=0.0, latency=0.1)
    mock_anthropic.status = MagicMock(error_rate=0.0, latency=0.2)
    mock_google.status = MagicMock(error_rate=0.0, latency=0.15)
    mock_cohere.status = MagicMock(error_rate=0.0, latency=0.25)

    mock_openai.generate = AsyncMock(return_value=InferenceResponse(response="OpenAI response", provider="openai", model="gpt-3.5-turbo", latency=0.1, tokens_used=10, cost=0.001, cache_hit=False, request_id="1", created_at="2023-01-01T00:00:00"))
    mock_anthropic.generate = AsyncMock(return_value=InferenceResponse(response="Anthropic response", provider="anthropic", model="claude-3-sonnet", latency=0.2, tokens_used=20, cost=0.002, cache_hit=False, request_id="2", created_at="2023-01-01T00:00:00"))
    mock_google.generate = AsyncMock(return_value=InferenceResponse(response="Google response", provider="google", model="gemini-pro", latency=0.15, tokens_used=15, cost=0.0015, cache_hit=False, request_id="3", created_at="2023-01-01T00:00:00"))
    mock_cohere.generate = AsyncMock(return_value=InferenceResponse(response="Cohere response", provider="cohere", model="command-r-plus", latency=0.25, tokens_used=25, cost=0.0025, cache_hit=False, request_id="4", created_at="2023-01-01T00:00:00"))

    # Set initial healthy status
    mock_openai.status.is_healthy = True
    mock_anthropic.status.is_healthy = True
    mock_google.status.is_healthy = True
    mock_cohere.status.is_healthy = True

    with patch.dict('src.models.providers.providers', {
        "openai": mock_openai,
        "anthropic": mock_anthropic,
        "google": mock_google,
        "cohere": mock_cohere,
    }):
        yield {
            "openai": mock_openai,
            "anthropic": mock_anthropic,
            "google": mock_google,
            "cohere": mock_cohere,
        }

@pytest.mark.asyncio
async def test_route_request_cache_hit(router_instance: Router):
    with patch.object(SemanticCache, 'get_similar_cached_response', return_value="Cached response") as mock_cache_get:
        with patch.object(SemanticCache, 'cache_response') as mock_cache_set:
            
            request = InferenceRequest(prompt="Test prompt", user_id="test_user")
            response = await router_instance.route_request(request)

            mock_cache_get.assert_called_once_with("Test prompt")
            mock_cache_set.assert_not_called()
            assert response.response == "Cached response"
            assert response.provider == "cache"

@pytest.mark.asyncio
async def test_route_request_no_cache_hit_provider_selection(router_instance: Router, mock_providers):
    with patch.object(SemanticCache, 'get_similar_cached_response', return_value=None) as mock_cache_get:
        with patch.object(SemanticCache, 'cache_response') as mock_cache_set:
            
            request = InferenceRequest(prompt="Test prompt", user_id="test_user", model_preference="openai")
            response = await router_instance.route_request(request)

            mock_cache_get.assert_called_once_with("Test prompt")
            mock_providers["openai"].generate.assert_called_once()
            mock_cache_set.assert_called_once()
            assert response.response == "OpenAI response"
            assert response.provider == "openai"

@pytest.mark.asyncio
async def test_route_request_circuit_breaker(router_instance: Router, mock_providers):
    with patch.object(SemanticCache, 'get_similar_cached_response', return_value=None):
        with patch.object(SemanticCache, 'cache_response'):
            
            # Simulate a provider failure
            mock_providers["openai"].generate.side_effect = Exception("API Error")
            mock_providers["openai"].status.record_failure()

            request = InferenceRequest(prompt="Test prompt", user_id="test_user", model_preference="openai")
            
            # First request should fail and trigger circuit breaker
            with pytest.raises(Exception):
                await router_instance.route_request(request)
            
            # Subsequent request should not call the failed provider immediately
            # It should fall back to another provider or raise an error if no healthy providers
            request_no_preference = InferenceRequest(prompt="Test prompt 2", user_id="test_user")
            response = await router_instance.route_request(request_no_preference)
            assert response.provider != "openai" # Should pick another provider

@pytest.mark.asyncio
async def test_calculate_similarity(router_instance: Router):
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast brown fox leaps over a sleepy canine."
    text3 = "A completely unrelated sentence."

    similarity12 = router_instance.calculate_similarity(text1, text2)
    similarity13 = router_instance.calculate_similarity(text1, text3)

    assert similarity12 > similarity13
    assert 0 <= similarity12 <= 1
    assert 0 <= similarity13 <= 1

@pytest.mark.asyncio
async def test_request_deduplication(router_instance: Router, mock_providers):
    with patch.object(SemanticCache, 'get_similar_cached_response', return_value=None):
        with patch.object(SemanticCache, 'cache_response'):
            
            prompt = "This is a unique prompt for deduplication."
            request1 = InferenceRequest(prompt=prompt, user_id="user1")
            request2 = InferenceRequest(prompt=prompt, user_id="user1")

            # First request should go to provider and be cached
            response1 = await router_instance.route_request(request1)
            mock_providers["openai"].generate.assert_called_once() # Assuming openai is selected

            # Second request with same prompt should be deduplicated and return cached response
            response2 = await router_instance.route_request(request2)
            mock_providers["openai"].generate.assert_called_once() # Should not be called again

            assert response1.response == response2.response
            assert response2.cache_hit == True
