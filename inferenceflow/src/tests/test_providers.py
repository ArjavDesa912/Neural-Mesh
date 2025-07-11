import pytest
from src.models.providers import OpenAIProvider, AnthropicProvider, GoogleProvider, CohereProvider
from src.models.response import InferenceResponse
import asyncio
from unittest.mock import AsyncMock, patch
import json

@pytest.mark.asyncio
async def test_openai_provider_generate():
    provider = OpenAIProvider()
    prompt = "Test prompt for OpenAI"
    model = "gpt-3.5-turbo"

    mock_response_payload = {
        "choices": [{"message": {"content": f"OpenAI response for prompt: {prompt}"}}],
        "usage": {"total_tokens": 50},
        "id": "test_openai_id"
    }
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_payload
    mock_response.raise_for_status = AsyncMock()

    with patch('aiohttp.ClientSession.post', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False))):
        response = await provider.generate(prompt, model)

    assert isinstance(response, InferenceResponse)
    assert response.provider == "openai"
    assert response.model == model
    assert response.response.startswith("OpenAI response for prompt:")
    assert response.tokens_used == 50
    assert response.request_id == "test_openai_id"

@pytest.mark.asyncio
async def test_anthropic_provider_generate():
    provider = AnthropicProvider()
    prompt = "Test prompt for Anthropic"
    model = "claude-3-sonnet"

    mock_response_payload = {
        "content": [{"text": f"Anthropic response for prompt: {prompt}"}],
        "usage": {"input_tokens": 20, "output_tokens": 30},
        "id": "test_anthropic_id"
    }
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_payload
    mock_response.raise_for_status = AsyncMock()

    with patch('aiohttp.ClientSession.post', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False))):
        response = await provider.generate(prompt, model)

    assert isinstance(response, InferenceResponse)
    assert response.provider == "anthropic"
    assert response.model == model
    assert response.response.startswith("Anthropic response for prompt:")
    assert response.tokens_used == 50
    assert response.request_id == "test_anthropic_id"

@pytest.mark.asyncio
async def test_google_provider_generate():
    provider = GoogleProvider()
    prompt = "Test prompt for Google"
    model = "gemini-pro"

    mock_response_payload = {
        "candidates": [{"content": {"parts": [{"text": f"Google response for prompt: {prompt}"}]}}]
    }
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_payload
    mock_response.raise_for_status = AsyncMock()

    with patch('aiohttp.ClientSession.post', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False))):
        response = await provider.generate(prompt, model)

    assert isinstance(response, InferenceResponse)
    assert response.provider == "google"
    assert response.model == model
    assert response.response.startswith("Google response for prompt:")
    # Token usage is estimated for Google, so we don't assert a specific value

@pytest.mark.asyncio
async def test_cohere_provider_generate():
    provider = CohereProvider()
    prompt = "Test prompt for Cohere"
    model = "command-r-plus"

    mock_response_payload = {
        "generations": [{"text": f"Cohere response for prompt: {prompt}"}],
        "id": "test_cohere_id"
    }
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_payload
    mock_response.raise_for_status = AsyncMock()

    with patch('aiohttp.ClientSession.post', return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=False))):
        response = await provider.generate(prompt, model)

    assert isinstance(response, InferenceResponse)
    assert response.provider == "cohere"
    assert response.model == model
    assert response.response.startswith("Cohere response for prompt:")
    assert response.request_id == "test_cohere_id"

def test_provider_estimate_cost():
    openai_provider = OpenAIProvider()
    anthropic_provider = AnthropicProvider()
    google_provider = GoogleProvider()
    cohere_provider = CohereProvider()

    prompt = "Short test prompt"
    max_tokens = 100

    assert isinstance(openai_provider.estimate_cost(prompt, max_tokens), float)
    assert isinstance(anthropic_provider.estimate_cost(prompt, max_tokens), float)
    assert isinstance(google_provider.estimate_cost(prompt, max_tokens), float)
    assert isinstance(cohere_provider.estimate_cost(prompt, max_tokens), float)
