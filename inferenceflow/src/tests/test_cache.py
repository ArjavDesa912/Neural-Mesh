import pytest
import redis.asyncio as redis
import pickle
import zlib
from src.core.cache import SemanticCache
from src.utils.config import settings
import time
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
import fnmatch

@pytest.fixture(name="test_cache")
async def test_cache_fixture():
    class MockRedisClient:
        def __init__(self):
            self._mock_store = {}

        async def get(self, key):
            return self._mock_store.get(key)

        async def setex(self, key, ttl, value):
            self._mock_store[key] = value

        async def scan_iter(self, pattern):
            for key in list(self._mock_store.keys()):
                if fnmatch.fnmatch(key.decode(), pattern):
                    yield key

        async def delete(self, key):
            if key in self._mock_store:
                del self._mock_store[key]

        async def dbsize(self):
            return len(self._mock_store)

        async def info(self):
            return {}

        async def flushdb(self):
            self._mock_store.clear()

        async def ping(self):
            return True

    with patch('redis.asyncio.Redis', return_value=MockRedisClient()) as MockRedis:
        cache = SemanticCache(redis_client=MockRedisClient()) # Pass the mocked client
        yield cache

@pytest.mark.asyncio
async def test_cache_response_and_get_similar(test_cache: SemanticCache):
    prompt = "Hello, how are you?"
    response = "I am fine, thank you."
    embedding = [0.1, 0.2, 0.3] # Dummy embedding

    await test_cache.cache_response(prompt, response, embedding)

    # Test exact match
    cached_response = await test_cache.get_similar_cached_response(prompt, threshold=0.99)
    assert cached_response == response

@pytest.mark.asyncio
async def test_cache_invalidation(test_cache: SemanticCache):
    prompt1 = "Test prompt 1"
    response1 = "Test response 1"
    embedding1 = [0.1, 0.1, 0.1]

    prompt2 = "Another prompt 2"
    response2 = "Another response 2"
    embedding2 = [0.2, 0.2, 0.2]

    await test_cache.cache_response(prompt1, response1, embedding1)
    await test_cache.cache_response(prompt2, response2, embedding2)

    assert await test_cache.get_similar_cached_response(prompt1) is not None
    assert await test_cache.get_similar_cached_response(prompt2) is not None

    await test_cache.invalidate_cache("cache:Test prompt*")

    assert await test_cache.get_similar_cached_response(prompt1) is None
    assert await test_cache.get_similar_cached_response(prompt2) is not None

@pytest.mark.asyncio
async def test_cache_stats(test_cache: SemanticCache):
    stats = await test_cache.get_cache_stats()
    assert "keys" in stats
    assert "info" in stats
    assert stats["keys"] == 0

    prompt = "Another prompt"
    response = "Another response"
    embedding = [0.5, 0.5, 0.5]
    await test_cache.cache_response(prompt, response, embedding)

    stats = await test_cache.get_cache_stats()
    assert stats["keys"] == 1

@pytest.mark.asyncio
async def test_cache_ttl(test_cache: SemanticCache):
    original_ttl = settings.CACHE_TTL
    settings.CACHE_TTL = 1 # Set TTL to 1 second for testing

    prompt = "Ephemeral prompt"
    response = "Ephemeral response"
    embedding = [0.9, 0.8, 0.7]

    await test_cache.cache_response(prompt, response, embedding)
    assert await test_cache.get_similar_cached_response(prompt) is not None

    await asyncio.sleep(2) # Wait for TTL to expire

    assert await test_cache.get_similar_cached_response(prompt) is None

    settings.CACHE_TTL = original_ttl # Restore original TTL
