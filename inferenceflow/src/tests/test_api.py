import pytest
from httpx import AsyncClient
from main import app
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_health_check():
    with patch('src.api.gateway.redis_client') as mock_redis_client:
        mock_redis_client.ping = AsyncMock(return_value=True)
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "redis_status": "connected"}

@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/v1/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text
    assert "# TYPE" in response.text

# Add more tests for /v1/inference and /v1/batch once authentication and rate limiting are properly mocked
