import redis
import pickle
import zlib
from typing import Optional, List, Dict, Any
from src.utils.config import settings
from src.services.embedding import embedding_service

class SemanticCache:
    def __init__(self, redis_client=None):
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=0
            )

    async def get_similar_cached_response(self, prompt: str, threshold: float = 0.85) -> Optional[str]:
        prompt_embedding = embedding_service.encode(prompt)
        # This is a simplified implementation. A real implementation would use a vector database for efficient similarity search.
        async for key in self.redis_client.scan_iter("cache:*"):
            cached_data = await self.redis_client.get(key)
            if cached_data:
                data = pickle.loads(zlib.decompress(cached_data))
                similarity = self._cosine_similarity(prompt_embedding, data['embedding'])
                if similarity > threshold:
                    return data['response']
        return None

    def cache_response(self, prompt: str, response: str, embedding: List[float]) -> None:
        key = f"cache:{prompt}"
        data = {
            'response': response,
            'embedding': embedding
        }
        compressed_data = zlib.compress(pickle.dumps(data))
        self.redis_client.setex(key, settings.CACHE_TTL, compressed_data)

    async def invalidate_cache(self, pattern: str) -> None:
        for key in await self.redis_client.scan_iter(f"cache:{pattern}"):
            self.redis_client.delete(key)

    async def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "keys": await self.redis_client.dbsize(),
            "info": await self.redis_client.info()
        }

    def _cosine_similarity(self, vec1, vec2):
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cache = SemanticCache()
