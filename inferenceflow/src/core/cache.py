import redis
import pickle
import zlib
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.utils.config import settings

class SemanticCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=0
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_similar_cached_response(self, prompt: str, threshold: float = 0.85) -> Optional[str]:
        prompt_embedding = self.model.encode(prompt)
        # This is a simplified implementation. A real implementation would use a vector database for efficient similarity search.
        for key in self.redis_client.scan_iter("cache:*"):
            cached_data = self.redis_client.get(key)
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

    def invalidate_cache(self, pattern: str) -> None:
        for key in self.redis_client.scan_iter(f"cache:{pattern}"):
            self.redis_client.delete(key)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "keys": self.redis_client.dbsize(),
            "info": self.redis_client.info()
        }

    def _cosine_similarity(self, vec1, vec2):
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cache = SemanticCache()
