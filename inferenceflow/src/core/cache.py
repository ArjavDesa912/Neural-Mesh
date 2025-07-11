import redis.asyncio as redis
import pickle
import zlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils.config import settings
from src.utils.logger import LoggerMixin
import asyncio

class SemanticCache(LoggerMixin):
    """Multi-layer caching system with semantic similarity matching"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
    async def initialize(self):
        """Initialize Redis connection and embedding model"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False  # We need bytes for pickle
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {str(e)}")
            raise
    
    async def get_similar_cached_response(
        self, 
        prompt: str, 
        threshold: float = None
    ) -> Optional[Tuple[str, float]]:
        """
        Find similar cached response based on semantic similarity
        
        Returns:
            Tuple of (cached_response, similarity_score) or None if no match found
        """
        if not self.redis_client or not self.embedding_model:
            return None
            
        threshold = threshold or settings.similarity_threshold
        self.total_requests += 1
        
        try:
            # Generate embedding for the input prompt
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Get all cached embeddings
            cache_keys = await self.redis_client.keys("embedding:*")
            
            best_match = None
            best_score = 0.0
            
            for key in cache_keys:
                # Get cached embedding and response
                cached_data = await self.redis_client.hgetall(key)
                
                if not cached_data:
                    continue
                    
                # Decode cached embedding
                cached_embedding_bytes = cached_data.get(b'embedding')
                if not cached_embedding_bytes:
                    continue
                    
                cached_embedding = pickle.loads(cached_embedding_bytes)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(prompt_embedding, cached_embedding)
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    response_key = cached_data.get(b'response_key', b'').decode('utf-8')
                    best_match = response_key
            
            if best_match:
                # Get the cached response
                cached_response = await self.redis_client.get(best_match)
                if cached_response:
                    # Decompress and deserialize
                    decompressed = zlib.decompress(cached_response)
                    response_data = pickle.loads(decompressed)
                    
                    self.cache_hits += 1
                    self.logger.info(f"Cache hit with similarity {best_score:.3f}")
                    
                    return response_data, best_score
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error in semantic cache lookup: {str(e)}")
            self.cache_misses += 1
            return None
    
    async def cache_response(
        self, 
        prompt: str, 
        response: str, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Cache a response with its embedding"""
        if not self.redis_client or not self.embedding_model:
            return
            
        try:
            # Generate embedding for the prompt
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Create unique keys
            response_key = f"response:{int(time.time() * 1000)}"
            embedding_key = f"embedding:{int(time.time() * 1000)}"
            
            # Prepare response data
            response_data = {
                "response": response,
                "prompt": prompt,
                "metadata": metadata or {},
                "created_at": time.time()
            }
            
            # Compress and serialize response
            serialized_response = pickle.dumps(response_data)
            compressed_response = zlib.compress(serialized_response)
            
            # Store response
            await self.redis_client.setex(
                response_key,
                settings.cache_ttl,
                compressed_response
            )
            
            # Store embedding with reference to response
            embedding_data = {
                "embedding": pickle.dumps(prompt_embedding),
                "response_key": response_key,
                "prompt": prompt,
                "created_at": time.time()
            }
            
            await self.redis_client.hset(embedding_key, mapping=embedding_data)
            await self.redis_client.expire(embedding_key, settings.cache_ttl)
            
            self.logger.info(f"Cached response with key: {response_key}")
            
        except Exception as e:
            self.logger.error(f"Error caching response: {str(e)}")
    
    async def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache entries matching the pattern"""
        if not self.redis_client:
            return 0
            
        try:
            # Find keys matching pattern
            keys_to_delete = await self.redis_client.keys(pattern)
            
            if keys_to_delete:
                # Delete the keys
                deleted_count = await self.redis_client.delete(*keys_to_delete)
                self.logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and analytics"""
        if not self.redis_client:
            return {}
            
        try:
            # Get basic stats
            total_keys = await self.redis_client.dbsize()
            
            # Get embedding keys count
            embedding_keys = await self.redis_client.keys("embedding:*")
            response_keys = await self.redis_client.keys("response:*")
            
            # Calculate hit rate
            hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            
            # Get memory usage
            info = await self.redis_client.info("memory")
            used_memory = info.get("used_memory_human", "0B")
            
            return {
                "total_keys": total_keys,
                "embedding_keys": len(embedding_keys),
                "response_keys": len(response_keys),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": self.total_requests,
                "hit_rate_percent": round(hit_rate, 2),
                "used_memory": used_memory,
                "cache_ttl_seconds": settings.cache_ttl
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    async def clear_cache(self) -> int:
        """Clear all cache entries"""
        if not self.redis_client:
            return 0
            
        try:
            # Get all keys
            all_keys = await self.redis_client.keys("*")
            
            if all_keys:
                deleted_count = await self.redis_client.delete(*all_keys)
                self.logger.info(f"Cleared {deleted_count} cache entries")
                return deleted_count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis connection closed")

# Global cache instance
semantic_cache = SemanticCache()
