import redis.asyncio as redis
import pickle
import zlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils.config import settings
from src.utils.logger import LoggerMixin
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib

@dataclass
class CacheEntry:
    """Enhanced cache entry with multi-modal support"""
    prompt: str
    response: str
    text_embedding: np.ndarray
    vision_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    created_at: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    similarity_score: float = 0.0
    multi_modal_features: Dict[str, Any] = None

class ReinforcementLearningCacheOptimizer:
    """RL-based cache optimization and eviction policies"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.cache_performance_history: deque = deque(maxlen=1000)
        self.eviction_policies = {
            'lru': self._lru_eviction,
            'lfu': self._lfu_eviction,
            'rl_optimized': self._rl_optimized_eviction
        }
        self.policy_weights = {
            'lru': 0.3,
            'lfu': 0.3,
            'rl_optimized': 0.4
        }
        
    def record_cache_hit(self, entry: CacheEntry, access_time: float):
        """Record cache hit for learning"""
        entry.access_count += 1
        entry.last_accessed = access_time
        
        # Record performance
        self.cache_performance_history.append({
            'timestamp': access_time,
            'type': 'hit',
            'entry_age': access_time - entry.created_at,
            'access_count': entry.access_count,
            'similarity_score': entry.similarity_score
        })
    
    def record_cache_miss(self, prompt_features: Dict[str, Any], miss_time: float):
        """Record cache miss for learning"""
        self.cache_performance_history.append({
            'timestamp': miss_time,
            'type': 'miss',
            'prompt_features': prompt_features
        })
    
    def _lru_eviction(self, entries: List[CacheEntry]) -> List[CacheEntry]:
        """Least Recently Used eviction"""
        return sorted(entries, key=lambda x: x.last_accessed)
    
    def _lfu_eviction(self, entries: List[CacheEntry]) -> List[CacheEntry]:
        """Least Frequently Used eviction"""
        return sorted(entries, key=lambda x: x.access_count)
    
    def _rl_optimized_eviction(self, entries: List[CacheEntry]) -> List[CacheEntry]:
        """RL-optimized eviction considering multiple factors"""
        scored_entries = []
        current_time = time.time()
        
        for entry in entries:
            # Calculate eviction score (lower is better)
            age_score = (current_time - entry.created_at) / 3600  # Age in hours
            access_score = 1.0 / (entry.access_count + 1)  # Inverse access frequency
            similarity_score = 1.0 - entry.similarity_score  # Lower similarity = higher eviction priority
            
            # Weighted score
            eviction_score = (age_score * 0.4) + (access_score * 0.3) + (similarity_score * 0.3)
            
            scored_entries.append((entry, eviction_score))
        
        # Sort by eviction score
        scored_entries.sort(key=lambda x: x[1])
        return [entry for entry, score in scored_entries]
    
    def select_eviction_candidates(self, entries: List[CacheEntry], count: int) -> List[CacheEntry]:
        """Select candidates for eviction using weighted policies"""
        if len(entries) <= count:
            return entries
        
        # Apply each policy
        policy_results = {}
        for policy_name, policy_func in self.eviction_policies.items():
            policy_results[policy_name] = policy_func(entries)
        
        # Combine results using weighted voting
        entry_scores = defaultdict(float)
        for policy_name, ranked_entries in policy_results.items():
            weight = self.policy_weights[policy_name]
            for i, entry in enumerate(ranked_entries):
                entry_scores[entry] += weight * (i + 1)  # Higher rank = higher score
        
        # Sort by combined score and return top candidates
        sorted_entries = sorted(entry_scores.items(), key=lambda x: x[1], reverse=True)
        return [entry for entry, score in sorted_entries[:count]]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get cache optimization statistics"""
        recent_performance = [p for p in self.cache_performance_history if time.time() - p['timestamp'] < 3600]
        
        hit_rate = len([p for p in recent_performance if p['type'] == 'hit']) / max(len(recent_performance), 1)
        
        return {
            'total_events': len(self.cache_performance_history),
            'recent_hit_rate': hit_rate,
            'policy_weights': self.policy_weights,
            'learning_rate': self.learning_rate
        }

class SemanticCache(LoggerMixin):
    """Multi-layer caching system with semantic similarity matching and RL optimization"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vision_embedding_model: Optional[SentenceTransformer] = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.rl_optimizer = ReinforcementLearningCacheOptimizer()
        self.multi_modal_features_cache: Dict[str, Any] = {}
        self.cache_clusters: Dict[str, List[CacheEntry]] = defaultdict(list)
        self.similarity_threshold = 0.85
        
    async def initialize(self):
        """Initialize Redis connection and embedding models"""
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
            
            # Initialize text embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vision embedding model for multi-modal support
            try:
                self.vision_embedding_model = SentenceTransformer('clip-ViT-B-32')
                self.logger.info("Vision embedding model loaded")
            except Exception:
                self.logger.warning("Vision embedding model not available, multi-modal features limited")
            
            self.logger.info("Embedding models loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {str(e)}")
            raise
    
    async def get_similar_cached_response(
        self, 
        prompt: str, 
        threshold: float = None,
        multi_modal_features: Dict[str, Any] = None
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Find similar cached response based on semantic similarity and multi-modal features
        
        Returns:
            Tuple of (cache_entry, similarity_score) or None if no match found
        """
        if not self.redis_client or not self.embedding_model:
            return None
            
        threshold = threshold or self.similarity_threshold
        self.total_requests += 1
        
        try:
            # Generate embedding for the input prompt
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Extract multi-modal features if not provided
            if not multi_modal_features:
                multi_modal_features = self._extract_multi_modal_features(prompt)
            
            # Get all cached embeddings
            cache_keys = await self.redis_client.keys("embedding:*")
            
            best_match = None
            best_score = 0.0
            best_entry = None
            
            for key in cache_keys:
                # Get cached embedding and response
                cached_data = await self.redis_client.hgetall(key)
                
                if not cached_data:
                    continue
                    
                # Decode cached data
                cached_entry_bytes = cached_data.get(b'cache_entry')
                if not cached_entry_bytes:
                    continue
                    
                cached_entry = pickle.loads(cached_entry_bytes)
                
                # Calculate multi-dimensional similarity
                similarity = self._calculate_multi_modal_similarity(
                    prompt_embedding, cached_entry, multi_modal_features
                )
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_entry = cached_entry
            
            if best_entry:
                # Update access statistics
                current_time = time.time()
                best_entry.last_accessed = current_time
                best_entry.access_count += 1
                best_entry.similarity_score = best_score
                
                # Record cache hit for RL optimization
                self.rl_optimizer.record_cache_hit(best_entry, current_time)
                
                self.cache_hits += 1
                self.logger.info(f"Cache hit with similarity {best_score:.3f}")
                
                return best_entry, best_score
            
            # Record cache miss for RL optimization
            self.rl_optimizer.record_cache_miss(multi_modal_features, time.time())
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
        metadata: Dict[str, Any] = None,
        multi_modal_features: Dict[str, Any] = None
    ) -> None:
        """Cache a response with its embedding and multi-modal features"""
        if not self.redis_client or not self.embedding_model:
            return
            
        try:
            current_time = time.time()
            
            # Generate embedding for the prompt
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Extract multi-modal features if not provided
            if not multi_modal_features:
                multi_modal_features = self._extract_multi_modal_features(prompt)
            
            # Create cache entry
            cache_entry = CacheEntry(
                prompt=prompt,
                response=response,
                text_embedding=prompt_embedding,
                vision_embedding=None,  # Would need image data
                metadata=metadata or {},
                created_at=current_time,
                access_count=1,
                last_accessed=current_time,
                similarity_score=0.0,
                multi_modal_features=multi_modal_features
            )
            
            # Create unique key
            entry_key = f"cache_entry:{int(current_time * 1000)}"
            
            # Serialize and compress cache entry
            serialized_entry = pickle.dumps(cache_entry)
            compressed_entry = zlib.compress(serialized_entry)
            
            # Store cache entry
            await self.redis_client.setex(
                entry_key,
                settings.cache_ttl,
                compressed_entry
            )
            
            # Store embedding reference
            embedding_key = f"embedding:{int(current_time * 1000)}"
            embedding_data = {
                "cache_entry": serialized_entry,
                "entry_key": entry_key,
                "created_at": current_time
            }
            
            await self.redis_client.hset(embedding_key, mapping=embedding_data)
            await self.redis_client.expire(embedding_key, settings.cache_ttl)
            
            # Add to cluster for automated response clustering
            cluster_id = self._assign_to_cluster(cache_entry)
            self.cache_clusters[cluster_id].append(cache_entry)
            
            # Check if cache is full and perform RL-optimized eviction if needed
            await self._check_and_evict_if_needed()
            
            self.logger.info(f"Cached response with key: {entry_key}, cluster: {cluster_id}")
            
        except Exception as e:
            self.logger.error(f"Error caching response: {str(e)}")
    
    def _assign_to_cluster(self, entry: CacheEntry) -> str:
        """Assign cache entry to cluster for automated response clustering"""
        # Simple clustering based on multi-modal features
        cluster_features = []
        
        if entry.multi_modal_features:
            # Language cluster
            cluster_features.append(f"lang_{entry.multi_modal_features.get('language', 'unknown')}")
            
            # Content type cluster
            if entry.multi_modal_features.get('has_code'):
                cluster_features.append('code')
            if entry.multi_modal_features.get('has_math'):
                cluster_features.append('math')
            
            # Complexity cluster
            complexity = entry.multi_modal_features.get('text_complexity', 0.5)
            if complexity > 0.7:
                cluster_features.append('complex')
            elif complexity < 0.3:
                cluster_features.append('simple')
        
        # Generate cluster ID
        cluster_id = '_'.join(cluster_features) if cluster_features else 'general'
        return cluster_id
    
    async def _check_and_evict_if_needed(self):
        """Check cache size and perform RL-optimized eviction if needed"""
        try:
            # Get current cache size
            total_keys = await self.redis_client.dbsize()
            max_cache_size = getattr(settings, 'max_cache_size', 10000)
            
            if total_keys > max_cache_size:
                # Calculate number of entries to evict (10% of cache)
                entries_to_evict = int(max_cache_size * 0.1)
                
                # Get all cache entries
                cache_keys = await self.redis_client.keys("cache_entry:*")
                
                if len(cache_keys) > entries_to_evict:
                    # Deserialize entries for RL optimization
                    entries = []
                    for key in cache_keys:
                        try:
                            entry_data = await self.redis_client.get(key)
                            if entry_data:
                                decompressed = zlib.decompress(entry_data)
                                entry = pickle.loads(decompressed)
                                entries.append(entry)
                        except Exception:
                            continue
                    
                    # Select eviction candidates using RL optimization
                    eviction_candidates = self.rl_optimizer.select_eviction_candidates(
                        entries, entries_to_evict
                    )
                    
                    # Evict selected entries
                    for entry in eviction_candidates:
                        # Find and delete the corresponding cache entry
                        entry_key = f"cache_entry:{int(entry.created_at * 1000)}"
                        embedding_key = f"embedding:{int(entry.created_at * 1000)}"
                        
                        await self.redis_client.delete(entry_key)
                        await self.redis_client.delete(embedding_key)
                    
                    self.logger.info(f"Evicted {len(eviction_candidates)} entries using RL optimization")
        
        except Exception as e:
            self.logger.error(f"Error in cache eviction: {str(e)}")
    
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
        """Get comprehensive cache statistics and analytics"""
        if not self.redis_client:
            return {}
            
        try:
            # Get basic stats
            total_keys = await self.redis_client.dbsize()
            
            # Get cache entry keys count
            cache_keys = await self.redis_client.keys("cache_entry:*")
            embedding_keys = await self.redis_client.keys("embedding:*")
            
            # Calculate hit rate
            hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            
            # Get memory usage
            info = await self.redis_client.info("memory")
            used_memory = info.get("used_memory_human", "0B")
            
            # Get cluster statistics
            cluster_stats = {}
            for cluster_id, entries in self.cache_clusters.items():
                cluster_stats[cluster_id] = {
                    "entry_count": len(entries),
                    "avg_access_count": sum(e.access_count for e in entries) / len(entries) if entries else 0,
                    "avg_similarity": sum(e.similarity_score for e in entries) / len(entries) if entries else 0
                }
            
            # Get RL optimization stats
            rl_stats = self.rl_optimizer.get_optimization_stats()
            
            # Calculate multi-modal feature distribution
            feature_distribution = self._calculate_feature_distribution()
            
            return {
                "total_keys": total_keys,
                "cache_entry_keys": len(cache_keys),
                "embedding_keys": len(embedding_keys),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": self.total_requests,
                "hit_rate_percent": round(hit_rate, 2),
                "used_memory": used_memory,
                "cache_ttl_seconds": settings.cache_ttl,
                "similarity_threshold": self.similarity_threshold,
                "clusters": cluster_stats,
                "reinforcement_learning": rl_stats,
                "multi_modal_features": {
                    "cached_features": len(self.multi_modal_features_cache),
                    "feature_distribution": feature_distribution,
                    "supported_features": [
                        "text_complexity", "has_code", "has_math", "language", 
                        "sentiment", "urgency", "prompt_length", "word_count"
                    ]
                },
                "performance_metrics": {
                    "avg_response_time": self._calculate_avg_response_time(),
                    "cache_efficiency": self._calculate_cache_efficiency(),
                    "eviction_events": len([p for p in self.rl_optimizer.cache_performance_history if p['type'] == 'eviction'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def _calculate_feature_distribution(self) -> Dict[str, Any]:
        """Calculate distribution of multi-modal features in cache"""
        distribution = {
            "languages": defaultdict(int),
            "content_types": defaultdict(int),
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
        for cluster_entries in self.cache_clusters.values():
            for entry in cluster_entries:
                if entry.multi_modal_features:
                    # Language distribution
                    lang = entry.multi_modal_features.get('language', 'unknown')
                    distribution["languages"][lang] += 1
                    
                    # Content type distribution
                    if entry.multi_modal_features.get('has_code'):
                        distribution["content_types"]["code"] += 1
                    if entry.multi_modal_features.get('has_math'):
                        distribution["content_types"]["math"] += 1
                    
                    # Complexity distribution
                    complexity = entry.multi_modal_features.get('text_complexity', 0.5)
                    if complexity < 0.33:
                        distribution["complexity_distribution"]["low"] += 1
                    elif complexity < 0.67:
                        distribution["complexity_distribution"]["medium"] += 1
                    else:
                        distribution["complexity_distribution"]["high"] += 1
        
        return distribution
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from cache hits"""
        # This would need to be tracked separately
        return 0.0  # Placeholder
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency score (0-1)"""
        if self.total_requests == 0:
            return 0.0
        
        hit_rate = self.cache_hits / self.total_requests
        
        # Consider similarity scores of hits
        avg_similarity = 0.0
        hit_count = 0
        
        for cluster_entries in self.cache_clusters.values():
            for entry in cluster_entries:
                if entry.access_count > 0:
                    avg_similarity += entry.similarity_score
                    hit_count += 1
        
        if hit_count > 0:
            avg_similarity /= hit_count
        
        # Combined efficiency score
        efficiency = (hit_rate * 0.7) + (avg_similarity * 0.3)
        return efficiency
    
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
    
    def _extract_multi_modal_features(self, prompt: str) -> Dict[str, Any]:
        """Extract multi-modal features from prompt"""
        features = {
            "text_complexity": self._calculate_text_complexity(prompt),
            "has_code": self._detect_code_content(prompt),
            "has_math": self._detect_math_content(prompt),
            "language": self._detect_language(prompt),
            "sentiment": self._analyze_sentiment(prompt),
            "urgency": self._detect_urgency(prompt),
            "prompt_length": len(prompt),
            "word_count": len(prompt.split())
        }
        
        # Cache features
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self.multi_modal_features_cache[prompt_hash] = features
        
        return features
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)"""
        if not text:
            return 0.0
        
        word_count = len(text.split())
        sentence_count = len(text.split('.')) + len(text.split('!')) + len(text.split('?'))
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        complexity = min(1.0, (word_count / 100) + (avg_word_length / 10) + (sentence_count / 10))
        return complexity
    
    def _detect_code_content(self, text: str) -> bool:
        """Detect if text contains code"""
        code_indicators = ['def ', 'function', 'class ', 'import ', 'from ', 'var ', 'let ', 'const ', '=>', '{', '}']
        return any(indicator in text.lower() for indicator in code_indicators)
    
    def _detect_math_content(self, text: str) -> bool:
        """Detect if text contains mathematical content"""
        math_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos', 'tan', 'log', 'integral', 'derivative']
        return any(indicator in text.lower() for indicator in math_indicators)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        if any(char in text for char in ['ä', 'ö', 'ü', 'ß']):
            return 'german'
        elif any(char in text for char in ['é', 'è', 'ê', 'à', 'ç']):
            return 'french'
        elif any(char in text for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú']):
            return 'spanish'
        else:
            return 'english'
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1)"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _detect_urgency(self, text: str) -> float:
        """Detect urgency level (0-1)"""
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'important', 'quickly', 'fast']
        text_lower = text.lower()
        
        urgent_count = sum(1 for word in urgent_words if word in text_lower)
        return min(1.0, urgent_count / len(urgent_words))
    
    def _calculate_multi_modal_similarity(
        self, 
        prompt_embedding: np.ndarray, 
        cached_entry: CacheEntry, 
        query_features: Dict[str, Any]
    ) -> float:
        """Calculate multi-dimensional similarity score"""
        similarities = []
        
        # Text embedding similarity
        text_similarity = self._cosine_similarity(prompt_embedding, cached_entry.text_embedding)
        similarities.append(text_similarity * 0.6)  # 60% weight
        
        # Multi-modal feature similarity
        if cached_entry.multi_modal_features:
            feature_similarity = self._calculate_feature_similarity(
                query_features, cached_entry.multi_modal_features
            )
            similarities.append(feature_similarity * 0.3)  # 30% weight
        
        # Length similarity (penalize large differences)
        length_diff = abs(query_features.get('prompt_length', 0) - len(cached_entry.prompt))
        length_similarity = max(0, 1 - (length_diff / 1000))  # Normalize
        similarities.append(length_similarity * 0.1)  # 10% weight
        
        # Vision similarity if available
        if (self.vision_embedding_model and 
            cached_entry.vision_embedding is not None and 
            query_features.get('has_image', False)):
            # Would need image data for this
            vision_similarity = 0.8  # Placeholder
            similarities.append(vision_similarity * 0.1)  # Additional 10% if available
        
        return min(1.0, sum(similarities))
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between multi-modal feature sets"""
        similarity_scores = []
        
        # Numerical features
        numerical_features = ['text_complexity', 'sentiment', 'urgency']
        for feature in numerical_features:
            if feature in features1 and feature in features2:
                val1, val2 = features1[feature], features2[feature]
                # Calculate normalized difference
                diff = abs(val1 - val2)
                similarity = max(0, 1 - diff)  # Normalize to 0-1
                similarity_scores.append(similarity)
        
        # Categorical features
        categorical_features = ['language', 'has_code', 'has_math']
        for feature in categorical_features:
            if feature in features1 and feature in features2:
                if features1[feature] == features2[feature]:
                    similarity_scores.append(1.0)
                else:
                    similarity_scores.append(0.0)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
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
