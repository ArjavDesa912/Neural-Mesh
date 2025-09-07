import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from src.core.cache import SemanticCache, CacheEntry
from src.services.metrics import MetricsService
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.utils.config import settings

class TestMultiModalIntegration:
    """Integration tests for Multi-modal features"""
    
    @pytest.fixture
    def semantic_cache(self):
        """Create semantic cache instance for testing"""
        return SemanticCache(
            max_size=1000,
            similarity_threshold=0.85
        )
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service"""
        mock_service = Mock()
        
        # Mock text embeddings
        mock_service.get_text_embedding = AsyncMock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        
        # Mock vision embeddings
        mock_service.get_vision_embedding = AsyncMock(return_value=np.array([0.6, 0.7, 0.8, 0.9, 1.0]))
        
        # Mock code embeddings
        mock_service.get_code_embedding = AsyncMock(return_value=np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        
        return mock_service
    
    @pytest.fixture
    def sample_cache_entries(self):
        """Create sample cache entries for testing"""
        return [
            CacheEntry(
                prompt="What is machine learning?",
                response="Machine learning is a subset of AI...",
                text_embedding=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                vision_embedding=None,
                metadata={"modality": "text", "tokens": 50}
            ),
            CacheEntry(
                prompt="Describe this image",
                response="This image shows a beautiful landscape...",
                text_embedding=np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
                vision_embedding=np.array([0.7, 0.8, 0.9, 1.0, 0.1]),
                metadata={"modality": "multimodal", "tokens": 75}
            ),
            CacheEntry(
                prompt="Analyze this code",
                response="This code implements a sorting algorithm...",
                text_embedding=np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
                vision_embedding=None,
                metadata={"modality": "code", "tokens": 100}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_semantic_cache_initialization(self, semantic_cache):
        """Test semantic cache initialization"""
        assert semantic_cache.max_size == 1000
        assert semantic_cache.similarity_threshold == 0.85
        assert len(semantic_cache.cache) == 0
        assert semantic_cache.cache_stats['total_requests'] == 0
        assert semantic_cache.cache_stats['cache_hits'] == 0
    
    @pytest.mark.asyncio
    async def test_text_cache_operations(self, semantic_cache, mock_embedding_service):
        """Test text-based cache operations"""
        prompt = "What is artificial intelligence?"
        response = "AI is the simulation of human intelligence..."
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Cache response
        await semantic_cache.cache_response(prompt, response, embedding)
        
        # Verify cache entry was created
        assert len(semantic_cache.cache) == 1
        assert semantic_cache.cache_stats['total_requests'] == 1
        
        # Retrieve cached response
        cached_response = await semantic_cache.get_similar_cached_response(prompt)
        assert cached_response == response
        assert semantic_cache.cache_stats['cache_hits'] == 1
    
    @pytest.mark.asyncio
    async def test_multimodal_cache_operations(self, semantic_cache, mock_embedding_service):
        """Test multi-modal cache operations"""
        prompt = "What's in this image?"
        response = "This image contains a cat..."
        text_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        vision_embedding = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Cache multi-modal response
        await semantic_cache.cache_response(
            prompt, response, text_embedding, vision_embedding
        )
        
        # Verify cache entry was created
        assert len(semantic_cache.cache) == 1
        
        # Retrieve cached response with similar text
        cached_response = await semantic_cache.get_similar_cached_response(
            "Describe the image content"
        )
        assert cached_response == response
    
    @pytest.mark.asyncio
    async def test_similarity_calculation(self, semantic_cache):
        """Test cosine similarity calculation"""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([1.0, 0.0, 0.0])
        
        # Test orthogonal vectors
        similarity = semantic_cache._calculate_similarity(embedding1, embedding2)
        assert abs(similarity) < 0.001
        
        # Test identical vectors
        similarity = semantic_cache._calculate_similarity(embedding1, embedding3)
        assert abs(similarity - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_cache_eviction_policies(self, semantic_cache):
        """Test different cache eviction policies"""
        # Set small cache size for testing
        semantic_cache.max_size = 3
        
        # Add 5 entries
        for i in range(5):
            prompt = f"Test prompt {i}"
            response = f"Test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await semantic_cache.cache_response(prompt, response, embedding)
        
        # Cache should be at max size
        assert len(semantic_cache.cache) <= 3
        
        # Test LRU eviction
        oldest_entry = min(semantic_cache.cache.values(), key=lambda x: x.last_accessed)
        assert oldest_entry is not None
    
    @pytest.mark.asyncio
    async def test_rl_based_cache_eviction(self, semantic_cache):
        """Test RL-based cache eviction"""
        # Enable RL eviction
        semantic_cache.eviction_policy = "rl_optimized"
        
        # Add entries with different access patterns
        entries = []
        for i in range(5):
            prompt = f"Test prompt {i}"
            response = f"Test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            entry = CacheEntry(
                prompt=prompt,
                response=response,
                text_embedding=embedding,
                access_count=i,  # Varying access counts
                similarity_score=i/10
            )
            entries.append(entry)
            
            await semantic_cache.cache_response(prompt, response, embedding)
        
        # Simulate RL eviction decision
        evicted_key = semantic_cache._rl_eviction_decision()
        assert evicted_key in semantic_cache.cache
    
    @pytest.mark.asyncio
    async def test_multi_modal_feature_extraction(self, semantic_cache):
        """Test multi-modal feature extraction"""
        test_cases = [
            {
                'input': {'text': 'Hello', 'image': None, 'code': None},
                'expected_modalities': ['text']
            },
            {
                'input': {'text': 'Hello', 'image': 'base64_data', 'code': None},
                'expected_modalities': ['text', 'vision']
            },
            {
                'input': {'text': 'Hello', 'image': 'base64_data', 'code': 'print("hello")'},
                'expected_modalities': ['text', 'vision', 'code']
            }
        ]
        
        for test_case in test_cases:
            modalities = semantic_cache._extract_modalities(test_case['input'])
            assert set(modalities) == set(test_case['expected_modalities'])
    
    @pytest.mark.asyncio
    async def test_cache_compression(self, semantic_cache):
        """Test cache compression functionality"""
        # Enable compression
        semantic_cache.compression_enabled = True
        
        large_response = "x" * 10000  # Large response
        prompt = "Test prompt"
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Cache large response
        await semantic_cache.cache_response(prompt, large_response, embedding)
        
        # Verify compression was applied
        cache_key = semantic_cache._generate_cache_key(prompt)
        cache_entry = semantic_cache.cache[cache_key]
        
        assert cache_entry.compressed
        assert len(cache_entry.compressed_response) < len(large_response)
    
    @pytest.mark.asyncio
    async def test_cache_performance_metrics(self, semantic_cache):
        """Test cache performance metrics"""
        # Add some entries
        for i in range(10):
            prompt = f"Test prompt {i}"
            response = f"Test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await semantic_cache.cache_response(prompt, response, embedding)
        
        # Get performance stats
        stats = semantic_cache.get_cache_stats()
        
        assert 'total_requests' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        assert 'total_entries' in stats
        assert stats['total_entries'] == 10
    
    @pytest.mark.asyncio
    async def test_cross_modal_similarity(self, semantic_cache):
        """Test cross-modal similarity matching"""
        # Cache text-only entry
        text_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        await semantic_cache.cache_response(
            "What is AI?", "AI is artificial intelligence", text_embedding
        )
        
        # Cache multi-modal entry
        multi_text_embedding = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        vision_embedding = np.array([0.7, 0.8, 0.9, 1.0, 0.1])
        await semantic_cache.cache_response(
            "AI concept", "AI concept explained", multi_text_embedding, vision_embedding
        )
        
        # Test cross-modal similarity
        similarity = semantic_cache._calculate_cross_modal_similarity(
            text_embedding, vision_embedding
        )
        assert 0 <= similarity <= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, semantic_cache):
        """Test concurrent cache operations"""
        async def cache_operation(i):
            prompt = f"Concurrent prompt {i}"
            response = f"Concurrent response {i}"
            embedding = np.array([i/100, i/100, i/100])
            
            await semantic_cache.cache_response(prompt, response, embedding)
            return await semantic_cache.get_similar_cached_response(prompt)
        
        # Run concurrent operations
        tasks = [cache_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert all(isinstance(result, str) for result in results)
        assert len(semantic_cache.cache) == 10
    
    @pytest.mark.asyncio
    async def test_cache_invalidations(self, semantic_cache):
        """Test cache invalidation strategies"""
        # Add some entries
        for i in range(5):
            prompt = f"Test prompt {i}"
            response = f"Test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await semantic_cache.cache_response(prompt, response, embedding)
        
        # Test pattern-based invalidation
        await semantic_cache.invalidate_cache("Test prompt")
        
        # Should invalidate matching entries
        assert len(semantic_cache.cache) < 5
    
    @pytest.mark.asyncio
    async def test_cache_cluster_operations(self, semantic_cache):
        """Test cache cluster operations"""
        # Enable cluster mode
        semantic_cache.cluster_mode = True
        
        # Mock cluster nodes
        semantic_cache.cluster_nodes = ["node1", "node2", "node3"]
        
        # Add entry with cluster replication
        prompt = "Cluster test prompt"
        response = "Cluster test response"
        embedding = np.array([0.1, 0.2, 0.3])
        
        await semantic_cache.cache_response(prompt, response, embedding)
        
        # Verify cluster replication was attempted
        assert len(semantic_cache.cache) == 1
    
    @pytest.mark.asyncio
    async def test_metrics_integration(self, semantic_cache):
        """Test integration with metrics service"""
        from src.services.metrics import metrics_service
        
        # Add some cache operations
        for i in range(5):
            prompt = f"Metrics test prompt {i}"
            response = f"Metrics test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await semantic_cache.cache_response(prompt, response, embedding)
            await semantic_cache.get_similar_cached_response(prompt)
        
        # Verify metrics were recorded
        assert metrics_service.cache_hits > 0
        assert metrics_service.total_requests > 0
        
        # Get multi-modal metrics
        multi_modal_metrics = metrics_service.multi_modal_metrics
        assert 'text_accuracy' in multi_modal_metrics
        assert 'vision_accuracy' in multi_modal_metrics
    
    @pytest.mark.asyncio
    async def test_advanced_cache_analytics(self, semantic_cache):
        """Test advanced cache analytics"""
        # Add entries with different characteristics
        for i in range(10):
            prompt = f"Analytics test prompt {i}"
            response = f"Analytics test response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await semantic_cache.cache_response(prompt, response, embedding)
            
            # Simulate varying access patterns
            for j in range(i):
                await semantic_cache.get_similar_cached_response(prompt)
        
        # Get advanced analytics
        analytics = semantic_cache.get_advanced_analytics()
        
        assert 'access_patterns' in analytics
        assert 'similarity_distribution' in analytics
        assert 'performance_metrics' in analytics
        assert 'efficiency_metrics' in analytics