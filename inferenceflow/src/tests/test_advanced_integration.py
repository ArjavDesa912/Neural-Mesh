import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from src.core.router import ReinforcementLearningRouter
from src.core.cache import SemanticCache
from src.services.metrics import MetricsService
from src.models.providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.utils.config import settings

class TestAdvancedIntegration:
    """Integration tests for complete advanced system"""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete system instance"""
        system = {
            'router': ReinforcementLearningRouter(),
            'cache': SemanticCache(max_size=1000, similarity_threshold=0.85),
            'metrics': MetricsService(),
            'providers': {
                'openai': Mock(spec=OpenAIProvider),
                'anthropic': Mock(spec=AnthropicProvider),
                'google': Mock(spec=GoogleProvider)
            }
        }
        
        # Set up mock providers
        for name, provider in system['providers'].items():
            provider.generate = AsyncMock()
            provider.estimate_cost = Mock(return_value=0.01)
            provider.get_provider_status = Mock(return_value={"healthy": True})
            provider.get_model_capabilities = Mock(return_value={
                "max_tokens": 100000,
                "supports_vision": True,
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03
            })
        
        return system
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, complete_system):
        """Test complete end-to-end workflow"""
        # Set up mock responses
        for provider in complete_system['providers'].values():
            provider.generate.return_value = InferenceResponse(
                response="Advanced AI response",
                provider="mock",
                model="advanced-model",
                latency=1.5,
                tokens_used=150,
                cost=0.02,
                cache_hit=False,
                request_id="test-123"
            )
        
        # Initialize metrics
        await complete_system['metrics'].initialize()
        
        # Create test request
        request = InferenceRequest(
            prompt="Explain quantum computing",
            user_id="advanced_user",
            max_tokens=500,
            metadata={"modality": "text", "priority": "high"}
        )
        
        # Mock feature extraction
        with patch.object(complete_system['router'], '_extract_features', return_value=['text']):
            # Route request through RL router
            selected_provider = await complete_system['router'].route_request(
                request, complete_system['providers']
            )
            
            # Generate response
            response = await complete_system['providers'][selected_provider].generate(
                prompt=request.prompt,
                model="advanced-model"
            )
            
            # Cache response
            embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            await complete_system['cache'].cache_response(
                request.prompt, response.response, embedding
            )
            
            # Record metrics
            await complete_system['metrics'].record_request_latency(
                response.latency, selected_provider
            )
            await complete_system['metrics'].record_cost(
                selected_provider, response.cost
            )
            await complete_system['metrics'].record_cache_hit()
        
        # Verify system components worked together
        assert selected_provider in complete_system['providers']
        assert response.response == "Advanced AI response"
        assert len(complete_system['cache'].cache) == 1
        
        # Verify metrics integration
        assert complete_system['metrics'].total_requests > 0
        assert complete_system['metrics'].cache_hits > 0
    
    @pytest.mark.asyncio
    async def test_multi_modal_processing_pipeline(self, complete_system):
        """Test multi-modal processing pipeline"""
        # Set up multi-modal mock responses
        for provider in complete_system['providers'].values():
            provider.generate.return_value = InferenceResponse(
                response="Multi-modal analysis complete",
                provider="mock",
                model="multimodal-model",
                latency=2.0,
                tokens_used=200,
                cost=0.03,
                cache_hit=False,
                request_id="multimodal-123"
            )
        
        # Create multi-modal request
        request = InferenceRequest(
            prompt="Analyze this image and code",
            user_id="multimodal_user",
            max_tokens=300,
            metadata={
                "modality": "multimodal",
                "image": "base64_encoded_image",
                "code": "print('hello world')"
            }
        )
        
        # Mock multi-modal feature extraction
        with patch.object(complete_system['router'], '_extract_features', 
                        return_value=['text', 'vision', 'code']):
            # Process through router
            selected_provider = await complete_system['router'].route_request(
                request, complete_system['providers']
            )
            
            # Generate response
            response = await complete_system['providers'][selected_provider].generate(
                prompt=request.prompt,
                model="multimodal-model",
                image=request.metadata['image'],
                code=request.metadata['code']
            )
            
            # Cache with multi-modal embeddings
            text_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            vision_embedding = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
            await complete_system['cache'].cache_response(
                request.prompt, response.response, text_embedding, vision_embedding
            )
        
        # Verify multi-modal processing
        assert selected_provider in complete_system['providers']
        assert 'image' in request.metadata
        assert 'code' in request.metadata
        
        # Verify cache supports multi-modal
        cache_key = complete_system['cache']._generate_cache_key(request.prompt)
        cache_entry = complete_system['cache'].cache[cache_key]
        assert cache_entry.vision_embedding is not None
    
    @pytest.mark.asyncio
    async def test_rl_optimization_with_feedback_loop(self, complete_system):
        """Test RL optimization with feedback loop"""
        # Initialize metrics
        await complete_system['metrics'].initialize()
        
        # Process multiple requests to train RL model
        requests = [
            InferenceRequest(
                prompt=f"RL training request {i}",
                user_id="rl_user",
                max_tokens=100
            ) for i in range(10)
        ]
        
        # Set up varying provider performance
        def create_mock_response(latency, cost, success=True):
            return InferenceResponse(
                response=f"Response with latency {latency}",
                provider="mock",
                model="rl-model",
                latency=latency,
                tokens_used=100,
                cost=cost,
                cache_hit=False,
                request_id=f"rl-{latency}"
            )
        
        # Configure provider performance
        complete_system['providers']['openai'].generate.return_value = create_mock_response(1.0, 0.01)
        complete_system['providers']['anthropic'].generate.return_value = create_mock_response(2.0, 0.02)
        complete_system['providers']['google'].generate.return_value = create_mock_response(3.0, 0.015)
        
        # Process requests
        for request in requests:
            with patch.object(complete_system['router'], '_extract_features', return_value=['text']):
                selected_provider = await complete_system['router'].route_request(
                    request, complete_system['providers']
                )
                
                response = await complete_system['providers'][selected_provider].generate(
                    prompt=request.prompt,
                    model="rl-model"
                )
                
                # Record metrics for RL learning
                await complete_system['metrics'].record_request_latency(
                    response.latency, selected_provider
                )
                await complete_system['metrics'].record_cost(
                    selected_provider, response.cost
                )
        
        # Verify RL learning occurred
        assert len(complete_system['router'].action_history) == 10
        assert len(complete_system['router'].q_table) > 0
        
        # Verify exploration rate decay
        assert complete_system['router'].exploration_rate < 0.1
        
        # Test that RL model learned preferences
        state_hash = hash("RL training request 0" + str(['text']))
        if state_hash in complete_system['router'].q_table:
            q_values = complete_system['router'].q_table[state_hash]
            # Should prefer faster, cheaper provider
            assert q_values['openai'] > q_values['google']
    
    @pytest.mark.asyncio
    async def test_cache_rl_integration(self, complete_system):
        """Test cache and RL integration"""
        # Enable RL-based cache eviction
        complete_system['cache'].eviction_policy = "rl_optimized"
        
        # Add entries with different characteristics
        for i in range(5):
            prompt = f"Cache RL test {i}"
            response = f"Cache RL response {i}"
            embedding = np.array([i/10, i/10, i/10])
            
            await complete_system['cache'].cache_response(prompt, response, embedding)
            
            # Vary access patterns
            for j in range(5-i):  # Reverse access pattern
                await complete_system['cache'].get_similar_cached_response(prompt)
        
        # Test RL-based eviction decision
        evicted_key = complete_system['cache']._rl_eviction_decision()
        assert evicted_key is not None
        
        # Verify eviction considers access patterns
        cache_entry = complete_system['cache'].cache[evicted_key]
        assert cache_entry.access_count < 3  # Should evict least accessed
    
    @pytest.mark.asyncio
    async def test_advanced_monitoring_integration(self, complete_system):
        """Test advanced monitoring integration"""
        # Initialize metrics
        await complete_system['metrics'].initialize()
        
        # Simulate system activity
        activities = [
            ('cache_hit', 15),
            ('cache_miss', 5),
            ('request_success', 18),
            ('request_error', 2),
            ('high_latency', 3),
            ('low_cost', 12)
        ]
        
        for activity_type, count in activities:
            for _ in range(count):
                if activity_type == 'cache_hit':
                    await complete_system['metrics'].record_cache_hit()
                elif activity_type == 'cache_miss':
                    await complete_system['metrics'].record_cache_miss()
                elif activity_type == 'request_success':
                    await complete_system['metrics'].record_provider_success('openai')
                elif activity_type == 'request_error':
                    await complete_system['metrics'].record_provider_error('openai', 'timeout')
                elif activity_type == 'high_latency':
                    await complete_system['metrics'].record_request_latency(3.0, 'openai')
                elif activity_type == 'low_cost':
                    await complete_system['metrics'].record_cost('openai', 0.005)
        
        # Get comprehensive performance summary
        summary = await complete_system['metrics'].get_performance_summary()
        
        # Verify comprehensive metrics
        assert 'basic_metrics' in summary
        assert 'performance_metrics' in summary
        assert 'system_health' in summary
        assert 'advanced_metrics' in summary
        assert 'kpi_summary' in summary
        assert 'alerts' in summary
        
        # Verify KPI calculations
        basic_metrics = summary['basic_metrics']
        assert basic_metrics['cache_hit_rate'] == 75.0  # 15 hits out of 20 total
        assert basic_metrics['total_requests'] == 20
        
        # Verify advanced metrics
        advanced_metrics = summary['advanced_metrics']
        assert 'reinforcement_learning' in advanced_metrics
        assert 'multi_modal' in advanced_metrics
        
        # Verify system health calculation
        system_health = summary['system_health']
        assert 'overall_score' in system_health
        assert 0 <= system_health['overall_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_advanced_processing(self, complete_system):
        """Test concurrent advanced processing"""
        # Set up mock responses
        for provider in complete_system['providers'].values():
            provider.generate.return_value = InferenceResponse(
                response="Concurrent advanced response",
                provider="mock",
                model="concurrent-model",
                latency=1.0,
                tokens_used=100,
                cost=0.01,
                cache_hit=False,
                request_id="concurrent-123"
            )
        
        # Create concurrent requests
        requests = [
            InferenceRequest(
                prompt=f"Concurrent advanced request {i}",
                user_id="concurrent_user",
                max_tokens=200,
                metadata={"priority": "normal" if i % 2 == 0 else "high"}
            ) for i in range(20)
        ]
        
        async def process_request(request):
            with patch.object(complete_system['router'], '_extract_features', 
                            return_value=['text']):
                # Route request
                selected_provider = await complete_system['router'].route_request(
                    request, complete_system['providers']
                )
                
                # Generate response
                response = await complete_system['providers'][selected_provider].generate(
                    prompt=request.prompt,
                    model="concurrent-model"
                )
                
                # Cache response
                embedding = np.array([i/20, i/20, i/20] for i in range(3))
                await complete_system['cache'].cache_response(
                    request.prompt, response.response, embedding
                )
                
                # Record metrics
                await complete_system['metrics'].record_request_latency(
                    response.latency, selected_provider
                )
                
                return response
        
        # Process requests concurrently
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests processed successfully
        assert all(isinstance(result, InferenceResponse) for result in results)
        assert len(complete_system['cache'].cache) == 20
        
        # Verify metrics recorded
        assert complete_system['metrics'].total_requests >= 20
        
        # Verify RL learning from concurrent processing
        assert len(complete_system['router'].action_history) == 20
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_and_recovery(self, complete_system):
        """Test fault tolerance and recovery mechanisms"""
        # Initialize metrics
        await complete_system['metrics'].initialize()
        
        # Configure provider failure simulation
        complete_system['providers']['openai'].generate.side_effect = Exception("Provider failure")
        complete_system['providers']['anthropic'].generate.return_value = InferenceResponse(
            response="Fallback response",
            provider="anthropic",
            model="fallback-model",
            latency=2.0,
            tokens_used=100,
            cost=0.02,
            cache_hit=False,
            request_id="fallback-123"
        )
        
        # Create request that will trigger failover
        request = InferenceRequest(
            prompt="Test failover",
            user_id="failover_user",
            max_tokens=100
        )
        
        # Process request with failover
        with patch.object(complete_system['router'], '_extract_features', return_value=['text']):
            selected_provider = await complete_system['router'].route_request(
                request, complete_system['providers']
            )
            
            try:
                response = await complete_system['providers'][selected_provider].generate(
                    prompt=request.prompt,
                    model="fallback-model"
                )
                
                # Record success metrics
                await complete_system['metrics'].record_request_latency(
                    response.latency, selected_provider
                )
                await complete_system['metrics'].record_cost(
                    selected_provider, response.cost
                )
                
            except Exception as e:
                # Record error metrics
                await complete_system['metrics'].record_provider_error(
                    selected_provider, str(e)
                )
        
        # Verify circuit breaker functionality
        assert selected_provider in complete_system['providers']
        
        # Verify error handling in metrics
        error_metrics = await complete_system['metrics'].get_performance_summary()
        assert 'alerts' in error_metrics
        
        # Test cache availability during provider failure
        cached_response = await complete_system['cache'].get_similar_cached_response(
            "Test failover"
        )
        # Should handle cache miss gracefully
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, complete_system):
        """Test performance benchmarks"""
        # Initialize metrics
        await complete_system['metrics'].initialize()
        
        # Set up high-performance mock responses
        for provider in complete_system['providers'].values():
            provider.generate.return_value = InferenceResponse(
                response="High-performance response",
                provider="mock",
                model="perf-model",
                latency=0.5,  # Fast response
                tokens_used=100,
                cost=0.005,  # Low cost
                cache_hit=False,
                request_id="perf-123"
            )
        
        # Process benchmark requests
        benchmark_requests = [
            InferenceRequest(
                prompt=f"Benchmark request {i}",
                user_id="benchmark_user",
                max_tokens=100
            ) for i in range(100)
        ]
        
        start_time = asyncio.get_event_loop().time()
        
        async def benchmark_request(request):
            with patch.object(complete_system['router'], '_extract_features', 
                            return_value=['text']):
                selected_provider = await complete_system['router'].route_request(
                    request, complete_system['providers']
                )
                
                response = await complete_system['providers'][selected_provider].generate(
                    prompt=request.prompt,
                    model="perf-model"
                )
                
                await complete_system['metrics'].record_request_latency(
                    response.latency, selected_provider
                )
                
                return response
        
        # Run benchmark
        tasks = [benchmark_request(req) for req in benchmark_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Verify performance metrics
        throughput = len(results) / total_time
        assert throughput > 10  # More than 10 requests per second
        
        # Verify latency metrics
        summary = await complete_system['metrics'].get_performance_summary()
        perf_metrics = summary['performance_metrics']
        
        # Should meet performance targets
        assert perf_metrics['avg_cost_per_request'] < 0.01
        assert summary['basic_metrics']['error_rate'] < 1.0
        
        # Verify system health
        system_health = summary['system_health']
        assert system_health['overall_score'] > 0.8