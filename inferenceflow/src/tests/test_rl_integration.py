import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from src.core.router import ReinforcementLearningRouter
from src.core.cache import SemanticCache
from src.services.metrics import MetricsService
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.utils.config import settings

class TestReinforcementLearningIntegration:
    """Integration tests for Reinforcement Learning features"""
    
    @pytest.fixture
    def rl_router(self):
        """Create RL router instance for testing"""
        return ReinforcementLearningRouter(
            learning_rate=0.01,
            discount_factor=0.95
        )
    
    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing"""
        providers = {
            'openai': Mock(),
            'anthropic': Mock(),
            'google': Mock()
        }
        
        # Set up mock responses
        for provider in providers.values():
            provider.generate = AsyncMock(return_value=InferenceResponse(
                response="Mock response",
                provider="mock",
                model="mock-model",
                latency=1.0,
                tokens_used=100,
                cost=0.01,
                cache_hit=False,
                request_id="test-123"
            ))
            provider.estimate_cost = Mock(return_value=0.01)
            provider.get_provider_status = Mock(return_value={"healthy": True})
        
        return providers
    
    @pytest.mark.asyncio
    async def test_rl_router_initialization(self, rl_router):
        """Test RL router initialization"""
        assert rl_router.learning_rate == 0.01
        assert rl_router.discount_factor == 0.95
        assert rl_router.exploration_rate == 0.1
        assert len(rl_router.q_table) == 0
        assert rl_router.action_history == []
    
    @pytest.mark.asyncio
    async def test_q_learning_update(self, rl_router):
        """Test Q-learning algorithm updates"""
        state = "test_state"
        action = "openai"
        reward = 1.0
        next_state = "next_state"
        
        # Initial Q-value should be 0
        assert rl_router.q_table[state][action] == 0.0
        
        # Update Q-value
        rl_router._update_q_value(state, action, reward, next_state)
        
        # Q-value should be updated
        assert rl_router.q_table[state][action] > 0.0
        
        # Verify Q-learning formula application
        expected_q = reward + rl_router.discount_factor * 0.0  # Assuming max_next_q is 0
        assert abs(rl_router.q_table[state][action] - expected_q) < 0.001
    
    @pytest.mark.asyncio
    async def test_provider_selection_with_exploration(self, rl_router):
        """Test provider selection with exploration"""
        # Set exploration rate to 1.0 for testing
        rl_router.exploration_rate = 1.0
        
        state = "test_state"
        available_providers = ['openai', 'anthropic', 'google']
        
        # With exploration rate 1.0, should select randomly
        selected_provider = rl_router._select_provider(state, available_providers)
        assert selected_provider in available_providers
        
        # Test multiple selections to ensure randomness
        selections = set()
        for _ in range(10):
            selected = rl_router._select_provider(state, available_providers)
            selections.add(selected)
        
        # Should have selected different providers due to exploration
        assert len(selections) > 1
    
    @pytest.mark.asyncio
    async def test_provider_selection_with_exploitation(self, rl_router):
        """Test provider selection with exploitation"""
        # Set exploration rate to 0.0 for testing
        rl_router.exploration_rate = 0.0
        
        state = "test_state"
        available_providers = ['openai', 'anthropic', 'google']
        
        # Set different Q-values
        rl_router.q_table[state]['openai'] = 0.5
        rl_router.q_table[state]['anthropic'] = 0.8
        rl_router.q_table[state]['google'] = 0.3
        
        # Should select provider with highest Q-value
        selected_provider = rl_router._select_provider(state, available_providers)
        assert selected_provider == 'anthropic'
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, rl_router):
        """Test multi-modal feature extraction"""
        test_cases = [
            {
                'input': {'text': 'Hello world', 'image': None, 'code': None},
                'expected_features': ['text']
            },
            {
                'input': {'text': 'Hello world', 'image': 'base64_image', 'code': None},
                'expected_features': ['text', 'vision']
            },
            {
                'input': {'text': 'Hello world', 'image': 'base64_image', 'code': 'print("hello")'},
                'expected_features': ['text', 'vision', 'code']
            }
        ]
        
        for test_case in test_cases:
            features = rl_router._extract_features(test_case['input'])
            assert all(feature in features for feature in test_case['expected_features'])
    
    @pytest.mark.asyncio
    async def test_reward_calculation(self, rl_router):
        """Test reward calculation for RL"""
        test_cases = [
            {
                'latency': 1.0,
                'cost': 0.01,
                'success': True,
                'expected_reward_range': (0.8, 1.0)
            },
            {
                'latency': 5.0,
                'cost': 0.05,
                'success': True,
                'expected_reward_range': (0.4, 0.6)
            },
            {
                'latency': 10.0,
                'cost': 0.1,
                'success': False,
                'expected_reward_range': (-1.0, -0.5)
            }
        ]
        
        for test_case in test_cases:
            reward = rl_router._calculate_reward(
                latency=test_case['latency'],
                cost=test_case['cost'],
                success=test_case['success']
            )
            
            min_reward, max_reward = test_case['expected_reward_range']
            assert min_reward <= reward <= max_reward
    
    @pytest.mark.asyncio
    async def test_complete_rl_workflow(self, rl_router, mock_providers):
        """Test complete RL workflow"""
        request = InferenceRequest(
            prompt="Test prompt",
            user_id="test_user",
            max_tokens=100
        )
        
        # Mock feature extraction
        with patch.object(rl_router, '_extract_features', return_value=['text']):
            # Route request
            provider = await rl_router.route_request(request, mock_providers)
            
            # Should return a valid provider
            assert provider in mock_providers
            
            # Verify Q-table was updated
            state_hash = hash("Test prompt" + str(['text']))
            assert state_hash in rl_router.q_table
    
    @pytest.mark.asyncio
    async def test_exploration_decay(self, rl_router):
        """Test exploration rate decay"""
        initial_rate = rl_router.exploration_rate
        
        # Decay multiple times
        for _ in range(10):
            rl_router._decay_exploration_rate()
        
        # Exploration rate should have decreased
        assert rl_router.exploration_rate < initial_rate
        assert rl_router.exploration_rate >= 0.01  # Minimum exploration rate
    
    @pytest.mark.asyncio
    async def test_action_history_tracking(self, rl_router, mock_providers):
        """Test action history tracking"""
        request = InferenceRequest(
            prompt="Test prompt",
            user_id="test_user",
            max_tokens=100
        )
        
        initial_history_length = len(rl_router.action_history)
        
        # Route request
        with patch.object(rl_router, '_extract_features', return_value=['text']):
            await rl_router.route_request(request, mock_providers)
        
        # History should have increased
        assert len(rl_router.action_history) == initial_history_length + 1
        
        # Verify history entry structure
        latest_action = rl_router.action_history[-1]
        assert 'state' in latest_action
        assert 'action' in latest_action
        assert 'reward' in latest_action
        assert 'timestamp' in latest_action
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, rl_router, mock_providers):
        """Test handling of concurrent requests"""
        requests = [
            InferenceRequest(prompt=f"Test prompt {i}", user_id="test_user", max_tokens=100)
            for i in range(5)
        ]
        
        # Mock feature extraction
        with patch.object(rl_router, '_extract_features', return_value=['text']):
            # Route requests concurrently
            tasks = [rl_router.route_request(req, mock_providers) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should be handled successfully
        assert all(isinstance(result, str) for result in results)
        assert all(result in mock_providers for result in results)
    
    @pytest.mark.asyncio
    async def test_rl_model_persistence(self, rl_router, tmp_path):
        """Test RL model persistence"""
        import pickle
        import os
        
        # Set up some Q-values
        rl_router.q_table['test_state']['openai'] = 0.5
        rl_router.q_table['test_state']['anthropic'] = 0.8
        
        # Save model
        model_path = tmp_path / "test_rl_model.pkl"
        rl_router.save_model(str(model_path))
        
        # Verify file was created
        assert os.path.exists(model_path)
        
        # Load model into new instance
        new_router = ReinforcementLearningRouter()
        new_router.load_model(str(model_path))
        
        # Verify Q-values were loaded
        assert new_router.q_table['test_state']['openai'] == 0.5
        assert new_router.q_table['test_state']['anthropic'] == 0.8
    
    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self, rl_router, mock_providers):
        """Test integration with metrics service"""
        from src.services.metrics import metrics_service
        
        request = InferenceRequest(
            prompt="Test prompt",
            user_id="test_user",
            max_tokens=100
        )
        
        # Mock feature extraction
        with patch.object(rl_router, '_extract_features', return_value=['text']):
            # Route request
            await rl_router.route_request(request, mock_providers)
        
        # Verify metrics were recorded
        assert metrics_service.total_requests > 0
        assert 'model_accuracy' in metrics_service.rl_metrics