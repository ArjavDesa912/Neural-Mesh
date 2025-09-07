from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from datetime import datetime, timedelta
from src.core.cache import cache
from src.models.providers import providers, LLMProvider
from src.services.metrics import metrics
from src.core.scaler import scaler
from src.utils.logger import logger
from src.utils.config import settings
from src.services.embedding import embedding_service
import random
import uuid
import time
import asyncio
from collections import deque
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from src.models.providers import BaseProvider, ProviderManager, ProviderStatus
from src.utils.config import settings
from src.utils.logger import LoggerMixin
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import json

class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"

@dataclass
class RoutingDecision:
    provider: BaseProvider
    confidence: float
    reasoning: str
    request_type: RequestType
    estimated_latency: float
    estimated_cost: float
    rl_score: float = 0.0
    multi_modal_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLState:
    provider_name: str
    request_type: RequestType
    prompt_complexity: float
    estimated_cost: float
    historical_latency: float
    error_rate: float

@dataclass
class RLAction:
    provider: BaseProvider
    model_selection: str
    confidence: float

class ReinforcementLearningRouter:
    """Reinforcement Learning-based provider selection and optimization"""
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.training_history: List[Dict[str, Any]] = []
        self.reward_history: deque = deque(maxlen=1000)
        
    def get_state_key(self, state: RLState) -> str:
        """Convert state to string key for Q-table"""
        return f"{state.provider_name}_{state.request_type.value}_{int(state.prompt_complexity*10)}_{int(state.estimated_cost*1000)}"
    
    def get_action_key(self, action: RLAction) -> str:
        """Convert action to string key for Q-table"""
        return f"{action.provider.__class__.__name__}_{action.model_selection}"
    
    def choose_action(self, state: RLState, available_actions: List[RLAction]) -> RLAction:
        """Choose action using epsilon-greedy strategy"""
        if random.random() < self.exploration_rate:
            # Exploration: choose random action
            return random.choice(available_actions)
        
        # Exploitation: choose best action
        state_key = self.get_state_key(state)
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            action_key = self.get_action_key(action)
            q_value = self.q_table[state_key][action_key]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action or available_actions[0]
    
    def update_q_value(self, state: RLState, action: RLAction, reward: float, next_state: RLState):
        """Update Q-value using Q-learning algorithm"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        # Get current Q-value
        current_q = self.q_table[state_key][action_key]
        
        # Get maximum Q-value for next state
        next_state_key = self.get_state_key(next_state)
        next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action_key] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        # Record training
        self.training_history.append({
            'timestamp': time.time(),
            'state': state_key,
            'action': action_key,
            'reward': reward,
            'q_value': new_q,
            'exploration_rate': self.exploration_rate
        })
    
    def calculate_reward(self, routing_decision: RoutingDecision, actual_latency: float, actual_cost: float, success: bool) -> float:
        """Calculate reward for reinforcement learning"""
        reward = 0.0
        
        # Success reward
        if success:
            reward += 10.0
        else:
            reward -= 20.0
        
        # Latency reward (lower is better)
        latency_penalty = max(0, (actual_latency - 2.0) * 2)  # Penalize > 2s latency
        reward -= latency_penalty
        
        # Cost reward (lower is better)
        cost_penalty = actual_cost * 10  # Penalize high cost
        reward -= cost_penalty
        
        # Confidence reward
        reward += routing_decision.confidence * 5.0
        
        # Cache efficiency bonus
        if hasattr(routing_decision, 'cache_hit') and routing_decision.cache_hit:
            reward += 5.0
        
        return reward
    
    def get_rl_stats(self) -> Dict[str, Any]:
        """Get reinforcement learning statistics"""
        return {
            'total_training_steps': len(self.training_history),
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'recent_rewards': list(self.reward_history)[-100:],  # Last 100 rewards
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }

class SmartRouter(LoggerMixin):
    """Intelligent request routing based on semantic similarity and reinforcement learning optimization"""
    
    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vision_embedding_model: Optional[SentenceTransformer] = None
        self.request_history: deque = deque(maxlen=1000)
        self.provider_latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        self.rl_router = ReinforcementLearningRouter()
        self.multi_modal_features_cache: Dict[str, Any] = {}
        
        # Request type patterns for classification
        self.request_patterns = {
            RequestType.CHAT: [
                "hello", "how are you", "what's up", "conversation", "chat"
            ],
            RequestType.COMPLETION: [
                "continue", "complete", "finish", "next", "following"
            ],
            RequestType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview", "tl;dr"
            ],
            RequestType.TRANSLATION: [
                "translate", "translation", "language", "convert"
            ],
            RequestType.CODE_GENERATION: [
                "code", "program", "function", "class", "algorithm", "python", "javascript"
            ],
            RequestType.ANALYSIS: [
                "analyze", "analysis", "examine", "evaluate", "assess"
            ]
        }
    
    async def initialize(self):
        """Initialize the router with embedding models and RL components"""
        try:
            # Initialize text embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vision embedding model for multi-modal support
            try:
                self.vision_embedding_model = SentenceTransformer('clip-ViT-B-32')
            except Exception:
                self.logger.warning("Vision embedding model not available, multi-modal features limited")
            
            # Initialize RL router
            self.rl_router = ReinforcementLearningRouter()
            
            self.logger.info("Router initialized with embedding models and RL optimization")
        except Exception as e:
            self.logger.error(f"Failed to initialize router: {str(e)}")
            raise
    
    async def route_request(self, request: InferenceRequest) -> RoutingDecision:
        """Route request to the optimal provider using RL optimization"""
        
        # Extract multi-modal features
        multi_modal_features = await self._extract_multi_modal_features(request)
        
        # Classify request type with enhanced analysis
        request_type = self._classify_request_type(request.prompt)
        
        # Get available providers
        available_providers = await self.provider_manager.get_healthy_providers()
        
        if not available_providers:
            raise Exception("No healthy providers available")
        
        # Apply circuit breaker checks
        available_providers = [
            p for p in available_providers 
            if not self._is_circuit_breaker_open(p)
        ]
        
        if not available_providers:
            # Reset circuit breakers and try again
            self._reset_circuit_breakers()
            available_providers = await self.provider_manager.get_healthy_providers()
        
        # Prepare RL state and actions
        rl_state = self._create_rl_state(request, request_type, available_providers)
        rl_actions = self._create_rl_actions(available_providers, request)
        
        # Use RL to select optimal provider
        optimal_action = self.rl_router.choose_action(rl_state, rl_actions)
        optimal_provider = optimal_action.provider
        
        # Calculate confidence and reasoning
        confidence, reasoning = self._calculate_routing_confidence(
            request, optimal_provider, request_type
        )
        
        # Estimate performance metrics
        estimated_latency = self._estimate_latency(optimal_provider)
        estimated_cost = optimal_provider.estimate_cost(
            request.prompt, request.max_tokens, optimal_action.model_selection if hasattr(optimal_action, 'model_selection') else None
        )
        
        # Calculate RL score
        rl_score = self._calculate_rl_score(rl_state, optimal_action)
        
        # Record routing decision
        self._record_routing_decision(request, optimal_provider, request_type)
        
        return RoutingDecision(
            provider=optimal_provider,
            confidence=confidence,
            reasoning=reasoning,
            request_type=request_type,
            estimated_latency=estimated_latency,
            estimated_cost=estimated_cost,
            rl_score=rl_score,
            multi_modal_features=multi_modal_features
        )
    
    def _classify_request_type(self, prompt: str) -> RequestType:
        """Classify request type based on prompt content"""
        prompt_lower = prompt.lower()
        
        # Calculate similarity scores for each request type
        type_scores = {}
        
        for req_type, patterns in self.request_patterns.items():
            max_similarity = 0.0
            for pattern in patterns:
                if pattern in prompt_lower:
                    max_similarity = max(max_similarity, 0.8)  # High similarity for keyword match
                else:
                    # Calculate semantic similarity if embedding model is available
                    if self.embedding_model:
                        try:
                            prompt_embedding = self.embedding_model.encode(prompt)
                            pattern_embedding = self.embedding_model.encode(pattern)
                            similarity = self._cosine_similarity(prompt_embedding, pattern_embedding)
                            max_similarity = max(max_similarity, similarity)
                        except Exception:
                            continue
            
            type_scores[req_type] = max_similarity
        
        # Return the request type with highest similarity
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.3:  # Threshold for classification
                return best_type[0]
        
        # Default to chat if no clear classification
        return RequestType.CHAT
    
    async def _select_optimal_provider(
        self, 
        request: InferenceRequest, 
        providers: List[BaseProvider], 
        request_type: RequestType
    ) -> BaseProvider:
        """Select the optimal provider based on multiple factors"""
        
        if len(providers) == 1:
            return providers[0]
        
        # Calculate scores for each provider
        provider_scores = []
        
        for provider in providers:
            score = await self._calculate_provider_score(
                provider, request, request_type
            )
            provider_scores.append((provider, score))
        
        # Sort by score (higher is better) and return the best
        provider_scores.sort(key=lambda x: x[1], reverse=True)
        return provider_scores[0][0]
    
    async def _calculate_provider_score(
        self, 
        provider: BaseProvider, 
        request: InferenceRequest, 
        request_type: RequestType
    ) -> float:
        """Calculate a score for a provider based on multiple factors"""
        
        score = 0.0
        
        # Health status (40% weight)
        if provider.status == ProviderStatus.HEALTHY:
            score += 0.4
        elif provider.status == ProviderStatus.DEGRADED:
            score += 0.2
        
        # Latency performance (30% weight)
        avg_latency = provider.total_latency / max(provider.total_requests, 1)
        if avg_latency > 0:
            # Lower latency = higher score
            latency_score = max(0, 1 - (avg_latency / 10))  # Normalize to 0-1
            score += latency_score * 0.3
        
        # Cost efficiency (20% weight)
        estimated_cost = provider.estimate_cost(request.prompt, request.max_tokens)
        # Lower cost = higher score (normalize to reasonable range)
        cost_score = max(0, 1 - (estimated_cost / 0.1))  # Assume $0.1 is high cost
        score += cost_score * 0.2
        
        # Request type compatibility (10% weight)
        type_compatibility = self._get_type_compatibility(provider, request_type)
        score += type_compatibility * 0.1
        
        return score
    
    def _get_type_compatibility(self, provider: BaseProvider, request_type: RequestType) -> float:
        """Get compatibility score between provider and request type"""
        
        # Simple compatibility matrix (could be more sophisticated)
        compatibility_matrix = {
            "OpenAIProvider": {
                RequestType.CHAT: 0.9,
                RequestType.COMPLETION: 0.8,
                RequestType.SUMMARIZATION: 0.7,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.9,
                RequestType.ANALYSIS: 0.8
            },
            "AnthropicProvider": {
                RequestType.CHAT: 0.9,
                RequestType.COMPLETION: 0.7,
                RequestType.SUMMARIZATION: 0.8,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.8,
                RequestType.ANALYSIS: 0.9
            },
            "GoogleProvider": {
                RequestType.CHAT: 0.7,
                RequestType.COMPLETION: 0.6,
                RequestType.SUMMARIZATION: 0.8,
                RequestType.TRANSLATION: 0.9,
                RequestType.CODE_GENERATION: 0.7,
                RequestType.ANALYSIS: 0.8
            },
            "CohereProvider": {
                RequestType.CHAT: 0.6,
                RequestType.COMPLETION: 0.8,
                RequestType.SUMMARIZATION: 0.7,
                RequestType.TRANSLATION: 0.6,
                RequestType.CODE_GENERATION: 0.6,
                RequestType.ANALYSIS: 0.7
            }
        }
        
        provider_name = provider.__class__.__name__
        return compatibility_matrix.get(provider_name, {}).get(request_type, 0.5)
    
    def _calculate_routing_confidence(
        self, 
        request: InferenceRequest, 
        provider: BaseProvider, 
        request_type: RequestType
    ) -> tuple[float, str]:
        """Calculate confidence score and reasoning for routing decision"""
        
        confidence = 0.0
        reasoning_parts = []
        
        # Provider health confidence
        if provider.status == ProviderStatus.HEALTHY:
            confidence += 0.4
            reasoning_parts.append("provider is healthy")
        elif provider.status == ProviderStatus.DEGRADED:
            confidence += 0.2
            reasoning_parts.append("provider is degraded but available")
        
        # Request type confidence
        type_compatibility = self._get_type_compatibility(provider, request_type)
        confidence += type_compatibility * 0.3
        reasoning_parts.append(f"good compatibility with {request_type.value} requests")
        
        # Historical performance confidence
        if provider.total_requests > 10:
            avg_latency = provider.total_latency / provider.total_requests
            if avg_latency < 2.0:  # Good historical performance
                confidence += 0.3
                reasoning_parts.append("good historical performance")
        
        reasoning = f"Selected {provider.__class__.__name__} because: {', '.join(reasoning_parts)}"
        
        return min(confidence, 1.0), reasoning
    
    def _estimate_latency(self, provider: BaseProvider) -> float:
        """Estimate expected latency for a provider"""
        if provider.total_requests > 0:
            return provider.total_latency / provider.total_requests
        return 2.0  # Default estimate
    
    def _record_routing_decision(
        self, 
        request: InferenceRequest, 
        provider: BaseProvider, 
        request_type: RequestType
    ):
        """Record routing decision for analytics"""
        decision = {
            "timestamp": time.time(),
            "provider": provider.__class__.__name__,
            "request_type": request_type.value,
            "prompt_length": len(request.prompt),
            "priority": request.priority.value
        }
        self.request_history.append(decision)
    
    def _is_circuit_breaker_open(self, provider: BaseProvider) -> bool:
        """Check if circuit breaker is open for a provider"""
        provider_name = provider.__class__.__name__
        
        if provider_name not in self.circuit_breaker_states:
            return False
        
        state = self.circuit_breaker_states[provider_name]
        
        # Check if circuit breaker is open
        if state.get("is_open", False):
            # Check if timeout has passed
            if time.time() - state.get("opened_at", 0) > settings.circuit_breaker_timeout:
                # Try to close circuit breaker
                state["is_open"] = False
                state["failure_count"] = 0
                return False
            return True
        
        return False
    
    def _reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        for state in self.circuit_breaker_states.values():
            state["is_open"] = False
            state["failure_count"] = 0
    
    def record_provider_failure(self, provider: BaseProvider):
        """Record a provider failure for circuit breaker"""
        provider_name = provider.__class__.__name__
        
        if provider_name not in self.circuit_breaker_states:
            self.circuit_breaker_states[provider_name] = {
                "failure_count": 0,
                "is_open": False,
                "opened_at": 0
            }
        
        state = self.circuit_breaker_states[provider_name]
        state["failure_count"] += 1
        
        # Open circuit breaker if failure threshold is reached
        if state["failure_count"] >= 5:  # Configurable threshold
            state["is_open"] = True
            state["opened_at"] = time.time()
            self.logger.warning(f"Circuit breaker opened for {provider_name}")
    
    def record_provider_success(self, provider: BaseProvider):
        """Record a provider success for circuit breaker"""
        provider_name = provider.__class__.__name__
        
        if provider_name in self.circuit_breaker_states:
            state = self.circuit_breaker_states[provider_name]
            state["failure_count"] = max(0, state["failure_count"] - 1)
            
            # Close circuit breaker if it was open
            if state["is_open"]:
                state["is_open"] = False
                self.logger.info(f"Circuit breaker closed for {provider_name}")
    
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
    
    async def _extract_multi_modal_features(self, request: InferenceRequest) -> Dict[str, Any]:
        """Extract multi-modal features from request"""
        features = {
            "text_complexity": self._calculate_text_complexity(request.prompt),
            "has_code": self._detect_code_content(request.prompt),
            "has_math": self._detect_math_content(request.prompt),
            "language": self._detect_language(request.prompt),
            "sentiment": self._analyze_sentiment(request.prompt),
            "urgency": self._detect_urgency(request.prompt)
        }
        
        # Cache features for reuse
        prompt_hash = hash(request.prompt)
        self.multi_modal_features_cache[prompt_hash] = features
        
        return features
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)"""
        if not text:
            return 0.0
        
        # Simple complexity metrics
        word_count = len(text.split())
        sentence_count = len(text.split('.')) + len(text.split('!')) + len(text.split('?'))
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        # Normalize complexity
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
        # Simple heuristic-based detection
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
    
    def _create_rl_state(self, request: InferenceRequest, request_type: RequestType, providers: List[BaseProvider]) -> RLState:
        """Create RL state from request and providers"""
        # Calculate average metrics across providers
        avg_latency = sum(p.total_latency / max(p.total_requests, 1) for p in providers) / len(providers)
        avg_error_rate = sum(p.error_count / max(p.total_requests, 1) for p in providers) / len(providers)
        
        return RLState(
            provider_name="multi_provider",
            request_type=request_type,
            prompt_complexity=self._calculate_text_complexity(request.prompt),
            estimated_cost=sum(p.estimate_cost(request.prompt, request.max_tokens) for p in providers) / len(providers),
            historical_latency=avg_latency,
            error_rate=avg_error_rate
        )
    
    def _create_rl_actions(self, providers: List[BaseProvider], request: InferenceRequest) -> List[RLAction]:
        """Create RL actions from available providers"""
        actions = []
        for provider in providers:
            for model in provider.get_available_models():
                confidence = self._calculate_provider_confidence(provider, request)
                actions.append(RLAction(
                    provider=provider,
                    model_selection=model,
                    confidence=confidence
                ))
        return actions
    
    def _calculate_provider_confidence(self, provider: BaseProvider, request: InferenceRequest) -> float:
        """Calculate confidence score for provider"""
        confidence = 0.0
        
        # Health confidence
        if provider.status == ProviderStatus.HEALTHY:
            confidence += 0.4
        elif provider.status == ProviderStatus.DEGRADED:
            confidence += 0.2
        
        # Performance confidence
        if provider.total_requests > 0:
            avg_latency = provider.total_latency / provider.total_requests
            error_rate = provider.error_count / provider.total_requests
            
            latency_score = max(0, 1 - (avg_latency / 5.0))  # Penalize > 5s latency
            error_score = max(0, 1 - (error_rate * 10))  # Penalize > 10% error rate
            
            confidence += (latency_score + error_score) * 0.3
        
        return min(1.0, confidence)
    
    def _calculate_rl_score(self, state: RLState, action: RLAction) -> float:
        """Calculate RL score for routing decision"""
        state_key = self.rl_router.get_state_key(state)
        action_key = self.rl_router.get_action_key(action)
        
        q_value = self.rl_router.q_table[state_key][action_key]
        
        # Normalize Q-value to 0-1 range for display
        return max(0.0, min(1.0, (q_value + 50) / 100))  # Assuming Q-values range -50 to 50
    
    def record_routing_outcome(self, decision: RoutingDecision, actual_latency: float, actual_cost: float, success: bool):
        """Record routing outcome for RL learning"""
        # Calculate reward
        reward = self.rl_router.calculate_reward(decision, actual_latency, actual_cost, success)
        
        # Update RL model
        state = self._create_rl_state_from_decision(decision)
        action = self._create_rl_action_from_decision(decision)
        next_state = self._create_rl_state_from_decision(decision)  # Simplified for now
        
        self.rl_router.update_q_value(state, action, reward, next_state)
        self.rl_router.reward_history.append(reward)
        
        self.logger.info(f"Recorded routing outcome: reward={reward:.2f}, success={success}")
    
    def _create_rl_state_from_decision(self, decision: RoutingDecision) -> RLState:
        """Create RL state from routing decision"""
        return RLState(
            provider_name=decision.provider.__class__.__name__,
            request_type=decision.request_type,
            prompt_complexity=decision.multi_modal_features.get('text_complexity', 0.5),
            estimated_cost=decision.estimated_cost,
            historical_latency=decision.estimated_latency,
            error_rate=0.0  # Would need to be tracked separately
        )
    
    def _create_rl_action_from_decision(self, decision: RoutingDecision) -> RLAction:
        """Create RL action from routing decision"""
        return RLAction(
            provider=decision.provider,
            model_selection=decision.provider.get_available_models()[0],  # Default to first model
            confidence=decision.confidence
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and analytics"""
        stats = {
            "total_routing_decisions": len(self.request_history),
            "circuit_breaker_states": {},
            "request_type_distribution": defaultdict(int),
            "provider_distribution": defaultdict(int),
            "reinforcement_learning": self.rl_router.get_rl_stats(),
            "multi_modal_features": {
                "cached_features": len(self.multi_modal_features_cache),
                "feature_types": ["text_complexity", "has_code", "has_math", "language", "sentiment", "urgency"]
            }
        }
        
        # Circuit breaker states
        for provider_name, state in self.circuit_breaker_states.items():
            stats["circuit_breaker_states"][provider_name] = {
                "is_open": state.get("is_open", False),
                "failure_count": state.get("failure_count", 0)
            }
        
        # Request type and provider distribution
        for decision in self.request_history:
            stats["request_type_distribution"][decision["request_type"]] += 1
            stats["provider_distribution"][decision["provider"]] += 1
        
        return stats

# Global router instance (will be initialized in orchestrator)
smart_router: Optional[SmartRouter] = None
