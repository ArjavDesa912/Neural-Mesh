from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_cluster_mode: bool = False
    redis_cluster_nodes: List[str] = []
    
    # Database
    database_url: Optional[str] = None
    
    # Application Settings
    log_level: str = "INFO"
    max_workers: int = 10
    cache_ttl: int = 3600
    similarity_threshold: float = 0.85
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Multi-Modal Configuration
    enable_multi_modal: bool = True
    vision_model_name: str = "openai/clip-vit-base-patch32"
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_image_size: int = 1024
    supported_modalities: List[str] = ["text", "image", "code"]
    
    # Reinforcement Learning Configuration
    enable_rl_optimization: bool = True
    rl_learning_rate: float = 0.01
    rl_discount_factor: float = 0.95
    rl_exploration_rate: float = 0.1
    rl_exploration_decay: float = 0.995
    rl_reward_threshold: float = 0.8
    rl_model_path: str = "models/rl_router.pkl"
    
    # Advanced Cache Configuration
    enable_semantic_cache: bool = True
    cache_max_size: int = 10000
    cache_eviction_policy: str = "rl_optimized"  # "lru", "lfu", "rl_optimized"
    cache_compression_enabled: bool = True
    cache_cluster_replication: bool = True
    cache_regions: List[str] = ["us-east-1", "us-west-1", "eu-west-1"]
    
    # Provider Configuration
    default_model: str = "gpt-5"
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Advanced Provider Settings
    provider_weights: Dict[str, float] = {
        "openai": 0.3,
        "anthropic": 0.3,
        "google": 0.25,
        "cohere": 0.15
    }
    
    provider_timeout_settings: Dict[str, int] = {
        "openai": 30,
        "anthropic": 45,
        "google": 60,
        "cohere": 40
    }
    
    # Monitoring and Observability
    prometheus_port: int = 8000
    grafana_url: str = "http://localhost:3000"
    enable_advanced_metrics: bool = True
    metrics_retention_hours: int = 168  # 7 days
    enable_distributed_tracing: bool = True
    tracing_sample_rate: float = 0.1
    
    # Alert Configuration
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = []
    alert_severity_thresholds: Dict[str, float] = {
        "info": 0.7,
        "warning": 0.5,
        "critical": 0.3
    }
    
    # Performance Configuration
    target_p95_latency: float = 2.0
    target_cache_hit_rate: float = 85.0
    target_error_rate: float = 1.0
    target_system_health: float = 0.8
    max_concurrent_requests: int = 1000
    request_timeout_seconds: int = 120
    
    # Scaling Configuration
    min_replicas: int = 3
    max_replicas: int = 20
    cpu_threshold: int = 70
    memory_threshold: int = 80
    enable_predictive_scaling: bool = True
    scaling_lookback_minutes: int = 30
    scaling_prediction_window: int = 15
    
    # Multi-Region Configuration
    enable_multi_region: bool = True
    primary_region: str = "us-east-1"
    secondary_regions: List[str] = ["us-west-1", "eu-west-1"]
    region_failover_enabled: bool = True
    cross_region_latency_threshold: float = 1.0
    
    # Security Configuration
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    max_request_size_mb: int = 10
    
    # Advanced Features
    enable_batch_processing: bool = True
    max_batch_size: int = 50
    batch_timeout_seconds: int = 5
    enable_streaming: bool = True
    enable_request_deduplication: bool = True
    
    # Cost Optimization
    enable_cost_optimization: bool = True
    cost_budget_monthly: float = 1000.0
    cost_alert_threshold: float = 0.8
    enable_provider_cost_tracking: bool = True
    
    # Model Configuration
    model_configs: Dict[str, Dict[str, Any]] = {
        "gpt-5": {
            "max_tokens": 200000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.04
        },
        "claude-4": {
            "max_tokens": 200000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075
        },
        "gemini-ultra": {
            "max_tokens": 32000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 0.00125,
            "cost_per_1k_output": 0.005
        },
        "command-r-plus": {
            "max_tokens": 128000,
            "supports_vision": False,
            "supports_streaming": True,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015
        }
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_nested_delimiter = "__"

# Global settings instance
settings = Settings()
