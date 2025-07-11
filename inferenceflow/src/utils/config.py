from pydantic_settings import BaseSettings
from typing import Optional
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
    
    # Database
    database_url: Optional[str] = None
    
    # Application Settings
    log_level: str = "INFO"
    max_workers: int = 10
    cache_ttl: int = 3600
    similarity_threshold: float = 0.85
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Monitoring
    prometheus_port: int = 8000
    grafana_url: str = "http://localhost:3000"
    
    # JWT Settings
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Provider Settings
    default_model: str = "gpt-3.5-turbo"
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Scaling Settings
    min_replicas: int = 3
    max_replicas: int = 20
    cpu_threshold: int = 70
    memory_threshold: int = 80
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
