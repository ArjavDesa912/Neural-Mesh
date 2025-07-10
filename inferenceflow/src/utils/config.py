from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    GOOGLE_API_KEY: str
    COHERE_API_KEY: str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    DATABASE_URL: str = "postgresql://user:pass@localhost/inferenceflow"
    LOG_LEVEL: str = "INFO"
    MAX_WORKERS: int = 10
    CACHE_TTL: int = 3600
    SIMILARITY_THRESHOLD: float = 0.85
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    PROMETHEUS_PORT: int = 8000
    GRAFANA_URL: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

settings = Settings()
