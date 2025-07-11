# Neural-Mesh InferenceFlow Implementation

This is a complete implementation of the Neural-Mesh InferenceFlow system as specified in the main README.md. The system provides intelligent AI inference orchestration with multi-provider support, semantic caching, and auto-scaling.

## 🚀 Features Implemented

### ✅ Phase 1: Core API Gateway and Basic Routing
- **FastAPI-based API Gateway** with rate limiting and JWT authentication
- **Smart Router** with semantic similarity clustering and load balancing
- **Circuit breaker pattern** for fault tolerance
- **Request validation** and sanitization

### ✅ Phase 2: Semantic Caching Implementation
- **Redis-based semantic cache** with similarity matching
- **Embedding-based similarity search** using sentence-transformers
- **Cache analytics** and hit rate tracking
- **Compression** for large responses

### ✅ Phase 3: Multi-Provider Integration
- **OpenAI Provider** (GPT-3.5-turbo, GPT-4)
- **Anthropic Provider** (Claude-3-sonnet, Claude-3-haiku)
- **Google Provider** (Gemini Pro)
- **Cohere Provider** (Command-R+)
- **Unified interface** with standardized request/response handling
- **Cost tracking** and latency monitoring

### ✅ Phase 4: Auto-Scaling and Monitoring
- **Predictive auto-scaling** based on traffic patterns
- **Comprehensive metrics** with Prometheus integration
- **System health monitoring**
- **Performance analytics**

### ✅ Phase 5: Advanced Analytics and Optimization
- **Structured logging** with JSON formatting
- **Performance benchmarking**
- **Cost optimization** algorithms
- **Real-time monitoring** dashboard

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│   Orchestrator  │───▶│   LLM Providers │
│   (FastAPI)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rate Limiting │    │  Smart Router   │    │  Semantic Cache │
│   (Redis)       │    │                 │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Authentication│    │ Auto Scaler     │    │   Metrics       │
│   (JWT)         │    │                 │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with your configuration:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Application Settings
LOG_LEVEL=INFO
CACHE_TTL=3600
SIMILARITY_THRESHOLD=0.85
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# JWT Settings
JWT_SECRET_KEY=your-secret-key-change-in-production
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or using docker-compose
docker-compose up redis -d
```

### 4. Run the Application

```bash
# Development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Using Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t neural-mesh .
docker run -p 8000:8000 --env-file .env neural-mesh
```

## 📡 API Endpoints

### Main Endpoints

- `POST /v1/inference` - Single inference request
- `POST /v1/batch` - Batch inference requests
- `GET /v1/health` - Health check
- `GET /v1/status` - System status
- `GET /v1/metrics` - Prometheus metrics
- `GET /v1/providers` - Provider information
- `POST /v1/cache/clear` - Clear cache (admin only)

### Example Request

```bash
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "user_id": "user123",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Example Response

```json
{
  "response": "The capital of France is Paris.",
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "latency": 1.234,
  "tokens_used": 15,
  "cost": 0.0003,
  "cache_hit": false,
  "request_id": "uuid-here",
  "created_at": "2024-01-01T12:00:00Z"
}
```

## 🔧 Configuration

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `CACHE_TTL` | 3600 | Cache time-to-live in seconds |
| `SIMILARITY_THRESHOLD` | 0.85 | Semantic similarity threshold |
| `RATE_LIMIT_REQUESTS` | 100 | Requests per rate limit window |
| `RATE_LIMIT_WINDOW` | 60 | Rate limit window in seconds |
| `MAX_WORKERS` | 10 | Maximum concurrent workers |
| `MIN_REPLICAS` | 3 | Minimum Kubernetes replicas |
| `MAX_REPLICAS` | 20 | Maximum Kubernetes replicas |

## 📊 Monitoring

### Prometheus Metrics

The application exposes comprehensive Prometheus metrics:

- `inference_requests_total` - Total requests by provider and status
- `inference_request_duration_seconds` - Request latency histogram
- `cache_hit_rate_percent` - Cache hit rate
- `provider_health_status` - Provider health status
- `cost_per_request_usd` - Cost per request
- `tokens_used_total` - Token usage

### Grafana Dashboard

A Grafana dashboard is provided in `monitoring/grafana-dashboard.json` for visualization.

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest src/tests/

# Run specific test file
pytest src/tests/test_basic.py

# Run with coverage
pytest --cov=src src/tests/
```

### Load Testing

```bash
# Basic load test
python -m pytest src/tests/test_load.py -v
```

## 🚀 Deployment

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Check deployment status
kubectl get pods -l app=inferenceflow
kubectl get services -l app=inferenceflow
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inferenceflow
```

## 🔍 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running and accessible
   - Check Redis host/port configuration

2. **Provider API Errors**
   - Verify API keys are correct
   - Check provider service status
   - Review rate limits

3. **High Latency**
   - Check cache hit rate
   - Monitor provider health
   - Review auto-scaling metrics

### Logs

Logs are written to both console and `logs/inferenceflow.log` with structured JSON formatting.

## 📈 Performance Benchmarks

The system is designed to meet these performance targets:

- **Latency**: P95 < 2 seconds
- **Throughput**: 10,000+ requests/second
- **Cache Hit Rate**: >85%
- **Availability**: 99.9%
- **Cost Efficiency**: 30% reduction vs direct API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for debugging information 