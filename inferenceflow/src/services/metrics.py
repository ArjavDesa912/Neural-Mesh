from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any

# Define Prometheus metrics
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds', ['provider'])
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
PROVIDER_ERRORS = Counter('provider_errors_total', 'Total errors from providers', ['provider', 'error_type'])
COST_PER_REQUEST = Gauge('cost_per_request_usd', 'Cost per request in USD', ['provider'])
QUEUE_DEPTH = Gauge('queue_depth', 'Current queue depth')
THROUGHPUT = Counter('throughput_total', 'Total requests processed')

class Metrics:
    def record_request_latency(self, latency: float, provider: str) -> None:
        REQUEST_LATENCY.labels(provider).observe(latency)

    def increment_cache_hit(self) -> None:
        CACHE_HITS.inc()

    def increment_cache_miss(self) -> None:
        CACHE_MISSES.inc()

    def record_provider_error(self, provider: str, error_type: str) -> None:
        PROVIDER_ERRORS.labels(provider, error_type).inc()

    def record_cost_per_request(self, cost: float, provider: str) -> None:
        COST_PER_REQUEST.labels(provider).set(cost)

    def set_queue_depth(self, depth: int) -> None:
        QUEUE_DEPTH.set(depth)

    def increment_throughput(self) -> None:
        THROUGHPUT.inc()

    def get_performance_summary(self) -> Dict[str, Any]:
        # This would typically involve querying Prometheus, but for now, we'll return dummy data
        return {
            "request_latency_p95": 0.0, # Placeholder
            "cache_hit_rate": 0.0, # Placeholder
            "error_rate": 0.0, # Placeholder
        }

metrics = Metrics()
