import time
import psutil
from typing import Dict, Any, List
from collections import defaultdict, deque
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST
)
from src.utils.logger import LoggerMixin
from src.utils.config import settings

class MetricsService(LoggerMixin):
    """Comprehensive metrics collection and monitoring service"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_counter = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['provider', 'status', 'cache_hit']
        )
        
        self.request_latency = Histogram(
            'inference_request_duration_seconds',
            'Inference request latency in seconds',
            ['provider', 'status'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate_percent',
            'Cache hit rate percentage'
        )
        
        self.provider_health = Gauge(
            'provider_health_status',
            'Provider health status (0=unhealthy, 1=degraded, 2=healthy)',
            ['provider']
        )
        
        self.cost_per_request = Summary(
            'cost_per_request_usd',
            'Cost per request in USD',
            ['provider']
        )
        
        self.tokens_used = Counter(
            'tokens_used_total',
            'Total tokens used',
            ['provider', 'type']  # type: input/output
        )
        
        self.error_counter = Counter(
            'inference_errors_total',
            'Total number of errors',
            ['provider', 'error_type']
        )
        
        self.queue_depth = Gauge(
            'request_queue_depth',
            'Current request queue depth'
        )
        
        self.active_requests = Gauge(
            'active_requests',
            'Number of currently active requests'
        )
        
        # Internal metrics storage
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cost_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
        }
    
    async def initialize(self):
        """Initialize the metrics service"""
        try:
            self.logger.info("Initializing metrics service")
            # Start system metrics collection
            await self._update_system_metrics()
            self.logger.info("Metrics service initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics service: {str(e)}")
            raise
    
    async def record_request_latency(self, latency: float, provider: str, status: str = "success"):
        """Record request latency"""
        try:
            # Update Prometheus histogram
            self.request_latency.labels(provider=provider, status=status).observe(latency)
            
            # Store in history for analytics
            self.latency_history[provider].append(latency)
            
            self.logger.debug(f"Recorded latency: {latency:.3f}s for {provider}")
            
        except Exception as e:
            self.logger.error(f"Error recording latency: {str(e)}")
    
    async def record_cache_hit(self):
        """Record a cache hit"""
        try:
            self.cache_hits += 1
            self.total_requests += 1
            
            # Update Prometheus metrics
            self.request_counter.labels(
                provider="cache", 
                status="success", 
                cache_hit="true"
            ).inc()
            
            # Update cache hit rate
            hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            self.cache_hit_rate.set(hit_rate)
            
        except Exception as e:
            self.logger.error(f"Error recording cache hit: {str(e)}")
    
    async def record_cache_miss(self):
        """Record a cache miss"""
        try:
            self.cache_misses += 1
            self.total_requests += 1
            
            # Update Prometheus metrics
            self.request_counter.labels(
                provider="cache", 
                status="success", 
                cache_hit="false"
            ).inc()
            
            # Update cache hit rate
            hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            self.cache_hit_rate.set(hit_rate)
            
        except Exception as e:
            self.logger.error(f"Error recording cache miss: {str(e)}")
    
    async def record_provider_success(self, provider: str):
        """Record a successful provider request"""
        try:
            self.total_requests += 1
            
            # Update Prometheus metrics
            self.request_counter.labels(
                provider=provider, 
                status="success", 
                cache_hit="false"
            ).inc()
            
            # Update provider health
            self.provider_health.labels(provider=provider).set(2)  # Healthy
            
        except Exception as e:
            self.logger.error(f"Error recording provider success: {str(e)}")
    
    async def record_provider_error(self, provider: str, error_type: str):
        """Record a provider error"""
        try:
            # Update Prometheus metrics
            self.request_counter.labels(
                provider=provider, 
                status="error", 
                cache_hit="false"
            ).inc()
            
            self.error_counter.labels(provider=provider, error_type=error_type).inc()
            
            # Update provider health
            self.provider_health.labels(provider=provider).set(0)  # Unhealthy
            
            # Store error in history
            self.error_history[provider].append({
                'timestamp': time.time(),
                'error_type': error_type
            })
            
        except Exception as e:
            self.logger.error(f"Error recording provider error: {str(e)}")
    
    async def record_cost(self, provider: str, cost: float):
        """Record cost for a request"""
        try:
            # Update Prometheus summary
            self.cost_per_request.labels(provider=provider).observe(cost)
            
            # Store in history
            self.cost_history[provider].append(cost)
            
        except Exception as e:
            self.logger.error(f"Error recording cost: {str(e)}")
    
    async def record_tokens(self, provider: str, input_tokens: int, output_tokens: int):
        """Record token usage"""
        try:
            if input_tokens > 0:
                self.tokens_used.labels(provider=provider, type="input").inc(input_tokens)
            
            if output_tokens > 0:
                self.tokens_used.labels(provider=provider, type="output").inc(output_tokens)
                
        except Exception as e:
            self.logger.error(f"Error recording tokens: {str(e)}")
    
    async def update_queue_depth(self, depth: int):
        """Update current queue depth"""
        try:
            self.queue_depth.set(depth)
        except Exception as e:
            self.logger.error(f"Error updating queue depth: {str(e)}")
    
    async def update_active_requests(self, count: int):
        """Update number of active requests"""
        try:
            self.active_requests.set(count)
        except Exception as e:
            self.logger.error(f"Error updating active requests: {str(e)}")
    
    async def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU usage
            self.system_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics['disk_usage'] = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            self.system_metrics['network_io'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {str(e)}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            await self._update_system_metrics()
            
            # Calculate latency percentiles
            latency_percentiles = {}
            for provider, latencies in self.latency_history.items():
                if latencies:
                    sorted_latencies = sorted(latencies)
                    latency_percentiles[provider] = {
                        'p50': sorted_latencies[int(len(sorted_latencies) * 0.5)],
                        'p95': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                        'p99': sorted_latencies[int(len(sorted_latencies) * 0.99)],
                        'avg': sum(sorted_latencies) / len(sorted_latencies)
                    }
            
            # Calculate cost statistics
            cost_stats = {}
            for provider, costs in self.cost_history.items():
                if costs:
                    cost_stats[provider] = {
                        'avg_cost': sum(costs) / len(costs),
                        'total_cost': sum(costs),
                        'min_cost': min(costs),
                        'max_cost': max(costs)
                    }
            
            # Calculate error rates
            error_rates = {}
            for provider, errors in self.error_history.items():
                if errors:
                    recent_errors = [e for e in errors if time.time() - e['timestamp'] < 3600]  # Last hour
                    error_rates[provider] = len(recent_errors)
            
            # Calculate throughput
            uptime = time.time() - self.start_time
            throughput = self.total_requests / max(uptime, 1)
            
            return {
                'total_requests': self.total_requests,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': (self.cache_hits / max(self.total_requests, 1)) * 100,
                'throughput_rps': throughput,
                'uptime_seconds': uptime,
                'latency_percentiles': latency_percentiles,
                'cost_statistics': cost_stats,
                'error_rates': error_rates,
                'system_metrics': self.system_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            return generate_latest()
        except Exception as e:
            self.logger.error(f"Error generating Prometheus metrics: {str(e)}")
            return ""
    
    async def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        try:
            # Reset internal counters
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.start_time = time.time()
            
            # Clear histories
            self.latency_history.clear()
            self.cost_history.clear()
            self.error_history.clear()
            
            # Reset system metrics
            self.system_metrics = {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
            }
            
            self.logger.info("Metrics reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {str(e)}")

# Global metrics service instance
metrics_service = MetricsService()
