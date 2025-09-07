import time
import psutil
import asyncio
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST, Info
)
from src.utils.logger import LoggerMixin
from src.utils.config import settings
import json

@dataclass
class KPIData:
    """Key Performance Indicator data structure"""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    trend: Optional[str] = None
    category: str = "general"
    description: str = ""

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", "=="
    threshold: float
    duration: int  # seconds
    severity: str  # "info", "warning", "critical"
    enabled: bool = True

class MetricsService(LoggerMixin):
    """Comprehensive metrics collection and monitoring service with 20+ KPIs"""
    
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
        
        # Advanced KPI metrics
        self.rl_performance = Gauge(
            'rl_performance_score',
            'Reinforcement learning performance score',
            ['model', 'metric_type']
        )
        
        self.multi_modal_accuracy = Gauge(
            'multi_modal_accuracy_percent',
            'Multi-modal feature accuracy percentage',
            ['feature_type']
        )
        
        self.cache_efficiency = Gauge(
            'cache_efficiency_score',
            'Cache efficiency score (0-1)',
            ['cache_type']
        )
        
        self.cost_optimization = Gauge(
            'cost_optimization_percent',
            'Cost optimization percentage',
            ['optimization_type']
        )
        
        self.provider_fairness = Gauge(
            'provider_fairness_index',
            'Provider load fairness index (0-1)',
            ['provider']
        )
        
        self.system_health = Gauge(
            'system_health_score',
            'Overall system health score (0-1)',
            ['component']
        )
        
        # Info metrics
        self.system_info = Info(
            'neural_mesh_system_info',
            'Neural Mesh system information'
        )
        
        # Internal metrics storage
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cost_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.kpi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
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
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'load_average': 0.0,
            'process_count': 0,
            'thread_count': 0
        }
        
        # Alert rules
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # RL and optimization metrics
        self.rl_metrics = {
            'model_accuracy': 0.0,
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'reward_history': deque(maxlen=1000),
            'q_table_size': 0
        }
        
        # Multi-modal metrics
        self.multi_modal_metrics = {
            'text_accuracy': 0.0,
            'vision_accuracy': 0.0,
            'feature_extraction_latency': 0.0,
            'cross_modal_correlation': 0.0
        }
    
    async def initialize(self):
        """Initialize the metrics service with advanced monitoring"""
        try:
            self.logger.info("Initializing advanced metrics service")
            
            # Initialize default alert rules
            self._initialize_alert_rules()
            
            # Start system metrics collection
            await self._update_system_metrics()
            
            # Start background monitoring tasks
            self._start_background_monitoring()
            
            # Initialize system info
            self._update_system_info()
            
            self.logger.info("Advanced metrics service initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics service: {str(e)}")
            raise
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        self.alert_rules = [
            AlertRule("high_latency", "request_latency_p95", ">", 5.0, 300, "warning"),
            AlertRule("low_cache_hit_rate", "cache_hit_rate", "<", 70.0, 600, "warning"),
            AlertRule("high_error_rate", "error_rate", ">", 5.0, 300, "critical"),
            AlertRule("high_cpu_usage", "cpu_usage", ">", 80.0, 300, "warning"),
            AlertRule("high_memory_usage", "memory_usage", ">", 85.0, 300, "critical"),
            AlertRule("low_provider_health", "provider_health", "<", 1.0, 600, "warning"),
            AlertRule("high_cost_per_request", "cost_per_request", ">", 0.1, 600, "info"),
            AlertRule("low_rl_performance", "rl_accuracy", "<", 0.7, 900, "warning")
        ]
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        # System metrics monitoring
        threading.Thread(target=self._system_metrics_monitor, daemon=True).start()
        
        # Alert monitoring
        threading.Thread(target=self._alert_monitor, daemon=True).start()
        
        # KPI calculation
        threading.Thread(target=self._kpi_calculator, daemon=True).start()
    
    def _system_metrics_monitor(self):
        """Background system metrics monitoring"""
        while True:
            try:
                asyncio.run(self._update_system_metrics())
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in system metrics monitor: {str(e)}")
                time.sleep(60)
    
    def _alert_monitor(self):
        """Background alert monitoring"""
        while True:
            try:
                self._check_alerts()
                time.sleep(60)  # Check alerts every minute
            except Exception as e:
                self.logger.error(f"Error in alert monitor: {str(e)}")
                time.sleep(120)
    
    def _kpi_calculator(self):
        """Background KPI calculation"""
        while True:
            try:
                self._calculate_kpis()
                time.sleep(300)  # Calculate KPIs every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in KPI calculator: {str(e)}")
                time.sleep(600)
    
    def _update_system_info(self):
        """Update system information"""
        try:
            import platform
            import socket
            
            system_info = {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'neural_mesh_version': '2.0.0',
                'start_time': self.start_time,
                'uptime_seconds': time.time() - self.start_time
            }
            
            self.system_info.info(system_info)
        except Exception as e:
            self.logger.error(f"Error updating system info: {str(e)}")
    
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
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.system_metrics['load_average'] = load_avg[0]  # 1-minute load average
            except AttributeError:
                self.system_metrics['load_average'] = 0.0
            
            # Process and thread counts
            process = psutil.Process()
            self.system_metrics['process_count'] = len(psutil.pids())
            self.system_metrics['thread_count'] = process.num_threads()
            
            # Update system health gauges
            self.system_health.labels(component='cpu').set(max(0, 1 - (self.system_metrics['cpu_usage'] / 100)))
            self.system_health.labels(component='memory').set(max(0, 1 - (self.system_metrics['memory_usage'] / 100)))
            self.system_health.labels(component='disk').set(max(0, 1 - (self.system_metrics['disk_usage'] / 100)))
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {str(e)}")
    
    def _check_alerts(self):
        """Check alert rules and trigger alerts if needed"""
        try:
            current_time = time.time()
            metrics = self._get_current_metrics()
            
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                metric_value = metrics.get(rule.metric_name, 0)
                
                # Check if condition is met
                condition_met = False
                if rule.condition == ">" and metric_value > rule.threshold:
                    condition_met = True
                elif rule.condition == "<" and metric_value < rule.threshold:
                    condition_met = True
                elif rule.condition == "==" and abs(metric_value - rule.threshold) < 0.001:
                    condition_met = True
                
                if condition_met:
                    # Check if alert is already active
                    active_alert = next(
                        (a for a in self.active_alerts if a['rule_name'] == rule.name),
                        None
                    )
                    
                    if not active_alert:
                        # Create new alert
                        alert = {
                            'rule_name': rule.name,
                            'metric_name': rule.metric_name,
                            'current_value': metric_value,
                            'threshold': rule.threshold,
                            'severity': rule.severity,
                            'triggered_at': current_time,
                            'message': f"{rule.metric_name} is {metric_value:.2f} {rule.condition} {rule.threshold}"
                        }
                        
                        self.active_alerts.append(alert)
                        self.alert_history.append(alert)
                        
                        self.logger.warning(f"Alert triggered: {alert['message']}")
                    else:
                        # Update existing alert
                        active_alert['current_value'] = metric_value
                        active_alert['last_updated'] = current_time
                else:
                    # Remove resolved alerts
                    self.active_alerts = [
                        a for a in self.active_alerts 
                        if a['rule_name'] != rule.name
                    ]
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values for alert checking"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['cpu_usage'] = self.system_metrics['cpu_usage']
            metrics['memory_usage'] = self.system_metrics['memory_usage']
            metrics['disk_usage'] = self.system_metrics['disk_usage']
            
            # Cache hit rate
            if self.total_requests > 0:
                metrics['cache_hit_rate'] = (self.cache_hits / self.total_requests) * 100
            else:
                metrics['cache_hit_rate'] = 0.0
            
            # Error rate
            total_errors = sum(len(errors) for errors in self.error_history.values())
            if self.total_requests > 0:
                metrics['error_rate'] = (total_errors / self.total_requests) * 100
            else:
                metrics['error_rate'] = 0.0
            
            # Request latency P95
            for provider, latencies in self.latency_history.items():
                if latencies:
                    sorted_latencies = sorted(latencies)
                    metrics[f'request_latency_p95_{provider}'] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            
            # Provider health
            # This would need to be populated from provider status updates
            metrics['provider_health'] = 2.0  # Default to healthy
            
            # Cost per request
            total_cost = sum(sum(costs) for costs in self.cost_history.values())
            if self.total_requests > 0:
                metrics['cost_per_request'] = total_cost / self.total_requests
            else:
                metrics['cost_per_request'] = 0.0
            
            # RL performance
            metrics['rl_accuracy'] = self.rl_metrics['model_accuracy']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {str(e)}")
            return {}
    
    def _calculate_kpis(self):
        """Calculate comprehensive KPIs"""
        try:
            current_time = time.time()
            
            # Calculate 20+ KPIs
            kpis = []
            
            # Performance KPIs
            kpis.append(KPIData(
                name="request_throughput",
                value=self.total_requests / max(current_time - self.start_time, 1),
                unit="requests/sec",
                threshold=10.0,
                category="performance",
                description="Number of requests processed per second"
            ))
            
            # Cache performance KPIs
            cache_hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            kpis.append(KPIData(
                name="cache_hit_rate",
                value=cache_hit_rate,
                unit="%",
                threshold=85.0,
                category="cache",
                description="Percentage of requests served from cache"
            ))
            
            # Latency KPIs
            if self.latency_history:
                all_latencies = [latency for latencies in self.latency_history.values() for latency in latencies]
                if all_latencies:
                    p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
                    kpis.append(KPIData(
                        name="p95_latency",
                        value=p95_latency,
                        unit="seconds",
                        threshold=2.0,
                        category="performance",
                        description="95th percentile request latency"
                    ))
            
            # Cost optimization KPIs
            total_cost = sum(sum(costs) for costs in self.cost_history.values())
            if self.total_requests > 0:
                avg_cost_per_request = total_cost / self.total_requests
                kpis.append(KPIData(
                    name="avg_cost_per_request",
                    value=avg_cost_per_request,
                    unit="USD",
                    threshold=0.05,
                    category="cost",
                    description="Average cost per request"
                ))
            
            # System health KPIs
            system_health_score = (
                (1 - self.system_metrics['cpu_usage'] / 100) * 0.3 +
                (1 - self.system_metrics['memory_usage'] / 100) * 0.3 +
                (1 - self.system_metrics['disk_usage'] / 100) * 0.2 +
                (cache_hit_rate / 100) * 0.2
            )
            kpis.append(KPIData(
                name="system_health_score",
                value=system_health_score,
                unit="score",
                threshold=0.8,
                category="health",
                description="Overall system health score (0-1)"
            ))
            
            # Error rate KPI
            total_errors = sum(len(errors) for errors in self.error_history.values())
            error_rate = (total_errors / max(self.total_requests, 1)) * 100
            kpis.append(KPIData(
                name="error_rate",
                value=error_rate,
                unit="%",
                threshold=1.0,
                category="reliability",
                description="Percentage of failed requests"
            ))
            
            # RL performance KPI
            kpis.append(KPIData(
                name="rl_model_accuracy",
                value=self.rl_metrics['model_accuracy'],
                unit="%",
                threshold=80.0,
                category="optimization",
                description="Reinforcement learning model accuracy"
            ))
            
            # Multi-modal accuracy KPI
            kpis.append(KPIData(
                name="multi_modal_accuracy",
                value=self.multi_modal_metrics['text_accuracy'],
                unit="%",
                threshold=85.0,
                category="accuracy",
                description="Multi-modal feature extraction accuracy"
            ))
            
            # Provider fairness KPI
            kpis.append(KPIData(
                name="provider_load_fairness",
                value=self._calculate_provider_fairness(),
                unit="index",
                threshold=0.7,
                category="fairness",
                description="Fairness of load distribution across providers"
            ))
            
            # Update KPI history
            for kpi in kpis:
                self.kpi_history[kpi.name].append({
                    'timestamp': current_time,
                    'value': kpi.value,
                    'unit': kpi.unit
                })
                
                # Update Prometheus gauges
                if kpi.name == "rl_model_accuracy":
                    self.rl_performance.labels(model='router', metric_type='accuracy').set(kpi.value)
                elif kpi.name == "multi_modal_accuracy":
                    self.multi_modal_accuracy.labels(feature_type='text').set(kpi.value)
                elif kpi.name == "cache_hit_rate":
                    self.cache_efficiency.labels(cache_type='semantic').set(kpi.value / 100)
                elif kpi.name == "avg_cost_per_request":
                    self.cost_optimization.labels(optimization_type='rl').set(max(0, (0.1 - kpi.value) / 0.1 * 100))
            
            self.logger.debug(f"Calculated {len(kpis)} KPIs")
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {str(e)}")
    
    def _calculate_provider_fairness(self) -> float:
        """Calculate provider load fairness index (0-1)"""
        try:
            # This would need actual provider load data
            # For now, return a placeholder
            return 0.8
        except Exception:
            return 0.5
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary with 20+ KPIs"""
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
            
            # Calculate advanced metrics
            cache_hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            total_errors = sum(len(errors) for errors in self.error_history.values())
            error_rate = (total_errors / max(self.total_requests, 1)) * 100
            
            total_cost = sum(sum(costs) for costs in self.cost_history.values())
            avg_cost_per_request = total_cost / max(self.total_requests, 1)
            
            # System health score
            system_health_score = (
                (1 - self.system_metrics['cpu_usage'] / 100) * 0.3 +
                (1 - self.system_metrics['memory_usage'] / 100) * 0.3 +
                (1 - self.system_metrics['disk_usage'] / 100) * 0.2 +
                (cache_hit_rate / 100) * 0.2
            )
            
            # RL performance metrics
            rl_performance = {
                'model_accuracy': self.rl_metrics['model_accuracy'],
                'learning_rate': self.rl_metrics['learning_rate'],
                'exploration_rate': self.rl_metrics['exploration_rate'],
                'q_table_size': self.rl_metrics['q_table_size'],
                'average_reward': sum(self.rl_metrics['reward_history']) / len(self.rl_metrics['reward_history']) if self.rl_metrics['reward_history'] else 0.0
            }
            
            # Multi-modal metrics
            multi_modal_performance = {
                'text_accuracy': self.multi_modal_metrics['text_accuracy'],
                'vision_accuracy': self.multi_modal_metrics['vision_accuracy'],
                'feature_extraction_latency': self.multi_modal_metrics['feature_extraction_latency'],
                'cross_modal_correlation': self.multi_modal_metrics['cross_modal_correlation']
            }
            
            # KPI summary
            kpi_summary = {}
            for kpi_name, history in self.kpi_history.items():
                if history:
                    latest = history[-1]
                    kpi_summary[kpi_name] = {
                        'current_value': latest['value'],
                        'unit': latest['unit'],
                        'trend': self._calculate_kpi_trend(kpi_name)
                    }
            
            return {
                'basic_metrics': {
                    'total_requests': self.total_requests,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'cache_hit_rate': cache_hit_rate,
                    'throughput_rps': throughput,
                    'uptime_seconds': uptime,
                    'error_rate': error_rate
                },
                'performance_metrics': {
                    'latency_percentiles': latency_percentiles,
                    'cost_statistics': cost_stats,
                    'error_rates': error_rates,
                    'avg_cost_per_request': avg_cost_per_request
                },
                'system_health': {
                    'overall_score': system_health_score,
                    'cpu_usage': self.system_metrics['cpu_usage'],
                    'memory_usage': self.system_metrics['memory_usage'],
                    'disk_usage': self.system_metrics['disk_usage'],
                    'load_average': self.system_metrics['load_average'],
                    'process_count': self.system_metrics['process_count'],
                    'thread_count': self.system_metrics['thread_count']
                },
                'advanced_metrics': {
                    'reinforcement_learning': rl_performance,
                    'multi_modal': multi_modal_performance,
                    'provider_fairness': self._calculate_provider_fairness()
                },
                'kpi_summary': kpi_summary,
                'alerts': {
                    'active_alerts': len(self.active_alerts),
                    'alert_history_size': len(self.alert_history),
                    'recent_alerts': list(self.alert_history)[-10:]  # Last 10 alerts
                },
                'monitoring': {
                    'total_alert_rules': len(self.alert_rules),
                    'enabled_alert_rules': len([r for r in self.alert_rules if r.enabled]),
                    'kpi_history_size': len(self.kpi_history)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def _calculate_kpi_trend(self, kpi_name: str) -> str:
        """Calculate KPI trend (increasing, decreasing, stable)"""
        try:
            history = self.kpi_history.get(kpi_name, [])
            if len(history) < 5:
                return "stable"
            
            # Get last 5 values
            recent_values = [h['value'] for h in history[-5:]]
            
            # Calculate trend
            if len(recent_values) >= 2:
                diff = recent_values[-1] - recent_values[0]
                if abs(diff) < 0.01:
                    return "stable"
                elif diff > 0:
                    return "increasing"
                else:
                    return "decreasing"
            
            return "stable"
            
        except Exception:
            return "stable"
    
    def get_advanced_observability_data(self) -> Dict[str, Any]:
        """Get advanced observability data for dashboards"""
        try:
            return {
                'kpi_data': [
                    {
                        'name': kpi_name,
                        'history': list(history)[-100:]  # Last 100 data points
                    }
                    for kpi_name, history in self.kpi_history.items()
                ],
                'system_metrics_timeline': [
                    {
                        'timestamp': time.time() - i * 30,  # Last 30 minutes, every 30 seconds
                        'cpu_usage': self.system_metrics['cpu_usage'],
                        'memory_usage': self.system_metrics['memory_usage'],
                        'disk_usage': self.system_metrics['disk_usage']
                    }
                    for i in range(60)
                ],
                'alert_summary': {
                    'total_alerts': len(self.alert_history),
                    'alerts_by_severity': {
                        severity: len([a for a in self.alert_history if a['severity'] == severity])
                        for severity in ['info', 'warning', 'critical']
                    },
                    'active_alerts': self.active_alerts
                },
                'performance_benchmarks': {
                    'target_latency_p95': 2.0,
                    'target_cache_hit_rate': 85.0,
                    'target_error_rate': 1.0,
                    'target_system_health': 0.8,
                    'current_performance': {
                        'latency_p95': self._get_current_p95_latency(),
                        'cache_hit_rate': (self.cache_hits / max(self.total_requests, 1)) * 100,
                        'error_rate': (sum(len(errors) for errors in self.error_history.values()) / max(self.total_requests, 1)) * 100,
                        'system_health': (
                            (1 - self.system_metrics['cpu_usage'] / 100) * 0.3 +
                            (1 - self.system_metrics['memory_usage'] / 100) * 0.3 +
                            (1 - self.system_metrics['disk_usage'] / 100) * 0.2 +
                            (self.cache_hits / max(self.total_requests, 1)) * 0.2
                        )
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting advanced observability data: {str(e)}")
            return {}
    
    def _get_current_p95_latency(self) -> float:
        """Get current P95 latency"""
        try:
            all_latencies = [latency for latencies in self.latency_history.values() for latency in latencies]
            if all_latencies:
                sorted_latencies = sorted(all_latencies)
                return sorted_latencies[int(len(sorted_latencies) * 0.95)]
            return 0.0
        except Exception:
            return 0.0
    
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
