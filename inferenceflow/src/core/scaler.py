import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta
from src.utils.config import settings
from src.utils.logger import LoggerMixin
from src.services.metrics import metrics_service

class AutoScaler(LoggerMixin):
    """Predictive auto-scaling based on traffic patterns and queue depth"""
    
    def __init__(self):
        self.traffic_history: deque = deque(maxlen=1000)
        self.queue_depth_history: deque = deque(maxlen=1000)
        self.scaling_history: deque = deque(maxlen=100)
        self.current_replicas = settings.min_replicas
        self.last_scale_time = time.time()
        self.scale_cooldown = 300  # 5 minutes cooldown between scaling operations
        
        # Scaling thresholds
        self.cpu_threshold = settings.cpu_threshold
        self.memory_threshold = settings.memory_threshold
        self.queue_depth_threshold = 10
        
        # Predictive scaling parameters
        self.prediction_window = 3600  # 1 hour
        self.traffic_pattern_window = 86400  # 24 hours
        
    async def initialize(self):
        """Initialize the auto scaler"""
        try:
            self.logger.info("Initializing auto scaler")
            
            # Start background scaling loop
            asyncio.create_task(self._scaling_loop())
            
            self.logger.info("Auto scaler initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize auto scaler: {str(e)}")
            raise
    
    async def _scaling_loop(self):
        """Background loop for continuous scaling decisions"""
        while True:
            try:
                await self._evaluate_and_scale()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _evaluate_and_scale(self):
        """Evaluate current conditions and scale if necessary"""
        
        # Check if we're in cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return
        
        # Get current metrics
        current_load = await self._get_current_load()
        predicted_load = await self.predict_traffic()
        
        # Calculate required capacity
        required_capacity = await self.calculate_required_capacity(current_load, predicted_load)
        
        # Determine if scaling is needed
        if required_capacity != self.current_replicas:
            await self.scale_infrastructure(required_capacity)
    
    async def predict_traffic(self, lookback_hours: int = 24) -> List[float]:
        """Predict traffic for the next hour based on historical patterns"""
        
        if len(self.traffic_history) < 10:
            return [1.0]  # Default prediction if insufficient data
        
        try:
            # Convert traffic history to numpy array
            traffic_data = np.array(list(self.traffic_history))
            
            # Simple moving average prediction
            window_size = min(60, len(traffic_data))  # Use last 60 data points or all available
            moving_avg = np.mean(traffic_data[-window_size:])
            
            # Add some trend analysis
            if len(traffic_data) >= 2:
                trend = np.polyfit(range(len(traffic_data[-10:])), traffic_data[-10:], 1)[0]
                trend_factor = 1 + (trend * 0.1)  # Apply trend with dampening
            else:
                trend_factor = 1.0
            
            # Generate predictions for next hour (60 minutes)
            predictions = []
            for i in range(60):
                # Add some randomness and time-of-day patterns
                time_factor = self._get_time_of_day_factor()
                prediction = moving_avg * trend_factor * time_factor * (0.9 + 0.2 * np.random.random())
                predictions.append(max(0.1, prediction))  # Ensure minimum prediction
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting traffic: {str(e)}")
            return [1.0]
    
    def _get_time_of_day_factor(self) -> float:
        """Get time-of-day scaling factor based on typical usage patterns"""
        hour = datetime.now().hour
        
        # Typical usage patterns (could be learned from data)
        if 9 <= hour <= 17:  # Business hours
            return 1.2
        elif 18 <= hour <= 22:  # Evening
            return 1.1
        elif 23 <= hour or hour <= 6:  # Night
            return 0.7
        else:  # Early morning
            return 0.9
    
    async def calculate_required_capacity(
        self, 
        current_load: float, 
        predicted_load: List[float]
    ) -> int:
        """Calculate required capacity based on current and predicted load"""
        
        try:
            # Get system metrics
            system_metrics = await self._get_system_metrics()
            
            # Calculate load factors
            cpu_factor = system_metrics.get('cpu_usage', 50) / 100
            memory_factor = system_metrics.get('memory_usage', 50) / 100
            queue_factor = min(1.0, system_metrics.get('queue_depth', 0) / self.queue_depth_threshold)
            
            # Weighted load calculation
            weighted_load = (cpu_factor * 0.4 + memory_factor * 0.3 + queue_factor * 0.3)
            
            # Use predicted load for forward-looking scaling
            avg_predicted_load = np.mean(predicted_load) if predicted_load else current_load
            
            # Calculate required replicas
            base_replicas = max(1, int(weighted_load * self.current_replicas))
            predicted_replicas = max(1, int(avg_predicted_load * self.current_replicas))
            
            # Use the higher of current and predicted requirements
            required_replicas = max(base_replicas, predicted_replicas)
            
            # Apply bounds
            required_replicas = max(settings.min_replicas, min(settings.max_replicas, required_replicas))
            
            self.logger.info(f"Load calculation: current={current_load:.2f}, predicted={avg_predicted_load:.2f}, "
                           f"required_replicas={required_replicas}")
            
            return required_replicas
            
        except Exception as e:
            self.logger.error(f"Error calculating required capacity: {str(e)}")
            return self.current_replicas
    
    async def scale_infrastructure(self, target_replicas: int) -> bool:
        """Scale infrastructure to target number of replicas"""
        
        if target_replicas == self.current_replicas:
            return True
        
        try:
            self.logger.info(f"Scaling from {self.current_replicas} to {target_replicas} replicas")
            
            # Record scaling decision
            scaling_decision = {
                'timestamp': time.time(),
                'from_replicas': self.current_replicas,
                'to_replicas': target_replicas,
                'reason': 'auto_scaling'
            }
            self.scaling_history.append(scaling_decision)
            
            # Update current replicas
            self.current_replicas = target_replicas
            self.last_scale_time = time.time()
            
            # In a real implementation, this would call Kubernetes API
            # For now, we'll just log the scaling action
            await self._apply_kubernetes_scaling(target_replicas)
            
            self.logger.info(f"Successfully scaled to {target_replicas} replicas")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scaling infrastructure: {str(e)}")
            return False
    
    async def _apply_kubernetes_scaling(self, target_replicas: int):
        """Apply scaling to Kubernetes deployment"""
        try:
            # This would integrate with Kubernetes API
            # For now, we'll simulate the scaling
            self.logger.info(f"Would scale Kubernetes deployment to {target_replicas} replicas")
            
            # Update metrics
            await metrics_service.update_active_requests(target_replicas)
            
        except Exception as e:
            self.logger.error(f"Error applying Kubernetes scaling: {str(e)}")
            raise
    
    async def _get_current_load(self) -> float:
        """Get current system load"""
        try:
            # Get performance summary
            performance = await metrics_service.get_performance_summary()
            
            # Calculate load based on throughput and latency
            throughput = performance.get('throughput_rps', 1.0)
            avg_latency = 0.0
            
            # Get average latency from all providers
            latency_percentiles = performance.get('latency_percentiles', {})
            if latency_percentiles:
                all_latencies = []
                for provider_latencies in latency_percentiles.values():
                    if 'avg' in provider_latencies:
                        all_latencies.append(provider_latencies['avg'])
                avg_latency = np.mean(all_latencies) if all_latencies else 1.0
            else:
                avg_latency = 1.0
            
            # Load = throughput * latency (higher means more load)
            load = throughput * avg_latency
            
            # Store in history
            self.traffic_history.append(load)
            
            return load
            
        except Exception as e:
            self.logger.error(f"Error getting current load: {str(e)}")
            return 1.0
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            performance = await metrics_service.get_performance_summary()
            system_metrics = performance.get('system_metrics', {})
            
            # Add queue depth
            queue_depth = len(self.traffic_history) if self.traffic_history else 0
            system_metrics['queue_depth'] = queue_depth
            
            return system_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {str(e)}")
            return {
                'cpu_usage': 50.0,
                'memory_usage': 50.0,
                'disk_usage': 50.0,
                'queue_depth': 0
            }
    
    async def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics and analytics"""
        try:
            current_load = await self._get_current_load()
            predicted_load = await self.predict_traffic()
            system_metrics = await self._get_system_metrics()
            
            return {
                'current_replicas': self.current_replicas,
                'current_load': current_load,
                'predicted_load_avg': np.mean(predicted_load) if predicted_load else 0,
                'system_metrics': system_metrics,
                'scaling_history': list(self.scaling_history)[-10:],  # Last 10 scaling events
                'traffic_history_length': len(self.traffic_history),
                'last_scale_time': self.last_scale_time,
                'scale_cooldown_remaining': max(0, self.scale_cooldown - (time.time() - self.last_scale_time))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scaling metrics: {str(e)}")
            return {}
    
    async def manual_scale(self, target_replicas: int, reason: str = "manual") -> bool:
        """Manually scale to target replicas"""
        try:
            self.logger.info(f"Manual scaling to {target_replicas} replicas: {reason}")
            
            # Apply bounds
            target_replicas = max(settings.min_replicas, min(settings.max_replicas, target_replicas))
            
            # Record manual scaling decision
            scaling_decision = {
                'timestamp': time.time(),
                'from_replicas': self.current_replicas,
                'to_replicas': target_replicas,
                'reason': reason
            }
            self.scaling_history.append(scaling_decision)
            
            # Apply scaling
            success = await self.scale_infrastructure(target_replicas)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in manual scaling: {str(e)}")
            return False
    
    async def update_queue_depth(self, depth: int):
        """Update current queue depth"""
        self.queue_depth_history.append({
            'timestamp': time.time(),
            'depth': depth
        })
        
        # Update metrics service
        await metrics_service.update_queue_depth(depth)

# Global auto scaler instance
auto_scaler = AutoScaler()
