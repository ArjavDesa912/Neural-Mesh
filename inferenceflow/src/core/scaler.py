from typing import Dict, Any, List
import pandas as pd
from sklearn.linear_model import LinearRegression
from kubernetes import client, config
from src.utils.logger import logger
from src.utils.config import settings

class AutoScaler:
    def __init__(self):
        self.traffic_history = []  # Store historical traffic data (e.g., requests per second)
        try:
            config.load_incluster_config()
        except config.ConfigException:
            try:
                config.load_kube_config()
            except config.ConfigException:
                logger.warning("Could not configure Kubernetes client. Auto-scaling will be disabled.")
                self.v1 = None
                return
        self.v1 = client.AppsV1Api()
        self.deployment_name = "inferenceflow"
        self.namespace = "default" # Assuming default namespace for now

    def predict_traffic(self, lookback_hours: int = 24) -> List[float]:
        if len(self.traffic_history) < 5: # Need at least 5 data points for a meaningful linear regression
            return [100.0] * lookback_hours # Assume a base traffic if not enough history

        # Use pandas for time series data and linear regression for prediction
        data = pd.DataFrame({'traffic': self.traffic_history})
        data['time'] = range(len(data))

        model = LinearRegression()
        model.fit(data[['time']], data['traffic'])

        future_time = np.array(range(len(self.traffic_history), len(self.traffic_history) + lookback_hours)).reshape(-1, 1)
        predictions = model.predict(future_time)

        logger.info(f"Predicted traffic for next {lookback_hours} hours: {predictions.tolist()}")
        return predictions.tolist()

    def get_current_replicas(self) -> int:
        if not self.v1:
            return 0
        try:
            deployment = self.v1.read_namespaced_deployment(self.deployment_name, self.namespace)
            return deployment.spec.replicas or 0
        except client.ApiException as e:
            logger.error(f"Error getting current replicas: {e}")
            return 0

    def calculate_required_capacity(self, current_load: float, predicted_load: float) -> int:
        current_replicas = self.get_current_replicas()
        if current_replicas == 0:
            return settings.MIN_REPLICAS # Start with min replicas if not running

        # Target CPU utilization from HPA config (e.g., 70%)
        target_cpu_utilization = 0.7 # This should ideally come from HPA configmap or settings

        # Simple calculation: scale based on predicted load vs current capacity
        # Assuming 1 unit of load requires 1 replica at 100% utilization
        # If predicted_load is in requests/second, and each replica can handle X requests/second
        # required_replicas = predicted_load / (X * target_cpu_utilization)

        # For now, a simplified logic based on percentage change
        if predicted_load > current_load * 1.1 and current_replicas < settings.MAX_REPLICAS:  # If predicted load is 10% higher, scale up
            new_replicas = min(settings.MAX_REPLICAS, current_replicas + 1)
            logger.info(f"Scaling up: predicted_load={predicted_load}, current_load={current_load}, new_replicas={new_replicas}")
            return new_replicas
        elif predicted_load < current_load * 0.9 and current_replicas > settings.MIN_REPLICAS: # If predicted load is 10% lower, scale down
            new_replicas = max(settings.MIN_REPLICAS, current_replicas - 1)
            logger.info(f"Scaling down: predicted_load={predicted_load}, current_load={current_load}, new_replicas={new_replicas}")
            return new_replicas
        else:
            logger.info(f"Maintaining replicas: predicted_load={predicted_load}, current_load={current_load}, current_replicas={current_replicas}")
            return current_replicas

    async def scale_infrastructure(self, target_replicas: int) -> None:
        if not self.v1:
            logger.warning("Kubernetes client not initialized. Cannot scale infrastructure.")
            return
        try:
            # Patch the deployment to update the replica count
            body = {"spec": {"replicas": target_replicas}}
            await self.v1.patch_namespaced_deployment(self.deployment_name, self.namespace, body)
            logger.info(f"Successfully scaled deployment {self.deployment_name} to {target_replicas} replicas.")
        except client.ApiException as e:
            logger.error(f"Error scaling infrastructure: {e}")

    def get_scaling_metrics(self) -> Dict[str, Any]:
        return {
            "current_replicas": self.get_current_replicas(),
            "traffic_history_length": len(self.traffic_history),
            "last_predicted_traffic": self.predict_traffic(1)[0] if self.traffic_history else 0.0
        }

scaler = AutoScaler()
