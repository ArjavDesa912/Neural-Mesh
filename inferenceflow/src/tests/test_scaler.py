import pytest
from src.core.scaler import AutoScaler
from unittest.mock import patch, MagicMock, AsyncMock
from kubernetes.client.rest import ApiException

@pytest.fixture
def scaler_instance():
    with patch('kubernetes.config.load_incluster_config'), \
         patch('kubernetes.config.load_kube_config'):
        scaler = AutoScaler()
        # Ensure v1 is always a MagicMock for testing purposes
        scaler.v1 = MagicMock()
        scaler.v1.patch_namespaced_deployment = AsyncMock()
        return scaler

def test_predict_traffic_empty_history(scaler_instance: AutoScaler):
    predictions = scaler_instance.predict_traffic(lookback_hours=3)
    assert len(predictions) == 3
    assert all(p == 100.0 for p in predictions)

def test_predict_traffic_with_history(scaler_instance: AutoScaler):
    scaler_instance.traffic_history = [100.0, 110.0, 120.0, 130.0, 140.0]
    predictions = scaler_instance.predict_traffic(lookback_hours=2)
    assert len(predictions) == 2
    assert predictions[0] > 100.0 # Should predict an increase

def test_calculate_required_capacity_scale_up(scaler_instance: AutoScaler):
    with patch('src.utils.config.settings') as mock_settings:
        mock_settings.MIN_REPLICAS = 1
        mock_settings.MAX_REPLICAS = 10
        with patch.object(scaler_instance, 'get_current_replicas', return_value=3):
            target_replicas = scaler_instance.calculate_required_capacity(current_load=100.0, predicted_load=120.0)
            assert target_replicas == 4 # Should scale up by 1

def test_calculate_required_capacity_scale_down(scaler_instance: AutoScaler):
    with patch('src.utils.config.settings') as mock_settings:
        mock_settings.MIN_REPLICAS = 1
        mock_settings.MAX_REPLICAS = 10
        with patch.object(scaler_instance, 'get_current_replicas', return_value=5):
            target_replicas = scaler_instance.calculate_required_capacity(current_load=100.0, predicted_load=80.0)
            assert target_replicas == 4 # Should scale down by 1

def test_calculate_required_capacity_maintain(scaler_instance: AutoScaler):
    with patch('src.utils.config.settings') as mock_settings:
        mock_settings.MIN_REPLICAS = 1
        mock_settings.MAX_REPLICAS = 10
        with patch.object(scaler_instance, 'get_current_replicas', return_value=3):
            target_replicas = scaler_instance.calculate_required_capacity(current_load=100.0, predicted_load=105.0)
            assert target_replicas == 3 # Should maintain

@pytest.mark.asyncio
async def test_scale_infrastructure_success(scaler_instance: AutoScaler):
    scaler_instance.v1.patch_namespaced_deployment.return_value = None
    await scaler_instance.scale_infrastructure(target_replicas=5)
    scaler_instance.v1.patch_namespaced_deployment.assert_called_once_with(
        scaler_instance.deployment_name, scaler_instance.namespace, {"spec": {"replicas": 5}}
    )

@pytest.mark.asyncio
async def test_scale_infrastructure_api_exception(scaler_instance: AutoScaler):
    scaler_instance.v1.patch_namespaced_deployment.side_effect = ApiException("Test API Error")
    with patch('src.utils.logger.logger.error') as mock_logger_error:
        await scaler_instance.scale_infrastructure(target_replicas=5)
        mock_logger_error.assert_called_once()

def test_get_scaling_metrics(scaler_instance: AutoScaler):
    with patch.object(scaler_instance, 'get_current_replicas', return_value=3):
        scaler_instance.traffic_history = [100.0, 110.0]
        metrics = scaler_instance.get_scaling_metrics()
        assert metrics["current_replicas"] == 3
        assert metrics["traffic_history_length"] == 2
        assert "last_predicted_traffic" in metrics
