from fastapi import FastAPI
from src.api.gateway import api_gateway
from src.core.orchestrator import orchestrator
from src.core.scaler import auto_scaler
from src.services.metrics import metrics_service
from prometheus_client import start_http_server
from src.utils.config import settings
from src.utils.logger import logger

# Use the FastAPI app from the API gateway
app = api_gateway.app

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    try:
        logger.info("Starting Neural-Mesh InferenceFlow...")
        
        # Start Prometheus metrics server
        start_http_server(settings.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
        
        # Initialize auto scaler
        await auto_scaler.initialize()
        logger.info("Auto scaler initialized")
        
        logger.info("Neural-Mesh InferenceFlow started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Shutting down Neural-Mesh InferenceFlow...")
        
        # Shutdown orchestrator (this will also shutdown other components)
        await orchestrator.shutdown()
        
        logger.info("Neural-Mesh InferenceFlow shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Health check endpoint for basic monitoring
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "neural-mesh-inferenceflow"}
