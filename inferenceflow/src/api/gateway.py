from fastapi import APIRouter, Depends, HTTPException, Request
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.core.router import route_request
from src.services.metrics import metrics
from prometheus_client import generate_latest
from src.utils.config import settings
import time

router = APIRouter()

# Dummy rate limiting
request_counts = {}

async def rate_limit(request: Request):
    user_id = request.headers.get("X-User-ID", "anonymous")
    current_time = time.time()
    
    if user_id not in request_counts:
        request_counts[user_id] = []
    
    # Remove old requests outside the window
    request_counts[user_id] = [t for t in request_counts[user_id] if current_time - t < settings.RATE_LIMIT_WINDOW]
    
    if len(request_counts[user_id]) >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[user_id].append(current_time)

# Dummy authentication
async def authenticate(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    # In a real app, validate JWT token here
    # For now, any Bearer token is considered valid
    pass

@router.post("/v1/inference", response_model=InferenceResponse, dependencies=[Depends(rate_limit), Depends(authenticate)])
async def inference(request: InferenceRequest):
    return await route_request(request)

@router.get("/v1/health")
async def health_check():
    return {"status": "ok"}

@router.get("/v1/metrics")
async def prometheus_metrics():
    return generate_latest().decode("utf-8")

@router.post("/v1/batch")
async def batch_inference():
    # Placeholder for batch inference
    return {"status": "not_implemented"}