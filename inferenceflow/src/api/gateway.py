import time
import jwt
import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from src.core.router import route_request
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.services.metrics import metrics
from src.utils.config import settings
from src.utils.logger import logger
from prometheus_client import generate_latest

router = APIRouter()

# Initialize Redis client
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True
)

async def rate_limit(request: Request):
    user_id = request.headers.get("X-User-ID", "anonymous")
    key = f"rate_limit:{user_id}"
    
    # Increment counter and set/update expiry
    current_requests = await redis_client.incr(key)
    if current_requests == 1:
        await redis_client.expire(key, settings.RATE_LIMIT_WINDOW)
    
    if current_requests > settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for user: {user_id}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

async def authenticate(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Authentication failed: No or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    try:
        # In a real application, you would use a secret key from config
        # For this example, we'll just decode without verification for simplicity
        # DO NOT USE THIS IN PRODUCTION WITHOUT PROPER SECRET KEY VALIDATION
        payload = jwt.decode(token, options={"verify_signature": False}) 
        request.state.user_id = payload.get("user_id")
        logger.info(f"User {request.state.user_id} authenticated successfully.")
    except jwt.ExpiredSignatureError:
        logger.warning("Authentication failed: Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        logger.warning("Authentication failed: Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/v1/inference", response_model=InferenceResponse, dependencies=[Depends(rate_limit), Depends(authenticate)])
async def inference(request: InferenceRequest, http_request: Request):
    logger.info(f"Received inference request from user {http_request.state.user_id}: {request.prompt[:50]}...")
    start_time = time.time()
    try:
        response = await route_request(request)
        end_time = time.time()
        logger.info(f"Inference request completed in {end_time - start_time:.4f} seconds. Request ID: {response.request_id}")
        return response
    except Exception as e:
        logger.error(f"Error processing inference request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/v1/health")
async def health_check():
    try:
        await redis_client.ping()
        return {"status": "ok", "redis_status": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: Redis connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: Redis not connected: {e}")

@router.get("/v1/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest().decode("utf-8"), media_type="text/plain")

@router.post("/v1/batch")
async def batch_inference(requests: list[InferenceRequest], http_request: Request):
    logger.info(f"Received batch inference request from user {http_request.state.user_id} for {len(requests)} requests.")
    responses = []
    for req in requests:
        try:
            response = await route_request(req)
            responses.append(response)
        except Exception as e:
            logger.error(f"Error processing batch inference request for prompt '{req.prompt[:50]}...': {e}")
            responses.append({"error": str(e), "prompt": req.prompt}) # Append error for individual request
    logger.info(f"Batch inference completed for {len(requests)} requests.")
    return responses