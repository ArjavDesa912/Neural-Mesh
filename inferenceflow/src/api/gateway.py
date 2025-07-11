from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time
import asyncio
from typing import List, Optional
import redis.asyncio as redis
from src.models.request import InferenceRequest
from src.models.response import InferenceResponse
from src.core.orchestrator import orchestrator
from src.utils.config import settings
from src.utils.logger import LoggerMixin
from src.services.metrics import metrics_service

class APIGateway(LoggerMixin):
    """Main API Gateway with rate limiting and authentication"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Neural-Mesh InferenceFlow",
            description="Intelligent AI inference orchestration platform",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize Redis for rate limiting
        self.redis_client: Optional[redis.Redis] = None
        
        # Security
        self.security = HTTPBearer()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )
        
        # Custom middleware for request logging and metrics
        @self.app.middleware("http")
        async def request_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Extract user ID for rate limiting
            user_id = self._extract_user_id(request)
            
            # Check rate limit
            if not await self._check_rate_limit(user_id):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": 60}
                )
            
            # Process request
            response = await call_next(request)
            
            # Record metrics
            latency = time.time() - start_time
            await metrics_service.record_request_latency(latency, "api_gateway")
            
            # Add response headers
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "")
            response.headers["X-Response-Time"] = str(latency)
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize components on startup"""
            try:
                self.logger.info("Starting API Gateway...")
                
                # Initialize Redis
                self.redis_client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    password=settings.redis_password,
                    db=settings.redis_db,
                    decode_responses=True
                )
                await self.redis_client.ping()
                self.logger.info("Redis connection established")
                
                # Initialize orchestrator
                await orchestrator.initialize()
                self.logger.info("Orchestrator initialized")
                
                self.logger.info("API Gateway started successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to start API Gateway: {str(e)}")
                raise
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            try:
                self.logger.info("Shutting down API Gateway...")
                await orchestrator.shutdown()
                
                if self.redis_client:
                    await self.redis_client.close()
                
                self.logger.info("API Gateway shutdown complete")
                
            except Exception as e:
                self.logger.error(f"Error during shutdown: {str(e)}")
        
        @self.app.post("/v1/inference", response_model=InferenceResponse)
        async def inference_endpoint(
            request: InferenceRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Main inference endpoint"""
            try:
                # Validate JWT token
                user_id = await self._validate_token(credentials.credentials)
                
                # Update request with validated user ID
                request.user_id = user_id
                
                # Process inference request
                response = await orchestrator.process_inference_request(request)
                
                # Record metrics in background
                background_tasks.add_task(
                    self._record_inference_metrics, response
                )
                
                return response
                
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            except Exception as e:
                self.logger.error(f"Inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/batch", response_model=List[InferenceResponse])
        async def batch_inference_endpoint(
            requests: List[InferenceRequest],
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Batch inference endpoint"""
            try:
                # Validate JWT token
                user_id = await self._validate_token(credentials.credentials)
                
                # Update all requests with validated user ID
                for request in requests:
                    request.user_id = user_id
                
                # Process batch requests
                responses = await orchestrator.process_batch_requests(requests)
                
                # Record metrics in background
                background_tasks.add_task(
                    self._record_batch_metrics, responses
                )
                
                return responses
                
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            except Exception as e:
                self.logger.error(f"Batch inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health_status = await orchestrator.health_check()
                return health_status
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
                return {"overall": "unhealthy", "error": str(e)}
        
        @self.app.get("/v1/status")
        async def system_status():
            """System status endpoint"""
            try:
                status = await orchestrator.get_system_status()
                return status
            except Exception as e:
                self.logger.error(f"Status check error: {str(e)}")
                return {"status": "error", "error": str(e)}
        
        @self.app.get("/v1/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            try:
                metrics = metrics_service.get_prometheus_metrics()
                return StreamingResponse(
                    iter([metrics]),
                    media_type="text/plain"
                )
            except Exception as e:
                self.logger.error(f"Metrics error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/v1/providers")
        async def providers_endpoint():
            """Get provider information"""
            try:
                provider_statuses = orchestrator.provider_manager.get_all_provider_status()
                return {"providers": provider_statuses}
            except Exception as e:
                self.logger.error(f"Providers error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/cache/clear")
        async def clear_cache_endpoint(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Clear cache endpoint"""
            try:
                # Validate admin token
                await self._validate_admin_token(credentials.credentials)
                
                from src.core.cache import semantic_cache
                deleted_count = await semantic_cache.clear_cache()
                
                return {"message": f"Cache cleared. Deleted {deleted_count} entries."}
                
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            except Exception as e:
                self.logger.error(f"Cache clear error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Neural-Mesh InferenceFlow API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/v1/health"
            }
    
    async def _validate_token(self, token: str) -> str:
        """Validate JWT token and return user ID"""
        try:
            payload = jwt.decode(
                token, 
                settings.jwt_secret_key, 
                algorithms=[settings.jwt_algorithm]
            )
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _validate_admin_token(self, token: str):
        """Validate admin JWT token"""
        try:
            payload = jwt.decode(
                token, 
                settings.jwt_secret_key, 
                algorithms=[settings.jwt_algorithm]
            )
            if payload.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request headers or JWT token"""
        # Try to get from header first
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Try to extract from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token, 
                    settings.jwt_secret_key, 
                    algorithms=[settings.jwt_algorithm]
                )
                return payload.get("user_id", "anonymous")
            except:
                pass
        
        return "anonymous"
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check rate limit for user"""
        if not self.redis_client:
            return True  # Allow if Redis is not available
        
        try:
            key = f"rate_limit:{user_id}"
            current_count = await self.redis_client.get(key)
            
            if current_count is None:
                # First request
                await self.redis_client.setex(
                    key, 
                    settings.rate_limit_window, 
                    1
                )
                return True
            
            count = int(current_count)
            if count >= settings.rate_limit_requests:
                return False
            
            # Increment counter
            await self.redis_client.incr(key)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {str(e)}")
            return True  # Allow if rate limiting fails
    
    async def _record_inference_metrics(self, response: InferenceResponse):
        """Record metrics for inference response"""
        try:
            await metrics_service.record_cost(response.provider, response.cost)
            await metrics_service.record_tokens(
                response.provider, 
                response.metadata.get("input_tokens", 0),
                response.tokens_used
            )
        except Exception as e:
            self.logger.error(f"Error recording inference metrics: {str(e)}")
    
    async def _record_batch_metrics(self, responses: List[InferenceResponse]):
        """Record metrics for batch responses"""
        try:
            for response in responses:
                await self._record_inference_metrics(response)
        except Exception as e:
            self.logger.error(f"Error recording batch metrics: {str(e)}")

# Create API Gateway instance
api_gateway = APIGateway()
app = api_gateway.app