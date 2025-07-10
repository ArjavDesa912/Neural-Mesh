from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class InferenceResponse(BaseModel):
    response: str
    provider: str
    model: str
    latency: float
    tokens_used: int
    cost: float
    cache_hit: bool
    request_id: str
    created_at: datetime
    metadata: Dict[str, Any] = {}