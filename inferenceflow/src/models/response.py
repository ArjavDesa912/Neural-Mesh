from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class InferenceResponse(BaseModel):
    response: str = Field(..., description="The generated response text")
    provider: str = Field(..., description="The LLM provider used")
    model: str = Field(..., description="The specific model used")
    latency: float = Field(..., ge=0.0, description="Response latency in seconds")
    tokens_used: int = Field(..., ge=0, description="Number of tokens used")
    cost: float = Field(..., ge=0.0, description="Cost of the inference in USD")
    cache_hit: bool = Field(False, description="Whether this was served from cache")
    request_id: str = Field(..., description="Unique request identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Response creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }