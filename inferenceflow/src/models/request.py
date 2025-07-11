from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The input prompt for inference")
    model_preference: Optional[str] = Field(None, description="Preferred model to use for inference")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")
    user_id: str = Field(..., description="User identifier for rate limiting and tracking")
    priority: Priority = Field(Priority.NORMAL, description="Request priority level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }