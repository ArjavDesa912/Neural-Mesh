from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class InferenceRequest(BaseModel):
    prompt: str
    model_preference: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: bool = False
    user_id: str
    priority: str = "normal"  # low, normal, high
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()