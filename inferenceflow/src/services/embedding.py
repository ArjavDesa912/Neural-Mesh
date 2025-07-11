from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text: str | List[str]) -> List[float] | List[List[float]]:
        return self.model.encode(text).tolist()

embedding_service = EmbeddingService()
