import requests
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class VectorSearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float

class VectorStore:
    def __init__(self):
        self.layers: Dict[int, Dict] = {}  # level -> {id -> (embedding, metadata)}
        
    def add_embedding(self, level: int, content_id: str, 
                     embedding: np.ndarray, metadata: Dict[str, Any]):
        if level not in self.layers:
            self.layers[level] = {}
        self.layers[level][content_id] = (embedding, metadata)
    
    def query_layer(self, level: int, query_embedding: np.ndarray, 
                   top_k: int = 5) -> List[Dict[str, Any]]:
        if level not in self.layers:
            return []
            
        # Perform similarity search within the layer
        results = []
        for content_id, (emb, metadata) in self.layers[level].items():
            similarity = np.dot(query_embedding, emb)
            results.append((similarity, content_id, metadata))
            
        results.sort(reverse=True)
        return results[:top_k]

    async def add_vectors(self, texts: List[str], metadata: List[Dict] = None):
        """Add vectors to storage"""
        response = requests.post(
            f"{self.api_url}/add",
            json={
                "texts": texts,
                "metadata": metadata or [{}] * len(texts)
            }
        )
        return response.json()
    
    async def search(self, query: str, filter_dict: Dict = None, top_k: int = 5) -> List[VectorSearchResult]:
        """Search vectors"""
        response = requests.post(
            f"{self.api_url}/search",
            json={
                "query": query,
                "filter": filter_dict,
                "top_k": top_k
            }
        )
        results = response.json()["results"]
        return [VectorSearchResult(**r) for r in results] 