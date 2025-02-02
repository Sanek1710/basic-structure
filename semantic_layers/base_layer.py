from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..storage.vector_store import VectorStore, VectorSearchResult
from ..embeddings.embedding_engine import EmbeddingEngine

class BaseSemanticLayer(ABC):
    def __init__(self, vector_store: VectorStore, layer_name: str, embedding_model: str):
        self.vector_store = vector_store
        self.layer_name = layer_name
        self.embedding_engine = EmbeddingEngine(embedding_model)
    
    @abstractmethod
    async def process_content(self, content: str) -> str:
        """Process content before embedding"""
        pass
    
    async def create_embedding(self, content: str) -> List[float]:
        """Create embedding for processed content"""
        return await self.embedding_engine.create_embedding(content)
    
    async def process_with_dependencies(
        self, 
        content: str, 
        previous_layer_results: Optional[List[Any]] = None
    ) -> List[Any]:
        """Process content with results from previous layer"""
        if previous_layer_results is None:
            # Base layer processing
            processed_content = await self.process_content(content)
        else:
            # Process using previous layer results
            processed_content = await self.process_from_previous(
                content, 
                previous_layer_results
            )
        
        # Create embedding for this layer
        embedding = await self.create_embedding(processed_content)
        
        # Store with layer-specific metadata
        metadata = {"layer": self.layer_name}
        await self.vector_store.add_vectors([processed_content], [metadata], [embedding])
        
        return await self.vector_store.search(processed_content, top_k=5)
    
    @abstractmethod
    async def process_from_previous(
        self, 
        content: str, 
        previous_results: List[Any]
    ) -> str:
        """Process content using results from previous layer"""
        pass
    
    async def search_with_dependencies(
        self, 
        query: str, 
        higher_layer_results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """Search within context of higher layer results"""
        # Modify search based on higher layer context
        modified_query = self.modify_query_with_context(query, higher_layer_results)
        return await self.vector_store.search(
            modified_query,
            filter_dict={"layer": self.layer_name},
            top_k=5
        )
    
    def modify_query_with_context(
        self, 
        query: str, 
        higher_results: List[VectorSearchResult]
    ) -> str:
        """Modify query based on higher layer context"""
        context = "\n".join([r.content for r in higher_results])
        return f"{query}\nContext from higher layer:\n{context}"
    
    async def search(self, query: str, top_k: int = 5) -> List[VectorSearchResult]:
        return await self.vector_store.search(
            query,
            filter_dict={"layer": self.layer_name},
            top_k=top_k
        ) 