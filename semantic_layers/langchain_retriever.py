from langchain.schema import BaseRetriever, Document
from typing import List
from .base_layer import BaseSemanticLayer
from ..context_manager.context_manager import DynamicContextManager

class SemanticLayerRetriever(BaseRetriever):
    def __init__(self, context_manager: DynamicContextManager):
        super().__init__()
        self.context_manager = context_manager
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents from semantic layers"""
        # Use existing context manager to get results
        results = await self.context_manager.query_top_down(query, "default")
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.content,
                    metadata={
                        "layer": result.metadata.get("layer"),
                        "score": result.score
                    }
                )
            )
        
        return documents
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Sync version required by LangChain"""
        import asyncio
        return asyncio.run(self._aget_relevant_documents(query)) 