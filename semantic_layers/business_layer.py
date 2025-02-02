from .base_layer import BaseSemanticLayer
from typing import List, Any

class BusinessLayer(BaseSemanticLayer):
    def __init__(self, vector_store):
        super().__init__(
            vector_store, 
            "business",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Model good for business context
        )
    
    async def process_content(self, content: str) -> str:
        # Process business-level content
        # Could include requirements, user stories, and business rules
        return content.strip()
    
    async def process_from_previous(
        self, 
        content: str, 
        previous_results: List[Any]
    ) -> str:
        module_summaries = [r.content for r in previous_results]
        
        summary = f"""Business level summary:
        Modules: {len(module_summaries)}
        
        Business functionality:
        {' '.join(module_summaries)}
        """
        
        return summary.strip() 