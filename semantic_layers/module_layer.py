from .base_layer import BaseSemanticLayer
from typing import List, Any

class ModuleLayer(BaseSemanticLayer):
    def __init__(self, vector_store):
        super().__init__(vector_store, "module")
    
    async def process_content(self, content: str) -> str:
        # Process module-level content
        # Could include module documentation, imports, and class relationships
        return content.strip()
    
    async def process_from_previous(
        self, 
        content: str, 
        previous_results: List[Any]
    ) -> str:
        # Group function-level content into module-level descriptions
        function_summaries = [r.content for r in previous_results]
        
        summary = f"""Module level summary:
        Functions: {len(function_summaries)}
        
        Module functionality:
        {' '.join(function_summaries)}
        """
        
        return summary.strip() 