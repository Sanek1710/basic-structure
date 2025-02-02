from .base_layer import BaseSemanticLayer
from typing import List, Any

class FunctionLayer(BaseSemanticLayer):
    def __init__(self, vector_store):
        super().__init__(
            vector_store, 
            "function",
            "sentence-transformers/all-mpnet-base-v2"  # Different model for function-level understanding
        )
    
    async def process_content(self, content: str) -> str:
        # Only used if processing without previous layer
        return content.strip()
    
    async def process_from_previous(
        self, 
        content: str, 
        previous_results: List[Any]
    ) -> str:
        # Group code snippets into function-level descriptions
        code_snippets = [r.content for r in previous_results]
        
        # Create function-level summary
        summary = f"""Function level summary:
        Code components: {len(code_snippets)}
        
        Combined functionality:
        {' '.join(code_snippets)}
        """
        
        return summary.strip() 