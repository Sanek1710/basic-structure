from .base_layer import BaseSemanticLayer
import ast

class CodeLayer(BaseSemanticLayer):
    def __init__(self, vector_store):
        super().__init__(
            vector_store, 
            "code",
            "microsoft/codebert-base"  # Code-specific embedding model
        )
    
    async def process_content(self, content: str) -> str:
        try:
            # Parse code and extract basic elements
            tree = ast.parse(content)
            return content.strip()
        except SyntaxError:
            return content.strip() 