from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .base_layer import BaseSemanticLayer

class SemanticLayer(ABC):
    def __init__(self, name: str, level: int):
        self.name = name
        self.level = level  # Lower number = closer to code
        
    @abstractmethod
    def process_content(self, content: Any) -> Dict[str, Any]:
        """Process content and create semantic embeddings with metadata"""
        pass
    
    @abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """Create summary for the next layer"""
        pass

class CodeLayer(SemanticLayer):
    def __init__(self):
        super().__init__("code_layer", 0)
        
    def process_content(self, content: str) -> Dict[str, Any]:
        # Process raw code, create embeddings
        return {
            "type": "code",
            "content": content,
            "embeddings": None  # Will be filled by vector store
        }
        
    def summarize(self) -> Dict[str, Any]:
        # Create summary for function/class layer
        return {
            "type": "code_summary",
            "content": None
        }

class FunctionLayer(SemanticLayer):
    def __init__(self):
        super().__init__("function_layer", 1)

class LayerProcessor:
    def __init__(self, layers: List[BaseSemanticLayer]):
        # Layers should be provided in order from bottom (code) to top (business)
        self.layers = layers
    
    async def process_bottom_up(self, initial_content: str) -> Dict[str, List[Any]]:
        results = {}
        previous_layer_results = None
        
        for layer in self.layers:
            if previous_layer_results is None:
                # First layer (code layer) processes raw content
                layer_results = await layer.process_with_dependencies(initial_content)
            else:
                # Higher layers process results from previous layer
                layer_results = await layer.process_with_dependencies(
                    initial_content, 
                    previous_layer_results
                )
            
            results[layer.layer_name] = layer_results
            previous_layer_results = layer_results
            
        return results
    
    async def query_top_down(self, query: str, starting_layer: str = "business") -> Dict[str, List[Any]]:
        # Find the starting layer index
        start_idx = next(
            (i for i, layer in enumerate(self.layers) 
             if layer.layer_name == starting_layer), 
            len(self.layers) - 1
        )
        
        results = {}
        current_results = None
        
        # Process from top layer down
        for layer in reversed(self.layers[:(start_idx + 1)]):
            if current_results is None:
                # Start with the top layer
                current_results = await layer.search(query)
            else:
                # Query lower layers using results from higher layer
                current_results = await layer.search_with_dependencies(
                    query, 
                    current_results
                )
            
            results[layer.layer_name] = current_results
            
        return results 