from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContextItem:
    content: str
    layer_level: int
    relevance_score: float
    metadata: Dict[str, Any]

class ContextManager:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.active_context: List[ContextItem] = []
        
    def add_context(self, items: List[ContextItem]):
        # Add new context items and sort by relevance
        self.active_context.extend(items)
        self.active_context.sort(key=lambda x: x.relevance_score, reverse=True)
        self._trim_context()
        
    def _trim_context(self):
        """Trim context to fit max length while keeping most relevant items"""
        current_length = sum(len(item.content) for item in self.active_context)
        while current_length > self.max_context_length and self.active_context:
            # Remove least relevant item
            removed = self.active_context.pop()
            current_length -= len(removed.content)
            
    def get_formatted_context(self) -> str:
        """Format context for LLM prompt"""
        formatted = []
        for item in self.active_context:
            formatted.append(f"[{item.metadata['type']}]: {item.content}")
        return "\n\n".join(formatted) 