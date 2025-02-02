from abc import ABC, abstractmethod
from ..llm.llm_interface import LLMInterface
from ..context_manager.context_manager import DynamicContextManager

class BaseAgent(ABC):
    def __init__(self, llm: LLMInterface, context_manager: DynamicContextManager):
        self.llm = llm
        self.context_manager = context_manager
    
    @abstractmethod
    async def process(self, query: str) -> str:
        pass 