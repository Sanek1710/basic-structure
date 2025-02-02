from .base_agent import BaseAgent

class CodeGeneratorAgent(BaseAgent):
    async def process(self, query: str) -> str:
        prompt = f"""Based on the following context:
        {self.context_manager.get_context_for_prompt()}
        
        Generate code for: {query}
        """
        
        return await self.llm.generate(prompt) 