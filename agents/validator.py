from .base_agent import BaseAgent

class ValidatorAgent(BaseAgent):
    async def process(self, query: str) -> str:
        # Update context with focus on validation requirements
        await self.context_manager.update_context(query, "validator")
        
        prompt = f"""Based on the following context:
        {self.context_manager.get_context_for_prompt()}
        
        Validate the following code or requirement: {query}
        
        Check for:
        1. Code correctness
        2. Best practices
        3. Potential issues
        4. Security concerns
        """
        
        return await self.llm.generate(prompt) 