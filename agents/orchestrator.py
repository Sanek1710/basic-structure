from typing import List, Dict, Any
from abc import ABC, abstractmethod
from .base_agent import BaseAgent
from .code_generator import CodeGeneratorAgent
from .validator import ValidatorAgent

class Agent(ABC):
    def __init__(self, name: str, required_layers: List[int]):
        self.name = name
        self.required_layers = required_layers
        
    @abstractmethod
    async def process(self, task: Dict[str, Any], context: str) -> Dict[str, Any]:
        pass

class AgentOrchestrator:
    def __init__(self, llm, context_manager):
        self.context_manager = context_manager
        self.agents: Dict[str, Agent] = {
            "code_generator": CodeGeneratorAgent(llm, context_manager),
            "validator": ValidatorAgent(llm, context_manager)
        }
    
    async def process_request(self, query: str, agent_type: str = "code_generator") -> str:
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Get context using top-down approach
        await self.context_manager.query_top_down(query, agent_type)
        
        # Process with appropriate agent
        agent = self.agents[agent_type]
        return await agent.process(query)

    def register_agent(self, agent: Agent):
        self.agents[agent.name] = agent
        
    async def execute_task(self, task: Dict[str, Any], context: str) -> Dict[str, Any]:
        # Determine which agents need to be involved
        required_agents = self._select_agents(task)
        
        results = {}
        for agent in required_agents:
            result = await agent.process(task, context)
            results[agent.name] = result
            
        return self._combine_results(results)
        
    def _select_agents(self, task: Dict[str, Any]) -> List[Agent]:
        # Logic to select relevant agents based on task
        task_type = task.get("type", "")
        if task_type == "code_generation":
            return [
                self.agents["generator"],
                self.agents["validator"]
            ]
        return []
        
    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Logic to combine results from multiple agents
        return results 