from typing import List, Dict, Any
from agents.base_agent import Agent

class AgentManager:
    """Manages the execution flow of various agents in the video processing pipeline."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes agents sequentially, passing the context from one to the next.
        
        Args:
            initial_context (Dict[str, Any]): The initial context to start the pipeline.

        Returns:
            Dict[str, Any]: The final context after all agents have executed.
        """
        context = initial_context
        for agent in self.agents:
            print(f"Executing {agent.name}...")
            context = agent.execute(context)
            print(f"Finished {agent.name}.")
        return context
