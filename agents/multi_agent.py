
from typing import List, Dict, Any
from agents.base_agent import Agent

class MultiAgent:
    """Orchestrates the execution of multiple agents in a pipeline."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the agents sequentially, passing the context from one to the next.

        Args:
            initial_context (Dict[str, Any]): The initial context to start the pipeline.

        Returns:
            Dict[str, Any]: The final context after all agents have executed.
        """
        context = initial_context
        for agent in self.agents:
            print(f"Executing {agent}...")
            context = agent.execute(context)
            print(f"Finished {agent}.")
        return context
