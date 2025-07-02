
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    """Base class for all agents in the video processing pipeline."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's task.

        Args:
            context (Dict[str, Any]): A dictionary containing the current state and data
                                       from previous agents.

        Returns:
            Dict[str, Any]: The updated context after the agent's execution.
        """
        pass

    def __str__(self):
        return f"🤖 Agent: {self.name}"
