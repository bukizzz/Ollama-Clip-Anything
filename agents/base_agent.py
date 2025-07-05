
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

class Agent(ABC):
    """Base class for all agents in the video processing pipeline."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, message):
        print(f"ℹ️ {message}")

    def log_error(self, message):
        print(f"❌ {message}")

    def log_warning(self, message):
        print(f"⚠️ {message}")

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
