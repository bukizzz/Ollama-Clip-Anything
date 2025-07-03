from typing import List, Dict, Any
from agents.base_agent import Agent
from core.config import config
from core.state_manager import update_state_file
from llm import llm_interaction
import time
import psutil
import os
import logging

class AgentManager:
    """Manages the execution flow of various agents in the video processing pipeline."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, initial_context: Dict[str, Any], pbar) -> Dict[str, Any]:
        """
        Executes agents sequentially, passing the context from one to the next.
        Agents are executed only if their corresponding setting in config.yaml is True.
        The state is updated after each agent's successful execution.
        
        Args:
            initial_context (Dict[str, Any]): The initial context to start the pipeline.

        Returns:
            Dict[str, Any]: The final context after all agents have executed.
        """
        context = initial_context
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024) # in MB
        logging.info(f"Initial RAM usage: {initial_memory:.2f} MB")
        for agent in self.agents:
            agent_name_snake_case = ''.join(['_' + i.lower() if i.isupper() else i for i in agent.name]).lstrip('_')
            
            # Check if the agent should be executed based on config and current stage
            print(f"DEBUG: Current stage before {agent.name}: {context.get("current_stage")}")
            if config.get(f'agents.{agent_name_snake_case}.enabled', True) and \
               context.get("current_stage") != f"{agent_name_snake_case}_complete":
                
                print(f"Executing {agent.name}...")
                context = agent.execute(context)
                
                # Update the current stage in the context and save the state
                context["current_stage"] = f"{agent_name_snake_case}_complete"
                update_state_file(context)
                
                print(f"✅ \033[92mFinished {agent.name}. State saved.\033[0m")
                current_memory = process.memory_info().rss / (1024 * 1024)
                logging.info(f"RAM usage after {agent.name}: {current_memory:.2f} MB (Change: {current_memory - initial_memory:.2f} MB)")
                llm_interaction.cleanup() # Clear VRAM after each agent
                time.sleep(2) # Pause for 2 seconds
            else:
                print(f"⏩ Skipping {agent.name} as it is disabled in config or already completed.")
            pbar.update(1)
        return context