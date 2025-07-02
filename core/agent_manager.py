from typing import List, Dict, Any
from agents.base_agent import Agent
from core.config import AGENT_CONFIG
from core.state_manager import update_state_file

class AgentManager:
    """Manages the execution flow of various agents in the video processing pipeline."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes agents sequentially, passing the context from one to the next.
        Agents are executed only if their corresponding setting in AGENT_CONFIG is True.
        The state is updated after each agent's successful execution.
        
        Args:
            initial_context (Dict[str, Any]): The initial context to start the pipeline.

        Returns:
            Dict[str, Any]: The final context after all agents have executed.
        """
        context = initial_context
        for agent in self.agents:
            agent_name_snake_case = ''.join(['_' + i.lower() if i.isupper() else i for i in agent.name]).lstrip('_')
            
            # Check if the agent should be executed based on config and current stage
            print(f"DEBUG: Current stage before {agent.name}: {context.get("current_stage")}")
            if AGENT_CONFIG.get(agent_name_snake_case, True) and \
               context.get("current_stage") != f"{agent_name_snake_case}_complete":
                
                print(f"Executing {agent.name}...")
                context = agent.execute(context)
                
                # Update the current stage in the context and save the state
                context["current_stage"] = f"{agent_name_snake_case}_complete"
                update_state_file(context)
                
                print(f"✅ \033[92mFinished {agent.name}. State saved.\033[0m")
            else:
                print(f"⏩ Skipping {agent.name} as it is disabled in config or already completed.")
        return context
