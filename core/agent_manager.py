import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from agents.base_agent import Agent
from core.state_manager import StateManager
from core.monitoring import Monitor
from core.resource_manager import resource_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentManager:
    def __init__(self, config: Dict[str, Any], state_manager: StateManager, monitor: Monitor):
        self.config = config
        self.state_manager = state_manager
        self.monitor = monitor
        self.parallel_groups = self.config.get('agents', {}).get('parallel_groups', {})

    def _run_sequential_agent(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_name = agent.name
        self.monitor.start_stage(stage_name)
        logging.info(f"Executing sequential agent {stage_name}...")
        try:
            logging.info(f"DEBUG: video_encoder in context before {stage_name}: {context.get('metadata', {}).get('processing_settings', {}).get('video_encoder')}")
            updated_context = agent.execute(context)
            self.state_manager.update_state_file(updated_context)
            logging.info(f"✅ Finished {stage_name}. State saved.")
            self.monitor.end_stage(stage_name)
            
            # Get current RAM/VRAM usage from resource_manager
            current_ram_gb = resource_manager.get_current_memory_usage_gb()
            current_vram_gb = resource_manager.get_current_vram_usage_gb()
            self.monitor.record_memory_usage(current_ram_gb, current_vram_gb)
            
            return updated_context
        except Exception as e:
            logging.error(f"Error in sequential agent {stage_name}: {e}")
            self.monitor.end_stage(stage_name)
            self.monitor.record_error(str(e), stage_name)
            resource_manager.unload_all_models()
            raise

    def _run_parallel_agent(self, agent_name: str, context_copy: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to run an agent in a separate thread for parallel execution."""
        agent = next((a for a in self.pipeline_agents if a.name == agent_name), None)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found for parallel execution.")

        stage_name = agent.name
        self.monitor.start_stage(stage_name)
        logging.info(f"Submitting {stage_name} for parallel execution.")
        try:
            logging.info(f"DEBUG: video_encoder in context before {stage_name}: {context_copy.get('metadata', {}).get('processing_settings', {}).get('video_encoder')}")
            updated_context = agent.execute(context_copy)
            self.monitor.end_stage(stage_name)
            
            # Get current RAM/VRAM usage from resource_manager
            current_ram_gb = resource_manager.get_current_memory_usage_gb()
            current_vram_gb = resource_manager.get_current_vram_usage_gb()
            self.monitor.record_memory_usage(current_ram_gb, current_vram_gb)
            
            return updated_context
        except Exception as e:
            logging.error(f"Error in parallel agent {stage_name}: {e}")
            self.monitor.end_stage(stage_name)
            self.monitor.record_error(str(e), stage_name)
            resource_manager.unload_all_models()
            raise

    def run(self, pipeline_agents: List[Agent], initial_context: Dict[str, Any], pbar: tqdm) -> Dict[str, Any]:
        self.pipeline_agents = pipeline_agents
        self.context = initial_context
        
        executed_parallel_agents = set()

        # Determine already completed stages from the context
        completed_stages = {stage for stage, status in self.context.get('pipeline_stages', {}).items() if status == 'complete'}
        
        # Update progress bar for already completed stages
        pbar.update(len(completed_stages))

        for agent in pipeline_agents:
            stage_name = agent.name
            if stage_name in completed_stages:
                logging.info(f"Skipping already completed stage: {stage_name}")
                continue

            if stage_name in executed_parallel_agents:
                continue

            is_parallel_group_start = False
            current_parallel_group_agents = []
            for group_name, agent_names in self.parallel_groups.items():
                if stage_name in agent_names:
                    is_parallel_group_start = True
                    current_parallel_group_agents = [a for a in self.pipeline_agents if a.name in agent_names]
                    break
            
            if is_parallel_group_start:
                logging.info(f"Executing parallel group '{group_name}': {[a.name for a in current_parallel_group_agents]}")
                with ThreadPoolExecutor(max_workers=len(current_parallel_group_agents)) as executor:
                    futures = {executor.submit(self._run_parallel_agent, a.name, self.context.copy()): a for a in current_parallel_group_agents}
                    
                    for future in as_completed(futures):
                        completed_agent = futures[future]
                        try:
                            updated_context_from_parallel_agent = future.result()
                            self.context = self.state_manager.update_state_file(updated_context_from_parallel_agent)
                            logging.info(f"✅ Finished parallel agent {completed_agent.name}. State saved.")
                            executed_parallel_agents.add(completed_agent.name)
                            pbar.update(1)
                        except Exception as e:
                            logging.error(f"Error in parallel agent {completed_agent.name}: {e}")
                            raise
            else:
                self.context = self._run_sequential_agent(agent, self.context)
                pbar.update(1)

        return self.context
