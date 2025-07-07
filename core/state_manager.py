import json
import os
import logging
from typing import Dict, Any

# Configure logging for the state manager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StateManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, state_file_path: str = 'temp_processing/.temp_clips.json'):
        if not self._initialized:
            self.state_file_path = state_file_path
            self.current_state: Dict[str, Any] = self._load_state_file()
            self._initialized = True

    def _load_state_file(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    state = json.load(f)
                    logging.info(f"Loaded state from {self.state_file_path}")
                    return state
            except json.JSONDecodeError:
                logging.warning(f"State file '{self.state_file_path}' is corrupted. Starting a fresh session.")
                return self._create_state_file()
        else:
            logging.info(f"State file '{self.state_file_path}' not found. Creating a new one.")
            return self._create_state_file()

    def _create_state_file(self) -> Dict[str, Any]:
        # Initialize with the new hierarchical structure
        initial_state = {
            "metadata": {},
            "current_analysis": {},
            "archived_data": {},
            "summaries": {},
            "pipeline_stages": {}
        }
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
        with open(self.state_file_path, 'w') as f:
            json.dump(initial_state, f, indent=4)
        logging.info(f"Created new state file at {self.state_file_path}")
        return initial_state

    @staticmethod
    def _deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merges source dictionary into destination dictionary.
        Lists are replaced, not merged.
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                destination[key] = StateManager._deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    def update_state_file(self, new_state_data: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"StateManager: Merging new state data: {new_state_data.keys()}")
        
        # Deep merge the new data into the current state
        self.current_state = self._deep_merge(new_state_data, self.current_state)
        
        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(self.current_state, f, indent=4)
            logging.info(f"State saved to {self.state_file_path}")
        except Exception as e:
            logging.error(f"Error saving state file: {e}")
        return self.current_state

    def get_state(self) -> Dict[str, Any]:
        return self.current_state

    def delete_state_file(self):
        """Deletes the state file."""
        if os.path.exists(self.state_file_path):
            try:
                os.remove(self.state_file_path)
                self.current_state = self._create_state_file() # Reset state in memory
                logging.info(f"State file '{self.state_file_path}' deleted.")
            except Exception as e:
                logging.error(f"Error deleting state file '{self.state_file_path}': {e}")
        else:
            logging.info(f"State file '{self.state_file_path}' does not exist, no need to delete.")

def set_stage_status(stage_name: str, status: str, details: Dict[str, Any] = None):
    state_manager = StateManager()
    current_pipeline_stages = state_manager.get_state().get('pipeline_stages', {})
    current_pipeline_stages[stage_name] = {'status': status, 'timestamp': os.path.getmtime(state_manager.state_file_path)}
    if details:
        current_pipeline_stages[stage_name].update(details)
    
    # Create a partial state update for pipeline_stages
    state_manager.update_state_file({'pipeline_stages': current_pipeline_stages})
