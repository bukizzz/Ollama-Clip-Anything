# core/config.py
"""
Central configuration file for the video processing application.
Loads settings from config.yaml.
"""
import yaml
import os
from typing import Any

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

class Config:
    """A class to hold and provide access to configuration settings."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Loads or reloads configuration from config.yaml."""
        with open(CONFIG_FILE_PATH, 'r') as f:
            self._config_data = yaml.safe_load(f)

    def _save_config(self):
        """Saves the current configuration data back to config.yaml."""
        with open(CONFIG_FILE_PATH, 'w') as f:
            yaml.safe_dump(self._config_data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key, with optional default."""
        keys = key.split('.')
        val = self._config_data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any):
        """Sets a configuration value by key and persists the change."""
        keys = key.split('.')
        d = self._config_data
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                d[k] = value
            else:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
        self._save_config()
        self._load_config() # Reload to ensure in-memory is consistent

    def update_llm_active_model(self, config_key: str):
        """
        Switches the current_active_model for a given LLM model type
        (e.g., 'llm_model' or 'image_model')
        by moving to the next model in its priority list and persists the change to config.yaml.
        """
        self._load_config() # Ensure we're working with the latest config
        
        llm_config = self._config_data['llm']
        
        priority_list_key = f"{config_key}s_priority" # e.g., 'llm_models_priority'
        current_active_model_key = f"current_active_{config_key}" # e.g., 'current_active_llm_model'

        priority_list = llm_config.get(priority_list_key)
        current_active_model_name = llm_config.get(current_active_model_key)

        if not priority_list or not isinstance(priority_list, list) or not priority_list:
            raise ValueError(f"Priority list '{priority_list_key}' not found or is empty in LLM configuration.")

        try:
            current_index = priority_list.index(current_active_model_name)
            next_index = (current_index + 1) % len(priority_list)
            new_active_model = priority_list[next_index]
            llm_config[current_active_model_key] = new_active_model
            print(f"Switched {config_key} from {current_active_model_name} to {new_active_model}.")
        except ValueError:
            # current_active_model_name not found in priority_list, default to first
            new_active_model = priority_list[0]
            llm_config[current_active_model_key] = new_active_model
            print(f"Resetting {config_key} active model to first in list ({new_active_model}) as current was unknown or not in list.")

        self._save_config() # Persist the change
        self._load_config() # Reload the in-memory config to reflect the saved changes

# Instantiate the Config class to load settings on import
config = Config()

# DEBUG: Verify video_encoder value immediately after config loading
print(f"DEBUG: core/config.py - video_encoder from config: {config.get('video_encoder')}")

# Expose commonly used config values directly for convenience
CLIP_DURATION_RANGE = (config.get('clip_duration_min'), config.get('clip_duration_max'))
