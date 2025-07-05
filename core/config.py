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
        with open(CONFIG_FILE_PATH, 'r') as f:
            self._config_data = yaml.safe_load(f)

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

# Instantiate the Config class to load settings on import
config = Config()

# Expose commonly used config values directly for convenience
CLIP_DURATION_RANGE = (config.get('clip_duration_min'), config.get('clip_duration_max'))


