# temp_manager.py
"""
Manages the temporary directory for intermediate files.
"""
import os
import shutil

from core.config import config

def ensure_temp_dir():
    """Create the temporary directory if it doesn't exist."""
    if not os.path.exists(config.get('temp_dir')):
        os.makedirs(config.get('temp_dir'))

def cleanup_temp_dir():
    """Remove the temporary directory and all its contents."""
    if os.path.exists(config.get('temp_dir')):
        try:
            shutil.rmtree(config.get('temp_dir'))
            print(f"🧹 Cleaned up temporary directory: {config.get('temp_dir')}")
        except Exception as e:
            print(f"⚠️ \033[93mWarning: Could not fully clean up temp directory: {e}\033[0m")

def get_temp_path(filename: str) -> str:
    """Get a full path for a file inside the temporary directory."""
    ensure_temp_dir()
    return os.path.join(config.get('temp_dir'), filename)

def get_temp_dir() -> str:
    """Returns the path to the temporary directory."""
    ensure_temp_dir()
    return config.get('temp_dir')


