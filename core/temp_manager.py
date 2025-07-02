# temp_manager.py
"""
Manages the temporary directory for intermediate files.
"""
import os
import shutil
import atexit
from core.config import TEMP_DIR

def ensure_temp_dir():
    """Create the temporary directory if it doesn't exist."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def cleanup_temp_dir():
    """Remove the temporary directory and all its contents."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"🧹 Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️ \033[93mWarning: Could not fully clean up temp directory: {e}\033[0m")

def get_temp_path(filename: str) -> str:
    """Get a full path for a file inside the temporary directory."""
    ensure_temp_dir()
    return os.path.join(TEMP_DIR, filename)

def register_cleanup():
    """Register the cleanup function to run on application exit."""
    atexit.register(cleanup_temp_dir)
