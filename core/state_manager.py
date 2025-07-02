# core/state_manager.py
"""
Manages the application's state for failure recovery and resumption.
"""
import os
import json
from core.config import TEMP_DIR, STATE_FILE
from core.temp_manager import cleanup_temp_dir as cleanup

def get_state_file_path():
    """Returns the full path to the state file."""
    return os.path.join(TEMP_DIR, STATE_FILE)

def create_state_file(initial_state):
    """Creates a new state file."""
    state_file = get_state_file_path()
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(initial_state, f, indent=4)

def load_state_file():
    """Loads the state file if it exists."""
    state_file = get_state_file_path()
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None

def update_state_file(updates):
    """Updates specific fields in the state file."""
    state = load_state_file()
    if state:
        state.update(updates)
        with open(get_state_file_path(), 'w') as f:
            json.dump(state, f, indent=4)

def delete_state_file():
    """Deletes the state file."""
    state_file = get_state_file_path()
    if os.path.exists(state_file):
        os.remove(state_file)