# core/state_manager.py
"""
Manages the application's state for failure recovery and resumption.
"""
import os
import json
from core.config import config
from core.temp_manager import cleanup_temp_dir as cleanup

def get_state_file_path():
    """Returns the full path to the state file."""
    return os.path.join(config.get('temp_dir'), config.get('state_file'))

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

def set_stage_status(stage_name: str, status: str, details: dict = None):
    """Sets the status of a specific pipeline stage."""
    state = load_state_file()
    if state:
        if "pipeline_stages" not in state:
            state["pipeline_stages"] = {}
        state["pipeline_stages"][stage_name] = {"status": status, "details": details, "timestamp": os.path.getmtime(get_state_file_path())}
        with open(get_state_file_path(), 'w') as f:
            json.dump(state, f, indent=4)

def get_stage_status(stage_name: str):
    """Gets the status of a specific pipeline stage."""
    state = load_state_file()
    if state and "pipeline_stages" in state and stage_name in state["pipeline_stages"]:
        return state["pipeline_stages"][stage_name]
    return None

def rollback_stage(stage_name: str):
    """Resets the status of a specific pipeline stage, effectively rolling back."""
    state = load_state_file()
    if state and "pipeline_stages" in state and stage_name in state["pipeline_stages"]:
        del state["pipeline_stages"][stage_name]
        with open(get_state_file_path(), 'w') as f:
            json.dump(state, f, indent=4)
        print(f"Rolled back stage: {stage_name}")

def delete_state_file():
    """Deletes the state file."""
    state_file = get_state_file_path()
    if os.path.exists(state_file):
        os.remove(state_file)

def handle_pipeline_completion(context: dict):
    """Handles state and temporary files based on pipeline completion status."""
    if context:
        processing_report = context.get("processing_report")
        if processing_report and processing_report.get("failed_clip_numbers"):
            print("\n⚠️ Some clips failed. State and temporary files preserved for resumption.")
        elif _check_all_stages_complete(context):
            print("\n✅ Pipeline completed successfully. Deleting state and temporary files.")
            delete_state_file()
            cleanup() # Explicitly clean up temp directory on successful completion
        elif context.get("failure_point"):
            print("\n⚠️ Pipeline failed. State and temporary files preserved for resumption.")
        else:
            print("\n👋 \u001b[94mExiting application.\u001b[0m")
    else:
        print("\n👋 \u001b[94mExiting application.\u001b[0m")

def _check_all_stages_complete(context: dict) -> bool:
    """Checks if all expected pipeline stages are marked as complete."""
    expected_stages = [
        "audio_rhythm_analysis_complete",
        "engagement_analysis_complete",
        "layout_detection_complete",
        "multimodal_analysis_complete",
        "speaker_tracking_complete",
        "intro_narration_generated",
        "qwen_vision_analysis_complete",
        "frame_feature_extraction_complete",
        "results_summary_complete" # Keep this as the final stage
    ]
    
    state = load_state_file()
    if not state or "pipeline_stages" not in state:
        return False

    for stage in expected_stages:
        if stage not in state["pipeline_stages"] or state["pipeline_stages"][stage]["status"] != "complete":
            return False
    return True
