import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging messages (1=INFO, 2=WARNING, 3=ERROR)
import traceback
import argparse
import logging
from tqdm import tqdm

from core.config import config
from core.utils import terminate_existing_processes
from core import temp_manager
from core.state_manager import StateManager
from llm import llm_interaction
from core import utils
from core.monitoring import monitor

# Import agents
from core.agent_manager import AgentManager
from agents.video_input_agent import VideoInputAgent
from agents.storyboarding_agent import StoryboardingAgent
from agents.content_director_agent import ContentDirectorAgent
from agents.audio_intelligence_agent import AudioIntelligenceAgent
from agents.llm_selection_agent import LLMSelectionAgent
from agents.broll_analysis_agent import BrollAnalysisAgent
from agents.frame_preprocessing_agent import FramePreprocessingAgent
from agents.layout_speaker_agent import LayoutSpeakerAgent
from agents.hook_identification_agent import HookIdentificationAgent
from agents.viral_potential_agent import ViralPotentialAgent
from agents.dynamic_editing_agent import DynamicEditingAgent
from agents.music_sync_agent import MusicSyncAgent
from agents.layout_optimization_agent import LayoutOptimizationAgent
from agents.subtitle_animation_agent import SubtitleAnimationAgent
from agents.video_production_agent import VideoProductionAgent
from agents.multimodal_analysis_agent import MultimodalAnalysisAgent
from agents.results_summary_agent import ResultsSummaryAgent
from core.prompt_parser import parse_user_prompt


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables
ffmpeg_path = config.get('ffmpeg_path')
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

def _get_default_state(args: dict) -> dict:
    """Returns the default state dictionary."""
    # Determine frame_analysis_rate: prioritize CLI arg, then config, then default
    frame_analysis_rate = args.get("image_analysis_fps")
    if frame_analysis_rate is None:
        frame_analysis_rate = config.get('qwen_vision.frame_extraction_rate_fps', 1)

    return {
        "input_source": None,
        "current_stage": "start",
        "completed_segments": [],
        "temp_files": {},
        "failure_point": None,
        "error_log": None,
        "segment_queue": [],
        "video_info": None,
        "transcription": None,
        "clips": None,
        "user_prompt": args.get("user_prompt"),
        "args": args,
        "pipeline_stages": {},
        "metadata": {
            "processing_settings": {
                "frame_analysis_rate": frame_analysis_rate,
                "frame_extraction_rate": 1.0 / frame_analysis_rate if frame_analysis_rate else 1, # Ensure it's not zero
                "video_encoder": config.get('video_encoder', 'libx264') # Add video_encoder from config
            }
        }
    }

def _log_troubleshooting_tips():
    """Logs common troubleshooting tips."""
    logging.info("\n💡 Troubleshooting:\n")
    logging.info("   - Ensure FFmpeg and Ollama are properly installed and running.\n")
    logging.info("   - Verify the input video is not corrupted.\n")
    logging.info("   - Check for sufficient disk space.\n")
    logging.info("   - Ensure all required Python packages are installed:\n")
    logging.info("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors Pillow TTS demucs\n")
    logging.info("     python -m spacy download en_core_web_sm\n")
    utils.print_system_info()

def main(args: dict):
    """Main function to run the video processing pipeline.
    Args:
        args (dict): A dictionary of arguments, typically from argparse or typer.
    """
    context = None
    
    state_manager_instance = StateManager()
    # monitor is already a global instance imported from core.monitoring

    monitor.start_stage("overall_pipeline")
    try:
        terminate_existing_processes()

        logging.info("🎬 === 60-Second Clips Generator ===\n")
        logging.info("🔍 Performing system checks...\n")
        utils.system_checks()

        logging.info("🧹 Initializing LLM and cleaning up VRAM...\n")
        llm_interaction.cleanup()
        logging.info("✅ LLM initialization and VRAM cleanup complete.\n")

        # --- Resume Mechanism ---
        state = state_manager_instance._load_state_file()
        
        # Check if new video input arguments are provided, which should override any previous state
        new_video_input_provided = args.get("video_path") or args.get("youtube_url")

        if args.get("nocache"):
            logging.warning("🗑️ --nocache flag detected. Deleting previous state and temporary files...\n")
            state_manager_instance.delete_state_file()
            temp_manager.cleanup_temp_dir()
            state = None
        elif new_video_input_provided and state and state.get('processed_video_path') and \
             (state['args'].get('video_path') != args.get('video_path') or \
              state['args'].get('youtube_url') != args.get('youtube_url')):
            # If new video input is provided and it's different from the one in the saved state,
            # force a fresh start to process the new video.
            logging.warning("🔄 New video input detected. Deleting previous state and temporary files to process new video...\n")
            state_manager_instance.delete_state_file()
            temp_manager.cleanup_temp_dir()
            state = None
        elif state:
            if args.get("retry"):
                logging.info("🔄 --retry flag detected. Attempting to resume previous session...\n")
            else:
                if state.get("failure_point"):
                    logging.warning(f"⚠️ Previous session failed at stage: {state.get("failure_point")}.\n")
                resume_choice = input("❓ Previous session detected. Resume? [y/n]: ").lower()
                if resume_choice != 'y':
                    logging.warning("🗑️ User declined resume. Deleting previous state and temporary files...\n")
                    state_manager_instance.delete_state_file()
                    temp_manager.cleanup_temp_dir()
                    state = None
                else:
                    logging.info("🔄 Resuming previous session...\n")
        
        if not state:
            state = _get_default_state(args)
            temp_manager.ensure_temp_dir() # Ensure temp directory exists before creating state file
            state_manager_instance.update_state_file(state)
        
        # Always ensure state['args'] reflects the current command-line arguments
        state["args"] = args
        state_manager_instance.update_state_file(state) # Save updated args to state

        # Ensure frame_extraction_rate is set from frame_analysis_rate
        # This block is now redundant as frame_analysis_rate and frame_extraction_rate are set in _get_default_state
        # if state.get("frame_analysis_rate"):
        #     state["frame_extraction_rate"] = 1.0 / state["frame_analysis_rate"]
        # else:
        #     # Fallback to default if not set for any reason
        #     state["frame_extraction_rate"] = config.get('qwen_vision.frame_extraction_rate_fps', 1)

        # Update metadata structure consistently
        state.setdefault("metadata", {})
        state["metadata"].setdefault("processing_settings", {})
        # These values are now set in _get_default_state, but we ensure they are updated if args change
        state["metadata"]["processing_settings"]["frame_analysis_rate"] = args.get("image_analysis_fps") or config.get('qwen_vision.frame_extraction_rate_fps', 1)
        state["metadata"]["processing_settings"]["frame_extraction_rate"] = 1.0 / state["metadata"]["processing_settings"]["frame_analysis_rate"] if state["metadata"]["processing_settings"]["frame_analysis_rate"] else 1
        state["metadata"]["processing_settings"]["video_encoder"] = config.get('video_encoder', 'libx264')
        # Add video output settings to processing_settings
        state["metadata"]["processing_settings"]["video_output"] = {
            "target_resolution_heights": config.get('video_output.target_resolution_heights'),
            "target_aspect_ratios": config.get('video_output.target_aspect_ratios')
        }
        state_manager_instance.update_state_file(state)

        # DEBUG: Check video_encoder after all state updates
        print(f"DEBUG: main.py - video_encoder in state after all updates: {state.get('metadata', {}).get('processing_settings', {}).get('video_encoder')}")


        user_prompt_arg = args.get("user_prompt")
        if user_prompt_arg:
            parsed_prompt = parse_user_prompt(user_prompt_arg)
            state["parsed_user_prompt"] = parsed_prompt
            state_manager_instance.update_state_file(state)
        
        pipeline_agents = [
            VideoInputAgent(config, state_manager_instance),
            StoryboardingAgent(config, state_manager_instance),
            MultimodalAnalysisAgent(config, state_manager_instance),
            AudioIntelligenceAgent(config, state_manager_instance),
            LayoutSpeakerAgent(config, state_manager_instance),
            BrollAnalysisAgent(config, state_manager_instance),
            LLMSelectionAgent(config, state_manager_instance),
            HookIdentificationAgent(config, state_manager_instance),
            ContentDirectorAgent(config, state_manager_instance),
            ViralPotentialAgent(config, state_manager_instance),
            DynamicEditingAgent(config, state_manager_instance),
            MusicSyncAgent(config, state_manager_instance),
            LayoutOptimizationAgent(config, state_manager_instance),
            SubtitleAnimationAgent(config, state_manager_instance),
            VideoProductionAgent(config, state_manager_instance),
            ResultsSummaryAgent()
        ]

        agent_manager = AgentManager(config._config_data, state_manager_instance, monitor) # Pass config._config_data
        with tqdm(total=len(pipeline_agents), desc="Processing Pipeline") as pbar:
            context = agent_manager.run(pipeline_agents, state, pbar)

        if 'storyboard_data' in context:
            logging.debug(f"DEBUG: Storyboard Data: {context['storyboard_data'][:2]}")
        else:
            logging.debug("DEBUG: Storyboard Data not found in context.")

        state_manager_instance.update_state_file(context)
        llm_interaction.cleanup()
        temp_manager.cleanup_temp_dir()
        monitor.finalize_metrics()

    except KeyboardInterrupt:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        logging.info("\n\n❌ Process interrupted by user. Cleaning up...\n")
        state_manager_instance.update_state_file({
            "failure_point": f"\nKeyboardInterrupt at stage: {current_stage}",
            "error_log": "\nProcess interrupted by user."
        })
    except Exception as e:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        error_message = f"\nA fatal error occurred at stage '{current_stage}': {e}"
        logging.error(f"\n\n❌ {error_message}\n")
        traceback.print_exc()
        state_manager_instance.update_state_file({
            "failure_point": current_stage,
            "error_log": error_message
        })
        _log_troubleshooting_tips()
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="60-Second Clips Generator")
    parser.add_argument("--video_path", type=str, help="Path to a local MP4 video file.")
    parser.add_argument("--youtube_url", type=str, help="URL of a YouTube video to download.")
    parser.add_argument("--youtube_quality", type=int, help="Desired YouTube video quality option (e.g., 0, 1, 2...)." )
    parser.add_argument("--user_prompt", type=str, help="Optional: A specific prompt for the LLM to guide clip selection.")
    parser.add_argument("--retry", action="store_true", help="Automatically resume from a previous failed session.")
    parser.add_argument("--nocache", action="store_true", help="Force a fresh start, deleting any existing state and temporary files.")
    parser.add_argument("--image_analysis_fps", type=float, help="Set the FPS for image analysis.")
    args = parser.parse_args()
    main(vars(args))
