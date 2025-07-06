import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging messages (1=INFO, 2=WARNING, 3=ERROR)
import traceback
import argparse
import logging
from tqdm import tqdm

from core.config import config # Re-importing to ensure recognition
from core.utils import terminate_existing_processes
from core import temp_manager
from core import state_manager
from llm import llm_interaction
from core import utils

# Import agents
from core.agent_manager import AgentManager
from agents.video_input_agent import VideoInputAgent
from agents.storyboarding_agent import StoryboardingAgent
from agents.content_alignment_agent import ContentAlignmentAgent
from agents.audio_transcription_agent import AudioTranscriptionAgent
from agents.llm_selection_agent import LLMSelectionAgent
from agents.video_analysis_agent import VideoAnalysisAgent
from agents.video_editing_agent import VideoEditingAgent
from agents.results_summary_agent import ResultsSummaryAgent
from agents.broll_analysis_agent import BrollAnalysisAgent
from agents.audio_analysis_agent import AudioAnalysisAgent
from agents.intro_narration_agent import IntroNarrationAgent
from agents.frame_preprocessing_agent import FramePreprocessingAgent
from agents.qwen_vision_agent import QwenVisionAgent
from agents.engagement_analysis_agent import EngagementAnalysisAgent
from agents.layout_detection_agent import LayoutDetectionAgent
from agents.speaker_tracking_agent import SpeakerTrackingAgent
from agents.hook_identification_agent import HookIdentificationAgent
from agents.llm_video_director_agent import LLMVideoDirectorAgent
from agents.viral_potential_agent import ViralPotentialAgent
from agents.dynamic_editing_agent import DynamicEditingAgent
from agents.music_sync_agent import MusicSyncAgent
from agents.layout_optimization_agent import LayoutOptimizationAgent
from agents.subtitle_animation_agent import SubtitleAnimationAgent
from agents.content_enhancement_agent import ContentEnhancementAgent
from agents.quality_assurance_agent import QualityAssuranceAgent
from core.prompt_parser import parse_user_prompt # Moved from inside main function


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables
ffmpeg_path = config.get('ffmpeg_path')
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

# TODO: Consider creating a dedicated 'tools/' directory for managing external models and libraries.

# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""

def _get_frame_analysis_rate(state: dict):
    """Interactively prompts the user for the frame analysis rate and updates the state."""
    options = {
        0: 1.0,  # Default
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 5.0,
        5: 10.0,
        6: 15.0,
        7: 20.0,
        8: 30.0
    }

    logging.info("\n🖼️ Choose the rate for frame analysis (in seconds per frame):")
    for key, value in options.items():
        if key == 0:
            logging.info(f"  [{key}] Default ({value:.2f} seconds) - Recommended for initial runs.")
        else:
            logging.info(f"  [{key}] {value:.0f} seconds - {'(Faster, less detailed)' if value > 5 else '(Slower, more detailed)'}")
    logging.info("  [9] Custom")

    while True:
        try:
            choice = input("Enter your choice (0-9): ").strip()
            if choice == '0':
                seconds_per_frame = options[0]
            elif choice == '9':
                custom_spf_str = input("Enter custom seconds per frame: ").strip()
                custom_spf = float(custom_spf_str)
                if custom_spf <= 0:
                    logging.warning("Seconds per frame must be a positive number.")
                    continue
                seconds_per_frame = custom_spf
            elif int(choice) in options:
                seconds_per_frame = options[int(choice)]
            else:
                logging.warning("Invalid choice. Please enter a number between 0 and 9.")
                continue
            
            state["frame_analysis_rate"] = seconds_per_frame
            state_manager.update_state_file(state) # Save the choice immediately
            return
        except ValueError:
            logging.warning("Invalid input. Please enter a valid number.")
        except Exception as e:
            logging.error(f"An error occurred during frame analysis rate selection: {e}")
            state["frame_analysis_rate"] = options[0] # Fallback to default
            state_manager.update_state_file(state)
            return # Exit on error, state updated with default

def _get_default_state(args: dict) -> dict:
    """Returns the default state dictionary."""
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
        "args": args
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
    context = None  # Initialize context to None
    try:
        terminate_existing_processes() # Terminate any other running instances
        # --- Setup ---
        logging.info("🎬 === 60-Second Clips Generator ===\n")

        logging.info("🔍 Performing system checks...\n")
        utils.system_checks()

        logging.info("🧹 Initializing LLM and cleaning up VRAM...\n")
        llm_interaction.cleanup() # Initial VRAM cleanup on app start
        logging.info("✅ LLM initialization and VRAM cleanup complete.\n")

        # --- Resume Mechanism ---
        state = state_manager.load_state_file()
        
        if args.get("nocache"):
            logging.warning("🗑️ --nocache flag detected. Deleting previous state and temporary files...\n")
            state_manager.delete_state_file()
            temp_manager.cleanup_temp_dir()
            state = None # Force a fresh start
        
        if state: # If state exists after nocache check
            if args.get("retry"):
                logging.info("🔄 --retry flag detected. Attempting to resume previous session...\n")
            else:
                if state.get("failure_point"):
                    logging.warning(f"⚠️ Previous session failed at stage: {state.get("failure_point")}.\n")
                resume_choice = input("❓ Previous session detected. Resume? [y/n]: ").lower()
                if resume_choice != 'y':
                    logging.warning("🗑️ User declined resume. Deleting previous state and temporary files...\n")
                    state_manager.delete_state_file()
                    temp_manager.cleanup_temp_dir()
                    state = None # User declined, force fresh start
                else:
                    logging.info("🔄 Resuming previous session...\n")
        
        if not state: # If state is still None (either no previous state, nocache, or user declined)
            state = _get_default_state(args)
            state_manager.create_state_file(state)
        
        # --- Frame Analysis Rate Configuration ---
        if "frame_analysis_rate" not in state:
            _get_frame_analysis_rate(state)
        
        # Convert seconds per frame to FPS for agents
        if state.get("frame_analysis_rate") is not None:
            state["frame_extraction_rate"] = 1.0 / state["frame_analysis_rate"]
        else:
            state["frame_extraction_rate"] = config.get('qwen_vision.frame_extraction_rate_fps', 1) # Fallback to config default

        # Parse user prompt if provided
        user_prompt_arg = args.get("user_prompt")
        if user_prompt_arg:
            parsed_prompt = parse_user_prompt(user_prompt_arg)
            state["parsed_user_prompt"] = parsed_prompt
            state_manager.update_state_file(state) # Persist parsed prompt immediately
        
        # Initialize MultiAgent with the pipeline
        pipeline_agents = [
            VideoInputAgent(state_manager),
            FramePreprocessingAgent(config, state_manager),
            StoryboardingAgent(config, state_manager),
            AudioTranscriptionAgent(state_manager),
            AudioAnalysisAgent(config, state_manager),
            BrollAnalysisAgent(config, state_manager),
            LLMSelectionAgent(config, state_manager),
            QwenVisionAgent(config, state_manager),
            VideoAnalysisAgent(config, state_manager),
            EngagementAnalysisAgent(config, state_manager),
            LayoutDetectionAgent(config, state_manager),
            SpeakerTrackingAgent(config, state_manager),
            HookIdentificationAgent(config, state_manager),
            LLMVideoDirectorAgent(config, state_manager), # Moved before ContentAlignmentAgent
            ContentAlignmentAgent(config, state_manager),
            ViralPotentialAgent(config, state_manager),
            DynamicEditingAgent(config, state_manager),
            MusicSyncAgent(config, state_manager),
            LayoutOptimizationAgent(config, state_manager),
            SubtitleAnimationAgent(config, state_manager),
            ContentEnhancementAgent(config, state_manager),
            VideoEditingAgent(config, state_manager),
            QualityAssuranceAgent(config, state_manager),
            ResultsSummaryAgent()
        ]

        # Initialize AgentManager with the full pipeline
        agent_manager = AgentManager(pipeline_agents)
        with tqdm(total=len(pipeline_agents), desc="Processing Pipeline") as pbar:
            context = agent_manager.run(state, pbar)

        # Debugging: Check storyboard_data after StoryboardingAgent
        if 'storyboard_data' in context:
            logging.debug(f"DEBUG: Storyboard Data: {context['storyboard_data'][:2]}") # Print first 2 elements
        else:
            logging.debug("DEBUG: Storyboard Data not found in context.")

        # Update state after pipeline run
        state_manager.update_state_file(context)
        llm_interaction.cleanup() # Clean up LLM models on successful completion
        temp_manager.cleanup_temp_dir() # Clean up temporary directory on successful completion

    except KeyboardInterrupt:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        logging.info("\n\n❌ Process interrupted by user. Cleaning up...\n")
        state_manager.update_state_file({
            "failure_point": f"\nKeyboardInterrupt at stage: {current_stage}",
            "error_log": "\nProcess interrupted by user."
        })
    except Exception as e:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        error_message = f"\nA fatal error occurred at stage '{current_stage}': {e}"
        logging.error(f"\n\n❌ {error_message}\n")
        traceback.print_exc()
        state_manager.update_state_file({
            "failure_point": current_stage,
            "error_log": error_message
        })
        _log_troubleshooting_tips() # Call the new helper function
    finally:
        # Ensure context is a dictionary, even if it's None
        state_manager.handle_pipeline_completion(context if context is not None else {})


if __name__ == "__main__":
    # This block will only be executed when main.py is run directly
    # For CLI usage, cli.py will call main with arguments
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
