import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging messages (1=INFO, 2=WARNING, 3=ERROR)
import traceback
import argparse
import logging
from tqdm import tqdm

from core.config import config
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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Set FFMPEG_BINARY before any MoviePy imports
os.environ["FFMPEG_BINARY"] = config.get('ffmpeg_path')


# TODO: Consider creating a dedicated 'tools/' directory for managing external models and libraries.

if config.get('ffmpeg_path'):
    os.environ["IMAGEIO_FFMPEG_EXE"] = config.get('ffmpeg_path')

# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""

def main(args: dict):
    """Main function to run the video processing pipeline.
    Args:
        args (dict): A dictionary of arguments, typically from argparse or typer.
    """
    context = None  # Initialize context to None
    try:
        terminate_existing_processes() # Terminate any other running instances
        config_obj = config # Use the imported config object
        # --- Setup ---
        print("🎬 === 60-Second Clips Generator ===\n")

        print("🔍 Performing system checks...\n")
        utils.system_checks()

        print("🧹 Initializing LLM and cleaning up VRAM...\n")
        llm_interaction.cleanup() # Initial VRAM cleanup on app start
        print("✅ LLM initialization and VRAM cleanup complete.\n")

        # --- Resume Mechanism ---
        state = state_manager.load_state_file()
        if args.get("nocache"):
            print("🗑️ \033[93m--nocache flag detected. Deleting previous state and temporary files...\033[0m\n")
            state_manager.delete_state_file()
            temp_manager.cleanup_temp_dir() # Clean up temp directory when --nocache is used
            state = None
        elif args.get("retry") and state:
            print("🔄 \033[94m--retry flag detected. Attempting to resume previous session...\033[0m\n")
        elif state:
            if args.get("retry"):
                print("🔄 \033[94m--retry flag detected. Attempting to resume previous session...\033[0m\n")
            else:
                if state.get("failure_point"):
                    print(f"⚠️ \033[91mPrevious session failed at stage: {state.get("failure_point")}.\033[0m\n")
                resume_choice = input("❓ Previous session detected. Resume? [y/n]: ").lower()
                if resume_choice != 'y':
                    print("🗑️ \033[93mUser declined resume. Deleting previous state and temporary files...\033[0m\n")
                    state_manager.delete_state_file()
                    temp_manager.cleanup_temp_dir() # Ensure temp files are cleaned when user declines resume
                    state = None
                else:
                    print("🔄 \033[94mResuming previous session...\033[0m\n")
        
        if not state:
            state = {
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
                "user_prompt": args.get("user_prompt"), # Add user_prompt to state
                "args": args # Pass args to the context for agents
            }
            state_manager.create_state_file(state)

        
        
        # Ensure state is always loaded/created before any potential exceptions
        state = state_manager.load_state_file() or state # Reload to ensure it's the latest from disk if created
        
        # Initialize MultiAgent with the pipeline
        # Initialize MultiAgent with the pipeline
        pipeline_agents = [
            VideoInputAgent(state_manager),
            StoryboardingAgent(config_obj, state_manager),
            AudioTranscriptionAgent(state_manager),
            AudioAnalysisAgent(config_obj, state_manager),
            LLMSelectionAgent(config_obj, state_manager),
            IntroNarrationAgent(config_obj, state_manager),
            ContentAlignmentAgent(config_obj, state_manager),
            BrollAnalysisAgent(config_obj, state_manager),
            FramePreprocessingAgent(config_obj, state_manager),
            QwenVisionAgent(config_obj, state_manager),
            VideoAnalysisAgent(config_obj, state_manager),
            EngagementAnalysisAgent(config_obj, state_manager),
            LayoutDetectionAgent(config_obj, state_manager),
            SpeakerTrackingAgent(config_obj, state_manager),
            HookIdentificationAgent(config_obj, state_manager),
            LLMVideoDirectorAgent(config_obj, state_manager),
            LLMSelectionAgent(config_obj, state_manager),
            ViralPotentialAgent(config_obj, state_manager),
            DynamicEditingAgent(config_obj, state_manager),
            MusicSyncAgent(config_obj, state_manager),
            LayoutOptimizationAgent(config_obj, state_manager),
            SubtitleAnimationAgent(config_obj, state_manager),
            ContentEnhancementAgent(config_obj, state_manager),
            VideoEditingAgent(config_obj, state_manager),
            QualityAssuranceAgent(config_obj, state_manager),
            ResultsSummaryAgent()
        ]

        # Initialize AgentManager with the full pipeline
        agent_manager = AgentManager(pipeline_agents)
        with tqdm(total=len(pipeline_agents), desc="Processing Pipeline") as pbar:
            context = agent_manager.run(state, pbar)

        # Debugging: Check storyboard_data after StoryboardingAgent
        if 'storyboard_data' in context:
            print(f"DEBUG: Storyboard Data: {context['storyboard_data'][:2]}") # Print first 2 elements
        else:
            print("DEBUG: Storyboard Data not found in context.")

        # Parse user prompt if provided
        user_prompt_arg = args.get("user_prompt")
        if user_prompt_arg:
            # from core.prompt_parser import parse_user_prompt # Moved to top
            parsed_prompt = parse_user_prompt(user_prompt_arg)
            state["parsed_user_prompt"] = parsed_prompt

        # Update state after pipeline run
        state_manager.update_state_file(context)
        llm_interaction.cleanup() # Clean up LLM models on successful completion
        temp_manager.cleanup_temp_dir() # Clean up temporary directory on successful completion

    except KeyboardInterrupt:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        print("\n\n❌ Process interrupted by user. Cleaning up...\n")
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
        logging.info("\n💡 Troubleshooting:\n")
        logging.info("   - Ensure FFmpeg and Ollama are properly installed and running.\n")
        logging.info("   - Verify the input video is not corrupted.\n")
        logging.info("   - Check for sufficient disk space.\n")
        logging.info("   - Ensure all required Python packages are installed:\n")
        logging.info("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors Pillow TTS demucs\n")
        logging.info("     python -m spacy download en_core_web_sm\n")
        utils.print_system_info()
    finally:
        state_manager.handle_pipeline_completion(context)


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
    args = parser.parse_args()
    main(vars(args))