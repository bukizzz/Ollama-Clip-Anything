import os
import traceback
import argparse
import logging

from core.config import FFMPEG_PATH
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
from core.prompt_parser import parse_user_prompt # Moved from inside main function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set FFMPEG_BINARY before any MoviePy imports
os.environ["FFMPEG_BINARY"] = "/usr/local/bin/ffmpeg"


# TODO: Consider creating a dedicated 'tools/' directory for managing external models and libraries.

if FFMPEG_PATH:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH

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
        # --- Setup ---
        print("🎬 === 60-Second Clips Generator ===")
        temp_manager.register_cleanup() # Ensure temp files are cleaned on exit
        
        # --- System Health Check ---
        print("\n🔍 0. Performing system checks...")
        utils.system_checks()
        print("🧹 Calling llm_interaction.cleanup() from main.py...")
        llm_interaction.cleanup() # Initial VRAM cleanup on app start
        print("✅ Returned from llm_interaction.cleanup() in main.py.")

        # --- Resume Mechanism ---
        state = state_manager.load_state_file()
        if args.get("nocache"):
            print("🗑️ --nocache flag detected. Deleting previous state and temporary files...")
            state_manager.delete_state_file()
            temp_manager.cleanup_temp_dir() # Clean up temp directory when --nocache is used
            state = None
        elif state:
            if args.get("retry"):
                print("🔄 --retry flag detected. Attempting to resume previous session...")
            else:
                resume_choice = input("Previous session failed. Resume? [y/n]: ").lower()
                if resume_choice != 'y':
                    print("🗑️ User declined resume. Deleting previous state and temporary files...")
                    state_manager.delete_state_file()
                    temp_manager.cleanup_temp_dir() # Clean up temp directory when resume is declined
                    state = None
                else:
                    print("🔄 Resuming previous session...")
        
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
        pipeline_agents = [
            VideoInputAgent(),
            StoryboardingAgent(),
            AudioTranscriptionAgent(), # Moved to run before ContentAlignmentAgent
            ContentAlignmentAgent(),
            BrollAnalysisAgent(),
            LLMSelectionAgent(),
            VideoAnalysisAgent(), # This agent initializes face_tracker and object_tracker
            VideoEditingAgent(),  # This agent needs face_tracker and object_tracker
            ResultsSummaryAgent()
        ]

        # Find the indices of VideoAnalysisAgent and VideoEditingAgent
        video_analysis_agent_index = -1
        video_editing_agent_index = -1
        for i, agent in enumerate(pipeline_agents):
            if isinstance(agent, VideoAnalysisAgent):
                video_analysis_agent_index = i
            elif isinstance(agent, VideoEditingAgent):
                video_editing_agent_index = i

        # Create sub-pipelines
        agents_before_analysis = pipeline_agents[:video_analysis_agent_index]
        agents_after_editing = pipeline_agents[video_editing_agent_index + 1:]

        # Initialize AgentManager for the first part of the pipeline
        pipeline_before_analysis = AgentManager(agents_before_analysis)
        context = pipeline_before_analysis.run(state)

        # Manually run VideoAnalysisAgent
        video_analysis_agent = pipeline_agents[video_analysis_agent_index]
        context = video_analysis_agent.execute(context)

        # Extract face_tracker and object_tracker instances
        face_tracker_instance = video_analysis_agent.face_tracker
        object_tracker_instance = video_analysis_agent.object_tracker

        # Manually run VideoEditingAgent
        video_editing_agent = pipeline_agents[video_editing_agent_index]
        context = video_editing_agent.execute(context, face_tracker_instance, object_tracker_instance)

        # Initialize AgentManager for the last part of the pipeline
        pipeline_after_editing = AgentManager(agents_after_editing)
        context = pipeline_after_editing.run(context)

        # Parse user prompt if provided
        user_prompt_arg = args.get("user_prompt")
        if user_prompt_arg:
            # from core.prompt_parser import parse_user_prompt # Moved to top
            parsed_prompt = parse_user_prompt(user_prompt_arg)
            state["parsed_user_prompt"] = parsed_prompt

        # Update state after pipeline run
        state_manager.update_state_file(context)

    except KeyboardInterrupt:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        print("\n\n❌ Process interrupted by user. Cleaning up...")
        state_manager.update_state_file({
            "failure_point": f"KeyboardInterrupt at stage: {current_stage}",
            "error_log": "Process interrupted by user."
        })
    except Exception as e:
        current_stage = context.get("current_stage", "unknown") if context else "unknown"
        error_message = f"A fatal error occurred at stage '{current_stage}': {e}"
        logging.error(f"\n\n❌ {error_message}")
        traceback.print_exc()
        state_manager.update_state_file({
            "failure_point": current_stage,
            "error_log": error_message
        })
        logging.info("\n💡 Troubleshooting:")
        logging.info("   - Ensure FFmpeg and Ollama are properly installed and running.")
        logging.info("   - Verify the input video is not corrupted.")
        logging.info("   - Check for sufficient disk space.")
        logging.info("   - Ensure all required Python packages are installed:")
        logging.info("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors Pillow TTS demucs")
        logging.info("     python -m spacy download en_core_web_sm")
        utils.print_system_info()
    finally:
        if context:
            processing_report = context.get("processing_report")
            if processing_report and processing_report.get("failed_clip_numbers"):
                print("\n⚠️ Some clips failed. State and temporary files preserved for resumption.")
            elif context.get("current_stage") == "results_summary_complete":
                print("\n✅ Pipeline completed successfully. Deleting state and temporary files.")
                state_manager.delete_state_file()
                temp_manager.cleanup_temp_dir() # Explicitly clean up temp directory on successful completion
            elif context.get("failure_point"):
                print("\n⚠️ Pipeline failed. State and temporary files preserved for resumption.")
            else:
                print("\n👋 \u001b[94mExiting application.\u001b[0m")
        else:
            print("\n👋 \u001b[94mExiting application.\u001b[0m")

        

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
