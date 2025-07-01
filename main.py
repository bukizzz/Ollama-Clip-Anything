import os
import traceback
import argparse

# Set FFMPEG_BINARY before any MoviePy imports
os.environ["FFMPEG_BINARY"] = "/usr/local/bin/ffmpeg"

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

# TODO: Consider creating a dedicated 'tools/' directory for managing external models and libraries.

if FFMPEG_PATH:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH

# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""

def main():
    """Main function to run the video processing pipeline."""
    context = None  # Initialize context to None
    try:
        terminate_existing_processes() # Terminate any other running instances
        # --- Setup ---
        print("=== 60-Second Clips Generator ===")
        temp_manager.register_cleanup() # Ensure temp files are cleaned on exit
        
        # --- System Health Check ---
        print("\n0. Performing system checks...")
        utils.system_checks()
        print("Calling llm_interaction.cleanup() from main.py...")
        llm_interaction.cleanup() # Initial VRAM cleanup on app start
        print("Returned from llm_interaction.cleanup() in main.py.")

        parser = argparse.ArgumentParser(description="60-Second Clips Generator")
        parser.add_argument("--video_path", type=str, help="Path to a local MP4 video file.")
        parser.add_argument("--youtube_url", type=str, help="URL of a YouTube video to download.")
        parser.add_argument("--youtube_quality", type=int, help="Desired YouTube video quality option (e.g., 0, 1, 2...)." )
        parser.add_argument("--user_prompt", type=str, help="Optional: A specific prompt for the LLM to guide clip selection.")
        parser.add_argument("--retry", action="store_true", help="Automatically resume from a previous failed session.")
        parser.add_argument("--nocache", action="store_true", help="Force a fresh start, deleting any existing state and temporary files.")
        args = parser.parse_args()

        # --- Resume Mechanism ---
        state = state_manager.load_state_file()
        if args.nocache:
            print("🗑️ --nocache flag detected. Deleting previous state and temporary files...")
            state_manager.delete_state_file()
            state = None
        elif state:
            if args.retry:
                print("🔄 --retry flag detected. Attempting to resume previous session...")
            else:
                resume_choice = input("Previous session failed. Resume? [y/n]: ").lower()
                if resume_choice != 'y':
                    print("🗑️ User declined resume. Deleting previous state and temporary files...")
                    state_manager.delete_state_file()
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
                "processed_video_path": None,
                "user_prompt": args.user_prompt, # Add user_prompt to state
                "args": vars(args) # Pass args to the context for agents
            }
            state_manager.create_state_file(state)
        
        # Ensure state is always loaded/created before any potential exceptions
        state = state_manager.load_state_file() or state # Reload to ensure it's the latest from disk if created
        
        # Initialize MultiAgent with the pipeline
        pipeline = AgentManager([
            VideoInputAgent(),
            StoryboardingAgent(),
            ContentAlignmentAgent(),
            AudioTranscriptionAgent(),
            LLMSelectionAgent(),
            VideoAnalysisAgent(),
            VideoEditingAgent(),
            ResultsSummaryAgent()
        ])

        # Parse user prompt if provided
        if args.user_prompt:
            from core.prompt_parser import parse_user_prompt
            parsed_prompt = parse_user_prompt(args.user_prompt)
            state["parsed_user_prompt"] = parsed_prompt

        # Run the pipeline
        context = pipeline.run(state)

        # Update state after pipeline run
        state_manager.update_state_file(context)

    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user. Cleaning up...")
        state_manager.update_state_file({
            "failure_point": "KeyboardInterrupt",
            "error_log": "Process interrupted by user."
        })
    except Exception as e:
        logging.error(f"\n\n❌ A fatal error occurred: {e}")
        traceback.print_exc()
        state_manager.update_state_file({
            "failure_point": state.get("current_stage", "unknown"),
            "error_log": str(e)
        })
        logging.info("\n💡 Troubleshooting:")
        logging.info("   - Ensure FFmpeg and Ollama are properly installed and running.")
        logging.info("   - Verify the input video is not corrupted.")
        logging.info("   - Check for sufficient disk space.")
        logging.info("   - Ensure all required Python packages are installed:")
        logging.info("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors Pillow")
        logging.info("     python -m spacy download en_core_web_sm")
        utils.print_system_info()
    finally:
        if context and context.get("current_stage") == "results_summary_complete": # Assuming successful completion means all stages before final clip creation are done
            print("\n✅ Pipeline completed successfully. Deleting state and temporary files.")
            state_manager.delete_state_file()
        elif context and context.get("failure_point"):
            print("\n⚠️ Pipeline failed. State and temporary files preserved for resumption.")
        else:
            print("\nExiting application.")

        

if __name__ == "__main__":
    main()