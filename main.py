import os
import traceback
import atexit
from core.config import FFMPEG_PATH
from core.utils import terminate_existing_processes

if FFMPEG_PATH:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH

# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""
import os
import traceback
from core import temp_manager
from core import state_manager
from video import video_input
import argparse
from audio import audio_processing
import ollama
from llm import llm_interaction
# Fixed import - use the correct function name from the enhanced video_editing module

from video import video_editing
from analysis import analysis_and_reporting
from core import utils

def main():
    """Main function to run the video processing pipeline."""
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

        # --- Step 1: Get Input Video ---
        print("\n1. Getting input video...")
        parser = argparse.ArgumentParser(description="60-Second Clips Generator")
        parser.add_argument("--video_path", type=str, help="Path to a local MP4 video file.")
        parser.add_argument("--youtube_url", type=str, help="URL of a YouTube video to download.")
        parser.add_argument("--youtube_quality", type=int, help="Desired YouTube video quality option (e.g., 0, 1, 2...)." )
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
                "processed_video_path": None
            }
            state_manager.create_state_file(state)
        
        # Ensure state is always loaded/created before any potential exceptions
        state = state_manager.load_state_file() or state # Reload to ensure it's the latest from disk if created
        
        processed_video_path = state.get("processed_video_path")
        transcription = state.get("transcription")
        clips = state.get("clips")
        video_info = state.get("video_info")
        current_stage = state.get("current_stage")
        
        # --- Step 1: Get Input Video ---
        print("\n1. Getting input video...")
        if current_stage == "start":
            if args.video_path or args.youtube_url:
                input_video = video_input.get_video_input(video_path=args.video_path, youtube_url=args.youtube_url, youtube_quality=args.youtube_quality)
            else:
                input_video = video_input.choose_input_video()
            
            print("\n🔍 Analyzing input video...")
            video_info, processed_video_path = utils.get_video_info(input_video)
            state_manager.update_state_file({
                "input_source": input_video,
                "processed_video_path": processed_video_path,
                "video_info": video_info,
                "current_stage": "video_input_complete",
                "temp_files": {"processed_video": processed_video_path}
            })
        else:
            print(f"⏩ Skipping video input. Loaded from state: {processed_video_path}")
            
        print(f"📹 Video info: {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
              f"codec: {video_info['codec']}")

        # --- Step 2: Transcribe Video ---
        print("\n2. Transcribing video...")
        if current_stage == "start" or current_stage == "video_input_complete":
            transcription = audio_processing.transcribe_video(processed_video_path)
            if not transcription:
                raise RuntimeError("Transcription failed. Video may have no audio.")
            state_manager.update_state_file({
                "transcription": transcription,
                "current_stage": "transcription_complete",
                "temp_files": {"processed_video": processed_video_path, "transcription": "transcription_data_in_state"} # Placeholder for actual transcription file if saved
            })
        else:
            print("⏩ Skipping transcription. Loaded from state.")
            
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        # --- Step 3: Select Clips with LLM ---
        print("\n3. Selecting coherent clips using LLM...")
        if current_stage == "start" or current_stage == "video_input_complete" or current_stage == "transcription_complete":
            clips = llm_interaction.get_clips_with_retry(transcription)
            state_manager.update_state_file({
                "clips": clips,
                "current_stage": "llm_selection_complete"
            })
            
        else:
            print("⏩ Skipping LLM clip selection. Loaded from state.")
            
        print(f"✅ Selected {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")

        # --- Step 4: Enhanced Video Analysis ---
        print("\n4. Performing enhanced video analysis...")
        # This stage doesn't produce critical intermediate files for resumption, so no state update needed here.
        video_analysis = analysis_and_reporting.analyze_video_content(processed_video_path)

        # --- Step 5: Create Enhanced Clips ---
        print(f"\n5. Creating {len(clips)} enhanced video clips...")
        # Use the new enhanced processing function
        created_clips, processing_report = video_editing.batch_process_with_analysis(
            processed_video_path, clips, transcription
        )
        
        # Extract failed clips from the report
        failed_clips = processing_report.get('failed_clip_numbers', [])

        # --- Step 6: Results Summary ---
        print("\n--- Generation Complete! ---")
        print(f"📊 Successfully created: {len(created_clips)}/{len(clips)} clips.")
        print(f"⏱️  Total processing time: {processing_report.get('total_processing_time', 0):.1f}s")
        print(f"📈 Success rate: {processing_report['results']['success_rate']:.1f}%")
        
        if created_clips:
            output_dir = os.path.dirname(created_clips[0])
            print(f"📂 Clips saved in: {output_dir}")
            print("📄 Processing report saved in output directory")
            
        if failed_clips:
            print(f"❌ Failed clip numbers: {failed_clips}")
            print("   Consider checking the source video at those timestamps or converting it to H.264 first.")
            
        # Display enhanced features used
        print("\n🎨 Enhanced Features Applied:")
        print(f"   - Face tracking: {'✅' if video_analysis.get('has_faces') else '❌'}")
        print(f"   - Object detection: {'✅' if video_analysis.get('has_objects') else '❌'}")
        print("   - Animated subtitles: ✅")
        print("   - Scene effects: ✅")
        print("   - Content analysis: ✅")

    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user. Cleaning up...")
        state_manager.update_state_file({
            "failure_point": "KeyboardInterrupt",
            "error_log": "Process interrupted by user."
        })
    except Exception as e:
        print(f"\n\n❌ A fatal error occurred: {e}")
        traceback.print_exc()
        state_manager.update_state_file({
            "failure_point": state.get("current_stage", "unknown"),
            "error_log": str(e)
        })
        print("\n💡 Troubleshooting:")
        print("   - Ensure FFmpeg and Ollama are properly installed and running.")
        print("   - Verify the input video is not corrupted.")
        print("   - Check for sufficient disk space.")
        print("   - Ensure all required Python packages are installed:")
        print("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors")
        print("     python -m spacy download en_core_web_sm")
        utils.print_system_info()
    finally:
        if state and state.get("current_stage") == "llm_selection_complete": # Assuming successful completion means all stages before final clip creation are done
            print("\n✅ Pipeline completed successfully. Deleting state and temporary files.")
            state_manager.delete_state_file()
        elif state and state.get("failure_point"):
            print("\n⚠️ Pipeline failed. State and temporary files preserved for resumption.")
        else:
            print("\nExiting application.")

        

if __name__ == "__main__":
    main()
