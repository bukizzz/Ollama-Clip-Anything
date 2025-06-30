import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""
import os
import traceback
from core import temp_manager
from video import video_input
import argparse
from audio import audio_processing
from llm import llm_interaction
# Fixed import - use the correct function name from the enhanced video_editing module

from video import video_editing
from analysis import analysis_and_reporting
from core import utils

def main():
    """Main function to run the video processing pipeline."""
    try:
        # --- Setup ---
        print("=== 60-Second Clips Generator ===")
        temp_manager.register_cleanup() # Ensure temp files are cleaned on exit
        
        # --- System Health Check ---
        print("\n0. Performing system checks...")
        utils.system_checks()

        # --- Step 1: Get Input Video ---
        print("\n1. Getting input video...")
        parser = argparse.ArgumentParser(description="60-Second Clips Generator")
        parser.add_argument("--video_path", type=str, help="Path to a local MP4 video file.")
        parser.add_argument("--youtube_url", type=str, help="URL of a YouTube video to download.")
        parser.add_argument("--youtube_quality", type=int, help="Desired YouTube video quality option (e.g., 0, 1, 2...).")
        args = parser.parse_args()

        if args.video_path or args.youtube_url:
            input_video = video_input.get_video_input(video_path=args.video_path, youtube_url=args.youtube_url, youtube_quality=args.youtube_quality)
        else:
            input_video = video_input.choose_input_video()
        
        print("\n🔍 Analyzing input video...")
        video_info, processed_video_path = utils.get_video_info(input_video)
        print(f"📹 Video info: {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
              f"codec: {video_info['codec']}")

        # --- Step 2: Transcribe Video ---
        print("\n2. Transcribing video...")
        transcription = audio_processing.transcribe_video(processed_video_path)
        if not transcription:
            raise RuntimeError("Transcription failed. Video may have no audio.")
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        # --- Step 3: Select Clips with LLM ---
        print("\n3. Selecting coherent clips using LLM...")
        clips = llm_interaction.get_clips_with_retry(transcription)
        print(f"✅ Selected {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")

        # --- Step 4: Enhanced Video Analysis ---
        print("\n4. Performing enhanced video analysis...")
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
    except Exception as e:
        print(f"\n\n❌ A fatal error occurred: {e}")
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   - Ensure FFmpeg and Ollama are properly installed and running.")
        print("   - Verify the input video is not corrupted.")
        print("   - Check for sufficient disk space.")
        print("   - Ensure all required Python packages are installed:")
        print("     pip install opencv-python torch torchvision mediapipe spacy scikit-learn librosa webcolors")
        print("     python -m spacy download en_core_web_sm")
        utils.print_system_info()
    finally:
        print("\nExiting application.")

if __name__ == "__main__":
    main()