# main.py
"""
Main entry point for the 60-Second Clips Generator application.
This script orchestrates the entire video processing pipeline.
"""
import os
import traceback
import temp_manager
from video_input import choose_input_video
from audio_processing import transcribe_video
from llm_interaction import get_clips_with_retry
# Fixed import - use the correct function name from the enhanced video_editing module
from video_editing import batch_process_with_analysis, analyze_video_content
from utils import get_video_info, system_checks, print_system_info

def main():
    """Main function to run the video processing pipeline."""
    try:
        # --- Setup ---
        print("=== 60-Second Clips Generator ===")
        temp_manager.register_cleanup() # Ensure temp files are cleaned on exit
        
        # --- System Health Check ---
        print("\n0. Performing system checks...")
        system_checks()

        # --- Step 1: Get Input Video ---
        print("\n1. Getting input video...")
        input_video = choose_input_video()
        
        print("\n🔍 Analyzing input video...")
        video_info = get_video_info(input_video)
        print(f"📹 Video info: {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
              f"codec: {video_info['codec']}")

        # --- Step 2: Transcribe Video ---
        print("\n2. Transcribing video...")
        transcription = transcribe_video(input_video)
        if not transcription:
            raise RuntimeError("Transcription failed. Video may have no audio.")
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        # --- Step 3: Select Clips with LLM ---
        print("\n3. Selecting coherent clips using LLM...")
        clips = get_clips_with_retry(transcription)
        print(f"✅ Selected {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")

        # --- Step 4: Enhanced Video Analysis ---
        print(f"\n4. Performing enhanced video analysis...")
        video_analysis = analyze_video_content(input_video)
        
        # --- Step 5: Create Enhanced Clips ---
        print(f"\n5. Creating {len(clips)} enhanced video clips...")
        # Use the new enhanced processing function
        created_clips, processing_report = batch_process_with_analysis(
            input_video, clips, transcription
        )
        
        # Extract failed clips from the report
        failed_clips = processing_report.get('failed_clip_numbers', [])

        # --- Step 6: Results Summary ---
        print(f"\n--- Generation Complete! ---")
        print(f"📊 Successfully created: {len(created_clips)}/{len(clips)} clips.")
        print(f"⏱️  Total processing time: {processing_report.get('total_processing_time', 0):.1f}s")
        print(f"📈 Success rate: {processing_report['results']['success_rate']:.1f}%")
        
        if created_clips:
            output_dir = os.path.dirname(created_clips[0])
            print(f"📂 Clips saved in: {output_dir}")
            print(f"📄 Processing report saved in output directory")
            
        if failed_clips:
            print(f"❌ Failed clip numbers: {failed_clips}")
            print("   Consider checking the source video at those timestamps or converting it to H.264 first.")
            
        # Display enhanced features used
        print(f"\n🎨 Enhanced Features Applied:")
        print(f"   - Face tracking: {'✅' if video_analysis.get('has_faces') else '❌'}")
        print(f"   - Object detection: {'✅' if video_analysis.get('has_objects') else '❌'}")
        print(f"   - Animated subtitles: ✅")
        print(f"   - Scene effects: ✅")
        print(f"   - Content analysis: ✅")

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
        print_system_info()
    finally:
        print("\nExiting application.")

if __name__ == "__main__":
    main()
