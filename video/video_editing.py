import os
import logging
import time
from typing import List, Dict, Tuple, Optional

from moviepy.editor import VideoFileClip

from core.temp_manager import get_temp_path
from core.config import OUTPUT_DIR, CLIP_PREFIX, VIDEO_ENCODER, FFMPEG_GLOBAL_PARAMS, FFMPEG_ENCODER_PARAMS
from core.ffmpeg_command_logger import FFMPEGCommandLogger
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker
from video.frame_processor import FrameProcessor
from analysis.analysis_and_reporting import analyze_video_content, optimize_processing_settings, create_processing_report, save_processing_report
from audio.subtitle_generation import create_ass_file

# Configure MoviePy's logger to capture FFmpeg commands
logging.setLoggerClass(FFMPEGCommandLogger)
logger = logging.getLogger('moviepy')
logger.setLevel(logging.INFO)

# Add a StreamHandler to ensure messages are printed to console
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def get_next_output_filename(source_video_path: str, clip_number: int) -> str:
    """Generate unique output filename"""
    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    folder_name = f"{source_name}_enhanced"
    output_folder = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return os.path.join(output_folder, f"{CLIP_PREFIX}_enhanced_{clip_number}.mp4")

def create_enhanced_individual_clip(
    original_video_path: str, 
    clip_data: Dict, 
    clip_number: int, 
    video_info: Dict,
    transcript: List[Dict],
    enable_face_tracking: bool = True,
    enable_object_tracking: bool = True,
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None
) -> str:
    """Create an individual clip with all enhanced features"""
    start, end = float(clip_data['start']), float(clip_data['end'])
    duration = end - start

    print(f"Creating enhanced clip {clip_number}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")

    try:
        # Load video clip
        original_clip = VideoFileClip(original_video_path).subclip(start, end)
        
        # Get video dimensions
        original_w, original_h = video_info['width'], video_info['height']
        
        # Initialize processors
        face_tracker = face_tracker_instance if enable_face_tracking else None
        object_tracker = object_tracker_instance if enable_object_tracking else None
        
        # Define the final output resolution (9:16 aspect ratio)
        output_h = original_h
        output_w = int(output_h * 9 / 16)

        if output_w > original_w:
            output_w = original_w
            output_h = int(original_w * 16 / 9)
        
        if output_w % 2 != 0:
            output_w += 1
        if output_h % 2 != 0:
            output_h += 1

        processor = FrameProcessor(original_w, original_h, output_w, output_h, face_tracker, object_tracker)
        processed_video_clip = original_clip.fl(processor.process_frame)
        
        # Generate subtitles for the clip
        ass_path = get_temp_path(f"subtitles_{clip_number}.ass")
        print(f"DEBUG: Transcript object: {transcript}")
        create_ass_file(transcript, ass_path, time_offset=start)

        # Output path
        output_path = get_next_output_filename(original_video_path, clip_number)
        
        # Write final video
        ffmpeg_params = list(FFMPEG_GLOBAL_PARAMS)
        
        # Add subtitle filter
        ffmpeg_params.extend(['-vf', f"subtitles={ass_path}"])

        print(f"DEBUG: FFmpeg parameters: {ffmpeg_params}")
        processed_video_clip.write_videofile(
            output_path,
            codec=VIDEO_ENCODER, # Explicitly set the video codec
            audio_codec='aac',
            temp_audiofile=get_temp_path(f'temp_audio_enhanced_{clip_number}.m4a'),
            remove_temp=True,
            fps=30,
            ffmpeg_params=ffmpeg_params
        )

        # Cleanup
        original_clip.close()
        processed_video_clip.close()
        
        print(f"Successfully created enhanced clip {clip_number}")
        return output_path

    except Exception as e:
        print(f"Failed to create enhanced clip {clip_number}: {e}")
        raise

def batch_create_enhanced_clips(
    original_video_path: str, 
    clips_data: List[Dict], 
    transcript: List[Dict],
    video_info: Dict,
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None,
    logger: logging.Logger = None,
    **enhancement_options
) -> Tuple[List[str], List[int]]:
    """Create multiple enhanced clips with all features"""
    print(f"Creating {len(clips_data)} enhanced clips with advanced features...")
    
    valid_clip_options = {
        'enable_face_tracking': enhancement_options.get('enable_face_tracking', True),
        'enable_object_tracking': enhancement_options.get('enable_object_tracking', True),
    }
    
    created_clips = []
    failed_clips = []

    for i, clip_data in enumerate(clips_data, 1):
        try:
            print(f"\n--- Processing Enhanced Clip {i}/{len(clips_data)} ---")
            clip_path = create_enhanced_individual_clip(
                original_video_path, clip_data, i, video_info, 
                transcript,
                face_tracker_instance=face_tracker_instance, 
                object_tracker_instance=object_tracker_instance, 
                **valid_clip_options
            )
            created_clips.append(clip_path)
        except Exception as e:
            print(f"Failed to create enhanced clip {i}: {e}")
            failed_clips.append(i)

    print(f"\nBatch processing complete: {len(created_clips)} successful, {len(failed_clips)} failed")
    if failed_clips:
        print(f"Failed clips: {failed_clips}")
    
    return created_clips, failed_clips



def detect_rhythm_and_beats(video_path: str) -> List[float]:
    """Placeholder for rhythm and beat detection.
    This would typically use audio analysis libraries like librosa to detect beats.
    Returns a list of beat timestamps.
    """
    print("Rhythm and beat detection is not yet implemented. Skipping this step.")
    # In a real implementation, you would analyze the audio track of the video
    # to detect beats or rhythm changes.
    # Example:
    # import librosa
    # y, sr = librosa.load(audio_path)
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # return librosa.frames_to_time(beats, sr=sr).tolist()
    return []

def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    custom_settings: Optional[Dict] = None
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    face_tracker_instance = FaceTracker()
    object_tracker_instance = ObjectTracker()

    print("\n=== Step 1: Video Content Analysis ===")
    video_analysis = analyze_video_content(video_path, face_tracker=face_tracker_instance, object_tracker=object_tracker_instance)
    
    print("\n=== Step 2: Processing Optimization ===")
    processing_settings = optimize_processing_settings(video_analysis)
    
    if custom_settings:
        processing_settings.update(custom_settings)
        print("Applied custom settings overrides")
    
    video_info = {
        'width': video_analysis['width'],
        'height': video_analysis['height'],
        'duration': video_analysis['duration'],
        'fps': video_analysis['fps']
    }

    print("\n=== Step 3: Rhythm and Beat Detection ===")
    rhythm_info = detect_rhythm_and_beats(video_path)
    # This rhythm_info could then be passed to create_enhanced_individual_clip
    # to influence dynamic cuts or transitions.
    
    print(f"\n=== Step 4: Processing {len(clips_data)} Clips ===")
    created_clips, failed_clips = batch_create_enhanced_clips(
        video_path,
        clips_data,
        transcript,
        video_info,
        face_tracker_instance=face_tracker_instance,
        object_tracker_instance=object_tracker_instance,
        **processing_settings
    )

    
    processing_time = time.time() - start_time
    print("\n=== Step 5: Generating Report ===")
    
    report = create_processing_report(
        video_path,
        created_clips,
        failed_clips,
        processing_time,
        video_analysis
    )
    
    if created_clips:
        output_dir = os.path.dirname(created_clips[0])
        save_processing_report(report, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Total time: {processing_time:.1f}s")
    print(f"Success rate: {report['results']['success_rate']:.1f}%")
    print(f"Average time per clip: {report['performance_metrics']['avg_time_per_clip']:.1f}s")
    
    return created_clips, report

if __name__ == "__main__":
    print("Enhanced video editing module loaded successfully!")
    print("Available features:")
    print("- Advanced face tracking with animations")
    print("- Object detection and tracking")
    print("- Scene change detection with zoom effects")
    print("- Word-by-word animated subtitles")
    print("- Intelligent content analysis")
    print("- Batch processing with optimization")
    print("- Quality validation and reporting")