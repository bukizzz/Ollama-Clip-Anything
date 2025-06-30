import os
import re
import json
import logging
import random

from typing import List, Dict, Tuple, Optional


import time
from moviepy.editor import VideoFileClip, CompositeVideoClip

from core.temp_manager import get_temp_path
from core.config import OUTPUT_DIR, CLIP_PREFIX, VIDEO_ENCODER, FFMPEG_GLOBAL_PARAMS, FFMPEG_ENCODER_PARAMS

from core.ffmpeg_command_logger import FFMPEGCommandLogger

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

from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker
from audio.subtitle_generation import SubtitleGenerator
from video.frame_processor import FrameProcessor
from analysis.analysis_and_reporting import analyze_video_content, optimize_processing_settings, create_processing_report, save_processing_report



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
    original_transcript: List[Dict], 
    video_info: Dict,
    enable_face_tracking: bool = True,
    enable_object_tracking: bool = True,
    enable_scene_effects: bool = True,
    enable_advanced_subtitles: bool = True,
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
        subtitle_generator = SubtitleGenerator() if enable_advanced_subtitles else None
        
        
        # Define the final output resolution (9:16 aspect ratio)
        # Define the final output resolution (9:16 aspect ratio)
        # Prioritize original height, adjust width to maintain aspect ratio
        output_h = original_h
        output_w = int(output_h * 9 / 16)

        # If the calculated width is greater than the original width,
        # scale based on width instead to avoid upscaling unnecessarily
        if output_w > original_w:
            output_w = original_w
            output_h = int(original_w * 16 / 9)
        
        # Ensure even dimensions
        if output_w % 2 != 0:
            output_w += 1
        if output_h % 2 != 0:
            output_h += 1

        processor = FrameProcessor(original_w, original_h, output_w, output_h, face_tracker, object_tracker)
        processed_video_clip = original_clip.fl(processor.process_frame)
        
        # Advanced subtitles
        subtitle_clips = []
        if subtitle_generator and original_transcript:
            # Filter transcript for this clip's timeframe
            clip_transcript = []
            for segment in original_transcript:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', seg_start + 1)
                
                # Check if segment overlaps with clip
                if seg_start < end and seg_end > start:
                    # Adjust timing relative to clip start
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] = max(0, seg_start - start)
                    adjusted_segment['end'] = min(duration, seg_end - start)
                    clip_transcript.append(adjusted_segment)
            
            if clip_transcript:
                subtitle_clips = subtitle_generator.create_word_by_word_subtitles(
                    clip_transcript, duration, (output_w, output_h)
                )
        
        # Combine all clips
        final_clips = [processed_video_clip]
        final_clips.extend(subtitle_clips)
        
        if len(final_clips) > 1:
            final_video = CompositeVideoClip(final_clips)
        else:
            final_video = processed_video_clip
        
        # Output path
        output_path = get_next_output_filename(original_video_path, clip_number)
        
        # Write final video
        ffmpeg_params = list(FFMPEG_GLOBAL_PARAMS)
        ffmpeg_params.extend(['-c:v', VIDEO_ENCODER])  # Fixed: removed leading space
        if VIDEO_ENCODER in FFMPEG_ENCODER_PARAMS:
            ffmpeg_params.extend(FFMPEG_ENCODER_PARAMS[VIDEO_ENCODER])

        print(f"DEBUG: FFmpeg parameters: {ffmpeg_params}")
        final_video.write_videofile(
            output_path,
            audio_codec='aac',
            temp_audiofile=get_temp_path(f'temp_audio_enhanced_{clip_number}.m4a'),
            remove_temp=True,
            fps=30,
            ffmpeg_params=ffmpeg_params
        )

        # Cleanup
        original_clip.close()
        final_video.close()
        for clip in subtitle_clips:
            if hasattr(clip, 'close'):
                clip.close()
        
        print(f"Successfully created enhanced clip {clip_number}")
        return output_path

    except Exception as e:
        print(f"Failed to create enhanced clip {clip_number}: {e}")
        raise

def batch_create_enhanced_clips(
    original_video_path: str, 
    clips_data: List[Dict], 
    original_transcript: List[Dict], 
    video_info: Dict,
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None,
    logger: logging.Logger = None,
    **enhancement_options
) -> Tuple[List[str], List[int]]:
    """Create multiple enhanced clips with all features"""
    print(f"Creating {len(clips_data)} enhanced clips with advanced features...")
    
    # Filter options to only include valid parameters for create_enhanced_individual_clip
    valid_clip_options = {
        'enable_face_tracking': enhancement_options.get('enable_face_tracking', True),
        'enable_object_tracking': enhancement_options.get('enable_object_tracking', True),
        'enable_scene_effects': enhancement_options.get('enable_scene_effects', True),
        'enable_advanced_subtitles': enhancement_options.get('enable_advanced_subtitles', True)
    }
    
    created_clips = []
    failed_clips = []

    for i, clip_data in enumerate(clips_data, 1):
        try:
            print(f"\n--- Processing Enhanced Clip {i}/{len(clips_data)} ---")
            clip_path = create_enhanced_individual_clip(
                original_video_path, clip_data, i, original_transcript, video_info, 
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


def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    custom_settings: Optional[Dict] = None
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    # Initialize trackers once for the entire batch
    face_tracker_instance = FaceTracker()
    object_tracker_instance = ObjectTracker()

    # Step 1: Analyze video content
    print("\n=== Step 1: Video Content Analysis ===")
    video_analysis = analyze_video_content(video_path, face_tracker=face_tracker_instance, object_tracker=object_tracker_instance)
    
    # Step 2: Optimize processing settings
    print("\n=== Step 2: Processing Optimization ===")
    processing_settings = optimize_processing_settings(video_analysis)
    
    # Override with custom settings if provided
    if custom_settings:
        processing_settings.update(custom_settings)
        print("Applied custom settings overrides")
    
    # Step 3: Extract video info for processing
    video_info = {
        'width': video_analysis['width'],
        'height': video_analysis['height'],
        'duration': video_analysis['duration'],
        'fps': video_analysis['fps']
    }
    
    # Step 4: Process clips in batches
    print(f"\n=== Step 3: Processing {len(clips_data)} Clips ===")
    created_clips, failed_clips = batch_create_enhanced_clips(
        video_path,
        clips_data,
        transcript,
        video_info,
        face_tracker_instance=face_tracker_instance,
        object_tracker_instance=object_tracker_instance,
        **processing_settings
    )
    
    # Step 5: Generate processing report
    processing_time = time.time() - start_time
    print("\n=== Step 4: Generating Report ===")
    
    report = create_processing_report(
        video_path,
        created_clips,
        failed_clips,
        processing_time,
        video_analysis
    )
    
    # Save report
    if created_clips:
        output_dir = os.path.dirname(created_clips[0])
        save_processing_report(report, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Total time: {processing_time:.1f}s")
    print(f"Success rate: {report['results']['success_rate']:.1f}%")
    print(f"Average time per clip: {report['performance_metrics']['avg_time_per_clip']:.1f}s")
    
    return created_clips, report

# Example usage function
def process_video_with_all_features(
    video_path: str,
    transcript_path: str,
    clips_data: List[Dict],
    output_settings: Optional[Dict] = None
):
    """
    Main entry point for processing a video with all enhanced features
    
    Args:
        video_path: Path to source video file
        transcript_path: Path to transcript JSON file
        clips_data: List of clip definitions with start/end times
        output_settings: Optional custom processing settings
    """
    try:
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        # Process with full pipeline
        created_clips, report = batch_process_with_analysis(
            video_path,
            clips_data,
            transcript,
            output_settings
        )
        
        return {
            'success': True,
            'created_clips': created_clips,
            'report': report,
            'message': f"Successfully processed {len(created_clips)} clips"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Processing failed: {e}"
        }

        # Cleanup
        original_clip.close()
        final_video.close()
        for clip in subtitle_clips:
            if hasattr(clip, 'close'):
                clip.close()
        
        print(f"Successfully created enhanced clip {clip_number}")
        return output_path

    except Exception as e:
        print(f"Failed to create enhanced clip {clip_number}: {e}")
        raise

def batch_create_enhanced_clips(
    original_video_path: str, 
    clips_data: List[Dict], 
    original_transcript: List[Dict], 
    video_info: Dict,
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None,
    logger: logging.Logger = None,
    **enhancement_options
) -> Tuple[List[str], List[int]]:
    """Create multiple enhanced clips with all features"""
    print(f"Creating {len(clips_data)} enhanced clips with advanced features...")
    
    # Filter options to only include valid parameters for create_enhanced_individual_clip
    valid_clip_options = {
        'enable_face_tracking': enhancement_options.get('enable_face_tracking', True),
        'enable_object_tracking': enhancement_options.get('enable_object_tracking', True),
        'enable_scene_effects': enhancement_options.get('enable_scene_effects', True),
        'enable_advanced_subtitles': enhancement_options.get('enable_advanced_subtitles', True)
    }
    
    created_clips = []
    failed_clips = []

    for i, clip_data in enumerate(clips_data, 1):
        try:
            print(f"\n--- Processing Enhanced Clip {i}/{len(clips_data)} ---")
            clip_path = create_enhanced_individual_clip(
                original_video_path, clip_data, i, original_transcript, video_info, 
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


def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    custom_settings: Optional[Dict] = None
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    # Initialize trackers once for the entire batch
    face_tracker_instance = FaceTracker()
    object_tracker_instance = ObjectTracker()

    # Step 1: Analyze video content
    print("\n=== Step 1: Video Content Analysis ===")
    video_analysis = analyze_video_content(video_path, face_tracker=face_tracker_instance, object_tracker=object_tracker_instance)
    
    # Step 2: Optimize processing settings
    print("\n=== Step 2: Processing Optimization ===")
    processing_settings = optimize_processing_settings(video_analysis)
    
    # Override with custom settings if provided
    if custom_settings:
        processing_settings.update(custom_settings)
        print("Applied custom settings overrides")
    
    # Step 3: Extract video info for processing
    video_info = {
        'width': video_analysis['width'],
        'height': video_analysis['height'],
        'duration': video_analysis['duration'],
        'fps': video_analysis['fps']
    }
    
    # Step 4: Process clips in batches
    print(f"\n=== Step 3: Processing {len(clips_data)} Clips ===")
    created_clips, failed_clips = batch_create_enhanced_clips(
        video_path,
        clips_data,
        transcript,
        video_info,
        face_tracker_instance=face_tracker_instance,
        object_tracker_instance=object_tracker_instance,
        **processing_settings
    )
    
    # Step 5: Generate processing report
    processing_time = time.time() - start_time
    print("\n=== Step 4: Generating Report ===")
    
    report = create_processing_report(
        video_path,
        created_clips,
        failed_clips,
        processing_time,
        video_analysis
    )
    
    # Save report
    if created_clips:
        output_dir = os.path.dirname(created_clips[0])
        save_processing_report(report, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Total time: {processing_time:.1f}s")
    print(f"Success rate: {report['results']['success_rate']:.1f}%")
    print(f"Average time per clip: {report['performance_metrics']['avg_time_per_clip']:.1f}s")
    
    return created_clips, report

# Example usage function
def process_video_with_all_features(
    video_path: str,
    transcript_path: str,
    clips_data: List[Dict],
    output_settings: Optional[Dict] = None
):
    """
    Main entry point for processing a video with all enhanced features
    
    Args:
        video_path: Path to source video file
        transcript_path: Path to transcript JSON file
        clips_data: List of clip definitions with start/end times
        output_settings: Optional custom processing settings
    """
    try:
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        # Process with full pipeline
        created_clips, report = batch_process_with_analysis(
            video_path,
            clips_data,
            transcript,
            output_settings
        )
        
        return {
            'success': True,
            'created_clips': created_clips,
            'report': report,
            'message': f"Successfully processed {len(created_clips)} clips"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Processing failed: {e}"
        }



if __name__ == "__main__":
    # Example usage
    sample_clips = [
        {'start': 0, 'end': 10},
        {'start': 10, 'end': 20},
        {'start': 20, 'end': 30}
    ]
    
    # This would be called from your main processing script
    print("Enhanced video editing module loaded successfully!")
    print("Available features:")
    print("- Advanced face tracking with animations")
    print("- Object detection and tracking")
    print("- Scene change detection with zoom effects")
    print("- Word-by-word animated subtitles")
    print("- Intelligent content analysis")
    print("- Batch processing with optimization")
    print("- Quality validation and reporting")
