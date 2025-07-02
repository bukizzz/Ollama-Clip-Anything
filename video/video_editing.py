import os
import logging
import time
from typing import List, Dict, Tuple, Optional

from moviepy.editor import VideoFileClip
from tqdm import tqdm

from core.temp_manager import get_temp_path
from core.config import OUTPUT_DIR, CLIP_PREFIX, VIDEO_ENCODER, FFMPEG_GLOBAL_PARAMS
import core.config
from core.ffmpeg_command_logger import FFMPEGCommandLogger
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker
from video.frame_processor import FrameProcessor
from video.scene_detection import SceneDetector
from analysis.analysis_and_reporting import create_processing_report, save_processing_report
from audio.subtitle_generation import create_ass_file
import librosa 
from audio.audio_processing import extract_audio 

tqdm.disable = True

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
    processing_settings: Dict, # Added processing_settings
    enable_face_tracking: bool = True,
    enable_object_tracking: bool = True,
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None, 
    split_screen_mode: bool = False,
    b_roll_image_path: Optional[str] = None
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
        
        # Detect scene changes within the current clip
        scene_detector = SceneDetector()
        clip_temp_path = get_temp_path(f"temp_clip_{clip_number}.mp4")
        original_clip.write_videofile(clip_temp_path, audio_codec='aac', verbose=False, logger=None)
        scene_changes = scene_detector.detect_scene_changes(clip_temp_path)
        os.remove(clip_temp_path) # Clean up temp clip

        # Define a function to check if a scene change occurred at a given time
        def is_scene_changed(current_time: float) -> bool:
            for sc_time in scene_changes:
                if abs(current_time - sc_time) < 0.1: # Within 0.1 seconds of a scene change
                    return True
            return False

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

        split_screen_mode = clip_data.get('split_screen', False)
        processor = FrameProcessor(original_w, original_h, output_w, output_h, face_tracker, object_tracker, split_screen_mode=split_screen_mode, b_roll_image_path=clip_data.get('b_roll_image', None))
        processed_video_clip = original_clip.fl(lambda gf, t: processor.process_frame(gf, t, scene_changed=is_scene_changed(t)))
        
        # Output path
        output_path = get_next_output_filename(original_video_path, clip_number)
        
        # Generate subtitles for the clip
        ass_path = get_temp_path(f"subtitles_{clip_number}.ass")
        create_ass_file(transcript, ass_path, time_offset=int(start), video_height=original_h, split_screen_mode=split_screen_mode) # Cast time_offset to int

        ffmpeg_params = list(FFMPEG_GLOBAL_PARAMS)
        ffmpeg_params.extend(processing_settings.get("ffmpeg_encoder_params", {}).get(processing_settings.get("video_encoder"), []))
        if not os.path.exists(ass_path):
            print(f"WARNING: Subtitle file not found at {ass_path}. Subtitles may not appear.")
        else:
            # Add subtitle filter
            ffmpeg_params.extend(['-vf', f"subtitles={ass_path}"])

        print(f"DEBUG: processing_settings: {processing_settings}")
        print(f"DEBUG: FFmpeg parameters: {ffmpeg_params}")
        processed_video_clip.write_videofile(
            output_path,
            
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
    processing_settings: Dict, # Added processing_settings
    face_tracker_instance: Optional[FaceTracker] = None,
    object_tracker_instance: Optional[ObjectTracker] = None,
    logger: Optional[logging.Logger] = None, # Changed to Optional
    custom_clip_themes: Optional[List[Dict]] = None, # Changed to Optional
    **enhancement_options
) -> Tuple[List[str], List[int]]:
    """Create multiple enhanced clips with all features"""
    print(f"Creating {len(clips_data)} enhanced clips with advanced features...")
    
    created_clips = []
    failed_clips = []

    for i, clip_data in enumerate(clips_data, 1):
        try:
            print(f"\n--- Processing Enhanced Clip {i}/{len(clips_data)} ---")
            clip_path = create_enhanced_individual_clip(
                original_video_path, clip_data, i, video_info, 
                transcript, processing_settings,
                face_tracker_instance=face_tracker_instance, 
                object_tracker_instance=object_tracker_instance, 
                split_screen_mode=clip_data.get('split_screen', False),
                b_roll_image_path=clip_data.get('b_roll_image', None)
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
    """Detects rhythm and beats in the video's audio using librosa.
    Returns a list of beat timestamps.
    """
    print("Detecting rhythm and beats...")
    audio_temp_path = get_temp_path("temp_audio_for_beats.wav")
    try:
        extract_audio(video_path, audio_temp_path)
        y, sr = librosa.load(audio_temp_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr).tolist()
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        print(f"Detected {len(beat_times)} beats.")
        return beat_times
    except Exception as e:
        print(f"Failed to detect rhythm and beats: {e}")
        return []

def sync_cuts_with_beats(clips: List[Dict], beat_times: List[float]) -> List[Dict]:
    """Adjusts clip start/end times to align with detected beats.
    This is a simplified example and can be made more sophisticated.
    """
    print("Syncing cuts with beats...")
    synced_clips = []
    for clip in clips:
        start = clip['start']
        end = clip['end']
        
        # Find the closest beat to the clip's start
        closest_start_beat = min(beat_times, key=lambda x: abs(x - start), default=start)
        # Find the closest beat to the clip's end
        closest_end_beat = min(beat_times, key=lambda x: abs(x - end), default=end)
        
        # Adjust clip to align with beats, ensuring minimum duration
        new_start = closest_start_beat
        new_end = closest_end_beat
        
        # Ensure the clip duration is still within reasonable bounds
        if new_end - new_start < 5: # Example: ensure minimum 5 seconds
            new_end = new_start + (end - start) # Revert to original duration if too short
        
        synced_clips.append({
            'start': new_start,
            'end': new_end,
            'text': clip['text']
        })
    print("Cuts synced with beats.")
    return synced_clips

def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    video_info: Dict, 
    processing_settings: Dict, 
    video_analysis: Dict, 
    custom_settings: Optional[Dict] = None,
    face_tracker_instance: Optional[FaceTracker] = None, 
    object_tracker_instance: Optional[ObjectTracker] = None 
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    if custom_settings:
        processing_settings.update(custom_settings)
        print("Applied custom settings overrides")
    
    print(f"\n=== Step 4: Processing {len(clips_data)} Clips ===")
    created_clips, failed_clips = batch_create_enhanced_clips(
        video_path,
        clips_data,
        transcript,
        video_info, 
        processing_settings, # Pass processing_settings
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
