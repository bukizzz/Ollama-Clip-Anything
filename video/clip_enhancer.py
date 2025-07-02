import os
import logging
from typing import List, Dict, Optional

from moviepy.editor import VideoFileClip

from core.temp_manager import get_temp_path
from core.config import CLIP_PREFIX, FFMPEG_GLOBAL_PARAMS
from video.frame_processor import FrameProcessor
from video.scene_detection import SceneDetector
from audio.subtitle_generation import create_ass_file
from video.tracking_manager import TrackingManager
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

def get_next_output_filename(source_video_path: str, clip_number: int, output_dir: str) -> str:
    """Generate unique output filename"""
    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    folder_name = f"{source_name}_enhanced"
    output_folder = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return os.path.join(output_folder, f"{CLIP_PREFIX}_enhanced_{clip_number}.mp4")

def create_enhanced_individual_clip(
    original_video_path: str, 
    clip_data: Dict, 
    clip_number: int, 
    video_info: Dict,
    transcript: List[Dict],
    processing_settings: Dict, 
    tracking_manager: TrackingManager,
    output_dir: str,
    audio_rhythm_data: Dict, # New parameter
    llm_cut_decisions: List[Dict] # New parameter
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
        face_tracker = tracking_manager.get_face_tracker() if processing_settings.get("enable_face_tracking") else None
        object_tracker = tracking_manager.get_object_tracker() if processing_settings.get("enable_object_tracking") else None
        
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
        # Apply dynamic editing based on audio rhythm and LLM cut decisions
        # This is a simplified integration. A full implementation would involve more complex logic
        # to precisely align effects with beats and LLM recommendations.
        def dynamic_frame_modifier(get_frame, t):
            frame = get_frame(t)
            
            # Apply rhythm-synced zoom (simplified example)
            if audio_rhythm_data and 'beat_times' in audio_rhythm_data:
                for beat_time in audio_rhythm_data['beat_times']:
                    if abs(t - beat_time) < 0.1: # If close to a beat
                        # Apply a subtle zoom effect
                        zoom_factor = 1.05 # Example zoom
                        h, w, _ = frame.shape
                        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        # Crop to original size from center
                        start_x = (new_w - w) // 2
                        start_y = (new_h - h) // 2
                        frame = frame[start_y:start_y+h, start_x:start_x+w]
                        break

            # Apply cuts determined by LLM video director (handled by subclip already, but for effects)
            # If LLM suggests a specific effect at a timestamp, it would be applied here.
            # For now, we assume LLM decisions primarily influence clip selection (start/end times).

            # Apply speaker-aware visual effects (placeholder)
            # This would involve using speaker_tracking_results to identify active speaker
            # and apply effects like highlighting or subtle blur to non-speakers.

            # Apply engagement-optimized cuts (handled by LLM selection, but for effects)
            # If a high engagement moment is detected, apply a specific visual flair.

            return processor.process_frame(frame, t, scene_changed=is_scene_changed(t))

        processor = FrameProcessor(original_w, original_h, output_w, output_h, face_tracker, object_tracker, split_screen_mode=split_screen_mode, b_roll_image_path=clip_data.get('b_roll_image', None))
        processed_video_clip = original_clip.fl(dynamic_frame_modifier)
        
        # Output path
        output_path = get_next_output_filename(original_video_path, clip_number, output_dir)
        
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
