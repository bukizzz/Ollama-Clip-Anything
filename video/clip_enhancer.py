import os
import logging
from typing import List, Dict

from moviepy.editor import VideoFileClip
from core.temp_manager import get_temp_path
from core.config import config
from video.frame_processor import FrameProcessor
from audio.subtitle_generation import create_ass_file
from video.tracking_manager import TrackingManager
from core.ffmpeg_command_logger import FFMPEGCommandLogger

logging.setLoggerClass(FFMPEGCommandLogger)
logger = logging.getLogger('moviepy')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_next_output_filename(source_video_path: str, clip_number: int, output_dir: str) -> str:
    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    folder_name = f"{source_name}_enhanced"
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, f"{config.get('clip_prefix')}_enhanced_{clip_number}.mp4")

def create_enhanced_individual_clip(
    original_video_path: str, 
    clip_data: Dict, 
    clip_number: int, 
    video_info: Dict,
    transcript: List[Dict],
    processing_settings: Dict, 
    tracking_manager: TrackingManager,
    output_dir: str,
    audio_rhythm_data: Dict,
    llm_cut_decisions: List[Dict],
    speaker_tracking_results: Dict,
    video_analysis: Dict
) -> str:
    start, end = float(clip_data['start']), float(clip_data['end'])
    print(f"Creating enhanced clip {clip_number}: {start:.1f}s - {end:.1f}s")

    try:
        original_clip = VideoFileClip(original_video_path).subclip(start, end)
        
        # Dynamic effects based on analysis data
        effects_to_apply = []
        if audio_rhythm_data and 'beat_times' in audio_rhythm_data:
            for beat_time in audio_rhythm_data['beat_times']:
                if start <= beat_time <= end:
                    effects_to_apply.append({'time': beat_time - start, 'effect': 'zoom', 'factor': 1.05})

        if llm_cut_decisions:
            for decision in llm_cut_decisions:
                if start <= decision['start_time'] <= end:
                    effects_to_apply.append({'time': decision['start_time'] - start, 'effect': 'cut'})

        # Process frames with effects
        frame_processor = FrameProcessor(video_info['width'], video_info['height'], video_info['width'], video_info['height'], tracking_manager.get_face_tracker(), tracking_manager.get_object_tracker())
        engagement_metrics = video_analysis.get('engagement_metrics', [])

        engagement_metrics = video_analysis.get('engagement_metrics', [])

        def dynamic_frame_modifier(t):
            frame = original_clip.get_frame(t)
            layout_info = {'recommended_layout': 'single_person_focus', 'active_speaker': None} # Placeholder, ideally from layout_optimization_recommendations
            return frame_processor.process_frame(frame, t, layout_info, engagement_metrics)

        processed_clip = original_clip.fl(dynamic_frame_modifier)

        # Subtitles and final output
        output_path = get_next_output_filename(original_video_path, clip_number, output_dir)
        ass_path = get_temp_path(f"subtitles_{clip_number}.ass")
        create_ass_file(transcript, ass_path, time_offset=int(start), video_height=video_info['height'])

        ffmpeg_params = list(config.get('ffmpeg_global_params'))
        ffmpeg_params.extend(processing_settings.get("ffmpeg_encoder_params", {}).get(processing_settings.get("video_encoder"), []))
        ffmpeg_params.extend(['-vf', f"subtitles={ass_path}"])

        processed_clip.write_videofile(
            output_path,
            audio_codec='aac',
            temp_audiofile=get_temp_path(f'temp_audio_enhanced_{clip_number}.m4a'),
            remove_temp=True,
            fps=30,
            ffmpeg_params=ffmpeg_params
        )

        original_clip.close()
        processed_clip.close()
        
        print(f"Successfully created enhanced clip {clip_number}")
        return output_path

    except Exception as e:
        print(f"Failed to create enhanced clip {clip_number}: {e}")
        raise
