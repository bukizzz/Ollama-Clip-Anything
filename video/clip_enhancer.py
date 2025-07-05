import os
import logging
from typing import List, Dict

from moviepy.editor import VideoFileClip, concatenate_videoclips
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
    scenes = clip_data.get('scenes', [])
    if not scenes:
        raise ValueError("Clip data must contain a 'scenes' list.")

    # Determine overall clip start and end times from the first and last scenes
    clip_start_time = scenes[0]['start_time']
    clip_end_time = scenes[-1]['end_time']
    
    print(f"Creating enhanced clip {clip_number}: {clip_start_time:.1f}s - {clip_end_time:.1f}s (composed of {len(scenes)} scenes)")

    try:
        final_clip_segments = []
        for i, scene in enumerate(scenes):
            scene_start = scene['start_time']
            scene_end = scene['end_time']
            print(f"  - Processing scene {i+1}/{len(scenes)}: {scene_start:.1f}s - {scene_end:.1f}s")

            # Extract subclip for the current scene
            subclip = VideoFileClip(original_video_path).subclip(scene_start, scene_end)

            # Dynamic effects based on analysis data (adjusted for subclip time)
            effects_to_apply = []
            if audio_rhythm_data and 'beat_times' in audio_rhythm_data:
                for beat_time in audio_rhythm_data['beat_times']:
                    if scene_start <= beat_time <= scene_end:
                        effects_to_apply.append({'time': beat_time - scene_start, 'effect': 'zoom', 'factor': 1.05})

            if llm_cut_decisions:
                for decision in llm_cut_decisions:
                    if scene_start <= decision['start_time'] <= scene_end:
                        effects_to_apply.append({'time': decision['start_time'] - scene_start, 'effect': 'cut'})

            # Process frames with effects
            frame_processor = FrameProcessor(video_info['width'], video_info['height'], video_info['width'], video_info['height'], tracking_manager.get_face_tracker(), tracking_manager.get_object_tracker())
            engagement_metrics = video_analysis.get('engagement_metrics', [])

            def dynamic_frame_modifier(t):
                frame = subclip.get_frame(t)
                layout_info = {'recommended_layout': 'single_person_focus', 'active_speaker': None} # Placeholder, ideally from layout_optimization_recommendations
                return frame_processor.process_frame(frame, t, layout_info, engagement_metrics)

            processed_subclip = subclip.fl(dynamic_frame_modifier)
            final_clip_segments.append(processed_subclip)
            subclip.close() # Close the subclip after processing

        # Concatenate all processed subclips
        if not final_clip_segments:
            raise ValueError("No valid scenes to concatenate for the clip.")
        
        final_clip = concatenate_videoclips(final_clip_segments)

        # Subtitles and final output
        output_path = get_next_output_filename(original_video_path, clip_number, output_dir)
        ass_path = get_temp_path(f"subtitles_{clip_number}.ass")
        
        # Filter transcript to only include segments within the overall clip duration
        clip_transcript = [
            seg for seg in transcript 
            if seg['start'] >= clip_start_time and seg['end'] <= clip_end_time
        ]
        create_ass_file(clip_transcript, ass_path, time_offset=int(clip_start_time), video_height=video_info['height'])

        ffmpeg_params = list(config.get('ffmpeg_global_params'))
        ffmpeg_params.extend(processing_settings.get("ffmpeg_encoder_params", {}).get(processing_settings.get("video_encoder"), []))
        ffmpeg_params.extend(['-vf', f"subtitles={ass_path}"])

        final_clip.write_videofile(
            output_path,
            audio_codec='aac',
            temp_audiofile=get_temp_path(f'temp_audio_enhanced_{clip_number}.m4a'),
            remove_temp=True,
            fps=30,
            ffmpeg_params=ffmpeg_params
        )

        final_clip.close()
        
        print(f"Successfully created enhanced clip {clip_number}")
        return output_path

    except Exception as e:
        print(f"Failed to create enhanced clip {clip_number}: {e}")
        raise
