import os
import logging
import time
import subprocess
from typing import List, Dict, Optional, Tuple

import cv2
from core.temp_manager import get_temp_path
from core.config import config
from video.frame_processor import FrameProcessor
from audio.subtitle_generation import create_ass_file
from video.tracking_manager import TrackingManager
from core.ffmpeg_command_logger import FFMPEGCommandLogger

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_next_output_filename(source_video_path: str, clip_number: int, output_dir: str) -> str:
    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    folder_name = f"{source_name}_enhanced"
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, f"{config.get('clip_prefix')}_enhanced_{clip_number}.mp4")

def _calculate_output_resolution(
    source_width: int, 
    source_height: int, 
    target_aspect_ratios: List[str]
) -> Tuple[int, int]:
    """
    Calculates the optimal output resolution based on source video height
    (which is the target height) and allowed aspect ratios.
    """
    
    target_height = source_height # As per user feedback, final clip height should be source height
    source_aspect_ratio = source_width / source_height

    # Calculate target width for each allowed aspect ratio
    possible_resolutions = []
    for ar_str in target_aspect_ratios:
        try:
            ar_parts = ar_str.split(':')
            ar_width = int(ar_parts[0])
            ar_height = int(ar_parts[1])
            
            # Calculate width based on target_height and target aspect ratio
            target_width = int(target_height * (ar_width / ar_height))
            
            # Ensure width is an even number for FFmpeg compatibility
            if target_width % 2 != 0:
                target_width += 1
            
            possible_resolutions.append((target_width, target_height, ar_str))
        except (ValueError, IndexError):
            logger.warning(f"Invalid aspect ratio format in config: {ar_str}. Skipping.")
            continue

    if not possible_resolutions:
        logger.warning("No valid target aspect ratios found in config. Falling back to source resolution.")
        return source_width, source_height

    # Choose the resolution that is closest to the source aspect ratio
    best_resolution = None
    min_aspect_ratio_diff = float('inf')

    for res_width, res_height, ar_str in possible_resolutions:
        current_aspect_ratio = res_width / res_height
        diff = abs(current_aspect_ratio - source_aspect_ratio)
        
        if diff < min_aspect_ratio_diff:
            min_aspect_ratio_diff = diff
            best_resolution = (res_width, res_height)
        elif diff == min_aspect_ratio_diff:
            # If aspect ratio difference is the same, prefer the one that maintains source height
            if best_resolution is None or res_height == source_height: # Prioritize maintaining source height
                best_resolution = (res_width, res_height)

    if best_resolution:
        logger.info(f"Calculated optimal output resolution: {best_resolution[0]}x{best_resolution[1]} (closest to source aspect ratio, maintaining source height).")
        return best_resolution
    else:
        logger.warning("Could not determine optimal output resolution. Falling back to source resolution.")
        return source_width, source_height


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
    video_analysis: Dict,
    zoom_events: Optional[List[Dict]] = None,
    video_output_settings: Optional[Dict] = None # New parameter for resolution and aspect ratio
) -> str:
    scenes = clip_data.get('scenes', [])
    if not scenes:
        raise ValueError("Clip data must contain a 'scenes' list.")

    clip_start_time = scenes[0].get('start_time')
    clip_end_time = scenes[-1].get('end_time')
    
    logger.info(f"Creating enhanced clip {clip_number}: {clip_start_time:.1f}s - {clip_end_time:.1f}s (composed of {len(scenes)} scenes)")

    ffmpeg_binary_path = config.get('ffmpeg_path')
    output_path = get_next_output_filename(original_video_path, clip_number, output_dir)
    ass_path = get_temp_path(f"subtitles_{clip_number}.ass")

    clip_transcript = [
        seg for seg in transcript 
        if seg['start'] >= clip_start_time and seg['end'] <= clip_end_time
    ]
    
    # Calculate target output resolution
    target_width, target_height = video_info['width'], video_info['height'] # Default to source resolution
    if video_output_settings:
        target_aspect_ratios = video_output_settings.get('target_aspect_ratios', [])
        if target_aspect_ratios: # Only check for target_aspect_ratios as target_height is source_height
            target_width, target_height = _calculate_output_resolution(
                video_info['width'], video_info['height'], target_aspect_ratios
            )
            logger.info(f"Output resolution set to: {target_width}x{target_height}")
        else:
            logger.warning("Video output settings found but missing target_aspect_ratios. Using source resolution.")
    else:
        logger.info("No video output settings provided. Using source resolution.")

    # Generate ASS file only if there's a transcript for the clip
    if clip_transcript:
        # Pass the calculated target_height and target_width to create_ass_file for correct subtitle scaling and positioning
        create_ass_file(clip_transcript, ass_path, time_offset=int(clip_start_time), video_height=target_height, video_width=target_width)
        # Verify if the ASS file was actually created and has content
        if not os.path.exists(ass_path) or os.path.getsize(ass_path) == 0:
            logger.warning(f"ASS file was not created or is empty for clip {clip_number}. Subtitles will not be applied.")
            ass_path = "" # Invalidate path if file is empty or not created
    else:
        logger.info(f"No transcript available for clip {clip_number}. Skipping ASS file generation.")
        ass_path = "" # Ensure ass_path is empty if no transcript


    # --- Segmented A/V Processing ---
    temp_av_segments_paths = []
    video_encoder_config = processing_settings.get("video_encoder")
    
    # Determine video codec and its parameters for temporary segments
    segment_ffmpeg_codec_params = []
    # Force libx264 if h264_nvenc or hevc_nvenc is specified in config due to driver issues
    if video_encoder_config in ['h264_nvenc', 'hevc_nvenc']:
        logger.warning(f"Forcing video encoder to 'libx264' instead of '{video_encoder_config}' due to potential driver compatibility issues.")
        video_encoder_actual = 'libx264'
    else:
        video_encoder_actual = video_encoder_config # Use whatever is configured if not NVENC
    
    segment_ffmpeg_codec_params.extend(['-c:v', video_encoder_actual])
    # Add encoder-specific parameters if they exist for the *actual* encoder being used
    if video_encoder_actual in ['h264_nvenc', 'hevc_nvenc', 'libx264']: # Add libx264 to this check
        segment_ffmpeg_codec_params.extend(config.get(f'ffmpeg_encoder_params.{video_encoder_actual}', []))
    
    frame_processor = FrameProcessor(
        video_info['width'], video_info['height'], target_width, target_height, # Pass target_width, target_height
        tracking_manager.get_face_tracker(), tracking_manager.get_object_tracker(), zoom_events=zoom_events
    )

    cap = None
    try:
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {original_video_path}")
        
        current_scene_index = 0
        for i, scene in enumerate(scenes):
            scene_start = scene.get('start_time')
            scene_end = scene.get('end_time')
            scene_duration = scene_end - scene_start

            if scene_start is None or scene_end is None:
                logger.warning(f"Skipping scene {i+1}/{len(scenes)} due to missing start or end time.")
                continue

            logger.info(f"Processing scene {i+1}/{len(scenes)}: {scene_start:.1f}s - {scene_end:.1f}s")

            # Check if this scene requires dynamic processing (e.g., has zoom events within its time range)
            requires_dynamic_processing = False
            if zoom_events:
                for event in zoom_events:
                    # Check for overlap between zoom event and current scene
                    if max(scene_start, event['start_time']) < min(scene_end, event['end_time']):
                        requires_dynamic_processing = True
                        break
            
            temp_scene_av_path = get_temp_path(f"temp_scene_av_{clip_number}_{i}.mp4")

            if requires_dynamic_processing:
                logger.info(f"  - Scene {i+1} requires dynamic processing.")
                # --- Dynamic Scene Processing (Frame-by-Frame) ---
                # First, extract the A/V segment for the scene
                extract_av_command = [
                    ffmpeg_binary_path, '-y',
                    '-v', 'debug', # Add verbose logging for debugging
                    '-ss', str(scene_start),
                    '-i', original_video_path,
                    '-t', str(scene_duration),
                    '-c:v', 'copy', # Copy video codec for initial extraction
                    '-c:a', 'copy', # Copy audio codec for initial extraction
                    get_temp_path(f"temp_raw_scene_av_{clip_number}_{i}.mp4")
                ]
                stdout, stderr, returncode = FFMPEGCommandLogger.log_command(extract_av_command, f"raw_av_extraction_scene_{i+1}")
                if returncode != 0:
                    logger.error(f"FFmpeg raw A/V extraction for scene {i+1} failed with exit code {returncode}. STDOUT: {stdout}. STDERR: {stderr}")
                    raise RuntimeError(f"FFmpeg raw A/V extraction for scene {i+1} failed with exit code {returncode}")
                
                # Now process frames from this raw A/V segment
                cap_scene = cv2.VideoCapture(get_temp_path(f"temp_raw_scene_av_{clip_number}_{i}.mp4"))
                if not cap_scene.isOpened():
                    raise IOError(f"Could not open temporary A/V file for scene {i+1}: {get_temp_path(f'temp_raw_scene_av_{clip_number}_{i}.mp4')}")

                scene_ffmpeg_command = [
                    ffmpeg_binary_path, '-y', '-v', 'debug', # Add verbose logging for debugging
                    '-f', 'rawvideo', '-vcodec', 'rawvideo',
                    '-s', f"{video_info['width']}x{video_info['height']}", '-pix_fmt', 'rgb24',
                    '-r', str(video_info['fps']), '-i', '-', # Input from pipe
                    '-i', get_temp_path(f"temp_raw_scene_av_{clip_number}_{i}.mp4"), # Audio input from raw A/V segment
                    *segment_ffmpeg_codec_params, # Use segment-specific video codec params
                    '-c:a', 'aac', '-b:a', '192k', # Re-encode audio
                    '-map', '0:v:0', '-map', '1:a:0', # Map video from pipe, audio from second input
                    '-pix_fmt', 'yuv420p', # Output pixel format
                    '-s', f"{target_width}x{target_height}", # Apply target resolution
                    temp_scene_av_path
                ]
                scene_process = subprocess.Popen(scene_ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Capture stdout/stderr for debugging
                if scene_process.stdin is None:
                    raise IOError("FFmpeg scene process stdin pipe is None.")

                frames_to_read = int(scene_duration * video_info['fps'])
                
                for frame_idx in range(frames_to_read):
                    ret, frame = cap_scene.read()
                    if not ret:
                        logger.warning(f"Could not read frame {frame_idx} for scene {i+1}. Breaking loop.")
                        break
                    
                    current_timestamp_original = scene_start + (frame_idx / video_info['fps'])
                    layout_info = {'recommended_layout': 'single_person_focus', 'active_speaker': None} # Placeholder
                    processed_frame = frame_processor.process_frame(frame, current_timestamp_original, layout_info)
                    
                    try:
                        scene_process.stdin.write(processed_frame.tobytes())
                    except (BrokenPipeError, IOError) as e:
                        logger.error(f"FFmpeg pipe broke for scene {i+1} while writing frame {frame_idx}: {e}")
                        break
                
                scene_process.stdin.close()
                scene_process.wait(timeout=60)
                if scene_process.returncode != 0:
                    logger.error(f"FFmpeg dynamic scene {i+1} encoding failed with exit code {scene_process.returncode}")
                    raise RuntimeError(f"FFmpeg dynamic scene {i+1} encoding failed with exit code {scene_process.returncode}")
                logger.info(f"  - Successfully processed and saved dynamic scene {i+1} to {temp_scene_av_path}")
                temp_av_segments_paths.append(temp_scene_av_path)
                cap_scene.release() # Release the temporary video capture

            else:
                logger.info(f"  - Scene {i+1} is static. Extracting A/V directly.")
                # --- Static Scene A/V Extraction (Direct FFmpeg) ---
                # Calculate duration for -t flag
                static_scene_duration = scene_end - scene_start
                
                static_extract_command = [
                    ffmpeg_binary_path, '-y', '-v', 'debug', # Add verbose logging for debugging
                    '-ss', str(scene_start), # Start time
                    '-i', original_video_path, # Input file
                    '-t', str(static_scene_duration), # Duration
                    '-c:v', video_encoder_actual, # Use the determined actual encoder
                    '-c:a', 'aac', '-b:a', '192k', # Re-encode audio
                    '-pix_fmt', 'yuv420p', # Ensure compatible pixel format
                    '-s', f"{target_width}x{target_height}", # Apply target resolution
                    temp_scene_av_path
                ]
                
                logger.info(f"  - Static scene A/V extraction command: {' '.join(static_extract_command)}")
                
                # Log the command and capture output for debugging
                stdout, stderr, returncode = FFMPEGCommandLogger.log_command(static_extract_command, "static_scene_av_extraction")

                if returncode != 0:
                    logger.error(f"FFmpeg static scene {i+1} A/V extraction failed with exit code {returncode}. STDOUT: {stdout}. STDERR: {stderr}")
                    raise RuntimeError(f"FFmpeg static scene {i+1} A/V extraction failed with exit code {returncode}")
                logger.info(f"  - Successfully extracted static scene {i+1} to {temp_scene_av_path}")
                temp_av_segments_paths.append(temp_scene_av_path)

    except Exception as e:
        logger.error(f"Error during segmented A/V processing: {e}")
        raise
    finally:
        if cap:
            cap.release()

    # --- Final Concatenation ---
    if not temp_av_segments_paths:
        raise ValueError("No A/V segments were processed for concatenation.")

    concat_list_path = get_temp_path(f"concat_list_{clip_number}.txt")
    with open(concat_list_path, 'w') as f:
        for p in temp_av_segments_paths:
            f.write(f"file '{os.path.abspath(p)}'\n") # Use absolute path for concat demuxer

    final_concat_command = [
        ffmpeg_binary_path, '-y', '-v', 'debug', # Add verbose logging for debugging
        '-f', 'concat', '-safe', '0', '-i', concat_list_path, # Concat demuxer
        '-c:v', video_encoder_actual, # Use the determined actual encoder
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0', '-map', '0:a:0', # Map video and audio from concatenated input
        '-s', f"{target_width}x{target_height}", # Apply target resolution
        *config.get('ffmpeg_global_params', []), # Apply global FFmpeg parameters
        output_path
    ]

    # Conditionally add subtitle filter if ass_path is valid
    if ass_path:
        final_concat_command.insert(-1, '-vf') # Insert before output_path
        final_concat_command.insert(-1, f"subtitles={ass_path}") # Insert before output_path

    logger.info(f"Final concatenation command: {' '.join(final_concat_command)}")
    final_process = None
    try:
        final_process = subprocess.Popen(final_concat_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_process.wait(timeout=180) # Increased timeout for final concat
        if final_process.returncode != 0:
            logger.error(f"FFmpeg final concatenation failed with exit code {final_process.returncode}")
            raise RuntimeError(f"FFmpeg final concatenation failed with exit code {final_process.returncode}")
        logger.info(f"Successfully concatenated segments to {output_path}")
    except Exception as e:
        logger.error(f"Error during final concatenation: {e}")
        raise
    finally:
        if final_process:
            if final_process.stdin:
                final_process.stdin.close()
            final_process.wait()

    # Clean up temporary files
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for p in temp_av_segments_paths:
        if os.path.exists(p):
            os.remove(p)

    return output_path
