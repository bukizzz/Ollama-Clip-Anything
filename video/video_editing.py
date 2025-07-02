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
from video.frame_processor import FrameProcessor
from video.scene_detection import SceneDetector
from analysis.analysis_and_reporting import create_processing_report, save_processing_report
import librosa 
from audio.audio_processing import extract_audio 
from video.clip_enhancer import create_enhanced_individual_clip 

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





def batch_create_enhanced_clips(
    original_video_path: str, 
    clips_data: List[Dict], 
    transcript: List[Dict],
    video_info: Dict, 
    processing_settings: Dict, 
    tracking_manager,
    logger: Optional[logging.Logger] = None, 
    custom_clip_themes: Optional[List[Dict]] = None, 
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
                tracking_manager,
                OUTPUT_DIR
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

from video.tracking_manager import TrackingManager

def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    video_info: Dict, 
    processing_settings: Dict, 
    video_analysis: Dict, 
    custom_settings: Optional[Dict] = None
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    if custom_settings:
        processing_settings.update(custom_settings)
        print("Applied custom settings overrides")
    
    print(f"\n=== Step 4: Processing {len(clips_data)} Clips ===")
    
    tracking_manager = TrackingManager()
    created_clips, failed_clips = batch_create_enhanced_clips(
        video_path,
        clips_data,
        transcript,
        video_info, 
        processing_settings, 
        tracking_manager
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
