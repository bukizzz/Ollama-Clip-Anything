import os
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.config import config
from core.ffmpeg_command_logger import FFMPEGCommandLogger
from analysis.analysis_and_reporting import create_processing_report, save_processing_report
from video.clip_enhancer import create_enhanced_individual_clip 
from video.tracking_manager import TrackingManager

# Configure MoviePy's logger to capture FFmpeg commands
logging.setLoggerClass(FFMPEGCommandLogger)
logger = logging.getLogger('moviepy')
logger.setLevel(logging.WARNING)

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
    audio_rhythm_data: Dict, # New parameter
    llm_cut_decisions: List[Dict], # New parameter
    speaker_tracking_results: Dict, # New parameter
    video_analysis: Dict,
    logger: Optional[logging.Logger] = None, 
    custom_clip_themes: Optional[List[Dict]] = None, 
    zoom_events: Optional[List[Dict]] = None, # New parameter
    video_output_settings: Optional[Dict] = None, # New parameter for resolution and aspect ratio
    **enhancement_options
) -> Tuple[List[str], List[int]]:
    """Create multiple enhanced clips with all features"""
    print(f"Creating {len(clips_data)} enhanced clips with advanced features...")
    
    created_clips = []
    failed_clips = []

    max_workers = config.get('video_production.max_render_workers', os.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {
            executor.submit(create_enhanced_individual_clip,
                original_video_path, clip_data, i, video_info,
                transcript, processing_settings,
                tracking_manager,
                output_dir=config.get('output_dir'),
                audio_rhythm_data=audio_rhythm_data,
                llm_cut_decisions=llm_cut_decisions,
                speaker_tracking_results=speaker_tracking_results,
                video_analysis=video_analysis,
                zoom_events=zoom_events, # Pass zoom_events
                video_output_settings=video_output_settings # Pass video_output_settings
            ): (clip_data, i) for i, clip_data in enumerate(clips_data, 1)
        }

        for future in tqdm(as_completed(future_to_clip), total=len(clips_data), desc="Rendering Clips"):
            clip_data, i = future_to_clip[future]
            try:
                clip_path = future.result()
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
    video_info: Dict, 
    processing_settings: Dict, 
    video_analysis: Dict, 
    audio_rhythm_data: Dict, # New parameter
    llm_cut_decisions: List[Dict], # New parameter
    speaker_tracking_results: Dict, # New parameter
    output_dir: str, # New parameter
    zoom_events: Optional[List[Dict]] = None, # New parameter
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
        tracking_manager,
        audio_rhythm_data=audio_rhythm_data, 
        llm_cut_decisions=llm_cut_decisions, 
        speaker_tracking_results=speaker_tracking_results,
        video_analysis=video_analysis,
        zoom_events=zoom_events # Pass zoom_events
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
    "- Scene change detection with zoom effects"
    print("- Word-by-word animated subtitles")
    print("- Intelligent content analysis")
    print("- Batch processing with optimization")
    print("- Quality validation and reporting")
