import os
import json
import datetime
from typing import Dict, List, Optional
import cv2
from moviepy.editor import VideoFileClip

from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker

def analyze_video_content(video_path: str, face_tracker: Optional[FaceTracker] = None, object_tracker: Optional[ObjectTracker] = None) -> Dict:
    """Analyze video content for optimal processing"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames for analysis
        sample_frames = []
        face_count_samples = []
        object_count_samples = []
        
        # Sample every 30 seconds or 10 frames max
        sample_interval = max(1, int(fps * 30))  # Every 30 seconds
        max_samples = min(10, frame_count // sample_interval)
        
        # Use provided trackers or create new ones
        _face_tracker = face_tracker if face_tracker else FaceTracker()
        _object_tracker = object_tracker if object_tracker else ObjectTracker()
        
        for i in range(0, frame_count, sample_interval):
            if len(sample_frames) >= max_samples:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            sample_frames.append(i / fps)  # timestamp
            
            # Analyze faces
            faces = _face_tracker.detect_faces_in_frame(frame)
            face_count_samples.append(len(faces))
            
            # Analyze objects
            objects = _object_tracker.detect_objects_in_frame(frame)
            object_count_samples.append(len(objects))
        
        cap.release()
        
        # Calculate averages
        avg_faces = sum(face_count_samples) / len(face_count_samples) if face_count_samples else 0
        avg_objects = sum(object_count_samples) / len(object_count_samples) if object_count_samples else 0
        
        analysis = {
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'aspect_ratio': width / height if height > 0 else 0,
            'avg_faces_per_frame': avg_faces,
            'avg_objects_per_frame': avg_objects,
            'has_faces': avg_faces > 0.1,
            'has_objects': avg_objects > 0.5,
            'recommended_face_tracking': avg_faces > 0.2,
            'recommended_object_tracking': avg_objects > 1.0,
            'processing_complexity': 'high' if (avg_faces > 1 or avg_objects > 3) else 'medium' if (avg_faces > 0 or avg_objects > 0) else 'low'
        }
        
        print("Video Analysis Complete:")
        print(f"  Duration: {analysis.get('duration', 0):.1f}s")
        print(f"  Resolution: {analysis.get('width', 0)}x{analysis.get('height', 0)}")
        print(f"  Average faces per frame: {analysis.get('avg_faces_per_frame', 0):.2f}")
        print(f"  Average objects per frame: {analysis.get('avg_objects_per_frame', 0):.2f}")
        print(f"  Processing complexity: {analysis.get('processing_complexity', 'low')}")
        
        return analysis
        
    except Exception as e:
        print(f"Video analysis failed: {e}")
        return {
            'duration': 0,
            'fps': 30,
            'width': 1920,
            'height': 1080,
            'frame_count': 0,
            'aspect_ratio': 16/9,
            'avg_faces_per_frame': 0,
            'avg_objects_per_frame': 0,
            'has_faces': False,
            'has_objects': False,
            'recommended_face_tracking': False,
            'recommended_object_tracking': False,
            'processing_complexity': 'low'
        }

def optimize_processing_settings(video_analysis: Dict, available_memory_gb: float = 8.0) -> Dict:
    """Optimize processing settings based on video analysis and system resources"""
    settings = {
        'enable_face_tracking': True,
        'enable_object_tracking': True,
        'enable_scene_effects': True,
        'enable_advanced_subtitles': True,
        'processing_quality': 'high',
        'batch_size': 5,
        'parallel_processing': False
    }
    
    complexity = video_analysis.get('processing_complexity', 'medium')
    
    
    # Adjust based on complexity
    if complexity == 'high':
        settings['batch_size'] = 3
        if available_memory_gb < 8:
            settings['enable_object_tracking'] = False
            settings['processing_quality'] = 'medium'
    elif complexity == 'low':
        settings['batch_size'] = 10
        settings['parallel_processing'] = available_memory_gb > 16
    
    # Disable features if not recommended
    if not video_analysis.get('recommended_face_tracking', True):
        settings['enable_face_tracking'] = False
    if not video_analysis.get('recommended_object_tracking', True):
        settings['enable_object_tracking'] = False
    
    print("Optimized processing settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    return settings

def create_processing_report(
    video_path: str,
    created_clips: List[str],
    failed_clips: List[int],
    processing_time: float,
    video_analysis: Dict
) -> Dict:
    """Create a comprehensive processing report"""
    report = {
        'source_video': os.path.basename(video_path),
        'processing_timestamp': datetime.datetime.now().isoformat(),
        'total_processing_time': processing_time,
        'video_analysis': video_analysis,
        'results': {
            'total_clips_attempted': len(created_clips) + len(failed_clips),
            'successful_clips': len(created_clips),
            'failed_clips': len(failed_clips),
            'success_rate': len(created_clips) / (len(created_clips) + len(failed_clips)) * 100 if (created_clips or failed_clips) else 0
        },
        'output_files': [os.path.basename(clip) for clip in created_clips],
        'failed_clip_numbers': failed_clips,
        'performance_metrics': {
            'avg_time_per_clip': processing_time / len(created_clips) if created_clips else 0,
            'clips_per_minute': len(created_clips) / (processing_time / 60) if processing_time > 0 else 0
        }
    }
    
    return report

def save_processing_report(report: Dict, output_dir: str) -> str:
    """Save processing report to JSON file"""
    report_filename = f"processing_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Processing report saved to: {report_path}")
        return report_path
    except Exception as e:
        print(f"Failed to save processing report: {e}")
        return ""

def validate_output_clips(clip_paths: List[str]) -> Dict:
    """Validate that output clips are properly created and playable"""
    validation_results = {
        'total_clips': len(clip_paths),
        'valid_clips': 0,
        'invalid_clips': [],
        'total_duration': 0,
        'average_filesize_mb': 0
    }
    
    file_sizes = []
    
    for clip_path in clip_paths:
        try:
            if not os.path.exists(clip_path):
                validation_results['invalid_clips'].append({
                    'path': clip_path,
                    'error': 'File does not exist'
                })
                continue
            
            # Check file size
            file_size = os.path.getsize(clip_path)
            if file_size < 1000:  # Less than 1KB
                validation_results['invalid_clips'].append({
                    'path': clip_path,
                    'error': 'File too small (possibly corrupted)'
                })
                continue
            
            file_sizes.append(file_size / (1024 * 1024))  # Convert to MB
            
            # Try to open with moviepy to verify it's playable
            with VideoFileClip(clip_path) as clip:
                validation_results['total_duration'] += clip.duration
                validation_results['valid_clips'] += 1
                
        except Exception as e:
            validation_results['invalid_clips'].append({
                'path': clip_path,
                'error': str(e)
            })
    
    if file_sizes:
        validation_results['average_filesize_mb'] = sum(file_sizes) / len(file_sizes)
    
    validation_results['success_rate'] = (validation_results['valid_clips'] / validation_results['total_clips']) * 100 if validation_results['total_clips'] > 0 else 0
    
    return validation_results
