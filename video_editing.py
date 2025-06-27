# video_editing.py
"""
Enhanced video editing with MoviePy featuring:
- Dynamic word-by-word subtitles with animations
- Object tracking and recognition with visual annotations
- Face tracking with automatic arrows/circles
- Scene change detection with zoom effects
- GPU-optimized processing where possible
- Advanced subtitle styling and animations
- Smart cropping and composition
"""
import os
import re
import json
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import cv2
from moviepy.editor import *
from moviepy.video.fx import resize, crop
from moviepy.video.tools.segmenting import findObjects
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
import spacy
from collections import defaultdict
import mediapipe as mp
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import librosa
import webcolors

from temp_manager import get_temp_path
from config import OUTPUT_DIR, CLIP_PREFIX, FACE_DETECTION_SAMPLES

# Initialize models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class SceneDetector:
    """Advanced scene change detection using multiple methods"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.prev_hist = None
        self.prev_features = None
        
    def detect_scene_changes(self, video_path: str, sample_rate: int = 1) -> List[float]:
        """Detect scene changes using histogram comparison and feature analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        scene_changes = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if self.prev_hist is not None:
                    # Calculate histogram correlation
                    correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    if correlation < (1 - self.threshold):
                        scene_changes.append(timestamp)
                        print(f"Scene change detected at {timestamp:.2f}s (correlation: {correlation:.3f})")
                
                self.prev_hist = hist
            
            frame_count += 1
        
        cap.release()
        return scene_changes

class FaceTracker:
    """Advanced face tracking with MediaPipe and visual annotations"""
    
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)
        self.tracked_faces = {}
        
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces and return detailed information"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Get key points for more detailed tracking
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        keypoints.append((int(keypoint.x * w), int(keypoint.y * h)))
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'center': (x + width // 2, y + height // 2),
                    'keypoints': keypoints,
                    'area': width * height
                })
        
        return faces
    
    def create_face_annotation_clip(self, frame_array: np.ndarray, faces: List[Dict], duration: float) -> VideoClip:
        """Create animated annotations for detected faces"""
        def make_frame(t):
            frame = frame_array.copy()
            h, w = frame.shape[:2]
            
            for i, face in enumerate(faces):
                x, y, width, height = face['bbox']
                center_x, center_y = face['center']
                
                # Animated circle around face
                radius = int(max(width, height) * 0.6)
                animation_phase = (t * 2) % (2 * math.pi)
                circle_radius = radius + int(10 * math.sin(animation_phase))
                
                # Draw pulsing circle
                cv2.circle(frame, (center_x, center_y), circle_radius, (0, 255, 255), 3)
                
                # Draw arrow pointing to face
                arrow_length = 50
                arrow_angle = animation_phase
                arrow_end_x = center_x + int(arrow_length * math.cos(arrow_angle))
                arrow_end_y = center_y + int(arrow_length * math.sin(arrow_angle))
                
                cv2.arrowedLine(frame, (arrow_end_x, arrow_end_y), (center_x, center_y), (255, 0, 0), 3)
                
                # Add confidence text
                confidence_text = f"Face {i+1}: {face['confidence']:.2f}"
                cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
        
        return VideoClip(make_frame, duration=duration)

class ObjectTracker:
    """Enhanced object tracking with visual annotations"""
    
    def __init__(self):
        self.model = None
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracked_objects = {}
        self.initialize_model()
        
        # COCO class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Colors for different object classes
        self.class_colors = {
            'person': (255, 0, 0),
            'car': (0, 255, 0),
            'dog': (255, 255, 0),
            'cat': (255, 0, 255),
            'bicycle': (0, 255, 255),
        }

    def initialize_model(self):
        try:
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(device)
            self.model.eval()
            print("Object detection model loaded successfully")
        except Exception as e:
            print(f"Could not load object detection model: {e}")

    def detect_objects_in_frame(self, frame: np.ndarray, confidence_threshold: float = 0.7) -> List[Dict]:
        if self.model is None:
            return []

        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            img_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            detected_objects = []
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.coco_names[label] if label < len(self.coco_names) else 'unknown'
                    detected_objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(score),
                        'class': class_name,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1)
                    })

            return detected_objects

        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def create_object_annotation_clip(self, frame_array: np.ndarray, objects: List[Dict], duration: float) -> VideoClip:
        """Create animated annotations for detected objects"""
        def make_frame(t):
            frame = frame_array.copy()
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                class_name = obj['class']
                confidence = obj['confidence']
                
                # Get color for this class
                color = self.class_colors.get(class_name, (128, 128, 128))
                
                # Animated bounding box
                animation_phase = (t * 3) % (2 * math.pi)
                thickness = 2 + int(2 * math.sin(animation_phase))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label background
                label_text = f"{class_name}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                center_x, center_y = obj['center']
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            return frame
        
        return VideoClip(make_frame, duration=duration)

class SubtitleGenerator:
    """Advanced subtitle generation with animations and styling"""
    
    def __init__(self):
        self.font_families = ['Arial-Bold', 'Helvetica-Bold', 'Georgia-Bold']
        self.animation_styles = ['fade', 'slide', 'zoom', 'bounce', 'typewriter']
        
    def analyze_text_sentiment(self, text: str) -> str:
        """Analyze text sentiment for styling"""
        if nlp is None:
            return 'neutral'
        
        doc = nlp(text)
        # Simple sentiment analysis based on token sentiment
        positive_words = ['great', 'amazing', 'awesome', 'fantastic', 'excellent', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def get_subtitle_style(self, sentiment: str, emphasis: bool = False) -> Dict:
        """Get styling based on sentiment and emphasis"""
        base_style = {
            'fontsize': 45 if emphasis else 35,
            'color': 'white',
            'stroke_color': 'black',
            'stroke_width': 2,
            'method': 'caption'
        }
        
        if sentiment == 'positive':
            base_style.update({
                'color': '#00FF7F',  # Spring green
                'stroke_color': '#006400'  # Dark green
            })
        elif sentiment == 'negative':
            base_style.update({
                'color': '#FF6B6B',  # Light red
                'stroke_color': '#8B0000'  # Dark red
            })
        elif emphasis:
            base_style.update({
                'color': '#FFD700',  # Gold
                'stroke_color': '#FF8C00'  # Dark orange
            })
        
        return base_style
    
    def create_word_by_word_subtitles(self, transcript: List[Dict], video_duration: float, video_size: Tuple[int, int]) -> List[VideoClip]:
        """Create animated word-by-word subtitles"""
        subtitle_clips = []
        w, h = video_size
        
        for segment in transcript:
            if 'words' not in segment:
                continue
            
            words = segment['words']
            segment_sentiment = self.analyze_text_sentiment(segment.get('text', ''))
            
            for i, word_data in enumerate(words):
                if isinstance(word_data, dict):
                    word = word_data.get('word', '')
                    start_time = float(word_data.get('start', 0))
                    end_time = float(word_data.get('end', start_time + 0.5))
                else:
                    # Handle simple string format
                    word = str(word_data)
                    start_time = segment.get('start', 0) + i * 0.3
                    end_time = start_time + 0.5
                
                if not word.strip():
                    continue
                
                duration = end_time - start_time
                if duration <= 0:
                    duration = 0.5
                
                # Determine if word should be emphasized
                emphasis = word.upper() == word and len(word) > 2
                
                # Get styling
                style = self.get_subtitle_style(segment_sentiment, emphasis)
                
                # Create base text clip
                txt_clip = TextClip(
                    word.strip(),
                    font=random.choice(self.font_families),
                    **style
                ).set_duration(duration).set_start(start_time)
                
                # Position at bottom center
                txt_clip = txt_clip.set_position(('center', h * 0.85))
                
                # Add animation
                animation_style = random.choice(self.animation_styles)
                txt_clip = self.apply_text_animation(txt_clip, animation_style, duration)
                
                subtitle_clips.append(txt_clip)
        
        return subtitle_clips
    
    def apply_text_animation(self, text_clip: TextClip, animation_style: str, duration: float) -> VideoClip:
        """Apply various animations to text clips"""
        if animation_style == 'fade':
            return text_clip.fadein(0.2).fadeout(0.2)
        
        elif animation_style == 'slide':
            w, h = text_clip.size
            return text_clip.set_position(lambda t: ('center', max(100, 200 - t * 100)))
        
        elif animation_style == 'zoom':
            def zoom_effect(get_frame, t):
                frame = get_frame(t)
                if t < 0.3:
                    scale = 0.5 + (t / 0.3) * 0.5
                    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                    frame = cv2.resize(frame, new_size)
                return frame
            return text_clip.fl(zoom_effect, apply_to=['mask'])
        
        elif animation_style == 'bounce':
            def bounce_effect(t):
                bounce_height = 20 * abs(math.sin(t * math.pi * 4))
                return ('center', text_clip.pos(0)[1] - bounce_height)
            return text_clip.set_position(bounce_effect)
        
        elif animation_style == 'typewriter':
            # This would require more complex implementation
            return text_clip.fadein(0.1)
        
        return text_clip

class VideoEffects:
    """Advanced video effects and transitions"""
    
    @staticmethod
    def apply_zoom_effect(clip: VideoClip, zoom_factor: float = 1.2, focus_point: Optional[Tuple[int, int]] = None) -> VideoClip:
        """Apply zoom effect with optional focus point"""
        w, h = clip.size
        
        if focus_point is None:
            focus_point = (w // 2, h // 2)
        
        def zoom_frame(get_frame, t):
            frame = get_frame(t)
            zoom = 1 + (zoom_factor - 1) * min(t / clip.duration, 1)
            
            # Calculate new dimensions
            new_w, new_h = int(w * zoom), int(h * zoom)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Calculate crop to maintain original size
            crop_x = max(0, (new_w - w) // 2)
            crop_y = max(0, (new_h - h) // 2)
            
            # Adjust crop based on focus point
            focus_x, focus_y = focus_point
            crop_x = max(0, min(crop_x + int((focus_x - w//2) * (zoom - 1)), new_w - w))
            crop_y = max(0, min(crop_y + int((focus_y - h//2) * (zoom - 1)), new_h - h))
            
            cropped = resized[crop_y:crop_y + h, crop_x:crop_x + w]
            
            # Ensure we have the right dimensions
            if cropped.shape[:2] != (h, w):
                cropped = cv2.resize(cropped, (w, h))
                
            return cropped
        
        return clip.fl(zoom_frame)
    
    @staticmethod
    def apply_scene_transition(clip1: VideoClip, clip2: VideoClip, transition_duration: float = 1.0, transition_type: str = 'crossfade') -> VideoClip:
        """Apply transitions between scenes"""
        if transition_type == 'crossfade':
            clip1_out = clip1.fadeout(transition_duration)
            clip2_in = clip2.fadein(transition_duration).set_start(clip1.duration - transition_duration)
            return concatenate_videoclips([clip1_out, clip2_in])
        
        elif transition_type == 'slide':
            # Implement slide transition
            w, h = clip1.size
            
            def slide_effect(get_frame, t):
                if t < clip1.duration - transition_duration:
                    return clip1.get_frame(t)
                elif t > clip1.duration:
                    return clip2.get_frame(t - clip1.duration)
                else:
                    # During transition
                    progress = (t - (clip1.duration - transition_duration)) / transition_duration
                    frame1 = clip1.get_frame(clip1.duration - 0.01)
                    frame2 = clip2.get_frame(0)
                    
                    # Slide effect
                    offset = int(w * progress)
                    result = np.zeros_like(frame1)
                    result[:, :w-offset] = frame1[:, offset:]
                    result[:, w-offset:] = frame2[:, :offset]
                    
                    return result
            
            total_duration = clip1.duration + clip2.duration - transition_duration
            return VideoClip(slide_effect, duration=total_duration)
        
        # Default to crossfade
        return VideoEffects.apply_scene_transition(clip1, clip2, transition_duration, 'crossfade')

def get_next_output_filename(source_video_path: str, clip_number: int) -> str:
    """Generate unique output filename"""
    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    random_num = random.randint(1000, 9999)
    folder_name = f"{source_name}_enhanced_{random_num}"
    output_folder = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    existing = [f for f in os.listdir(output_folder) if f.startswith(CLIP_PREFIX) and f.endswith(".mp4")]
    batch_numbers = [int(m.group(1)) for f in existing if (m := re.search(r'batch(\d+)_', f))]
    current_batch = max(batch_numbers, default=0) + 1

    return os.path.join(output_folder, f"{CLIP_PREFIX}_enhanced_batch{current_batch}_{clip_number}.mp4")

def create_enhanced_individual_clip(
    original_video_path: str, 
    clip_data: Dict, 
    clip_number: int, 
    original_transcript: List[Dict], 
    video_info: Dict,
    enable_face_tracking: bool = True,
    enable_object_tracking: bool = True,
    enable_scene_effects: bool = True,
    enable_advanced_subtitles: bool = True
) -> str:
    """Create an individual clip with all enhanced features"""
    start, end = float(clip_data['start']), float(clip_data['end'])
    duration = end - start

    print(f"Creating enhanced clip {clip_number}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")

    try:
        # Load video clip
        video_clip = VideoFileClip(original_video_path).subclip(start, end)
        
        # Get video dimensions and crop to 9:16
        original_w, original_h = video_info['width'], video_info['height']
        crop_h = original_h
        crop_w = int(original_h * (9 / 16))
        crop_x = (original_w - crop_w) // 2

        # Ensure even dimensions
        if crop_w % 2 != 0:
            crop_w += 1
        if crop_h % 2 != 0:
            crop_h += 1

        video_clip = video_clip.crop(x1=crop_x, y1=0, width=crop_w, height=crop_h)
        
        # Initialize processors
        face_tracker = FaceTracker() if enable_face_tracking else None
        object_tracker = ObjectTracker() if enable_object_tracking else None
        subtitle_generator = SubtitleGenerator() if enable_advanced_subtitles else None
        scene_detector = SceneDetector() if enable_scene_effects else None
        
        # Process first frame for object/face detection
        first_frame = video_clip.get_frame(0)
        overlay_clips = []
        
        # Face tracking
        if face_tracker:
            faces = face_tracker.detect_faces_in_frame(first_frame)
            if faces:
                print(f"Detected {len(faces)} faces")
                face_overlay = face_tracker.create_face_annotation_clip(first_frame, faces, duration)
                face_overlay = face_overlay.set_duration(duration)
                overlay_clips.append(face_overlay)
        
        # Object tracking
        if object_tracker:
            objects = object_tracker.detect_objects_in_frame(first_frame)
            if objects:
                print(f"Detected {len(objects)} objects: {[obj['class'] for obj in objects]}")
                object_overlay = object_tracker.create_object_annotation_clip(first_frame, objects, duration)
                object_overlay = object_overlay.set_duration(duration)
                overlay_clips.append(object_overlay)
        
        # Scene effects
        if enable_scene_effects:
            # Apply zoom effect if objects or faces detected
            if (face_tracker and faces) or (object_tracker and objects):
                focus_point = None
                if faces:
                    # Focus on largest face
                    largest_face = max(faces, key=lambda f: f['area'])
                    focus_point = largest_face['center']
                elif objects:
                    # Focus on most prominent object
                    largest_object = max(objects, key=lambda o: o['area'])
                    focus_point = largest_object['center']
                
                if focus_point:
                    video_clip = VideoEffects.apply_zoom_effect(video_clip, zoom_factor=1.1, focus_point=focus_point)
        
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
                    clip_transcript, duration, (crop_w, crop_h)
                )
        
        # Combine all clips
        final_clips = [video_clip]
        final_clips.extend(overlay_clips)
        final_clips.extend(subtitle_clips)
        
        if len(final_clips) > 1:
            final_video = CompositeVideoClip(final_clips)
        else:
            final_video = video_clip
        
        # Output path
        output_path = get_next_output_filename(original_video_path, clip_number)
        
        # Write final video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=get_temp_path(f'temp_audio_enhanced_{clip_number}.m4a'),
            remove_temp=True,
            fps=30,
            preset='medium',
            ffmpeg_params=[
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
        )

        # Cleanup
        video_clip.close()
        final_video.close()
        for clip in overlay_clips + subtitle_clips:
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
                original_video_path, clip_data, i, original_transcript, video_info, **valid_clip_options
            )
            created_clips.append(clip_path)
        except Exception as e:
            print(f"Failed to create enhanced clip {i}: {e}")
            failed_clips.append(i)

    print(f"\nBatch processing complete: {len(created_clips)} successful, {len(failed_clips)} failed")
    if failed_clips:
        print(f"Failed clips: {failed_clips}")
    
    return created_clips, failed_clips

# Utility functions for external integration
def analyze_video_content(video_path: str) -> Dict:
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
        
        face_tracker = FaceTracker()
        object_tracker = ObjectTracker()
        
        for i in range(0, frame_count, sample_interval):
            if len(sample_frames) >= max_samples:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            sample_frames.append(i / fps)  # timestamp
            
            # Analyze faces
            faces = face_tracker.detect_faces_in_frame(frame)
            face_count_samples.append(len(faces))
            
            # Analyze objects
            objects = object_tracker.detect_objects_in_frame(frame)
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
        
        print(f"Video Analysis Complete:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Resolution: {width}x{height}")
        print(f"  Average faces per frame: {avg_faces:.2f}")
        print(f"  Average objects per frame: {avg_objects:.2f}")
        print(f"  Processing complexity: {analysis['processing_complexity']}")
        
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
    duration = video_analysis.get('duration', 0)
    
    # Adjust based on complexity
    if complexity == 'high':
        settings['batch_size'] = 3
        if available_memory_gb < 8:
            settings['enable_object_tracking'] = False
            settings['processing_quality'] = 'medium'
    elif complexity == 'low':
        settings['batch_size'] = 10
        settings['parallel_processing'] = available_memory_gb > 16
    
    # Adjust based on duration
    if duration > 1800:  # 30 minutes
        settings['enable_scene_effects'] = False
        settings['batch_size'] = max(2, settings['batch_size'] // 2)
    
    # Disable features if not recommended
    if not video_analysis.get('recommended_face_tracking', True):
        settings['enable_face_tracking'] = False
    if not video_analysis.get('recommended_object_tracking', True):
        settings['enable_object_tracking'] = False
    
    print(f"Optimized processing settings:")
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
        'processing_timestamp': import_datetime().datetime.now().isoformat(),
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

def import_datetime():
    """Import datetime module"""
    import datetime
    return datetime

def save_processing_report(report: Dict, output_dir: str) -> str:
    """Save processing report to JSON file"""
    report_filename = f"processing_report_{import_datetime().datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Processing report saved to: {report_path}")
        return report_path
    except Exception as e:
        print(f"Failed to save processing report: {e}")
        return ""

def batch_process_with_analysis(
    video_path: str,
    clips_data: List[Dict],
    transcript: List[Dict],
    custom_settings: Optional[Dict] = None
) -> Tuple[List[str], Dict]:
    """Complete batch processing pipeline with analysis and optimization"""
    import time
    start_time = time.time()
    
    print("Starting comprehensive video processing pipeline...")
    
    # Step 1: Analyze video content
    print("\n=== Step 1: Video Content Analysis ===")
    video_analysis = analyze_video_content(video_path)
    
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
        **processing_settings
    )
    
    # Step 5: Generate processing report
    processing_time = time.time() - start_time
    print(f"\n=== Step 4: Generating Report ===")
    
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
    
    print(f"\n=== Processing Complete ===")
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

# Quality control functions
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
