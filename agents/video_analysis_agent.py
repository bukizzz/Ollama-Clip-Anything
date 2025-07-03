from agents.base_agent import Agent
from typing import Dict, Any
import cv2
import mediapipe as mp
import numpy as np
from core.state_manager import set_stage_status
from video.tracking_manager import TrackingManager
from deepface import DeepFace
import time

class VideoAnalysisAgent(Agent):
    """Agent for comprehensive video analysis including engagement metrics."""

    def __init__(self, config, state_manager):
        super().__init__("VideoAnalysisAgent")
        self.config = config
        self.state_manager = state_manager
        self.face_tracker = TrackingManager().get_face_tracker()
        self.object_tracker = TrackingManager().get_object_tracker()
        self.mp_holistic = mp.solutions.holistic

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        video_path = context.get("processed_video_path")
        qwen_results = context.get('qwen_vision_analysis_results', [])

        if not video_path:
            self.log_error("Video path missing. Cannot perform video analysis.")
            set_stage_status('video_analysis', 'failed', {'reason': 'Missing video path'})
            return context

        self.log_info("Starting enhanced video analysis...")
        set_stage_status('video_analysis', 'running')

        video_analysis = {
            "facial_expressions": [], "gesture_recognition": [],
            "visual_complexity": [], "energy_levels": [], "engagement_metrics": [],
            "qwen_features": qwen_results
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_error(f"Could not open video file: {video_path}")
            set_stage_status('video_analysis', 'failed', {'reason': 'Could not open video file'})
            return context

        # Get video properties for debugging and frame limiting
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        self.log_info(f"Video properties - Total frames: {total_frames}, FPS: {fps}, Duration: {duration}s")
        
        # Add safety limits
        max_frames = min(total_frames, 3000)  # Limit to 3000 frames max
        max_duration = 300  # 5 minutes max
        start_time = time.time()
        
        frame_count = 0
        prev_frame = None
        
        try:
            with self.mp_holistic.Holistic(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5,
                model_complexity=1  # Use lighter model for faster processing
            ) as holistic:
                
                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    
                    # Check if frame reading failed
                    if not ret:
                        self.log_info(f"End of video reached at frame {frame_count}")
                        break
                    
                    # Safety check for processing time
                    if time.time() - start_time > max_duration:
                        self.log_warning(f"Processing timeout reached after {max_duration}s")
                        break
                    
                    # Progress logging
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        self.log_info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        self.log_warning(f"Invalid frame at {timestamp}s")
                        frame_count += 1
                        continue
                    
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Facial Expression Analysis with error handling
                    faces = []
                    expression = "neutral"
                    try:
                        faces = self.face_tracker.detect_faces_in_frame(frame)
                        if faces:
                            try:
                                result = DeepFace.analyze(
                                    frame, 
                                    actions=['emotion'], 
                                    enforce_detection=False, 
                                    detector_backend='mediapipe',
                                    silent=True  # Reduce DeepFace logging
                                )
                                if result and len(result) > 0:
                                    expression = result[0]['dominant_emotion']
                            except Exception as e:
                                self.log_warning(f"DeepFace analysis failed at {timestamp}s: {e}")
                    except Exception as e:
                        self.log_warning(f"Face detection failed at {timestamp}s: {e}")
                    
                    video_analysis["facial_expressions"].append({
                        "timestamp": timestamp, 
                        "expression": expression
                    })

                    # Gesture Recognition with error handling
                    gestures = []
                    try:
                        # Convert frame for MediaPipe
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(rgb_frame)
                        
                        if results.pose_landmarks:
                            gestures.append("pose")
                        if results.left_hand_landmarks:
                            gestures.append("left_hand")
                        if results.right_hand_landmarks:
                            gestures.append("right_hand")
                    except Exception as e:
                        self.log_warning(f"Gesture recognition failed at {timestamp}s: {e}")
                    
                    video_analysis["gesture_recognition"].append({
                        "timestamp": timestamp, 
                        "gestures": gestures
                    })

                    # Visual Complexity (using Laplacian variance)
                    try:
                        complexity = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                    except Exception as e:
                        self.log_warning(f"Complexity calculation failed at {timestamp}s: {e}")
                        complexity = 0
                    
                    video_analysis["visual_complexity"].append({
                        "timestamp": timestamp, 
                        "score": float(complexity)
                    })

                    # Energy Levels (using optical flow)
                    energy = 0
                    if prev_frame is not None:
                        try:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_frame, gray_frame, None, 
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            energy = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                        except Exception as e:
                            self.log_warning(f"Optical flow calculation failed at {timestamp}s: {e}")
                    
                    video_analysis["energy_levels"].append({
                        "timestamp": timestamp, 
                        "level": energy
                    })
                    prev_frame = gray_frame.copy()

                    # Engagement Metrics
                    try:
                        engagement_score = (
                            (len(faces) * 0.4) + 
                            (len(gestures) * 0.2) + 
                            (energy * 0.3) + 
                            (complexity / 1000 * 0.1)
                        )
                    except Exception as e:
                        self.log_warning(f"Engagement calculation failed at {timestamp}s: {e}")
                        engagement_score = 0
                    
                    video_analysis["engagement_metrics"].append({
                        "timestamp": timestamp, 
                        "score": float(engagement_score)
                    })
                    
                    frame_count += 1

        except Exception as e:
            self.log_error(f"Critical error during video analysis: {e}")
            set_stage_status('video_analysis', 'failed', {'reason': str(e)})
            return context
        finally:
            # Always release the video capture
            if cap.isOpened():
                cap.release()

        processing_time = time.time() - start_time
        self.log_info(f"Enhanced video analysis complete. Processed {frame_count} frames in {processing_time:.2f}s")
        
        context['video_analysis_results'] = video_analysis
        set_stage_status('video_analysis_complete', 'complete', {
            'metrics_collected': True,
            'frames_processed': frame_count,
            'processing_time': processing_time
        })
        
        return context  # Fixed: was returning True instead of context
