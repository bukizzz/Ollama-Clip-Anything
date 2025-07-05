from agents.base_agent import Agent
from typing import Dict, Any, List, Optional, Tuple
import cv2
from mediapipe.python.solutions import holistic as mp_holistic_solutions
import numpy as np
from core.state_manager import set_stage_status
from deepface import DeepFace
import time

class VideoAnalysisAgent(Agent):
    """Agent for comprehensive video analysis including engagement metrics."""

    # Constants for configuration and thresholds
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    HOLISTIC_MODEL_COMPLEXITY = 1 # Lighter model for faster processing
    PROGRESS_LOG_INTERVAL = 100 # Log progress every N frames

    # Engagement score weights
    FACE_WEIGHT = 0.4
    GESTURE_WEIGHT = 0.2
    ENERGY_WEIGHT = 0.3
    COMPLEXITY_WEIGHT = 0.1
    COMPLEXITY_NORMALIZATION_FACTOR = 1000.0 # Factor to normalize complexity score

    def __init__(self, config, state_manager):
        super().__init__("VideoAnalysisAgent")
        self.config = config
        self.state_manager = state_manager
        # self.face_tracker = TrackingManager().get_face_tracker() # Not used in execute, removed
        # self.object_tracker = TrackingManager().get_object_tracker() # Not used in execute, removed
        self.mp_holistic = mp_holistic_solutions.Holistic

    def _process_frame_data(self, frame_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Reads a frame from path and performs basic validation."""
        current_frame = cv2.imread(frame_path)
        if current_frame is None or current_frame.size == 0:
            self.log_warning(f"Invalid frame at {timestamp}s from path {frame_path}")
            return None
        return current_frame

    def _perform_deepface_analysis(self, frame: np.ndarray, timestamp: float) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Performs facial emotion analysis using DeepFace."""
        dominant_emotion = "neutral"
        deepface_results = None
        try:
            deepface_results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='mediapipe',
                silent=True
            )
            if deepface_results and len(deepface_results) > 0:
                dominant_emotion = deepface_results[0]['dominant_emotion']
        except Exception as e:
            self.log_warning(f"DeepFace analysis failed for frame at {timestamp}s: {e}")
        return dominant_emotion, deepface_results # Return deepface_results for face count

    def _perform_gesture_recognition(self, frame: np.ndarray, holistic_processor: mp_holistic_solutions.Holistic) -> List[str]:
        """Performs gesture recognition using MediaPipe Holistic."""
        gestures = []
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_processor.process(rgb_frame)
            if results.pose_landmarks: # type: ignore
                gestures.append("pose")
            if results.left_hand_landmarks: # type: ignore
                gestures.append("left_hand")
            if results.right_hand_landmarks: # type: ignore
                gestures.append("right_hand")
        except Exception as e:
            self.log_warning(f"Gesture recognition failed: {e}")
        return gestures

    def _calculate_visual_complexity(self, gray_frame: np.ndarray, timestamp: float) -> float:
        """Calculates visual complexity using Laplacian variance."""
        complexity = 0.0
        try:
            complexity = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        except Exception as e:
            self.log_warning(f"Complexity calculation failed at {timestamp}s: {e}")
        return float(complexity)

    def _calculate_energy_level(self, prev_gray_frame: Optional[np.ndarray], current_gray_frame: np.ndarray, timestamp: float) -> float:
        """Calculates energy level using optical flow."""
        energy = 0.0
        if prev_gray_frame is not None:
            try:
                # The 'flow' parameter is an output array, passing None is correct for allocation
                flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, current_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0) # type: ignore
                energy = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            except Exception as e:
                self.log_warning(f"Optical flow calculation failed at {timestamp}s: {e}")
        return float(energy)

    def _calculate_engagement_score(self, faces_detected_count: int, gestures: List[str], energy: float, complexity: float) -> float:
        """Calculates an engagement score based on various metrics."""
        engagement_score = 0.0
        try:
            normalized_complexity = complexity / self.COMPLEXITY_NORMALIZATION_FACTOR
            engagement_score = (
                (faces_detected_count * self.FACE_WEIGHT) +
                (len(gestures) * self.GESTURE_WEIGHT) +
                (energy * self.ENERGY_WEIGHT) +
                (normalized_complexity * self.COMPLEXITY_WEIGHT)
            )
        except Exception as e:
            self.log_warning(f"Engagement calculation failed: {e}")
        return float(engagement_score)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        extracted_frames_info = context.get("extracted_frames_info")
        qwen_results = context.get('qwen_vision_analysis_results', [])

        if not extracted_frames_info:
            self.log_error("Extracted frames info missing. Cannot perform video analysis.")
            set_stage_status('video_analysis', 'failed', {'reason': 'Missing extracted_frames_info'})
            return context

        print("👁️ Starting enhanced video analysis on pre-processed frames...")
        set_stage_status('video_analysis', 'running')

        video_analysis = {
            "facial_expressions": [], "gesture_recognition": [],
            "visual_complexity": [], "energy_levels": [], "engagement_metrics": [],
            "qwen_features": qwen_results
        }

        processed_frame_count = 0
        prev_gray_frame = None
        start_time = time.time()

        try:
            with self.mp_holistic(
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
                model_complexity=self.HOLISTIC_MODEL_COMPLEXITY
            ) as holistic:
                
                for frame_info in extracted_frames_info:
                    frame_path = frame_info['frame_path']
                    timestamp = frame_info['timestamp_sec']
                    
                    current_frame = self._process_frame_data(frame_path, timestamp)
                    if current_frame is None:
                        continue
                    
                    # Progress logging
                    if processed_frame_count % self.PROGRESS_LOG_INTERVAL == 0:
                        progress = (processed_frame_count / len(extracted_frames_info)) * 100
                        print(f"🎞️ Processing frame {processed_frame_count}/{len(extracted_frames_info)} ({progress:.1f}%)")

                    # DeepFace analysis
                    dominant_emotion, deepface_results = self._perform_deepface_analysis(current_frame, timestamp)
                    video_analysis["facial_expressions"].append({
                        "timestamp": timestamp,
                        "expression": dominant_emotion
                    })

                    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                    # Gesture recognition
                    gestures = self._perform_gesture_recognition(current_frame, holistic)
                    video_analysis["gesture_recognition"].append({
                        "timestamp": timestamp,
                        "gestures": gestures
                    })

                    # Visual complexity
                    complexity = self._calculate_visual_complexity(gray_frame, timestamp)
                    video_analysis["visual_complexity"].append({
                        "timestamp": timestamp,
                        "score": complexity
                    })

                    # Energy levels
                    energy = self._calculate_energy_level(prev_gray_frame, gray_frame, timestamp)
                    video_analysis["energy_levels"].append({
                        "timestamp": timestamp,
                        "level": energy
                    })
                    prev_gray_frame = gray_frame.copy()

                    # Engagement score
                    faces_detected_count = len(deepface_results) if deepface_results else 0
                    engagement_score = self._calculate_engagement_score(faces_detected_count, gestures, energy, complexity)
                    video_analysis["engagement_metrics"].append({
                        "timestamp": timestamp,
                        "score": engagement_score
                    })
                    
                    processed_frame_count += 1
                        
        except Exception as e:
            self.log_error(f"Critical error during video analysis: {e}")
            set_stage_status('video_analysis', 'failed', {'reason': str(e)})
            return context

        processing_time = time.time() - start_time
        print(f"✅ Enhanced video analysis complete. Processed {processed_frame_count} frames in {processing_time:.2f}s")
        
        context['video_analysis_results'] = video_analysis
        set_stage_status('video_analysis_complete', 'complete', {
            'metrics_collected': True,
            'frames_processed': processed_frame_count,
            'processing_time': processing_time
        })
        
        return context
