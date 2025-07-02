from agents.base_agent import BaseAgent
from typing import Dict, Any
from analysis import analysis_and_reporting
from video.tracking_manager import TrackingManager
import cv2
import mediapipe as mp
from core.state_manager import set_stage_status, get_stage_status

class VideoAnalysisAgent(BaseAgent):
    """Agent responsible for performing enhanced video analysis and optimizing processing settings."""

    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.face_tracker = TrackingManager().get_face_tracker()
        self.object_tracker = TrackingManager().get_object_tracker()
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        video_path = context.get("processed_video_path")
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')

        if not video_path:
            self.log_error("Processed video path is missing from context. Cannot perform video analysis.")
            set_stage_status('video_analysis', 'failed', {'reason': 'Missing video path'})
            return False

        if not qwen_vision_analysis_results:
            self.log_warning("Qwen2.5-VL analysis results not found. Proceeding with limited video analysis.")

        self.log_info("Starting enhanced video analysis...")
        set_stage_status('video_analysis', 'running')

        video_analysis_results = {
            "facial_expressions": [],
            "gesture_recognition": [],
            "visual_complexity": [],
            "energy_levels": [],
            "engagement_metrics": [],
            "qwen_features": qwen_vision_analysis_results # Integrate Qwen-VL results directly
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_error(f"Could not open video file: {video_path}")
            set_stage_status('video_analysis', 'failed', {'reason': 'Could not open video file'})
            return False

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Facial Expression Analysis (Placeholder - requires a dedicated model or more complex logic)
                # For now, we'll just check for face presence
                face_landmarks = results.face_landmarks
                if face_landmarks:
                    video_analysis_results["facial_expressions"].append({
                        "timestamp": timestamp_sec,
                        "has_face": True,
                        "expression": "neutral" # Placeholder
                    })
                else:
                    video_analysis_results["facial_expressions"].append({
                        "timestamp": timestamp_sec,
                        "has_face": False,
                        "expression": "N/A"
                    })

                # Gesture Recognition (using MediaPipe Holistic for pose and hand landmarks)
                gestures = []
                if results.pose_landmarks:
                    # Simple check for pose presence as a gesture indicator
                    gestures.append("pose_detected")
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    gestures.append("hand_gesture_detected")
                video_analysis_results["gesture_recognition"].append({
                    "timestamp": timestamp_sec,
                    "gestures": gestures
                })

                # Visual Complexity Scores (Placeholder - could use edge detection, color variance, etc.)
                # Simple example: count non-black pixels as a proxy for complexity
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                non_black_pixels = cv2.countNonZero(gray_frame)
                visual_complexity = non_black_pixels / (frame.shape[0] * frame.shape[1])
                video_analysis_results["visual_complexity"].append({
                    "timestamp": timestamp_sec,
                    "score": visual_complexity
                })

                # Energy Levels (Placeholder - could combine movement, facial expressions, audio cues)
                # Simple example: based on overall pixel change from previous frame (requires storing prev_frame)
                # For now, a placeholder.
                energy_level = 0.5 # Placeholder
                video_analysis_results["energy_levels"].append({
                    "timestamp": timestamp_sec,
                    "level": energy_level
                })

                # Frame-level Engagement Metrics (Combination of above, and more advanced logic)
                # This would be a weighted sum or ML model output based on all features.
                engagement_score = (visual_complexity + (1 if face_landmarks else 0) + (1 if gestures else 0) + energy_level) / 4.0
                video_analysis_results["engagement_metrics"].append({
                    "timestamp": timestamp_sec,
                    "score": engagement_score
                })

                frame_idx += 1

        cap.release()

        context['video_analysis_results'] = video_analysis_results
        self.log_info("Enhanced video analysis complete.")
        set_stage_status('video_analysis_complete', 'complete', {'metrics_collected': True})
        return True

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This method is kept for compatibility with AgentManager, but the core logic is in run()
        # In a refactored system, AgentManager would call run() directly.
        return self.run(context)
