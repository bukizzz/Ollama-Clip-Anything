import cv2
import numpy as np
import time # Import time for delays
from typing import Callable, Optional, List, Dict
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker
from video.layout_manager import LayoutManager

from core.temp_manager import get_temp_path # Import get_temp_path
import logging # Import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to INFO to see detailed frame extraction logs

class FrameProcessor:
    def __init__(self, original_w: int, original_h: int, output_w: int, output_h: int, face_tracker: Optional[FaceTracker] = None, object_tracker: Optional[ObjectTracker] = None):
        self.original_w = original_w
        self.original_h = original_h
        self.output_w = output_w
        self.output_h = output_h
        self.face_tracker = face_tracker
        self.object_tracker = object_tracker
        self.layout_manager = LayoutManager()
        self.last_layout = 'single_person_focus' # Initialize to a default string
        self.transition_progress = -1

    def process_frame(self, get_frame: Callable[[float], np.ndarray], t: float, layout_info: dict, engagement_metrics: Optional[List[Dict]] = None) -> np.ndarray:
        frame = get_frame(t).copy()
        
        # Ensure current_layout is always a string
        recommended_layout_val = layout_info.get('recommended_layout')
        if recommended_layout_val is None:
            # If no specific layout is recommended, try to infer based on face count
            if self.face_tracker:
                faces = self.face_tracker.detect_faces_in_frame(frame)
                if len(faces) == 2:
                    current_layout = 'two_person_split_screen'
                else:
                    current_layout = 'single_person_focus' # Default for other cases
            else:
                current_layout = 'single_person_focus'
        else:
            current_layout = str(recommended_layout_val) # Explicitly cast to str to be safe
            
        active_speaker_id = layout_info.get('active_speaker')

        # Handle layout transitions
        if self.last_layout and self.last_layout != current_layout:
            self.transition_progress = 0
        
        if self.transition_progress >= 0:
            # In transition
            from_frame = self.layout_manager.apply_layout(frame, self.last_layout, self.output_w, self.output_h, self.face_tracker, self.object_tracker, active_speaker_id, engagement_metrics, t)
            to_frame = self.layout_manager.apply_layout(frame, current_layout, self.output_w, self.output_h, self.face_tracker, self.object_tracker, active_speaker_id, engagement_metrics, t)
            
            # Simple cross-fade transition
            alpha = self.transition_progress / 10.0 # 10 frames transition
            final_frame = cv2.addWeighted(from_frame, 1 - alpha, to_frame, alpha, 0)
            
            self.transition_progress += 1
            if self.transition_progress > 10:
                self.transition_progress = -1
                self.last_layout = current_layout
        else:
            # No transition
            final_frame = self.layout_manager.apply_layout(frame, current_layout, self.output_w, self.output_h, self.face_tracker, self.object_tracker, active_speaker_id, engagement_metrics, t)
            self.last_layout = current_layout

        # Add speaker labels
        if active_speaker_id and self.face_tracker:
            faces = self.face_tracker.detect_faces_in_frame(frame)
            for face in faces:
                if face.get('id') == active_speaker_id:
                    x, y, w, h = face['bbox']
                    # Adjust bbox to final frame coordinates (this is complex and needs a robust mapping)
                    # For now, let's assume the label is added to the corner
                    cv2.putText(final_frame, f"Speaker: {active_speaker_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    break

        return final_frame

    def extract_frame_at_timestamp(self, video_path: str, timestamp_sec: float, max_retries: int = 3, retry_delay_sec: float = 1.0) -> Optional[str]:
        """
        Extracts a single frame from a video at a specific timestamp and saves it to a temporary file.
        Includes retry logic for robustness.
        Returns the path to the saved frame.
        """
        logger.info(f"Attempting to extract frame at {timestamp_sec:.2f}s from {video_path}")
        for attempt in range(max_retries):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Error: Could not open video file {video_path} (Attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec)
                continue

            # Check if video has any frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                logger.error(f"Error: Video file {video_path} has no frames (Attempt {attempt + 1}/{max_retries}).")
                cap.release()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec)
                continue

            # Set position by milliseconds for better precision
            target_msec = timestamp_sec * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
            
            # Verify if the position was set correctly
            current_msec_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            logger.info(f"Requested timestamp {timestamp_sec:.2f}s ({target_msec:.2f}ms), actual position set to {current_msec_pos:.2f}ms.")

            ret, frame = cap.read()
            cap.release()

            if ret:
                temp_frame_path = get_temp_path(f"frame_{int(timestamp_sec * 1000)}.jpg")
                try:
                    cv2.imwrite(temp_frame_path, frame)
                    logger.info(f"Successfully extracted and saved frame to {temp_frame_path}")
                    return temp_frame_path
                except Exception as e:
                    logger.error(f"Error saving frame to {temp_frame_path}: {e} (Attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay_sec)
            else:
                logger.error(f"Error: Could not read frame at timestamp {timestamp_sec:.2f}s from {video_path} (Attempt {attempt + 1}/{max_retries}). Check if timestamp is out of bounds or video is corrupted.")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec)
        
        logger.error(f"Failed to extract frame at timestamp {timestamp_sec:.2f}s after {max_retries} attempts.")
        return None
