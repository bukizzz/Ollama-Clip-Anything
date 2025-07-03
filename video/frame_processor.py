import cv2
import numpy as np
from typing import Callable, Optional, List, Dict
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker
from video.layout_manager import LayoutManager

class FrameProcessor:
    def __init__(self, original_w: int, original_h: int, output_w: int, output_h: int, face_tracker: Optional[FaceTracker] = None, object_tracker: Optional[ObjectTracker] = None):
        self.original_w = original_w
        self.original_h = original_h
        self.output_w = output_w
        self.output_h = output_h
        self.face_tracker = face_tracker
        self.object_tracker = object_tracker
        self.layout_manager = LayoutManager()
        self.last_layout = None
        self.transition_progress = -1

    def process_frame(self, get_frame: Callable[[float], np.ndarray], t: float, layout_info: dict, engagement_metrics: Optional[List[Dict]] = None) -> np.ndarray:
        frame = get_frame(t).copy()
        
        current_layout = layout_info.get('recommended_layout', 'single_person_focus')
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
