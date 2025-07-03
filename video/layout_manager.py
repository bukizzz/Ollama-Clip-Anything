import cv2
import numpy as np
from typing import Optional, List, Dict

class LayoutManager:
    def __init__(self):
        self.templates = {
            "single_person_focus": self._apply_single_person_focus,
            "multi_person_grid": self._apply_multi_person_grid,
            "presentation_with_speaker": self._apply_presentation_with_speaker,
        }

    def apply_layout(self, frame: np.ndarray, layout_type: str, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str] = None, engagement_metrics: Optional[List[Dict]] = None, frame_timestamp_in_seconds: float = 0.0) -> np.ndarray:
        """Applies the specified layout to the frame."""
        layout_function = self.templates.get(layout_type, self._center_crop_and_resize)
        return layout_function(frame, output_w, output_h, face_tracker, object_tracker, active_speaker_id, engagement_metrics, frame_timestamp_in_seconds)

    def _center_crop_and_resize(self, frame: np.ndarray, output_w: int, output_h: int, *args) -> np.ndarray:
        h, w, _ = frame.shape
        start_x = max(0, (w - output_w) // 2)
        start_y = max(0, (h - output_h) // 2)
        cropped_frame = frame[start_y:start_y + output_h, start_x:start_x + output_w]
        return cv2.resize(cropped_frame, (output_w, output_h))

    def _apply_single_person_focus(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str], engagement_metrics: Optional[List[Dict]], frame_timestamp_in_seconds: float) -> np.ndarray:
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        target_face = next((f for f in faces if f.get('id') == active_speaker_id), None) or (faces[0] if faces else None)

        if target_face:
            x, y, w, h = target_face['bbox']
            center_x, center_y = x + w // 2, y + h // 2
            
            # Engagement-driven zoom
            zoom = 1.0
            if engagement_metrics:
                current_engagement = next((m['score'] for m in engagement_metrics if abs(m['timestamp'] - frame_timestamp_in_seconds) < 0.1), 0)
                zoom += current_engagement * 0.2 # Zoom up to 20% based on engagement

            crop_w = int(output_w / zoom)
            crop_h = int(output_h / zoom)
            
            crop_x1 = max(0, center_x - crop_w // 2)
            crop_y1 = max(0, center_y - crop_h // 2)
            
            cropped = frame[crop_y1:crop_y1+crop_h, crop_x1:crop_x1+crop_w]
            return cv2.resize(cropped, (output_w, output_h))
        return self._center_crop_and_resize(frame, output_w, output_h)

    def _apply_multi_person_grid(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str], engagement_metrics: Optional[List[Dict]]) -> np.ndarray:
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        if not faces:
            return self._center_crop_and_resize(frame, output_w, output_h)

        grid_size = int(np.ceil(np.sqrt(len(faces))))
        cell_w, cell_h = output_w // grid_size, output_h // grid_size
        output_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for i, face in enumerate(faces):
            row, col = i // grid_size, i % grid_size
            x, y, w, h = face['bbox']
            face_img = cv2.resize(frame[y:y+h, x:x+w], (cell_w, cell_h))
            
            if face.get('id') == active_speaker_id:
                cv2.rectangle(face_img, (0,0), (cell_w, cell_h), (0,255,0), 5)
            
            output_frame[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = face_img
        return output_frame

    def _detect_screen_content(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return frame[y:y+h, x:x+w]
        return None

    def _apply_presentation_with_speaker(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str], engagement_metrics: Optional[List[Dict]]) -> np.ndarray:
        screen_area = self._detect_screen_content(frame)
        if screen_area is None:
            screen_area = frame
        
        speaker_inset = np.zeros((output_h // 4, output_w // 4, 3), dtype=np.uint8)
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        speaker_face = next((f for f in faces if f.get('id') == active_speaker_id), None) or (faces[0] if faces else None)
        
        if speaker_face:
            x,y,w,h = speaker_face['bbox']
            speaker_inset = cv2.resize(frame[y:y+h, x:x+w], (output_w // 4, output_h // 4))

        main_content = cv2.resize(screen_area, (output_w, output_h))
        main_content[-speaker_inset.shape[0]:, -speaker_inset.shape[1]:] = speaker_inset
        return main_content
