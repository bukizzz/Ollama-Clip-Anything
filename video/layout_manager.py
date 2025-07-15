import cv2
import numpy as np
from typing import Optional, List, Dict

import cv2
import numpy as np
from typing import Optional, List, Dict
import time # Import time for current timestamp

class LayoutManager:
    def __init__(self, output_w: int, output_h: int):
        self.output_w = output_w
        self.output_h = output_h
        self.templates = {
            "single_person_focus": self._apply_single_person_focus,
            "multi_person_grid": self._apply_multi_person_grid,
            "presentation_with_speaker": self._apply_presentation_with_speaker,
            "two_person_split_screen": self._apply_two_person_split_screen,
        }
        self._current_zoom_level = 1.0 # Current actual zoom level
        self._target_zoom_level = 1.0  # Desired zoom level
        self._zoom_transition_start_time = 0.0 # When the current zoom transition started
        self._zoom_transition_duration = 0.25 # seconds for zoom transition (very fast: 0.25s)
        self._last_zoom_event_end_time = 0.0 # Timestamp when the last zoom-in/out finished
        self._zoom_cooldown_duration = 5.0 # seconds for zoom cooldown after any zoom event

        # Pre-calculated zoom events will be set externally
        self.zoom_events: List[Dict] = [] # Expected format: [{'start_time': X, 'end_time': Y, 'zoom_level': Z}]

    def set_zoom_events(self, events: List[Dict]):
        self.zoom_events = sorted(events, key=lambda x: x['start_time'])

    def apply_layout(self, frame: np.ndarray, layout_type: str, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str] = None, engagement_metrics: Optional[List[Dict]] = None, frame_timestamp_in_seconds: float = 0.0) -> np.ndarray:
        """Applies the specified layout to the frame."""
        layout_function = self.templates.get(layout_type, self._center_crop_and_resize)
        # Pass zoom_events to the layout function if needed, or handle internally
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
            
            # Determine target zoom based on pre-calculated zoom events and cooldown
            new_target_zoom = 1.0 # Default to no zoom

            # Check for active zoom event
            active_event = None
            for event in self.zoom_events:
                if event['start_time'] <= frame_timestamp_in_seconds < event['end_time']:
                    active_event = event
                    break
            
            # Apply zoom if an active event is found and not in cooldown
            if active_event and (frame_timestamp_in_seconds - self._last_zoom_event_end_time) >= self._zoom_cooldown_duration:
                new_target_zoom = active_event['zoom_level']
            else:
                # If no active event or in cooldown, ensure we zoom out to base level
                new_target_zoom = 1.0

            # Update target zoom and transition start time if target changes
            if new_target_zoom != self._target_zoom_level:
                self._target_zoom_level = new_target_zoom
                self._zoom_transition_start_time = frame_timestamp_in_seconds
                # If we are transitioning to 1.0 (zoom out), record the end time for cooldown
                if self._target_zoom_level == 1.0:
                    self._last_zoom_event_end_time = frame_timestamp_in_seconds + self._zoom_transition_duration # Cooldown starts after transition

            # Smoothly interpolate current zoom level towards target zoom level
            elapsed_time = frame_timestamp_in_seconds - self._zoom_transition_start_time
            if elapsed_time < self._zoom_transition_duration:
                progress = elapsed_time / self._zoom_transition_duration
                # Use linear interpolation for simplicity and speed
                self._current_zoom_level = self._current_zoom_level + (self._target_zoom_level - self._current_zoom_level) * progress
            else:
                self._current_zoom_level = self._target_zoom_level # Snap to target if transition finished

            zoom_factor = self._current_zoom_level

            # Calculate crop dimensions based on the zoom factor
            crop_w = int(output_w / zoom_factor)
            crop_h = int(output_h / zoom_factor)

            # Ensure crop dimensions are at least 1 pixel to avoid errors
            crop_w = max(1, crop_w)
            crop_h = max(1, crop_h)
            
            # Calculate crop coordinates, ensuring they stay within frame boundaries
            crop_x1 = max(0, min(frame.shape[1] - crop_w, center_x - crop_w // 2))
            crop_y1 = max(0, min(frame.shape[0] - crop_h, center_y - crop_h // 2))
            
            # Ensure crop_x2 and crop_y2 do not exceed frame dimensions
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h

            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Handle cases where cropped frame might be empty due to invalid dimensions
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                return self._center_crop_and_resize(frame, output_w, output_h) # Fallback

            return cv2.resize(cropped, (output_w, output_h))
        
        # If no target face is found, or face_tracker is None, fall back to center crop
        # Reset zoom state if no face is detected to prevent lingering zoom
        self._target_zoom_level = 1.0
        self._zoom_transition_start_time = frame_timestamp_in_seconds # Reset transition start
        self._current_zoom_level = 1.0 # Immediately snap to base zoom
        self._last_zoom_event_end_time = frame_timestamp_in_seconds # Reset cooldown
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

    def _apply_presentation_with_speaker(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str], engagement_metrics: Optional[List[Dict]], frame_timestamp_in_seconds: float) -> np.ndarray:
        # Temporarily disabled for debugging
        return self._center_crop_and_resize(frame, output_w, output_h)

    def _apply_two_person_split_screen(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str], engagement_metrics: Optional[List[Dict]], frame_timestamp_in_seconds: float) -> np.ndarray:
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        
        if len(faces) == 2:
            # Sort faces by their x-coordinate to ensure consistent left/right placement
            faces = sorted(faces, key=lambda f: f['bbox'][0])

            # Calculate dimensions for each half of the split screen
            half_w = output_w // 2
            
            # Create an empty canvas for the split screen
            split_screen_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)

            for i, face in enumerate(faces):
                x, y, w, h = face['bbox']
                
                # Extract the region of interest around the face
                # Add some padding around the face for better framing
                padding_x = int(w * 0.2)
                padding_y = int(h * 0.2)
                
                crop_x1 = max(0, x - padding_x)
                crop_y1 = max(0, y - padding_y)
                crop_x2 = min(frame.shape[1], x + w + padding_x)
                crop_y2 = min(frame.shape[0], y + h + padding_y)
                
                face_roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # Resize the face ROI to fit into half of the output screen
                resized_face_roi = cv2.resize(face_roi, (half_w, output_h))

                # Place the resized face ROI into the split screen frame
                if i == 0: # Left side
                    split_screen_frame[:, :half_w] = resized_face_roi
                else: # Right side
                    split_screen_frame[:, half_w:] = resized_face_roi
            
            return split_screen_frame
        else:
            # Fallback if not exactly two faces, or if face_tracker is not available
            return self._center_crop_and_resize(frame, output_w, output_h)
