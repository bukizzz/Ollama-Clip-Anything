import cv2
import numpy as np
from typing import Optional

class LayoutManager:
    def __init__(self):
        pass

    def apply_layout(self, frame: np.ndarray, layout_type: str, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str] = None) -> np.ndarray:
        """Applies the specified layout to the frame."""
        if layout_type == "single_person_focus":
            return self._apply_single_person_focus(frame, output_w, output_h, face_tracker, active_speaker_id)
        elif layout_type == "multi_person_grid":
            return self._apply_multi_person_grid(frame, output_w, output_h, face_tracker, active_speaker_id)
        elif layout_type == "presentation_with_speaker":
            return self._apply_presentation_with_speaker(frame, output_w, output_h, face_tracker, object_tracker, active_speaker_id)
        else:
            # Default to simple center crop if layout type is unknown
            return self._center_crop_and_resize(frame, output_w, output_h)

    def _center_crop_and_resize(self, frame: np.ndarray, output_w: int, output_h: int) -> np.ndarray:
        h, w, _ = frame.shape
        start_x = max(0, (w - output_w) // 2)
        start_y = max(0, (h - output_h) // 2)
        cropped_frame = frame[start_y:start_y + output_h, start_x:start_x + output_w]
        return cv2.resize(cropped_frame, (output_w, output_h))

    def _apply_single_person_focus(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, active_speaker_id: Optional[str]) -> np.ndarray:
        """Focuses on a single person, ideally the active speaker."""
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        target_face = None

        if active_speaker_id:
            for face in faces:
                if face.get('id') == active_speaker_id:
                    target_face = face
                    break
        
        if not target_face and faces:
            target_face = max(faces, key=lambda f: f.get('area', 0)) # Largest face if no active speaker

        if target_face:
            x, y, w, h = target_face['bbox']
            center_x, center_y = x + w // 2, y + h // 2

            # Calculate crop region around the face
            crop_width = output_w
            crop_height = output_h

            # Ensure the face is within the cropped area
            crop_x1 = max(0, center_x - crop_width // 2)
            crop_y1 = max(0, center_y - crop_height // 2)

            crop_x1 = min(crop_x1, frame.shape[1] - crop_width)
            crop_y1 = min(crop_y1, frame.shape[0] - crop_height)

            cropped_frame = frame[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]
            return cv2.resize(cropped_frame, (output_w, output_h))
        else:
            return self._center_crop_and_resize(frame, output_w, output_h)

    def _apply_multi_person_grid(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, active_speaker_id: Optional[str]) -> np.ndarray:
        """Arranges multiple people in a grid layout."""
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        num_faces = len(faces)

        if num_faces == 0:
            return self._center_crop_and_resize(frame, output_w, output_h)

        # Simple grid: 2x2 for up to 4 faces, then 3x3, etc.
        grid_size = int(np.ceil(np.sqrt(num_faces)))
        cell_w = output_w // grid_size
        cell_h = output_h // grid_size

        output_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for i, face in enumerate(faces):
            row = i // grid_size
            col = i % grid_size

            x, y, w, h = face['bbox']
            cropped_face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(cropped_face, (cell_w, cell_h))

            output_frame[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized_face

            # Highlight active speaker
            if active_speaker_id and face.get('id') == active_speaker_id:
                cv2.rectangle(output_frame, (col * cell_w, row * cell_h), ((col + 1) * cell_w, (row + 1) * cell_h), (0, 255, 0), 5)

        return output_frame

    def _apply_presentation_with_speaker(self, frame: np.ndarray, output_w: int, output_h: int, face_tracker, object_tracker, active_speaker_id: Optional[str]) -> np.ndarray:
        """Layout for presentation content with a speaker inset."""
        # Main area for presentation (e.g., 70% width, 100% height)
        presentation_w = int(output_w * 0.7)
        presentation_h = output_h

        # Speaker inset area (e.g., 30% width, 30% height, bottom right)
        speaker_w = output_w - presentation_w
        speaker_h = int(output_h * 0.3)

        # Resize main frame for presentation area (simple resize for now)
        resized_presentation = cv2.resize(frame, (presentation_w, presentation_h))

        # Extract speaker frame
        faces = face_tracker.detect_faces_in_frame(frame) if face_tracker else []
        speaker_frame = None
        if active_speaker_id:
            for face in faces:
                if face.get('id') == active_speaker_id:
                    x, y, w, h = face['bbox']
                    speaker_frame = frame[y:y+h, x:x+w]
                    break
        if speaker_frame is None and faces:
            speaker_frame = frame[faces[0]['bbox'][1]:faces[0]['bbox'][1]+faces[0]['bbox'][3], faces[0]['bbox'][0]:faces[0]['bbox'][0]+faces[0]['bbox'][2]]
        
        if speaker_frame is None:
            speaker_frame = np.zeros((speaker_h, speaker_w, 3), dtype=np.uint8) # Black frame if no speaker
        else:
            speaker_frame = cv2.resize(speaker_frame, (speaker_w, speaker_h))

        # Create final frame
        final_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        final_frame[0:presentation_h, 0:presentation_w] = resized_presentation
        final_frame[output_h - speaker_h:output_h, output_w - speaker_w:output_w] = speaker_frame

        return final_frame
