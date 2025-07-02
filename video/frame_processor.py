import cv2
import numpy as np
from typing import Callable, Optional
from core.config import SMOOTHING_FACTOR
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker

class FrameProcessor:
    def __init__(self, original_w: int, original_h: int, output_w: int, output_h: int, face_tracker: Optional[FaceTracker] = None, object_tracker: Optional[ObjectTracker] = None, split_screen_mode: bool = False, b_roll_image_path: Optional[str] = None, layout_manager: Optional['LayoutManager'] = None):
        self.original_w = original_w
        self.original_h = original_h
        self.output_w = output_w
        self.output_h = output_h
        self.face_tracker = face_tracker
        self.object_tracker = object_tracker
        self.smoothed_center_x = self.original_w / 2
        self.smoothed_center_y = self.original_h / 2
        self.split_screen_mode = split_screen_mode
        self.b_roll_image_path = b_roll_image_path
        self.layout_manager = layout_manager # New: LayoutManager instance

    def process_frame(self, get_frame: Callable[[float], np.ndarray], t: float, scene_changed: bool = False, active_speaker_id: Optional[str] = None, current_layout_recommendation: Optional[str] = None) -> np.ndarray:
        frame = get_frame(t).copy()
        current_w, current_h = frame.shape[1], frame.shape[0]

        # If LayoutManager is present, use it to determine the frame processing logic
        if self.layout_manager and current_layout_recommendation:
            final_frame = self.layout_manager.apply_layout(
                frame, 
                current_layout_recommendation, 
                self.output_w, 
                self.output_h, 
                self.face_tracker, 
                self.object_tracker, 
                active_speaker_id=active_speaker_id
            )
            return final_frame

        # Existing logic for dynamic framing, split-screen, and b-roll
        center_x, center_y = current_w / 2, current_h / 2
        main_face = None
        faces = []
        if self.face_tracker:
            faces = self.face_tracker.detect_faces_in_frame(frame)
            if faces:
                main_face = max(faces, key=lambda f: f.get('area', 0))
                center_x, center_y = main_face['center']

        if scene_changed:
            self.smoothed_center_x = center_x
            self.smoothed_center_y = center_y
        else:
            self.smoothed_center_x = self.smoothed_center_x * (1 - SMOOTHING_FACTOR) + center_x * SMOOTHING_FACTOR
            self.smoothed_center_y = self.smoothed_center_y * (1 - SMOOTHING_FACTOR) + center_y * SMOOTHING_FACTOR

        center_x = int(self.smoothed_center_x)
        center_y = int(self.smoothed_center_y)

        potential_crop_height = int(current_w / (self.output_w / self.output_h))

        if potential_crop_height <= current_h:
            crop_width = current_w
            crop_height = potential_crop_height
        else:
            crop_height = current_h
            crop_width = int(current_h * (self.output_w / self.output_h))

        if crop_width % 2 != 0:
            crop_width -= 1
        if crop_height % 2 != 0:
            crop_height -= 1

        if self.split_screen_mode and len(faces) >= 2:
            # Sort faces by y-coordinate to determine top and bottom
            faces.sort(key=lambda f: f['center'][1])
            top_face = faces[0]
            bottom_face = faces[1]

            # Calculate crop regions for top and bottom faces
            # Top half for top_face
            top_crop_y1 = int(top_face['center'][1] - (current_h / 4))
            top_crop_y1 = max(0, min(top_crop_y1, current_h // 2 - crop_height // 2))
            top_crop_x1 = int(top_face['center'][0] - crop_width // 2)
            top_crop_x1 = max(0, min(top_crop_x1, current_w - crop_width))

            # Bottom half for bottom_face
            bottom_crop_y1 = int(bottom_face['center'][1] - (current_h / 4))
            bottom_crop_y1 = max(current_h // 2, min(bottom_crop_y1, current_h - crop_height // 2))
            bottom_crop_x1 = int(bottom_face['center'][0] - crop_width // 2)
            bottom_crop_x1 = max(0, min(bottom_crop_x1, current_w - crop_width))

            # Create two cropped frames
            cropped_top = frame[top_crop_y1:top_crop_y1 + crop_height // 2, top_crop_x1:top_crop_x1 + crop_width]
            cropped_bottom = frame[bottom_crop_y1:bottom_crop_y1 + crop_height // 2, bottom_crop_x1:bottom_crop_x1 + crop_width]

            # Resize to output dimensions
            resized_top = cv2.resize(cropped_top, (self.output_w, self.output_h // 2))
            resized_bottom = cv2.resize(cropped_bottom, (self.output_w, self.output_h // 2))

            # Combine into a single frame
            final_frame = np.vstack((resized_top, resized_bottom))

        elif self.b_roll_image_path:
            # B-roll rendering logic
            # Top half for main video content
            crop_x1 = int(self.smoothed_center_x - crop_width // 2)
            crop_y1 = int(self.smoothed_center_y - crop_height // 2)

            if main_face:
                face_x, face_y, face_w, face_h = main_face['bbox']
                if face_x < crop_x1:
                    crop_x1 = face_x
                if face_x + face_w > crop_x1 + crop_width:
                    crop_x1 = (face_x + face_w) - crop_width
                if face_y < crop_y1:
                    crop_y1 = face_y
                if face_y + face_h > crop_y1 + crop_height:
                    crop_y1 = (face_y + face_h) - crop_height

            crop_x1 = max(0, min(crop_x1, current_w - crop_width))
            crop_y1 = max(0, min(crop_y1, current_h - crop_height))

            cropped_main_video = frame[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]
            resized_main_video = cv2.resize(cropped_main_video, (self.output_w, self.output_h // 2))

            # Bottom half for B-roll image
            b_roll_image = cv2.imread(self.b_roll_image_path)
            if b_roll_image is None:
                print(f"⚠️ Warning: Could not load B-roll image from {self.b_roll_image_path}. Using black frame.")
                b_roll_image = np.zeros((self.output_h // 2, self.output_w, 3), dtype=np.uint8)
            else:
                b_roll_image = cv2.resize(b_roll_image, (self.output_w, self.output_h // 2))
            
            final_frame = np.vstack((resized_main_video, b_roll_image))

        else:
            # Existing single-person cropping logic
            # Calculate initial crop based on smoothed center
            crop_x1 = int(self.smoothed_center_x - crop_width // 2)
            crop_y1 = int(self.smoothed_center_y - crop_height // 2)

            # Adjust crop to ensure main face is within bounds
            if main_face:
                face_x, face_y, face_w, face_h = main_face['bbox']
                
                # Ensure face is not cut off on the left
                if face_x < crop_x1:
                    crop_x1 = face_x
                # Ensure face is not cut off on the right
                if face_x + face_w > crop_x1 + crop_width:
                    crop_x1 = (face_x + face_w) - crop_width
                
                # Ensure face is not cut off on the top
                if face_y < crop_y1:
                    crop_y1 = face_y
                # Ensure face is not cut off on the bottom
                if face_y + face_h > crop_y1 + crop_height:
                    crop_y1 = (face_y + face_h) - crop_height

            # Ensure crop stays within original frame boundaries
            crop_x1 = max(0, min(crop_x1, current_w - crop_width))
            crop_y1 = max(0, min(crop_y1, current_h - crop_height))

            cropped_frame = frame[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]
            
            final_frame = cv2.resize(cropped_frame, (self.output_w, self.output_h))
        
        if self.face_tracker and faces:
            # Active speaker highlighting and speaker labels
            for face in faces:
                x, y, w, h = face['bbox']
                if active_speaker_id and face.get('id') == active_speaker_id: # Assuming face has an 'id'
                    # Draw a green rectangle around the active speaker
                    cv2.rectangle(final_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Add speaker label
                    cv2.putText(final_frame, f"Speaker {active_speaker_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    # Draw a subtle grey rectangle for non-active speakers
                    cv2.rectangle(final_frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        if self.object_tracker:
            objects = self.object_tracker.detect_objects_in_frame(frame)
            for obj in objects:
                pass

        return final_frame
