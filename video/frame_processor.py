import cv2
import numpy as np
from typing import Callable
from core.config import SMOOTHING_FACTOR
from video.face_tracking import FaceTracker
from video.object_tracking import ObjectTracker

class FrameProcessor:
    def __init__(self, original_w: int, original_h: int, output_w: int, output_h: int, face_tracker: FaceTracker = None, object_tracker: ObjectTracker = None):
        self.original_w = original_w
        self.original_h = original_h
        self.output_w = output_w
        self.output_h = output_h
        self.face_tracker = face_tracker
        self.object_tracker = object_tracker
        self.smoothed_center_x = self.original_w / 2
        self.smoothed_center_y = self.original_h / 2

    def process_frame(self, get_frame: Callable[[float], np.ndarray], t: float) -> np.ndarray:
        frame = get_frame(t).copy()
        current_w, current_h = frame.shape[1], frame.shape[0]

        center_x, center_y = current_w / 2, current_h / 2
        main_face = None
        faces = []
        if self.face_tracker:
            faces = self.face_tracker.detect_faces_in_frame(frame)
            if faces:
                main_face = max(faces, key=lambda f: f.get('area', 0))
                center_x, center_y = main_face['center']

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

        if self.face_tracker and faces:
            pass
        
        if self.object_tracker:
            objects = self.object_tracker.detect_objects_in_frame(frame)
            for obj in objects:
                pass

        cropped_frame = frame[crop_y1:crop_y1 + crop_height, crop_x1:crop_x1 + crop_width]
        
        final_frame = cv2.resize(cropped_frame, (self.output_w, self.output_h))
        
        return final_frame
