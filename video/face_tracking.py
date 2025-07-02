import cv2
import mediapipe as mp
from typing import List, Dict, Tuple, Any

from core.llm_models import query_image_embedding # Import query_image_embedding


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class FaceTracker:
    """Advanced face tracking with MediaPipe and visual annotations"""
    
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)
        self.tracked_faces = {} # This was a placeholder, now it will store actual tracked data
        self.active_trackers: Dict[int, Tuple[Any, List[int]]] = {} # Stores {face_id: (tracker_instance, last_bbox)}
        self.next_face_id = 0
        self.face_db = self._load_face_db()
        self.image_embedding_model = query_image_embedding # Assign the new function

    def _get_face_db_path(self):
        return "./video/face_db.json"

    def _load_face_db(self):
        import json
        try:
            with open(self._get_face_db_path(), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            print("⚠️ Warning: face_db.json is corrupted. Starting with empty database.")
            return {}

    def _save_face_db(self):
        import json
        with open(self._get_face_db_path(), 'w') as f:
            json.dump(self.face_db, f, indent=4)

    def save_face_embedding(self, name: str, embedding: List[float]):
        """Saves a face embedding to the database."""
        self.face_db[name] = embedding
        self._save_face_db()
        print(f"💾 \033[92mSaved embedding for face: {name}\033[0m")

    def load_face_db(self) -> Dict[str, List[float]]:
        """Loads the face database."""
        return self.face_db

    def cleanup(self):
        """Release resources held by the face tracker."""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        # print("Face detection and mesh models released.")
        
    def detect_faces_in_frame(self, frame: cv2.typing.MatLike) -> List[Dict]:
        """Detect faces and return detailed information, incorporating persistent tracking."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Update existing trackers
        updated_active_trackers = {}
        current_frame_tracked_faces = []
        for face_id, (tracker, last_bbox) in self.active_trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                updated_active_trackers[face_id] = (tracker, (x, y, w, h))
                current_frame_tracked_faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0, # Assume high confidence if tracked
                    'center': (x + w // 2, y + h // 2),
                    'area': w * h,
                    'id': face_id # Add ID for persistent tracking
                })
            # else: tracker failed, face is lost for now

        self.active_trackers = updated_active_trackers

        # 2. Run MediaPipe detection for new faces or re-acquiring lost ones
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            h, w_frame = frame.shape[:2]
            for detection in results.detections:
                bbox_mp = detection.location_data.relative_bounding_box
                x_mp = int(bbox_mp.xmin * w_frame)
                y_mp = int(bbox_mp.ymin * h)
                width_mp = int(bbox_mp.width * w_frame)
                height_mp = int(bbox_mp.height * h)
                
                new_detection_bbox = (x_mp, y_mp, width_mp, height_mp)

                # Check for overlap with already tracked faces
                is_already_tracked = False
                for tracked_face in current_frame_tracked_faces:
                    tx, ty, tw, th = tracked_face['bbox']
                    # Simple overlap check (could be improved with IoU)
                    if not (x_mp > tx + tw or x_mp + width_mp < tx or y_mp > ty + th or y_mp + height_mp < ty):
                        is_already_tracked = True
                        break
                
                if not is_already_tracked:
                    # This is a new face or a re-acquired lost face
                    new_tracker = cv2.legacy.TrackerCSRT_create()
                    new_tracker.init(frame, new_detection_bbox)
                    self.active_trackers[self.next_face_id] = (new_tracker, new_detection_bbox)
                    current_frame_tracked_faces.append({
                        'bbox': new_detection_bbox,
                        'confidence': detection.score[0],
                        'center': (x_mp + width_mp // 2, y_mp + height_mp // 2),
                        'area': width_mp * height_mp,
                        'id': self.next_face_id
                    })
                    self.next_face_id += 1
        
        return current_frame_tracked_faces
