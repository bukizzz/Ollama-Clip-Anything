import cv2
import mediapipe as mp
from typing import List, Dict

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class FaceTracker:
    """Advanced face tracking with MediaPipe and visual annotations"""
    
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)
        self.tracked_faces = {}
        # Placeholder for a database to store character images for recognition
        self.face_db = self._load_face_db()
        # Placeholder for an image embedding model (e.g., ImageBind) for better recognition
        self.image_embedding_model = None # Initialize as None, to be loaded if needed

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
            print("Warning: face_db.json is corrupted. Starting with empty database.")
            return {}

    def _save_face_db(self):
        import json
        with open(self._get_face_db_path(), 'w') as f:
            json.dump(self.face_db, f, indent=4)

    def save_face_embedding(self, name: str, embedding: List[float]):
        """Saves a face embedding to the database."""
        self.face_db[name] = embedding
        self._save_face_db()
        print(f"Saved embedding for face: {name}")

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
        """Detect faces and return detailed information"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Get key points for more detailed tracking
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        keypoints.append((int(keypoint.x * w), int(keypoint.y * h)))
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'center': (x + width // 2, y + height // 2),
                    'keypoints': keypoints,
                    'area': width * height
                })
        
        return faces
