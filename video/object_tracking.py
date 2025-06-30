import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
from typing import List, Dict
import math
from moviepy.editor import VideoClip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ObjectTracker:
    """Enhanced object tracking with visual annotations"""
    
    def __init__(self):
        self.model = None
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracked_objects = {}
        self.initialize_model()
        
        # COCO class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Colors for different object classes
        self.class_colors = {
            'person': (255, 0, 0),
            'car': (0, 255, 0),
            'dog': (255, 255, 0),
            'cat': (255, 0, 255),
            'bicycle': (0, 255, 255),
        }

    def initialize_model(self):
        try:
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(device)
            self.model.eval()
            print("Object detection model loaded successfully")
        except Exception as e:
            print(f"Could not load object detection model: {e}")

    def detect_objects_in_frame(self, frame: cv2.typing.MatLike, confidence_threshold: float = 0.7) -> List[Dict]:
        if self.model is None:
            return []

        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            img_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            detected_objects = []
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.coco_names[label] if label < len(self.coco_names) else 'unknown'
                    detected_objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(score),
                        'class': class_name,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1)
                    })

            return detected_objects

        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def create_object_annotation_clip(self, frame_array: cv2.typing.MatLike, objects: List[Dict], duration: float) -> VideoClip:
        """Create animated annotations for detected objects"""
        def make_frame(t):
            frame = frame_array.copy()
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                class_name = obj['class']
                confidence = obj['confidence']
                
                # Get color for this class
                color = self.class_colors.get(class_name, (128, 128, 128))
                
                # Animated bounding box
                animation_phase = (t * 3) % (2 * math.pi)
                thickness = 2 + int(2 * math.sin(animation_phase))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label background
                label_text = f"{class_name}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                center_x, center_y = obj['center']
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            return frame
        
        return VideoClip(make_frame, duration=duration)

    def cleanup(self):
        """Release resources held by the object detection model."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Object detection model unloaded and GPU memory released.")
