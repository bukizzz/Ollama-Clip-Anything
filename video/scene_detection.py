import cv2
from typing import List

class SceneDetector:
    """Advanced scene change detection using multiple methods"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.prev_hist = None
        self.prev_features = None
        
    def detect_scene_changes(self, video_path: str, sample_rate: int = 1) -> List[float]:
        """Detect scene changes using histogram comparison and feature analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        scene_changes = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if self.prev_hist is not None:
                    # Calculate histogram correlation
                    correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    if correlation < (1 - self.threshold):
                        scene_changes.append(timestamp)
                        print(f"Scene change detected at {timestamp:.2f}s (correlation: {correlation:.3f})")
                
                self.prev_hist = hist
            
            frame_count += 1
        
        cap.release()
        return scene_changes
