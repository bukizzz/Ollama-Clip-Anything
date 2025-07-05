import cv2
from typing import List

class SceneDetector:
    """Advanced scene change detection using multiple methods"""
    
    def __init__(self, config):
        self.config = config
        self.threshold = self.config.get('scene_detection.threshold', 0.3)
        self.min_scene_duration = self.config.get('scene_detection.min_scene_duration', 2.0)
        self.prev_hist = None
        self.prev_features = None
        
    def detect_scene_changes(self, video_path: str) -> List[float]:
        """Detect scene changes using histogram comparison and feature analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        raw_scene_changes = []
        frame_count = 0
        sample_rate = self.config.get('scene_detection.sample_rate', 1)
        
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
                        raw_scene_changes.append(timestamp)
                        print(f"🎬 \033[93mScene change detected at {timestamp:.2f}s (correlation: {correlation:.3f})\033[0m")
                
                self.prev_hist = hist
            
            frame_count += 1
        
        cap.release()

        # Filter scene changes: if multiple within 1 second, use only one
        filtered_scene_changes = []
        if raw_scene_changes:
            filtered_scene_changes.append(raw_scene_changes[0])
            for i in range(1, len(raw_scene_changes)):
                if raw_scene_changes[i] - filtered_scene_changes[-1] > self.min_scene_duration: # Configurable threshold
                    filtered_scene_changes.append(raw_scene_changes[i])
        
        return filtered_scene_changes
