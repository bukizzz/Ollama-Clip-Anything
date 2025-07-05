import cv2
import numpy as np # Added numpy import
from agents.base_agent import Agent
from core.state_manager import set_stage_status

class LayoutDetectionAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("LayoutDetectionAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.layout_config = agent_config.get('layout_detection', {})
        
        # Replaced CascadeClassifier with YuNet DNN model
        self.face_detector = cv2.dnn.readNetFromONNX("weights/face_detection_yunet_2023mar.onnx")
        self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.input_size = (320, 320) # YuNet input size, can be adjusted

    def execute(self, context):
        video_path = context.get('processed_video_path')
        # Modified check to use face_detector instead of face_cascade
        if not video_path or self.face_detector.empty():
            self.log_error("Processed video path not found in context or face detector not loaded.")
            set_stage_status('layout_detection', 'failed', {'reason': 'Missing video path or detector'})
            return context

        print("🔍 Starting layout detection...")
        set_stage_status('layout_detection', 'running')

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            layout_analysis_results = []
            prev_num_faces = 0
            confidence_threshold = 0.9 # Confidence threshold for face detection

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                # Preprocess frame for YuNet
                blob = cv2.dnn.blobFromImage(frame, 1.0, self.input_size, (0, 0, 0), swapRB=True, crop=False)
                self.face_detector.setInput(blob)

                # Inference
                detections = self.face_detector.forward()

                # Post-process detections
                faces = []
                # detections is a 1x1xNx15 array, where N is the number of detections
                # Each detection is [batch_id, class_id, score, x1, y1, x2, y2, ...]
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        # Extract bounding box coordinates (x1, y1, x2, y2)
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append([x1, y1, x2 - x1, y2 - y1]) # Convert to x, y, w, h format if needed

                num_faces = len(faces)

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Detect screen sharing (simple heuristic: if no faces and high visual complexity/static content)
                # This is a very basic placeholder and would need more sophisticated logic
                is_screen_share = False
                if num_faces == 0 and timestamp_sec > 5: # After initial few seconds
                    # More advanced screen share detection would involve analyzing image entropy, text detection, etc.
                    is_screen_share = True 

                # Classify layout types
                layout_type = "unknown"
                if num_faces == 0 and is_screen_share:
                    layout_type = "presentation_mode"
                elif num_faces == 1:
                    layout_type = "single_person"
                elif num_faces > 1:
                    layout_type = "multi_person"
                
                # Detect speaker transitions and layout changes
                layout_change = False
                if num_faces != prev_num_faces:
                    layout_change = True

                layout_analysis_results.append({
                    'timestamp': timestamp_sec,
                    'num_faces': num_faces,
                    'is_screen_share': is_screen_share,
                    'layout_type': layout_type,
                    'layout_change': layout_change
                })
                prev_num_faces = num_faces

            cap.release()
            context['layout_detection_results'] = layout_analysis_results
            print("✅ Layout detection complete.")
            set_stage_status('layout_detection_complete', 'complete', {'num_frames_analyzed': len(layout_analysis_results)})
            return context

        except Exception as e:
            self.log_error(f"Error during layout detection: {e}")
            set_stage_status('layout_detection', 'failed', {'reason': str(e)})
            return context
