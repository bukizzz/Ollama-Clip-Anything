import cv2
import numpy as np
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
        self.input_size = (320, 320) # YuNet input size, can be adjusted

        # Force CPU backend to avoid CUDA assertion errors
        self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.log_info("Using CPU backend for layout detection.")


    def execute(self, context):
        video_path = context.get('processed_video_path')
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

            layout_segments = []
            current_layout_type = None
            current_num_faces = -1
            segment_start_time = 0.0
            confidence_threshold = 0.9 # Confidence threshold for face detection

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 1.0, self.input_size, (0, 0, 0), swapRB=True, crop=False)
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()

                faces = []
                for i in range(detections.shape[1]):
                    confidence = detections[0, i, 2]
                    if confidence > confidence_threshold:
                        x1 = int(detections[0, i, 3] * w)
                        y1 = int(detections[0, i, 4] * h)
                        x2 = int(detections[0, i, 5] * w)
                        y2 = int(detections[0, i, 6] * h)
                        faces.append([x1, y1, x2 - x1, y2 - y1])

                num_faces = len(faces)
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                is_screen_share = False
                if num_faces == 0 and timestamp_sec > 5:
                    is_screen_share = True 

                layout_type = "unknown"
                if num_faces == 0 and is_screen_share:
                    layout_type = "presentation_mode"
                elif num_faces == 1:
                    layout_type = "single_person"
                elif num_faces > 1:
                    layout_type = "multi_person"
                
                # Check for layout change
                if layout_type != current_layout_type or num_faces != current_num_faces:
                    if current_layout_type is not None: # Not the very first frame
                        layout_segments.append({
                            'start_time': segment_start_time,
                            'end_time': timestamp_sec, # End at the current frame's timestamp
                            'layout_type': current_layout_type,
                            'num_faces': current_num_faces
                        })
                    # Start new segment
                    segment_start_time = timestamp_sec
                    current_layout_type = layout_type
                    current_num_faces = num_faces
                
                frame_count += 1

            # Add the last segment after the loop finishes
            if current_layout_type is not None:
                layout_segments.append({
                    'start_time': segment_start_time,
                    'end_time': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, # End at video's total duration
                    'layout_type': current_layout_type,
                    'num_faces': current_num_faces
                })

            cap.release()
            context['layout_detection_results'] = layout_segments
            print("✅ Layout detection complete.")
            set_stage_status('layout_detection_complete', 'complete', {'num_layout_segments': len(layout_segments)})
            return context

        except Exception as e:
            self.log_error(f"Error during layout detection: {e}")
            set_stage_status('layout_detection', 'failed', {'reason': str(e)})
            return context
