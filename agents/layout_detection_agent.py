import cv2
from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import LAYOUT_DETECTION_CONFIG

class LayoutDetectionAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.layout_config = LAYOUT_DETECTION_CONFIG
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def run(self, context):
        video_path = context.get('processed_video_path')
        if not video_path or not cv2.CascadeClassifier().empty():
            self.log_error("Processed video path not found in context or face cascade not loaded.")
            set_stage_status('layout_detection', 'failed', {'reason': 'Missing video path or cascade'})
            return False

        self.log_info("Starting layout detection...")
        set_stage_status('layout_detection', 'running')

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            layout_analysis_results = []
            prev_num_faces = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
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
            self.log_info("Layout detection complete.")
            set_stage_status('layout_detection_complete', 'complete', {'num_frames_analyzed': len(layout_analysis_results)})
            return True

        except Exception as e:
            self.log_error(f"Error during layout detection: {e}")
            set_stage_status('layout_detection', 'failed', {'reason': str(e)})
            return False
