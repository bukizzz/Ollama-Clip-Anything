import os
import cv2
from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.temp_manager import get_temp_path
from core.config import QWEN_VISION_CONFIG

class FramePreprocessingAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.qwen_vision_config = QWEN_VISION_CONFIG

    def run(self, context):
        video_path = context.get('processed_video_path')
        if not video_path or not os.path.exists(video_path):
            self.log_error("Processed video path not found in context or does not exist.")
            set_stage_status('frame_preprocessing', 'failed', {'reason': 'Missing or invalid video path'})
            return False

        self.log_info(f"Starting frame preprocessing for {video_path}")
        set_stage_status('frame_preprocessing', 'running')

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.qwen_vision_config.get('frame_extraction_rate_fps', 1)) # Default 1 fps
            frame_count = 0
            extracted_frames_info = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp_sec = timestamp_ms / 1000.0

                    # Resize frame if resolution settings are specified
                    resolution = self.qwen_vision_config.get('resolution_settings')
                    if resolution:
                        if resolution == "720p":
                            frame = cv2.resize(frame, (1280, 720))
                        elif resolution == "1080p":
                            frame = cv2.resize(frame, (1920, 1080))
                        # Add more resolutions as needed

                    frame_filename = get_temp_path(f"frame_{frame_count:07d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    extracted_frames_info.append({
                        'frame_path': frame_filename,
                        'timestamp_sec': timestamp_sec,
                        'frame_number': frame_count
                    })

                frame_count += 1

            cap.release()
            self.log_info(f"Extracted {len(extracted_frames_info)} frames.")
            context['extracted_frames_info'] = extracted_frames_info
            set_stage_status('frame_feature_extraction_complete', 'complete', {'num_frames': len(extracted_frames_info)})
            return True

        except Exception as e:
            self.log_error(f"Error during frame preprocessing: {e}")
            set_stage_status('frame_preprocessing', 'failed', {'reason': str(e)})
            return False
