import os
import cv2
from agents.base_agent import Agent
from core.state_manager import set_stage_status
from collections import defaultdict
from core.cache_manager import cache_manager # Import cache_manager
from core.resource_manager import resource_manager # Import resource_manager

class LayoutSpeakerAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("LayoutSpeakerAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.layout_config = agent_config.get('layout_detection', {})
        
        # YuNet model for face detection
        # Construct absolute path to the ONNX file
        script_dir = os.path.dirname(__file__)
        onnx_model_path = os.path.join(script_dir, "..", "weights", "face_detection_yunet_2023mar.onnx")
        
        self.face_detector = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.input_size = (320, 320) # YuNet input size, can be adjusted

        # Force CPU backend to avoid CUDA assertion errors
        self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.log_info("Using CPU backend for layout detection.")

    def execute(self, context):
        video_path = context.get('processed_video_path')
        audio_analysis = context.get('current_analysis', {}).get('audio_analysis_results', {})
        video_analysis = context.get('current_analysis', {}).get('multimodal_analysis_results', {})

        diarization = audio_analysis.get('speaker_diarization')
        faces_over_time = video_analysis.get('facial_expressions') # This contains timestamp and expression, not direct face info

        if not video_path or self.face_detector.empty():
            self.log_error("Processed video path not found or face detector not loaded.")
            set_stage_status('layout_speaker_analysis', 'failed', {'reason': 'Missing video path or detector'})
            return context

        print("🔍🗣️ Starting layout and speaker analysis...")
        set_stage_status('layout_speaker_analysis', 'running')

        cache_key = f"layout_speaker_{os.path.basename(video_path)}"
        cached_results = cache_manager.get(cache_key, level="disk")

        if cached_results:
            print("⏩ Skipping layout and speaker analysis. Loaded from cache.")
            context.update(cached_results)
            set_stage_status('layout_speaker_analysis_complete', 'complete', {'loaded_from_cache': True})
            return context

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
            context['current_analysis']['layout_detection_results'] = layout_segments
            print("✅ Layout detection complete.")

            # Speaker Tracking Logic (from SpeakerTrackingAgent)
            speaker_tracking_results = {}
            if diarization:
                print("🗣️ Performing speaker tracking...")
                speaker_map = defaultdict(list)
                
                time_to_num_faces = {}
                for layout_seg in layout_segments:
                    mid_time = (layout_seg['start_time'] + layout_seg['end_time']) / 2
                    time_to_num_faces[mid_time] = layout_seg['num_faces']

                sorted_layout_segments = sorted(layout_segments, key=lambda x: x['start_time'])

                for segment in diarization:
                    start, end = segment['start'], segment['end']
                    # Ensure speaker_label is always a string, default to "Unknown Speaker" if None
                    speaker_label = str(segment.get('speaker', "Unknown Speaker"))
                    
                    face_present_in_speaker_segment = False
                    for layout_seg in sorted_layout_segments:
                        if max(start, layout_seg['start_time']) < min(end, layout_seg['end_time']):
                            if layout_seg['num_faces'] > 0:
                                face_present_in_speaker_segment = True
                                break

                    if face_present_in_speaker_segment:
                        speaker_map[speaker_label].append(True)

                final_speaker_to_face = {k: True for k, v in speaker_map.items() if v}

                speaker_profiles = {}
                transitions = []
                last_speaker = None
                for segment in sorted(diarization, key=lambda x: x['start']):
                    # Ensure speaker_label is always a string here too
                    speaker_label = str(segment.get('speaker', "Unknown Speaker"))
                    if speaker_label != last_speaker and last_speaker is not None:
                        transitions.append({'timestamp': segment['start'], 'from': last_speaker, 'to': speaker_label})
                    last_speaker = speaker_label

                    if speaker_label not in speaker_profiles:
                        speaker_profiles[speaker_label] = {'visual_profile': {}}

                speaker_tracking_results = {
                    "speaker_to_face_map": final_speaker_to_face,
                    "speaker_profiles": speaker_profiles,
                    "speaker_transitions": transitions
                }
                context['current_analysis']['speaker_tracking_results'] = speaker_tracking_results
                print("✅ Speaker tracking complete.")
            else:
                self.log_warning("Diarization results missing. Skipping speaker tracking.")
                context['current_analysis']['speaker_tracking_results'] = {}

            set_stage_status('layout_speaker_analysis_complete', 'complete', {
                'num_layout_segments': len(layout_segments),
                'num_speakers_tracked': len(speaker_tracking_results.get('speaker_to_face_map', {}))
            })
            
            # Cache the results before returning
            cache_manager.set(cache_key, {
                'current_analysis': context.get('current_analysis')
            }, level="disk")

            return context

        except Exception as e:
            self.log_error(f"Error during layout and speaker analysis: {e}")
            set_stage_status('layout_speaker_analysis', 'failed', {'reason': str(e)})
            resource_manager.unload_all_models() # Unload models on error
            return context
