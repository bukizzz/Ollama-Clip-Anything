from agents.base_agent import Agent
from core.state_manager import set_stage_status
from video.face_tracking import FaceTracker
from collections import defaultdict

class SpeakerTrackingAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("SpeakerTrackingAgent")
        self.config = config
        self.state_manager = state_manager
        self.face_tracker = FaceTracker()

    def execute(self, context):
        audio_analysis = context.get('audio_analysis_results', {})
        video_analysis = context.get('video_analysis_results', {})
        
        diarization = audio_analysis.get('speaker_diarization')
        faces_over_time = video_analysis.get('facial_expressions')

        if not diarization or not faces_over_time:
            self.log_error("Diarization or face tracking data missing. Cannot perform speaker tracking.")
            set_stage_status('speaker_tracking', 'failed', {'reason': 'Missing dependencies'})
            return context

        self.log_info("Starting speaker tracking...")
        set_stage_status('speaker_tracking', 'running')

        try:
            # This is a complex task. A robust solution would involve voiceprints and face embeddings.
            # Here's a simplified approach:
            speaker_map = defaultdict(list)
            for segment in diarization:
                start, end, speaker_label = segment['start'], segment['end'], segment['speaker']
                
                # Find faces that appear during this speech segment
                active_faces = [f for f in faces_over_time if f['timestamp'] >= start and f['timestamp'] <= end and f.get('has_face')]
                
                if active_faces:
                    # Simplistic: assume the most prominent face is the speaker
                    # A better approach would use face embeddings to re-identify speakers
                    face_ids = [face.get('id', -1) for face in active_faces]
                    if face_ids:
                        most_common_face_id = max(set(face_ids), key=face_ids.count)
                        speaker_map[speaker_label].append(most_common_face_id)

            # Consolidate mappings
            final_speaker_to_face = {k: max(set(v), key=v.count) for k, v in speaker_map.items() if v}

            # Generate speaker profiles and transition timestamps
            speaker_profiles = {}
            transitions = []
            last_speaker = None
            for segment in sorted(diarization, key=lambda x: x['start']):
                speaker_label = segment['speaker']
                if speaker_label != last_speaker and last_speaker is not None:
                    transitions.append({'timestamp': segment['start'], 'from': last_speaker, 'to': speaker_label})
                last_speaker = speaker_label

                if speaker_label not in speaker_profiles:
                    face_id = final_speaker_to_face.get(speaker_label)
                    # Conceptual: get visual profile (e.g., average bbox) from face tracking data
                    speaker_profiles[speaker_label] = {'face_id': face_id, 'visual_profile': {}}

            speaker_tracking_results = {
                "speaker_to_face_map": final_speaker_to_face,
                "speaker_profiles": speaker_profiles,
                "speaker_transitions": transitions
            }

            context['speaker_tracking_results'] = speaker_tracking_results
            self.log_info("Speaker tracking complete.")
            set_stage_status('speaker_tracking_complete', 'complete', {'num_speakers': len(final_speaker_to_face)})
            return True

        except Exception as e:
            self.log_error(f"Error during speaker tracking: {e}")
            set_stage_status('speaker_tracking', 'failed', {'reason': str(e)})
            return context
