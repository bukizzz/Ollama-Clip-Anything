import librosa
import numpy as np
from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status

class AudioRhythmAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.audio_rhythm_config = config.get('audio_rhythm', {})

    def run(self, context):
        audio_path = context.get('audio_path')
        if not audio_path:
            self.log_error("Audio path not found in context.")
            set_stage_status('audio_rhythm_analysis', 'failed', {'reason': 'Audio path missing'})
            return False

        self.log_info(f"Starting audio rhythm analysis for {audio_path}")
        set_stage_status('audio_rhythm_analysis', 'running')

        try:
            y, sr = librosa.load(audio_path)

            # Extract audio tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            self.log_info(f"Detected tempo: {tempo:.2f} BPM")

            # Detect beat positions
            onset_env = librosa.onset.onset_detect(y=y, sr=sr)
            beats = librosa.beat.beat_track(onset_env=onset_env, sr=sr)[1]
            beat_times = librosa.frames_to_time(beats, sr=sr)
            self.log_info(f"Detected {len(beat_times)} beats.")

            # Placeholder for speech rhythm patterns (pause detection, emphasis detection)
            # This would involve more advanced audio analysis, potentially using a separate model
            # For now, we'll just store basic rhythm data.
            rhythm_data = {
                'tempo': tempo,
                'beat_times': beat_times.tolist(),
                # Add more rhythm analysis data here as implemented
            }

            context['audio_rhythm_data'] = rhythm_data
            self.log_info("Audio rhythm analysis complete.")
            set_stage_status('audio_rhythm_analysis', 'complete', {'tempo': tempo, 'beats_detected': len(beat_times)})
            return True

        except Exception as e:
            self.log_error(f"Error during audio rhythm analysis: {e}")
            set_stage_status('audio_rhythm_analysis', 'failed', {'reason': str(e)})
            return False
