import librosa
import numpy as np
from agents.base_agent import Agent
from core.state_manager import set_stage_status

class AudioRhythmAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("AudioRhythmAgent")
        self.config = config
        self.state_manager = state_manager
        self.audio_rhythm_config = config.get('audio_rhythm', {})

    def execute(self, context):
        audio_path = context.get('audio_path')
        if not audio_path:
            self.log_error("Audio path not found in context.")
            set_stage_status('audio_rhythm_analysis', 'failed', {'reason': 'Audio path missing'})
            return context

        self.log_info(f"Starting audio rhythm analysis for {audio_path}")
        set_stage_status('audio_rhythm_analysis', 'running')

        try:
            y, sr = librosa.load(audio_path)

            # Extract audio tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
            self.log_info(f"Detected tempo: {tempo:.2f} BPM")

            # Detect beat positions
            beat_times = librosa.frames_to_time(beats, sr=sr)
            self.log_info(f"Detected {len(beat_times)} beats.")

            # Analyze speech rhythm patterns (emphasis detection)
            rms = librosa.feature.rms(y=y)[0]
            emphasis_threshold = float(np.mean(rms) + self.audio_rhythm_config.get('emphasis_threshold_std', 1.5) * np.std(rms))
            emphasized_segments = np.where(rms > emphasis_threshold)[0]
            
            # Convert frame indices to time
            emphasized_times = librosa.frames_to_time(emphasized_segments, sr=sr)
            
            # Generate rhythm map for dynamic editing synchronization
            rhythm_map = {
                'tempo': float(tempo),
                'beat_times': beat_times.tolist(),
                'emphasized_times': emphasized_times.tolist(),
                'emphasis_threshold': emphasis_threshold
            }

            context['audio_rhythm_data'] = rhythm_map
            self.log_info("Audio rhythm analysis complete.")
            set_stage_status('audio_rhythm_analysis', 'complete', {'tempo': tempo, 'beats_detected': len(beat_times), 'emphasized_segments': len(emphasized_times), 'emphasis_threshold': float(emphasis_threshold)})
            return context

        except Exception as e:
            self.log_error(f"Error during audio rhythm analysis: {e}")
            set_stage_status('audio_rhythm_analysis', 'failed', {'reason': str(e)})
            return context
