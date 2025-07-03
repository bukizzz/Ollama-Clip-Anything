from agents.base_agent import Agent
from core.state_manager import set_stage_status
import librosa
from llm import llm_interaction

class MusicSyncAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("MusicSyncAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.music_integration_config = agent_config.get('music_integration', {})
        # Conceptual: music library with metadata
        self.music_library = {
            "upbeat_electronic": {"path": "path/to/upbeat.mp3", "tempo": 128},
            "ambient_chill": {"path": "path/to/ambient.mp3", "tempo": 80},
            "cinematic_action": {"path": "path/to/action.mp3", "tempo": 140}
        }

    def execute(self, context):
        audio_analysis = context.get('audio_analysis_results', {})
        audio_rhythm = context.get('audio_rhythm_data', {})

        if not audio_analysis or not audio_rhythm:
            self.log_error("Audio analysis or rhythm data missing. Cannot sync music.")
            set_stage_status('music_sync', 'failed', {'reason': 'Missing dependencies'})
            return context

        self.log_info("Starting music synchronization...")
        set_stage_status('music_sync', 'running')

        try:
            # 1. Select music based on mood
            sentiment = audio_analysis.get('sentiment', {}).get('label', 'NEUTRAL').lower()
            prompt = f"Choose a music genre for a video with a '{sentiment}' mood from this list: {list(self.music_library.keys())}. Respond with JSON: {{\"genre\": \"...\"}}"
            response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [{"role": "user", "content": prompt}])
            genre = llm_interaction.extract_json_from_text(response).get('genre', 'ambient_chill')
            
            selected_track = self.music_library.get(genre)
            if not selected_track:
                self.log_warning(f"Genre '{genre}' not found in library, using default.")
                selected_track = self.music_library['ambient_chill']

            # 2. Match tempo
            # video_tempo = audio_rhythm.get('tempo', 120)
            # music_tempo = selected_track['tempo']
            # Conceptual: could implement time-stretching here if tempos don't match
            
            # 3. Synchronize beats
            music_beats, _ = librosa.beat.beat_track(path=selected_track['path'])
            music_beat_times = librosa.frames_to_time(music_beats, sr=librosa.get_samplerate(selected_track['path']))

            # 4. Volume adjustment (conceptual, to be used in editing)
            volume_automation = [
                {'time': 0, 'level': 0.1}, # Start low
                {'time': 2, 'level': 0.3}, # Fade in
                {'time': -2, 'level': 0.1} # Fade out at the end
            ]

            context['music_sync_results'] = {
                'track_path': selected_track['path'],
                'beat_times': music_beat_times.tolist(),
                'volume_automation': volume_automation
            }
            self.log_info(f"Music synchronization complete. Selected genre: {genre}")
            set_stage_status('music_integration_complete', 'complete', {'genre': genre})
            return True

        except Exception as e:
            self.log_error(f"Error during music synchronization: {e}")
            set_stage_status('music_sync', 'failed', {'reason': str(e)})
            return context
