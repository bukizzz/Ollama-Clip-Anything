from agents.base_agent import Agent
from core.state_manager import set_stage_status
import librosa
import numpy as np # Import numpy
from llm import llm_interaction
from pydantic import BaseModel, Field
import os # Import os for path checking

class MusicSyncAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("MusicSyncAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.music_integration_config = agent_config.get('music_integration', {})
        # Conceptual: music library with metadata
        # For now, using placeholder paths. In a real scenario, these would be actual audio files.
        self.music_library = {
            "upbeat_electronic": {"path": "path/to/upbeat.mp3", "tempo": 128},
            "ambient_chill": {"path": "path/to/ambient.mp3", "tempo": 80},
            "cinematic_action": {"path": "path/to/action.mp3", "tempo": 140}
        }

    def execute(self, context):
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        # If music sync results already exist in the context, skip this stage.
        if context.get('music_sync_results') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Music synchronization already complete.")
            return context

        # Updated paths to retrieve data from the hierarchical context
        audio_analysis_results = context.get('current_analysis', {}).get('audio_analysis_results', {})
        audio_rhythm = audio_analysis_results.get('audio_rhythm', {})

        if not audio_analysis_results or not audio_rhythm:
            self.log_error("Audio analysis or rhythm data missing. Cannot sync music.")
            set_stage_status('music_sync', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("🎵 Starting music synchronization...")
        set_stage_status('music_sync', 'running')

        try:
            # 1. Select music based on mood
            class MusicGenre(BaseModel):
                genre: str = Field(description="The chosen music genre from the provided list.")

            # Access sentiment from the correct hierarchical path
            sentiment = audio_analysis_results.get('sentiment_analysis', {}).get('label', 'NEUTRAL').lower()
            prompt = f"Choose a music genre for a video with a '{sentiment}' mood from this list: {list(self.music_library.keys())}."
            
            genre_selection = llm_interaction.robust_llm_json_extraction(
                system_prompt="You are an expert music selector. Your task is to choose the most appropriate music genre based on the video's mood.",
                user_prompt=prompt,
                output_schema=MusicGenre
            )
            genre = genre_selection.genre
            
            selected_track = self.music_library.get(genre)
            if not selected_track:
                self.log_warning(f"Genre '{genre}' not found in library, using default 'ambient_chill'.")
                selected_track = self.music_library['ambient_chill']

            music_beat_times = np.array([]) # Initialize as empty numpy array

            # Check if the music file path is valid before proceeding with librosa
            if not os.path.exists(selected_track['path']):
                self.log_warning(f"Music file not found at {selected_track['path']}. Skipping beat tracking.")
            else:
                # Load audio file
                y, sr = librosa.load(selected_track['path'])
                
                # 2. Match tempo (conceptual, not fully implemented here)
                # video_tempo = audio_rhythm.get('tempo', 120)
                # music_tempo = selected_track['tempo']
                # Conceptual: could implement time-stretching here if tempos don't match
                
                # 3. Synchronize beats
                # librosa.beat.beat_track returns a tuple (tempo, beat_frames)
                # We need the beat_frames, which is the second element [1]
                _, music_beats_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
                
                # Ensure music_beats_frames is a numpy array before calling .astype(int)
                if not isinstance(music_beats_frames, np.ndarray):
                    music_beats_frames = np.array(music_beats_frames)
                
                music_beat_times = librosa.frames_to_time(music_beats_frames.astype(int), sr=sr)

            # 4. Volume adjustment (conceptual, to be used in editing)
            volume_automation = [
                {'time': 0, 'level': 0.1}, # Start low
                {'time': 2, 'level': 0.3}, # Fade in
                {'time': -2, 'level': 0.1} # Fade out at the end
            ]

            context['music_sync_results'] = {
                'track_path': selected_track['path'],
                'beat_times': music_beat_times.tolist(), # .tolist() is now safe
                'volume_automation': volume_automation
            }
            print(f"✅ Music synchronization complete. Selected genre: {genre}")
            set_stage_status('music_integration_complete', 'complete', {'genre': genre})
            return context

        except Exception as e:
            self.log_error(f"Error during music synchronization: {e}")
            set_stage_status('music_sync', 'failed', {'reason': str(e)})
            return context
