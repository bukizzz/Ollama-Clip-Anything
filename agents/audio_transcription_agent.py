from agents.base_agent import Agent
from typing import Dict, Any
from audio import audio_processing
from agents.audio_rhythm_agent import AudioRhythmAgent
from agents.audio_analysis_agent import AudioAnalysisAgent
from core.config import load_config
from core.state_manager import set_stage_status, get_stage_status

class AudioTranscriptionAgent(Agent):
    """Agent responsible for transcribing the video's audio and performing rhythm analysis."""

    def __init__(self, state_manager):
        super().__init__("AudioTranscriptionAgent")
        self.config = load_config()
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        transcription = context.get("transcription")

        print("\n🎤 \u001b[94m2. Transcribing video...\u001b[0m")
        
        if processed_video_path is None:
            raise RuntimeError("Processed video path is missing from context. Cannot transcribe.")

        if transcription is None: # Only transcribe if not already loaded from state
            transcription = audio_processing.transcribe_video(processed_video_path)
            if not transcription:
                raise RuntimeError("Transcription failed. Video may have no audio.")
            context.update({
                "transcription": transcription,
                "current_stage": "transcription_complete", # Update stage after successful transcription
                "temp_files": context.get("temp_files", {}).update({"transcription": "transcription_data_in_state"}) # Placeholder for actual transcription file if saved
            })
            
            # Analyze transcript with LLM
            print("Analyzing transcript with LLM...")
            llm_analysis = audio_processing.analyze_transcript_with_llm(transcription)
            context.update({"llm_transcript_analysis": llm_analysis})
        else:
            print("⏩ Skipping transcription. Loaded from state.")
            
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        # Run AudioRhythmAgent
        audio_rhythm_agent = AudioRhythmAgent(self.config, self.state_manager)
        audio_rhythm_agent.run(context)

        # Run AudioAnalysisAgent
        audio_analysis_agent = AudioAnalysisAgent(self.config, self.state_manager)
        audio_analysis_agent.run(context)

        return context

