from agents.base_agent import Agent
from typing import Dict, Any
from audio import audio_processing
from core.config import config
from core.temp_manager import get_temp_path

class AudioTranscriptionAgent(Agent):
    """Agent responsible for transcribing the video's audio and performing rhythm analysis."""

    def __init__(self, state_manager):
        super().__init__("AudioTranscriptionAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        transcription = context.get("transcription")

        print("\n🎤 \u001b[94m2. Transcribing video...\u001b[0m")
        
        if processed_video_path is None:
            raise RuntimeError("Processed video path is missing from context. Cannot transcribe.")

        # Extract audio and get path
        audio_path = get_temp_path("temp_audio_normalized.wav")
        if not context.get("audio_path"):
            audio_processing.extract_audio(processed_video_path, audio_path)
            context["audio_path"] = audio_path

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
            print("🧠 \u001b[94mAnalyzing transcript with LLM...\u001b[0m")
            llm_analysis = audio_processing.analyze_transcript_with_llm(transcription)
            context.update({"llm_transcript_analysis": llm_analysis})
        else:
            print("⏩ Skipping transcription. Loaded from state.")
            
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        return context

