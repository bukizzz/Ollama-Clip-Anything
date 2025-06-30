
from agents.base_agent import Agent
from typing import Dict, Any
from audio import audio_processing

class AudioTranscriptionAgent(Agent):
    """Agent responsible for transcribing the video's audio."""

    def __init__(self):
        super().__init__("AudioTranscriptionAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        current_stage = context.get("current_stage")
        transcription = context.get("transcription")

        print("\n2. Transcribing video...")
        if current_stage == "video_input_complete":
            transcription = audio_processing.transcribe_video(processed_video_path)
            if not transcription:
                raise RuntimeError("Transcription failed. Video may have no audio.")
            context.update({
                "transcription": transcription,
                "current_stage": "transcription_complete",
                "temp_files": context.get("temp_files", {}).update({"transcription": "transcription_data_in_state"}) # Placeholder for actual transcription file if saved
            })
        else:
            print("⏩ Skipping transcription. Loaded from state.")
            
        print(f"✅ Transcription complete: {len(transcription)} segments found.")
        
        return context
