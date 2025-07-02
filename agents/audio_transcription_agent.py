from agents.base_agent import Agent
from typing import Dict, Any
from audio import audio_processing

class AudioTranscriptionAgent(Agent):
    """Agent responsible for transcribing the video's audio."""

    def __init__(self):
        super().__init__("AudioTranscriptionAgent")

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
        
        return context
