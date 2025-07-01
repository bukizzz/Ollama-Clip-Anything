from agents.base_agent import Agent
from typing import Dict, Any
from llm import llm_interaction
import json

class ContentAlignmentAgent(Agent):
    """Agent responsible for synchronizing audio and video elements."""

    def __init__(self):
        super().__init__("ContentAlignmentAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        storyboard_data = context.get("storyboard_data")

        print("ContentAlignmentAgent: Synchronizing audio and video elements...")

        if not transcription or not storyboard_data:
            print("Skipping content alignment: transcription or storyboard_data not available.")
            context["content_alignment_data"] = "N/A"
            return context

        # Prepare data for LLM
        simplified_transcript = [{
            "start": round(seg['start'], 1),
            "end": round(seg['end'], 1),
            "text": seg['text']
        } for seg in transcription]

        simplified_storyboard = [{
            "timestamp": round(sb['timestamp'], 1),
            "description": sb['description']
        } for sb in storyboard_data]

        llm_prompt = f"""
        Given the following video transcript and storyboard data, identify key moments where the spoken content aligns with visual scene changes or significant visual elements.
        Provide a list of these aligned moments, including the timestamp, a brief description of the visual, and the corresponding spoken text.

        Transcript:
        {json.dumps(simplified_transcript, indent=2)}

        Storyboard:
        # {json.dumps(simplified_storyboard, indent=2)}

        Return the output as a JSON array of objects, each with "timestamp", "visual_description", and "spoken_text".
        """

        try:
            response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [
                {"role": "system", "content": "You are an expert in video content analysis and synchronization."},
                {"role": "user", "content": llm_prompt.strip()}
            ])
            
            alignment_results = llm_interaction.extract_json_from_text(response)
            context["content_alignment_data"] = alignment_results
            print("✅ Content alignment by LLM complete.")
        except Exception as e:
            print(f"Failed to perform content alignment with LLM: {e}")
            context["content_alignment_data"] = "Error during alignment."

        return context