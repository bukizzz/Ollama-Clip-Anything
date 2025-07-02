from agents.base_agent import Agent
from typing import Dict, Any
from llm import llm_interaction

class LLMSelectionAgent(Agent):
    """Agent responsible for selecting engaging clips using an LLM."""

    def __init__(self):
        super().__init__("LLMSelectionAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        # current_stage = context.get("current_stage") # Not needed for this logic
        clips = context.get("clips")

        print("\n🧠 \u001b[94m3. Selecting coherent clips using LLM...\u001b[0m")
        
        if clips is None: # Only select clips if not already loaded from state
            if transcription is None:
                raise RuntimeError("Transcription data is missing from context. Cannot select clips.")
            
            user_prompt = context.get("user_prompt")
            b_roll_data = context.get("b_roll_data")
            clips = llm_interaction.get_clips_from_llm(transcription, user_prompt=user_prompt, b_roll_data=b_roll_data)
            context.update({
                "clips": clips,
                "current_stage": "llm_selection_complete" # Update stage after successful clip selection
            })
            
        else:
            print("⏩ Skipping LLM clip selection. Loaded from state.")
            
        print(f"✅ Selected {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")
        
        return context
